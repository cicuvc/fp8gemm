"""CUDA FP8 linear backward with independently quantized X^T as dW's B.

The two GEMMs and layout/bias kernels are CUDA. The standalone transpose +
row/K32 activation quantizer uses Triton, as selected by the prior benchmark.
Forward weights retain their original 32x32 scales for dX. No input activation
cache is required. dX uses swapAB: original weights are transposed by LDSM into
the native A operand, while dY is quantized online as B. Only an incompatible
weight row stride needs a non-transposing row-padding copy. All required
preparation is included in each backward call.

dW reads the same original-orientation BF16 dY buffer via transposing LDSM,
then quantizes per output-feature/M32. The default no longer materializes dY^T.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import subprocess

import torch
import triton
import triton.language as tl
import mxtest
from experiments.tk_persistent_gemm import ROOT, FLAGS, CONFIGS as GEMM_CONFIGS, GemmLibrary, build as build_gemm, DEFAULT_OUTPUT as GEMM_DIR
from experiments.tk_dual_online_gemm import DualOnlineLibrary, build as build_dual, DEFAULT_OUTPUT as DUAL_DIR

OUTPUT = ROOT/'runs/tk_linear_backward'


def build(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-shared','-Xcompiler=-fPIC',
                        'cuda/linear_backward_prepare.cu','-o',str(output/'libprepare.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)


@lru_cache(None)
def libraries():
    # Explicit warm-up before CUDA graph capture also compiles missing binaries.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm up the CUDA backward once before CUDA graph capture')
    headers=list((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))
    def stale(binary,sources):
        return not binary.exists() or any(s.stat().st_mtime>binary.stat().st_mtime for s in sources)
    if stale(GEMM_DIR/'libgemm.so',[ROOT/'cuda/persistent_fp8_gemm.cu',ROOT/'cuda/tma_ring_pipe.cuh',*headers]):build_gemm()
    if stale(DUAL_DIR/'libdual.so',[ROOT/'cuda/persistent_dual_online_gemm.cu',ROOT/'cuda/tma_ring_pipe.cuh',*headers]):build_dual()
    if stale(OUTPUT/'libprepare.so',[ROOT/'cuda/linear_backward_prepare.cu']):build()
    lib=ct.CDLL(str(OUTPUT/'libprepare.so'))
    lib.linear_prepare_dy.argtypes=[ct.c_void_p]*4+[ct.c_int]*4+[ct.c_int64]*2+[ct.c_int,ct.c_void_p]
    lib.linear_transpose_weight.argtypes=[ct.c_void_p]*4+[ct.c_int]*4+[ct.c_void_p]
    lib.linear_pad_weight.argtypes=[ct.c_void_p]*2+[ct.c_int]*3+[ct.c_void_p]
    lib.linear_finish.argtypes=[ct.c_void_p]*6+[ct.c_int]*8+[ct.c_void_p]
    for name in ['linear_prepare_dy','linear_transpose_weight','linear_pad_weight','linear_finish']:
        getattr(lib,name).restype=ct.c_int
    return lib,GemmLibrary(),DualOnlineLibrary()


@triton.jit
def _quantize_xt(X,Q,S,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,
                 S0:tl.constexpr,S1:tl.constexpr):
    mi=tl.program_id(0)*64+tl.arange(0,64)
    ki=tl.program_id(1)*64+tl.arange(0,64)
    x=tl.load(X+mi[:,None]*S0+ki[None,:]*S1,(mi[:,None]<M)&(ki[None,:]<K),0)
    q,s=mxtest.online_scale(tl.trans(x))
    tl.store(Q+ki[:,None]*MP+mi[None,:],q)
    groups=tl.program_id(0)*2+tl.arange(0,2)
    tl.store(S+ki[:,None]*(MP//32)+groups[None,:],s)


def _check(error):
    if error:raise RuntimeError(f'CUDA backward preparation error {error}')


class LinearBackwardPlan:
    """Reusable buffers; call the plan after changing dy/x/weights in place.

    Positive M,N,K, CUDA BF16 x/dy, contiguous E4M3 weight[N,K] and uint8
    block scales[ceil(N/32),ceil(K/32)]. Strided x/dy are accepted. SM120 only.
    Returns BF16 dX, FP32 dW, and FP32 db (empty when bias=False).
    transpose_weight=True retains the previous dX path for paired measurements.
    swap_ab=False retains direct AB with FP8 W as B. By default swapAB is used
    whenever transpose_weight=False; all three paths return the same layout.
    dw_native_dy=True (default) avoids materializing dY^T by using BF16 transposing
    LDSM; dw_swap_ab=True also exchanges dW's MMA operands (experimental).
    dw_raster_group groups dW output-row tiles for L2 reuse; 1 is row-major.
    Keep the plan alive until its GPU work finishes, including graph replay.
    """
    def __init__(self,dy,x,w,ws,bias=True,dx_config=None,dw_config=None,*,transpose_weight=False,
                 swap_ab=None,dw_native_dy=True,dw_swap_ab=False,dw_raster_group=1):
        if x.ndim!=2 or dy.ndim!=2 or w.ndim!=2 or dy.shape[0]!=x.shape[0] or w.shape!=(dy.shape[1],x.shape[1]):
            raise ValueError('Expected x[M,K], dy[M,N], weight[N,K]')
        m,k=x.shape;n=dy.shape[1]
        if min(m,n,k)<=0:raise ValueError('Dimensions must be positive')
        if x.dtype!=torch.bfloat16 or dy.dtype!=torch.bfloat16 or w.dtype!=torch.float8_e4m3fn:
            raise TypeError('Expected BF16 x/dy and E4M3 weight')
        if not x.is_cuda or any(t.device!=x.device for t in [dy,w,ws]):raise ValueError('Expected one CUDA device')
        if torch.cuda.get_device_capability(x.device)!=(12,0):raise RuntimeError('CUDA WASP backward requires SM120')
        if ws.dtype!=torch.uint8 or ws.shape!=((n+31)//32,(k+31)//32) or not ws.is_contiguous() or not w.is_contiguous():
            raise ValueError('Expected contiguous FP8 weight and 32x32 block scales')
        self.dy,self.x,self.w,self.ws=dy,x,w,ws
        self.m,self.n,self.k=m,n,k
        self.mp,self.np,self.kp=mp,np,kp=tuple((v+63)//64*64 for v in (m,n,k))
        self.bias=bool(bias)
        self.transpose_weight=transpose_weight
        self.dw_native_dy=bool(dw_native_dy)
        if dw_swap_ab and not dw_native_dy:
            raise ValueError('dW swapAB experiment requires native dY')
        self.swap_ab=not transpose_weight if swap_ab is None else bool(swap_ab)
        if transpose_weight and self.swap_ab:
            raise ValueError('swapAB requires original weight orientation')
        def empty(shape,dtype):return torch.empty(shape,device=x.device,dtype=dtype)
        self.dyp=empty((m,np),torch.bfloat16)
        self.dyt=None if dw_native_dy else empty((np,mp),torch.bfloat16)
        if transpose_weight:
            self.wt=empty((kp,np),torch.float8_e4m3fn)
            self.wst=empty((kp//32,np//32),torch.uint8)
            self.dx_stride=kp
        else:
            self.dx_stride=(k+31)//32*32
            self.w_direct=w if k==self.dx_stride else empty((n,self.dx_stride),torch.float8_e4m3fn)
        self.xq=empty((kp,mp),torch.float8_e4m3fn)
        self.xs=empty((kp,mp//32),torch.uint8)
        self.partial=empty((mp//32,np) if bias else (0,),torch.float32)
        self.dx=empty((m,k),torch.bfloat16)
        self.dw=empty((n,k),torch.float32)
        self.db=empty((n,) if bias else (0,),torch.float32)
        self.dxp=self.dx if k==self.dx_stride else empty((m,self.dx_stride),torch.bfloat16)
        self.dwp=self.dw if (n,k)==(np,kp) else empty((np,kp),torch.float32)
        with torch.cuda.device(x.device):self.lib,g,d=libraries()
        # Conservative defaults from the existing tile sweeps; no timing inside autograd.
        if dx_config is None:
            dx_config=7 if m<=256 else 8
            if self.swap_ab:
                # Both choices use R16/BK64; C64 creates twice as many CTAs
                # when C128 cannot cover the device's SMs.
                tasks_c128=((m+127)//128)*((self.dx_stride+127)//128)
                sms=torch.cuda.get_device_properties(x.device).multi_processor_count
                dx_config=7 if tasks_c128<sms else 8
            if not transpose_weight and self.dx_stride%GEMM_CONFIGS[dx_config][1]:
                dx_config=7 if self.dx_stride%64==0 else 0
        dw_config=(0 if np<=256 and kp<=2048 else 3) if dw_config is None else dw_config
        if dw_config==7 and mp%128:raise ValueError('dw config7 requires padded batch divisible by128')
        if transpose_weight:
            self.dx_plan=g.plan(self.dyp,self.wt,dx_config,out=self.dxp,scales=self.wst)
        else:
            self.dx_plan=g.plan_ab(self.dyp,self.w_direct,ws,dx_config,out=self.dxp,swap_ab=self.swap_ab)
        if dw_native_dy:
            self.dw_plan=d.plan_native_dy(self.dyp,self.xq,self.xs,dw_config,out=self.dwp,
                                          swap_ab=dw_swap_ab,raster_group=dw_raster_group)
        else:
            self.dw_plan=d.plan(self.dyt,self.xq,dw_config,out=self.dwp,output_dtype=torch.float32,
                               b_scales=self.xs,raster_group=dw_raster_group)
        self.info={'shape_mnk':[m,n,k],'padded_mnk':[mp,np,kp], 'dx':self.dx_plan.info,'dw':self.dw_plan.info,
                   'bias':self.bias,'weight_transpose_each_call':transpose_weight,
                   'weight_row_padding':not transpose_weight and k!=self.dx_stride,
                   'weight_row_stride':self.dx_stride,'swap_ab':self.swap_ab,
                   'dy_transpose_each_call':not dw_native_dy,'dw_swap_ab':bool(dw_swap_ab)}

    def prepare_dy(self):
        _check(self.lib.linear_prepare_dy(self.dy.data_ptr(),self.dyp.data_ptr(),
            self.dyt.data_ptr() if self.dyt is not None else 0,self.partial.data_ptr(),
            self.m,self.n,self.mp,self.np,*self.dy.stride(),int(self.bias),torch.cuda.current_stream(self.x.device).cuda_stream))

    def prepare_weight(self):
        stream=torch.cuda.current_stream(self.x.device).cuda_stream
        if self.transpose_weight:
            _check(self.lib.linear_transpose_weight(self.w.data_ptr(),self.ws.data_ptr(),self.wt.data_ptr(),self.wst.data_ptr(),
                self.n,self.k,self.np,self.kp,stream))
        elif self.w_direct is not self.w:
            _check(self.lib.linear_pad_weight(self.w.data_ptr(),self.w_direct.data_ptr(),self.n,self.k,self.dx_stride,stream))

    def quantize_x(self):
        _quantize_xt[(self.mp//64,self.kp//64)](self.x,self.xq,self.xs,self.m,self.k,self.mp,*self.x.stride(),num_warps=4)

    def finish(self):
        _check(self.lib.linear_finish(self.dxp.data_ptr(),self.dwp.data_ptr(),self.partial.data_ptr(),
            self.dx.data_ptr(),self.dw.data_ptr(),self.db.data_ptr(),self.m,self.n,self.k,self.mp,self.np,self.kp,
            self.dx_stride,int(self.bias),torch.cuda.current_stream(self.x.device).cuda_stream))

    def __call__(self):
        with torch.cuda.device(self.x.device):
            self.prepare_dy()
            self.prepare_weight()
            self.dx_plan()
            self.quantize_x()
            self.dw_plan()
            self.finish()
        return self.dx,self.dw,self.db


def backward(dy,x,w,ws,bias=True):
    return LinearBackwardPlan(dy.to(torch.bfloat16),x,w,ws,bias)()


@torch.library.custom_op('fp8gemm::linear_cuda_bwd',mutates_args=())
def _cuda_bwd(dy:torch.Tensor,x:torch.Tensor,w:torch.Tensor,ws:torch.Tensor,bias:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return backward(dy,x,w,ws,bias)


@_cuda_bwd.register_fake
def _cuda_bwd_fake(dy,x,w,ws,bias):
    return (torch.empty_like(x,memory_format=torch.contiguous_format),
            torch.empty(w.shape,device=x.device,dtype=torch.float32),
            torch.empty((w.shape[0],) if bias else (0,),device=x.device,dtype=torch.float32))


@torch.library.custom_op('fp8gemm::linear_cuda_bwd_fwd',mutates_args=())
def linear(x:torch.Tensor,master_weight:torch.Tensor,w:torch.Tensor,ws:torch.Tensor,bias:torch.Tensor|None)->torch.Tensor:
    return mxtest.launch_fp8gemm(x,w,ws,bias)


@linear.register_fake
def _linear_fake(x,master_weight,w,ws,bias):
    return torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=torch.bfloat16)


def _setup(ctx,inputs,output):
    x,_,w,ws,bias=inputs
    ctx.save_for_backward(x,w,ws)
    ctx.bias=bias is not None


def _backward(ctx,dy):
    x,w,ws=ctx.saved_tensors
    dx,dw,db=_cuda_bwd(dy,x,w,ws,ctx.bias)
    return dx,dw,None,None,db if ctx.bias else None


torch.library.register_autograd(linear,_backward,setup_context=_setup)

if __name__=='__main__':
    build()
    libraries()
    print('CUDA linear backward libraries ready')
