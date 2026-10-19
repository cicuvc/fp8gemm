"""SM120a WASP fused gate/up/SwiGLU. Quantize weights with mxtest (Triton)."""
import ctypes as ct
from pathlib import Path
import subprocess
import torch
from experiments.tk_persistent_gemm import ROOT,FLAGS

DEFAULT_OUTPUT=ROOT/'runs/tk_persistent_swiglu'
CONFIGS=[(16,32,64,2),(16,64,64,2),(16,128,32,2),(16,128,64,2),
         (32,64,32,2),(32,32,64,2),(16,64,64,3),(16,32,128,2)]


def build(output=DEFAULT_OUTPUT,cuda=Path('/usr/local/cuda'),fused_silu=True):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'),*FLAGS,f'-DFP8_SWIGLU_FMA={int(fused_silu)}','-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_fp8_swiglu.cu','-lcuda','-o',str(output/'libswiglu.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)


class SwiGLULibrary:
    def __init__(self,output=DEFAULT_OUTPUT):
        self.lib=ct.CDLL(str(Path(output).resolve()/'libswiglu.so'))
        self.lib.tk_swiglu_create.argtypes=[ct.c_void_p]*8+[ct.c_int]*5
        self.lib.tk_swiglu_create.restype=ct.c_void_p
        self.lib.tk_swiglu_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_swiglu_launch.restype=ct.c_int
        self.lib.tk_swiglu_error.restype=ct.c_char_p
        self.lib.tk_swiglu_destroy.argtypes=[ct.c_void_p]
        self.lib.tk_swiglu_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]

    def plan(self,x,gate_weight,gate_scale,up_weight,up_scale,config=3,
             out=None,gate_out=None,up_out=None,sms=0):
        return SwiGLUPlan(self,x,gate_weight,gate_scale,up_weight,up_scale,
                          config,out,gate_out,up_out,sms)


class SwiGLUPlan:
    """Reusable, stream-aware forward plan, with optional BF16 backward auxiliaries.

    Inputs are contiguous CUDA BF16 X[M,K], E4M3 gate/up weights[N,K], and
    uint8 E8M0 scales[ceil(N/32),ceil(K/32)]. Weight quantization stays in Triton.
    N must be divisible by 64; K by 64 (128 for config 7). M may have tails.
    Matches Triton online-scale/SiLU conventions, including scale-zero clipping.
    By default the SiLU epilogue uses FMA; build(fused_silu=False) preserves
    Triton's add/multiply ordering for rounding-sensitive comparisons.
    This experimental entry point has no autograd registration. Allocations are
    retained by the plan and must remain alive until asynchronous work completes.
    """
    def __init__(self,owner,x,g,gs,u,us,config,out,go,uo,sms):
        self.handle=None
        if not 0<=config<len(CONFIGS) or x.ndim!=2 or g.ndim!=2 or u.shape!=g.shape:
            raise ValueError('Invalid configuration or matrix shapes')
        m,k=x.shape;n,gk=g.shape
        if min(m,n,k)<=0 or gk!=k or n%64 or k%(128 if config==7 else 64):
            raise ValueError('Expected positive M,N,K; N/K aligned to the selected TMA panels')
        for tensor,dtype,shape in [(x,torch.bfloat16,(m,k)),(g,torch.float8_e4m3fn,(n,k)),
                                    (u,torch.float8_e4m3fn,(n,k)),(gs,torch.uint8,((n+31)//32,(k+31)//32)),
                                    (us,torch.uint8,((n+31)//32,(k+31)//32))]:
            if tensor.dtype!=dtype or tensor.shape!=shape or not tensor.is_cuda or tensor.device!=x.device or not tensor.is_contiguous():
                raise ValueError('Invalid input dtype, shape, device or strides')
        if (go is None)!=(uo is None):
            raise ValueError('gate_out and up_out must be supplied together')
        if out is None:out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
        for tensor in [out]+([] if go is None else [go,uo]):
            if tensor.shape!=(m,n) or tensor.dtype!=torch.bfloat16 or tensor.device!=x.device or not tensor.is_contiguous():
                raise ValueError('Invalid output buffer')
        if any(t.data_ptr()%16 for t in [x,g,u,out]+([] if go is None else [go,uo])):
            raise ValueError('TMA tensor base pointers must be 16-byte aligned')
        # Output aliasing can race across persistent CTAs or overwrite auxiliaries.
        tensors=[x,g,gs,u,us];outputs=[out]+([] if go is None else [go,uo])
        for i,tensor in enumerate(outputs):
            lo=tensor.data_ptr();hi=lo+tensor.numel()*tensor.element_size()
            for other in tensors+outputs[:i]:
                ol=other.data_ptr();oh=ol+other.numel()*other.element_size()
                if lo<oh and ol<hi:raise ValueError('Output buffers must not overlap inputs or each other')
        self.owner,self.x,self.g,self.gs,self.u,self.us=owner,x,g,gs,u,us
        self.out,self.gate_out,self.up_out=out,go,uo
        with torch.cuda.device(x.device):
            self.handle=owner.lib.tk_swiglu_create(x.data_ptr(),g.data_ptr(),gs.data_ptr(),u.data_ptr(),us.data_ptr(),
                out.data_ptr(),None if go is None else go.data_ptr(),None if uo is None else uo.data_ptr(),m,n,k,config,sms)
        if not self.handle:raise RuntimeError(owner.lib.tk_swiglu_error().decode())
        info=(ct.c_int*8)();owner.lib.tk_swiglu_info(self.handle,info)
        self.info=dict(zip(['R','C','BK','stages','shared_bytes','initial_regs','grid','store_aux'],info))

    def __call__(self):
        with torch.cuda.device(self.x.device):
            if self.owner.lib.tk_swiglu_launch(self.handle,torch.cuda.current_stream(self.x.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_swiglu_error().decode())
        return self.out

    def __del__(self):
        if getattr(self,'handle',None):
            self.owner.lib.tk_swiglu_destroy(self.handle);self.handle=None
