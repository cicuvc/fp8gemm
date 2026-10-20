"""WASP AB^T: BF16 A/B -> online row/K32 E4M3 + E8M0 -> FP32 accumulation."""
import ctypes as ct
from pathlib import Path
import subprocess
import torch
from experiments.tk_persistent_gemm import ROOT,FLAGS

DEFAULT_OUTPUT=ROOT/'runs/tk_dual_online_gemm'
CONFIGS=[(16,32,64,2),(16,64,64,2),(16,128,32,2),(16,128,64,2),
         (32,64,32,2),(32,32,64,2),(16,64,64,3),(16,32,128,2)]
NATIVE_DW_CONFIGS=CONFIGS+[
    (32,128,32,1),(16,256,32,1),(32,64,32,3),(16,128,32,3),
    (64,32,32,1),(32,64,32,1),(16,256,32,2),(16,128,32,4),(16,64,64,4)]


def build(output=DEFAULT_OUTPUT,cuda=Path('/usr/local/cuda')):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'),*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
            'cuda/persistent_dual_online_gemm.cu','-lcuda','-o',str(output/'libdual.so')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)


class DualOnlineLibrary:
    def __init__(self,output=DEFAULT_OUTPUT):
        self.lib=ct.CDLL(str(Path(output).resolve()/'libdual.so'))
        self.lib.tk_dual_create.argtypes=[ct.c_void_p]*4+[ct.c_int]*7
        self.lib.tk_dual_create.restype=ct.c_void_p
        self.lib.tk_dual_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_dual_launch.restype=ct.c_int
        self.lib.tk_dual_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_dual_destroy.argtypes=[ct.c_void_p]
        self.lib.tk_dual_error.restype=ct.c_char_p
        if hasattr(self.lib,'tk_dual_set_raster_group'):
            self.lib.tk_dual_set_raster_group.argtypes=[ct.c_void_p,ct.c_int]
            self.lib.tk_dual_set_raster_group.restype=ct.c_int
        if hasattr(self.lib,'tk_dual_create_native_dy'):
            self.lib.tk_dual_create_native_dy.argtypes=[ct.c_void_p]*4+[ct.c_int]*7
            self.lib.tk_dual_create_native_dy.restype=ct.c_void_p

    def plan(self,a,b,config=3,out=None,output_dtype=torch.bfloat16,sms=0,b_scales=None,*,raster_group=1):
        return DualOnlinePlan(self,a,b,config,out,output_dtype,sms,b_scales,raster_group=raster_group)

    def plan_native_dy(self,dy,xq,scales,config=3,out=None,sms=0,*,swap_ab=False,raster_group=1):
        """FP32 dW[N,K] from BF16 dy[M,N], FP8 xq[K,Mp], scales[K,Mp/32].

        TMA zero-fills missing dy rows (M<=Mp); N/K/Mp align to standard panels.
        Native transposing BF16 LDSM replaces the global dY transpose.
        swap_ab additionally exchanges the MMA operands and transposes FP32
        output into shared memory before TMA. No dY^T allocation is needed.
        raster_group groups output rows (N tiles); 1 retains row-major order.
        Configurations >=8 are additional native-dY candidates and require
        swap_ab=False; their (R,C,BK,stages) are listed in NATIVE_DW_CONFIGS.
        """
        return DualOnlinePlan(self,dy,xq,config,out,torch.float32,sms,scales,
                              native_dy=True,swap_ab=swap_ab,raster_group=raster_group)


class DualOnlinePlan:
    """Contiguous CUDA A[M,K] and B[N,K] compute AB^T, with optional FP32 output.

    Default A and B are BF16; both use the existing online_scale exponent rule
    independently per row/K32 (including its exponent-254 reciprocal behavior).
    N is a multiple of 64, K of 64 (128 for config 7); M may have tails.
    A cached-B comparison accepts E4M3 B plus uint8 b_scales[N,K/32], with
    identical quantization semantics. It does NOT use 32x32 weight scales.
    All data moves through TMA; only the cached-B producer reads global scales.
    No implicit input transpose, weight refresh or autograd registration.
    The plan retains all allocations; keep it alive until GPU work completes.
    """
    def __init__(self,owner,a,b,config,out,dtype,sms,scales,*,native_dy=False,swap_ab=False,raster_group=1):
        self.handle=None
        if config>=8 and (not native_dy or swap_ab):
            raise ValueError('Configurations >=8 require native dY without swapAB')
        if isinstance(raster_group,bool) or not isinstance(raster_group,int) or not 1<=raster_group<=2**31-1:
            raise ValueError('raster_group must be a positive int32')
        configs=NATIVE_DW_CONFIGS if native_dy and not swap_ab else CONFIGS
        if (not 0<=config<len(configs) or a.ndim!=2 or b.ndim!=2 or
                (not native_dy and a.shape[1]!=b.shape[1])):
            raise ValueError('Expected A[M,K] and B[N,K], valid configuration')
        m,k=(a.shape[1],b.shape[1]) if native_dy else a.shape
        n=b.shape[0];online=scales is None
        if swap_ab and not native_dy:
            raise ValueError('swapAB requires native dY')
        if native_dy and (online or dtype!=torch.float32 or not 0<a.shape[0]<=k or m%64):
            raise ValueError('Native dY requires FP32 output, cached FP8 X, aligned dY width, and rows <= reduction width')
        if min(m,n,k)<=0 or n%64 or k%(128 if config==7 else 64):
            raise ValueError('N/K must align to the selected TMA panels')
        if a.dtype!=torch.bfloat16 or b.dtype!=(torch.bfloat16 if online else torch.float8_e4m3fn):
            raise ValueError('Expected BF16 A and either BF16 B or cached E4M3 B with per-row scales')
        if not a.is_cuda or b.device!=a.device or not a.is_contiguous() or not b.is_contiguous():
            raise ValueError('Inputs must be contiguous on the same CUDA device')
        if dtype not in [torch.bfloat16,torch.float32]:raise ValueError('Output dtype must be BF16 or FP32')
        if out is None:out=torch.empty((m,n),device=a.device,dtype=dtype)
        if out.shape!=(m,n) or out.dtype!=dtype or out.device!=a.device or not out.is_contiguous():
            raise ValueError('Invalid output buffer')
        if any(t.data_ptr()%16 for t in [a,b,out]):raise ValueError('TMA base pointers must be 16-byte aligned')
        if scales is not None and (scales.shape!=(n,k//32) or scales.dtype!=torch.uint8 or scales.device!=a.device or not scales.is_contiguous()):
            raise ValueError('Expected contiguous uint8 B scales[N,K/32]')
        lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
        for t in [a,b]+([] if scales is None else [scales]):
            if lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi:
                raise ValueError('Output must not overlap inputs')
        self.owner,self.a,self.b,self.scales,self.out=owner,a,b,scales,out
        with torch.cuda.device(a.device):
            if native_dy:
                self.handle=owner.lib.tk_dual_create_native_dy(a.data_ptr(),b.data_ptr(),scales.data_ptr(),
                    out.data_ptr(),m,n,k,a.shape[0],config,sms,int(swap_ab))
            else:
                self.handle=owner.lib.tk_dual_create(a.data_ptr(),b.data_ptr(),None if scales is None else scales.data_ptr(),
                    out.data_ptr(),m,n,k,config,sms,int(online),int(dtype==torch.float32))
        if not self.handle:raise RuntimeError(owner.lib.tk_dual_error().decode())
        if raster_group!=1:
            if not hasattr(owner.lib,'tk_dual_set_raster_group'):
                raise RuntimeError('Rebuild CUDA library for grouped rasterization')
            if owner.lib.tk_dual_set_raster_group(self.handle,raster_group):
                raise RuntimeError(owner.lib.tk_dual_error().decode())
        info=(ct.c_int*9)();owner.lib.tk_dual_info(self.handle,info)
        self.info=dict(zip(['R','C','BK','stages','shared_bytes','initial_regs','grid','online_b','fp32_output'],info))
        self.info.update(native_dy=bool(native_dy),swap_ab=bool(swap_ab),
                         raster_group=min(raster_group,(m+8*configs[config][0]-1)//(8*configs[config][0])))

    def __call__(self):
        with torch.cuda.device(self.a.device):
            if self.owner.lib.tk_dual_launch(self.handle,torch.cuda.current_stream(self.a.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_dual_error().decode())
        return self.out

    def __del__(self):
        if getattr(self,'handle',None):self.owner.lib.tk_dual_destroy(self.handle);self.handle=None
