"""Torch-side pointer/stream adapter for the Torch-free CUDA WASP GEMM."""
import ctypes as ct
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'runs/tk_persistent_gemm'
CONFIGS = [(16, 32, 32, 2), (16, 64, 32, 2), (16, 128, 32, 2),
           (32, 32, 32, 2), (32, 64, 32, 2), (64, 16, 32, 2), (64, 32, 32, 2),
           (16, 64, 64, 2), (16, 128, 64, 2), (32, 32, 64, 2)]
FLAGS = ['-std=c++20', '-gencode=arch=compute_120a,code=sm_120a',
         '-DKITTENS_RTX_BLACKWELL', '-Ithird_party/thunderkittens/include',
         '--expt-relaxed-constexpr', '--extended-lambda', '-O3', '-lineinfo']


def build(output=DEFAULT_OUTPUT, cuda=Path('/usr/local/cuda')):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cmd = [str(cuda / 'bin/nvcc'), *FLAGS, '-Xptxas=-v', '-shared', '-Xcompiler=-fPIC',
           'cuda/persistent_fp8_gemm.cu', '-lcuda', '-o', str(output / 'libgemm.so')]
    with (output / 'build.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)


class GemmLibrary:
    def __init__(self, output=DEFAULT_OUTPUT):
        self.lib = ct.CDLL(str(Path(output).resolve() / 'libgemm.so'))
        self.lib.tk_gemm_create.argtypes = [ct.c_void_p] * 3 + [ct.c_int] * 5
        self.lib.tk_gemm_create.restype = ct.c_void_p
        self.lib.tk_gemm_create_scaled.argtypes = [ct.c_void_p] * 4 + [ct.c_int] * 5
        self.lib.tk_gemm_create_scaled.restype = ct.c_void_p
        self.lib.tk_gemm_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_gemm_launch.restype = ct.c_int
        self.lib.tk_gemm_error.restype = ct.c_char_p
        self.lib.tk_gemm_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_gemm_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        if hasattr(self.lib, 'tk_gemm_set_raster_group'):
            self.lib.tk_gemm_set_raster_group.argtypes = [ct.c_void_p, ct.c_int]
            self.lib.tk_gemm_set_raster_group.restype = ct.c_int
        if hasattr(self.lib, 'tk_gemm_create_prefetch'):
            self.lib.tk_gemm_create_prefetch.argtypes = [ct.c_void_p]*4+[ct.c_int]*5
            self.lib.tk_gemm_create_prefetch.restype = ct.c_void_p
        if hasattr(self.lib, 'tk_gemm_create_ab'):
            self.lib.tk_gemm_create_ab.argtypes = [ct.c_void_p]*4+[ct.c_int]*8
            self.lib.tk_gemm_create_ab.restype = ct.c_void_p
        if hasattr(self.lib, 'tk_gemm_create_ab_swapped'):
            self.lib.tk_gemm_create_ab_swapped.argtypes = [ct.c_void_p]*4+[ct.c_int]*8
            self.lib.tk_gemm_create_ab_swapped.restype = ct.c_void_p

    def plan(self, a, b, config=8, out=None, sms=0, scales=None, *, raster_group=1, prefetch=False):
        return GemmPlan(self, a, b, config, out, sms, scales, raster_group=raster_group, prefetch=prefetch)

    def plan_ab(self, a, b, scales, config=8, out=None, sms=0, *, swap_ab=False):
        return GemmABPlan(self, a, b, scales, config, out, sms, swap_ab)


class GemmABPlan:
    """A[M,Kp] @ B[K,N], FP8 B in its original row-major global layout.

    A is BF16 and quantized online. B is E4M3 with 32x32 block scales.
    K<=Kp: TMA zero-fills missing B rows without a padded B allocation.
    A's contiguous row width is a multiple64; B/output width must align to
    the selected shared swizzle panel (32/64/128 elements for C32/64/128).
    Scale rows/columns may cover less than padded B, with missing blocks zero.
    swap_ab=True computes B^T @ A^T with FP8 weights as the native A operand;
    BF16 conversion + transposing STSM + TMA still return output[M,N].
    R/C in info always refer to the returned output; swapped accumulators are CxR.
    """
    def __init__(self, owner, a, b, scales, config, out, sms, swap_ab=False):
        self.handle=None
        if config not in [0,1,2,3,4,6,7,8,9] or a.ndim!=2 or b.ndim!=2:
            raise ValueError('Invalid AB configuration or shapes')
        m,k=a.shape;bk,n=b.shape
        c=CONFIGS[config][1]
        if min(m,k,bk,n)<=0 or bk>k or k%64 or n%c:
            raise ValueError('A width must align64; B/output width must align to C')
        if a.dtype!=torch.bfloat16 or b.dtype!=torch.float8_e4m3fn:
            raise ValueError('Expected BF16 A and E4M3 B')
        if (not a.is_cuda or b.device!=a.device or scales.device!=a.device or
                not all(t.is_contiguous() for t in [a,b,scales])):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if (scales.dtype!=torch.uint8 or scales.ndim!=2 or scales.shape[0]!=(bk+31)//32
                or not 0<scales.shape[1]<=(n+31)//32):
            raise ValueError('Invalid 32x32 scale grid')
        if out is None:out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
        if out.shape!=(m,n) or out.dtype!=torch.bfloat16 or out.device!=a.device or not out.is_contiguous():
            raise ValueError('Invalid output')
        if any(t.data_ptr()%16 for t in [a,b,out]):raise ValueError('TMA pointers must align16')
        lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
        if any(lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi for t in [a,b,scales]):
            raise ValueError('Output overlaps inputs')
        self.owner,self.a,self.b,self.scales,self.out=owner,a,b,scales,out
        with torch.cuda.device(a.device):
            create = owner.lib.tk_gemm_create_ab_swapped if swap_ab else owner.lib.tk_gemm_create_ab
            self.handle=create(a.data_ptr(),b.data_ptr(),scales.data_ptr(),out.data_ptr(),
                m,n,k,bk,*scales.shape,config,sms)
        if not self.handle:raise RuntimeError(owner.lib.tk_gemm_error().decode())
        info=(ct.c_int*7)();owner.lib.tk_gemm_info(self.handle,info)
        self.info=dict(zip(['R','C','BK','stages','shared_bytes','initial_regs','grid'],info))
        self.info['direct_b']=True
        self.info['swap_ab']=bool(swap_ab)
        self.info['accumulator_rows']=self.info['C'] if swap_ab else self.info['R']
        self.info['accumulator_cols']=self.info['R'] if swap_ab else self.info['C']

    def __call__(self):
        with torch.cuda.device(self.a.device):
            if self.owner.lib.tk_gemm_launch(self.handle,torch.cuda.current_stream(self.a.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_gemm_error().decode())
        return self.out

    def __del__(self):
        if getattr(self,'handle',None):self.owner.lib.tk_gemm_destroy(self.handle);self.handle=None


class GemmPlan:
    """Reusable descriptors for fixed allocations, shapes, and configuration.

    A: contiguous BF16 [M,K], B: contiguous E4M3 [N,K], output BF16 [M,N].
    B has unit block scale unless uint8 E8M0 scales[ceil(N/32),ceil(K/32)]
    are supplied. Only the producer reads those global metadata bytes;
    consumers load them from shared memory before releasing their input slot.
    A uses the existing online_scale exponent rule.
    K/N must be multiples of the selected TMA panel widths (64 always works).
    raster_group groups output M tiles for L2 reuse (1 is row-major).
    prefetch loads the next shared A/B/scale packet into registers before
    issuing the current packet's MMA, with explicit register ping-pong.
    No autograd registration; inputs/outputs must remain alive until work ends.
    """
    def __init__(self, owner, a, b, config, out, sms, scales=None, *, raster_group=1, prefetch=False):
        self.handle = None
        if isinstance(raster_group,bool) or not isinstance(raster_group,int) or not 1<=raster_group<=2**31-1:
            raise ValueError('raster_group must be a positive int32')
        if not isinstance(prefetch,bool):
            raise ValueError('prefetch must be bool')
        if not 0 <= config < len(CONFIGS):
            raise ValueError('Invalid configuration')
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError('Expected A[M,K] and B[N,K]')
        if a.dtype != torch.bfloat16 or b.dtype != torch.float8_e4m3fn:
            raise ValueError('Expected BF16 A and E4M3 B')
        if not a.is_cuda or a.device != b.device or not a.is_contiguous() or not b.is_contiguous():
            raise ValueError('Inputs must be contiguous on the same CUDA device')
        m, k = a.shape
        n = b.shape[0]
        if out is None:
            out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        if out.shape != (m, n) or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
            raise ValueError('Invalid output tensor')
        if scales is not None and (scales.shape != ((n+31)//32,(k+31)//32) or scales.dtype != torch.uint8
                                   or scales.device != a.device or not scales.is_contiguous()):
            raise ValueError('Invalid weight block scales')
        self.owner, self.a, self.b, self.out, self.scales = owner, a, b, out, scales
        self.handle = None
        with torch.cuda.device(a.device):
            if prefetch:
                if not hasattr(owner.lib, 'tk_gemm_create_prefetch'):
                    raise RuntimeError('Rebuild CUDA library for register prefetch')
                self.handle = owner.lib.tk_gemm_create_prefetch(
                    a.data_ptr(), b.data_ptr(), scales.data_ptr() if scales is not None else 0,
                    out.data_ptr(), m,n,k,config,sms)
            elif scales is None:
                self.handle = owner.lib.tk_gemm_create(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, config, sms)
            else:
                self.handle = owner.lib.tk_gemm_create_scaled(a.data_ptr(), b.data_ptr(), scales.data_ptr(),
                                                             out.data_ptr(), m, n, k, config, sms)
        if not self.handle:
            raise RuntimeError(owner.lib.tk_gemm_error().decode())
        if raster_group != 1:
            if not hasattr(owner.lib, 'tk_gemm_set_raster_group'):
                raise RuntimeError('Rebuild CUDA library for grouped rasterization')
            if owner.lib.tk_gemm_set_raster_group(self.handle,raster_group):
                raise RuntimeError(owner.lib.tk_gemm_error().decode())
        info = (ct.c_int * 7)()
        owner.lib.tk_gemm_info(self.handle, info)
        self.info = dict(zip(['R', 'C', 'BK', 'stages', 'shared_bytes', 'initial_regs', 'grid'], info))
        self.info.update(raster_group=min(raster_group,(m+8*CONFIGS[config][0]-1)//(8*CONFIGS[config][0])),
                         prefetch=prefetch)

    def __call__(self):
        with torch.cuda.device(self.a.device):
            stream = torch.cuda.current_stream(self.a.device).cuda_stream
            if self.owner.lib.tk_gemm_launch(self.handle, stream):
                raise RuntimeError(self.owner.lib.tk_gemm_error().decode())
        return self.out

    def __del__(self):
        if getattr(self, 'handle', None):
            self.owner.lib.tk_gemm_destroy(self.handle)
            self.handle = None
