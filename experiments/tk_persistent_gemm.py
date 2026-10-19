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
        self.lib.tk_gemm_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_gemm_launch.restype = ct.c_int
        self.lib.tk_gemm_error.restype = ct.c_char_p
        self.lib.tk_gemm_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_gemm_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]

    def plan(self, a, b, config=8, out=None, sms=0):
        return GemmPlan(self, a, b, config, out, sms)


class GemmPlan:
    """Reusable descriptors for fixed allocations, shapes, and configuration.

    A: contiguous BF16 [M,K], B: contiguous E4M3 [N,K], output BF16 [M,N].
    B has unit block scale. A uses the existing online_scale exponent rule.
    K/N must be multiples of the selected TMA panel widths (64 always works).
    No autograd registration; inputs/outputs must remain alive until work ends.
    """
    def __init__(self, owner, a, b, config, out, sms):
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
        self.owner, self.a, self.b, self.out = owner, a, b, out
        self.handle = None
        with torch.cuda.device(a.device):
            self.handle = owner.lib.tk_gemm_create(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, config, sms)
        if not self.handle:
            raise RuntimeError(owner.lib.tk_gemm_error().decode())
        info = (ct.c_int * 7)()
        owner.lib.tk_gemm_info(self.handle, info)
        self.info = dict(zip(['R', 'C', 'BK', 'stages', 'shared_bytes', 'initial_regs', 'grid'], info))

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
