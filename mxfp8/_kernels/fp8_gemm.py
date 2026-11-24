"""SM120 E4M3 GEMM with FP32 scales at K=128 granularity."""

import ctypes as ct
from pathlib import Path
import subprocess
import re

import torch

from .._core import FLAGS, ROOT, nvcc_path

OUTPUT = ROOT / "runs/tk_fp8_gemm"
CONFIGS = [
    (128, int(m[1]), 128, int(m[2]))
    for m in re.findall(
        r"^CFG\((\d+),(\d+),(\d+)\);",
        (ROOT / "cuda/fp8_configs.inc").read_text(),
        re.M,
    )
]
_library = None


def build(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "build.log").open("w") as log:
        subprocess.run(
            [
                nvcc_path(),
                *FLAGS,
                "-Xptxas=-v",
                "-shared",
                "-Xcompiler=-fPIC",
                "cuda/persistent_fp8_gemm.cu",
                "-lcuda",
                "-o",
                str(output / "libfp8_gemm.so"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )


def _load_library():
    global _library
    if _library is None:
        from ..runtime import ensure_extensions

        ensure_extensions("fp8_gemm")
        lib = ct.CDLL(str(OUTPUT / "libfp8_gemm.so"))
        lib.tk_fp8_create.argtypes = [ct.c_void_p] * 5 + [ct.c_int] * 7
        lib.tk_fp8_create.restype = ct.c_void_p
        lib.tk_fp8_training_create.argtypes = [ct.c_void_p] * 11 + [ct.c_int] * 4
        lib.tk_fp8_training_create.restype = ct.c_void_p
        lib.tk_fp8_streamk_create.argtypes = [ct.POINTER(ct.c_void_p)] + [ct.c_int] * 10
        lib.tk_fp8_streamk_create.restype = ct.c_void_p
        lib.tk_fp8_set_output.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.tk_fp8_set_output.restype = None
        lib.tk_fp8_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.tk_fp8_launch.restype = ct.c_int
        lib.tk_fp8_error.restype = ct.c_char_p
        lib.tk_fp8_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        lib.tk_fp8_info.restype = None
        lib.tk_fp8_destroy.argtypes = [ct.c_void_p]
        lib.tk_fp8_destroy.restype = None
        _library = lib
    return _library


class FP8GemmPlan:
    """Reusable forward-only GEMM of prequantized operands on SM120.

    A[M,K] and B[N,K] are contiguous float8_e4m3fn. sfA[M,K//128]
    and sfB[N//128,K//128] are contiguous FP32 dequantization multipliers.
    With b_scale_mode="row", sfB is [N,K//128] and output must be FP32;
    this is the weight-gradient path (default persistent launch settings only).
    N and K must be positive multiples of 128; any positive M is supported.
    Each unscaled K128 dot product is accumulated in FP32, then multiplied
    by sfA*sfB and accumulated into a separate FP32 result. The output is
    BF16 by default, or FP32 when requested. No quantization or autograd.
    QMMA.SF uses internal unit scales (E8M0 127); external FP32 scales
    are applied at K128 granularity.

    streamk="atomic" or "deterministic" enables hybrid Stream-K (config 0).
    Splits occur only at K128 boundaries, after FP32 scaling. Atomic uses TMA
    reduce-add; deterministic fixes the order of FP32 partial sums. Neither
    promises bitwise equality with the unsplit dot product. The default None
    keeps the persistent kernel. Optional tail_waves controls the split region.

    Construct before CUDA Graph capture. Calls read current tensor contents
    and overwrite the same output. Keep tensor storage/metadata fixed and
    serialize calls to a plan; use separate plans for concurrent outputs.
    """

    def __init__(
        self,
        a,
        b,
        sfA,
        sfB,
        *,
        out=None,
        dtype=None,
        config=0,
        sms=0,
        raster_group=8,
        b_scale_mode="block",
        streamk=None,
        tail_waves=None,
    ):
        self._handle = None
        if streamk not in (None, "atomic", "deterministic"):
            raise ValueError("streamk must be None, 'atomic', or 'deterministic'")
        if tail_waves is not None and (type(tail_waves) is not int or tail_waves < 0 or streamk is None):
            raise ValueError("tail_waves requires Stream-K and a nonnegative integer")
        if streamk is not None and config != 0:
            raise ValueError("Stream-K currently requires config=0")
        self.streamk = streamk
        self.tail_waves = (0 if streamk == "atomic" else 1) if tail_waves is None else tail_waves
        inputs = (a, b, sfA, sfB)
        if any(not isinstance(t, torch.Tensor) for t in inputs):
            raise TypeError("a, b, sfA and sfB must be tensors")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("Expected A[M,K] and B[N,K]")
        m, k = a.shape
        n = b.shape[0]
        if min(m, n, k) <= 0 or n % 128 or k % 128:
            raise ValueError(
                "M must be positive; N and K must be positive multiples of 128"
            )
        if max(m, n, k) > 2**31 - 128 or ((m + 127) // 128) * (n // 64) >= 2**31:
            raise ValueError("Dimensions exceed the kernel's int32 indexing limits")
        if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
            raise ValueError("A and B must have dtype float8_e4m3fn")
        if sfA.dtype != torch.float32 or sfA.shape != (m, k // 128):
            raise ValueError("sfA must be FP32 [M,K//128]")
        if b_scale_mode not in ("block", "row"):
            raise ValueError("b_scale_mode must be 'block' or 'row'")
        b_rows = n // 128 if b_scale_mode == "block" else n
        if sfB.dtype != torch.float32 or sfB.shape != (b_rows, k // 128):
            raise ValueError(f"sfB must be FP32 [{b_rows},K//128]")
        self.b_scale_mode = b_scale_mode
        if not a.is_cuda or any(
            t.device != a.device or not t.is_contiguous() for t in inputs
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        if any(t.requires_grad for t in inputs):
            raise ValueError(
                "FP8 GEMM is forward-only; detach inputs requiring gradients"
            )
        if type(config) is not int or config not in range(len(CONFIGS)):
            raise ValueError("Invalid config")
        if type(sms) is not int or not 0 <= sms < 2**31:
            raise ValueError("sms must be a nonnegative int32")
        if type(raster_group) is not int or not 0 < raster_group < 2**31:
            raise ValueError("raster_group must be a positive int32")
        from ..runtime import require_sm120

        require_sm120(a.device)
        if out is not None and not isinstance(out, torch.Tensor):
            raise TypeError("out must be a tensor")
        if dtype is None:
            dtype = out.dtype if out is not None else torch.bfloat16
        if b_scale_mode == "row" and (
            dtype != torch.float32 or config != 0
            or (streamk is None and (sms or raster_group != 8))
        ):
            raise ValueError("Rowwise B scales require FP32 output and default launch settings")
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Output dtype must be BF16 or FP32")
        if out is None:
            out = torch.empty((m, n), device=a.device, dtype=dtype)
        if (
            out.shape != (m, n)
            or out.dtype != dtype
            or out.device != a.device
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError(
                "Invalid output: expected contiguous [M,N] on the input device without gradients"
            )
        if any(t.data_ptr() % 16 for t in (a, b, out)):
            raise ValueError("A, B and output pointers must be 16-byte aligned")
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        if any(
            lo < t.data_ptr() + t.numel() * t.element_size() and t.data_ptr() < hi
            for t in inputs
        ):
            raise ValueError("Output must not alias an input")
        self._lib = _load_library()
        self._refs = (*inputs, out)
        self.out = out
        self.device = a.device
        with torch.cuda.device(self.device):
            self._handle = self._create_handle(m, n, k, config, sms, raster_group, dtype)
        if not self._handle:
            raise RuntimeError(self._lib.tk_fp8_error().decode())
        values = (ct.c_int * 8)()
        self._lib.tk_fp8_info(self._handle, values)
        self.info = dict(
            zip(
                (
                    "block_m",
                    "block_n",
                    "block_k",
                    "stages",
                    "shared_bytes",
                    "initial_regs",
                    "local_bytes",
                    "grid",
                ),
                values,
            )
        )
        if self.streamk is not None:
            self.info.update(self._schedule_info)
        else:
            self.info["backend"] = "persistent"

    def _create_handle(self, m, n, k, config, sms, raster_group, dtype):
        if self.streamk is not None:
            return self._create_streamk_handle(m,n,k,sms,raster_group,dtype,0,(None,)*6)
        if self.b_scale_mode == "row":
            return self._lib.tk_fp8_training_create(
                *(t.data_ptr() for t in self._refs), *([None] * 6), m, n, k, 0
            )
        return self._lib.tk_fp8_create(
            *(t.data_ptr() for t in self._refs), m, n, k, config, sms,
            raster_group, int(dtype == torch.float32),
        )

    def _create_streamk_handle(self, m,n,k,sms,raster_group,dtype,mode,extra):
        from .streamk_schedule import schedule

        cap = torch.cuda.get_device_properties(self.device).multi_processor_count
        sched = schedule(m,n,k,min(sms or cap,cap),raster_group,self.tail_waves,
                         tile_cols=64,block_k=128)
        info = sched['info']
        slots = info['partial_slots']
        atomic = self.streamk == "atomic"
        if atomic:
            destinations = {(f[0],f[1]): i for i,f in enumerate(sched['fixups'])}
            for work in sched['work']:
                if work[4] >= 0: work[4] = destinations[(work[0],work[1])]
            slots = len(destinations)
        components = 2 if mode == 1 else 1
        self.workspace = torch.empty((max(1,slots),components,128,64),
                                     device=self.device,dtype=torch.float32)
        self._metadata = [torch.tensor(sched[key] or [0],device=self.device,dtype=torch.int32)
                          for key in ('work','offsets','fixups','contributors')]
        self._schedule_info = dict(info, backend="streamk", reduction=self.streamk,
            workspace_bytes=self.workspace.numel()*4, workspace_slots=slots)
        refs = (*self._refs, *extra, self.workspace, *self._metadata)
        pointers = (ct.c_void_p * 16)(*(t.data_ptr() if t is not None else 0 for t in refs))
        handle = self._lib.tk_fp8_streamk_create(pointers,m,n,k,mode,
            int(self.b_scale_mode == "row"),int(dtype == torch.float32),int(atomic),
            info['grid'],slots,info['fixup_tiles'])
        self._refs += (self.workspace,*self._metadata)
        return handle

    def _new_output(self):
        """Rebind direct stores without rebuilding the input TMA descriptors."""
        self.out = torch.empty_like(self.out)
        self._refs = (*self._refs[:4], self.out, *self._refs[5:])
        self._lib.tk_fp8_set_output(self._handle, self.out.data_ptr())
        return self.out

    def __call__(self):
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self._lib.tk_fp8_launch(self._handle, stream.cuda_stream):
                raise RuntimeError(self._lib.tk_fp8_error().decode())
            for t in self._refs:
                t.record_stream(stream)
        return self.out

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.tk_fp8_destroy(self._handle)
            self._handle = None


def fp8_gemm(a, b, sfA, sfB, *, out=None, dtype=None, b_scale_mode="block", streamk=None):
    """Compute scaled A @ B.T; see FP8GemmPlan for layouts and semantics.

    For repeated calls or CUDA Graph capture, construct FP8GemmPlan once.
    """
    return FP8GemmPlan(a, b, sfA, sfB, out=out, dtype=dtype, b_scale_mode=b_scale_mode, streamk=streamk)()



class _FP8DualPlan(FP8GemmPlan):
    """Internal fused gate/up forward (1) or summed dX (2)."""

    def __init__(self, a, b, sa, sb, *, a2, b2, sa2, sb2, mode, out, gate=None, up=None,
                 streamk=None, sms=0, tail_waves=None):
        self._mode = mode
        self._extra = (a2, b2, sa2, sb2, gate, up)
        super().__init__(a, b, sa, sb, out=out, streamk=streamk, sms=sms, tail_waves=tail_waves)
        self._refs += tuple(t for t in self._extra if t is not None)

    def _create_handle(self, m, n, k, config, sms, raster_group, dtype):
        if self.streamk is not None:
            return self._create_streamk_handle(m,n,k,sms,raster_group,dtype,self._mode,self._extra)
        return self._lib.tk_fp8_training_create(
            *(t.data_ptr() for t in self._refs),
            *(t.data_ptr() if t is not None else None for t in self._extra),
            m, n, k, self._mode,
        )


if __name__ == "__main__":
    build()
