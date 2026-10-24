"""Experimental inference-only MXFP8 topK linear on SM120, 1 <= K <= 16.

Scores equal the ordinary MXFP8 GEMM's BF16 output, sorted descending with
ascending original token IDs for ties (signed zero is canonicalized). Inputs
must produce non-NaN logits. No full [M,V] logits tensor is allocated or stored.
TMA permutes full N tiles; partial tiles use ordinary bounds-checked loads.
The associated weight scales follow the same permutation.

Build: python -m experiments.tk_mxfp8_topk
Example: values, ids = MXFP8TopKLinear(top_k=16)(x_bf16, weight)
Optional quantized_weight=(E4M3, E8M0) reuses the existing weight cache.
Low-level plans reuse output storage; module outputs have independent storage.
Warm each shape before graph capture. Scratch is for sequential-stream use.
"""

import ctypes as ct
from pathlib import Path
import re
import subprocess

import torch
from torch import nn

from experiments.mxfp8_core import FLAGS, ROOT, nvcc_path, quantize_weight
from experiments.mxfp8_topk import MergePlan, validate_k
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

OUTPUT = ROOT / "runs/tk_mxfp8_topk"
CONFIGS = {int(m[0]): tuple(map(int, m[1:])) for m in re.findall(
    r"^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);",
    (ROOT / "cuda/mxfp8_topk_configs.inc").read_text(), re.M)}


def build(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "build.log").open("w") as log:
        subprocess.run([nvcc_path(), *FLAGS, "-Xptxas=-v", "-shared",
                        "-Xcompiler=-fPIC", "cuda/persistent_mxfp8_topk.cu", "-lcuda",
                        "-o", str(output / "libtopk.so")], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=600)


def _validate_options(top_k, config, raster_group, sms, permute_n):
    validate_k(top_k)
    if type(config) is not int or config not in CONFIGS:
        raise ValueError("Invalid topK config")
    if type(raster_group) is not int or not 0 < raster_group < 2**31:
        raise ValueError("Invalid raster_group")
    if type(sms) is not int or not 0 <= sms < 2**31:
        raise ValueError("Invalid sms")
    if type(permute_n) is not bool:
        raise ValueError("permute_n must be bool")


class Library:
    def __init__(self, output=OUTPUT):
        self.lib = ct.CDLL(str(Path(output) / "libtopk.so"))
        self.lib.tk_mxt_create.argtypes = [ct.c_void_p] * 5 + [ct.c_int] * 8
        self.lib.tk_mxt_create.restype = ct.c_void_p
        self.lib.tk_mxt_launch.argtypes = [ct.c_void_p] * 2
        self.lib.tk_mxt_launch.restype = ct.c_int
        self.lib.tk_mxt_rebind.argtypes = [ct.c_void_p] * 6
        self.lib.tk_mxt_rebind.restype = ct.c_int
        self.lib.tk_mxt_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_mxt_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_mxt_error.restype = ct.c_char_p

    def plan(self, x, w, xs, ws, *, top_k=16, config=8, raster_group=13, sms=0, permute_n=True):
        return Plan(self, x, w, xs, ws, top_k, config, raster_group, sms, permute_n)


class Plan:
    def __init__(self, owner, x, w, xs, ws, top_k, config, raster_group, sms, permute_n):
        self.handle = None
        _validate_options(top_k, config, raster_group, sms, permute_n)
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Expected X[M,H], W[V,H]")
        self.m, self.h = x.shape
        self.v = w.shape[0]
        validate_k(top_k, self.v)
        panel = min(CONFIGS[config][2], 128)
        if min(self.m, self.v, self.h) <= 0 or max(self.m, self.v, self.h) >= 2**31 or self.h % panel:
            raise ValueError(f"Positive int32 dimensions and H divisible by {panel} required")
        self.device, self.owner, self.top_k = x.device, owner, top_k
        self.bucket = 1 << (top_k - 1).bit_length()
        self.parts = (self.v + CONFIGS[config][1] - 1) // CONFIGS[config][1]
        self.local_keys = torch.empty((self.m, self.parts, self.bucket), device=x.device, dtype=torch.int64)
        self._validate(x, w, xs, ws)
        self.merge = MergePlan(self.m, self.parts, top_k, x.device)
        with torch.cuda.device(x.device):
            self.handle = owner.lib.tk_mxt_create(
                *[t.data_ptr() for t in (x, w, xs, ws, self.local_keys)],
                self.m, self.v, self.h, config, self.bucket, sms, raster_group, int(permute_n))
        if not self.handle:
            raise RuntimeError(owner.lib.tk_mxt_error().decode())
        self.refs = (x, w, xs, ws)
        info = (ct.c_int * 7)()
        owner.lib.tk_mxt_info(self.handle, info)
        self.info = dict(zip(("R", "C", "BK", "stages", "shared_bytes", "initial_regs", "grid"), info))
        self.info.update(top_k=top_k, bucket=self.bucket, permute_n=permute_n)

    def _validate(self, x, w, xs, ws):
        shapes = ((self.m, self.h), (self.v, self.h),
                  ((self.h + 127) // 128, (self.m + 15) // 16 * 16),
                  ((self.v + 31) // 32, self.h // 32))
        for t, shape, dtype in zip((x, w, xs, ws), shapes,
                                  (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.int32, torch.uint8)):
            if t.shape != shape or t.dtype != dtype:
                raise ValueError(f"Expected {dtype} {shape}")
            if not t.is_cuda or t.device != self.device or not t.is_contiguous():
                raise ValueError("Expected contiguous tensors on the plan CUDA device")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise RuntimeError("SM120 required")
        if any(t.data_ptr() % 16 for t in (x, w, xs)):
            raise ValueError("Unaligned TMA pointer")

    def rebind(self, x, w, xs, ws):
        self._validate(x, w, xs, ws)
        with torch.cuda.device(self.device):
            status = self.owner.lib.tk_mxt_rebind(
                self.handle, *[t.data_ptr() for t in (x, w, xs, ws, self.local_keys)])
        if status:
            raise RuntimeError(self.owner.lib.tk_mxt_error().decode())
        self.refs = (x, w, xs, ws)
        return self

    def local(self):
        with torch.cuda.device(self.device):
            if self.owner.lib.tk_mxt_launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mxt_error().decode())
        return self.local_keys

    def __call__(self, *, out=None):
        return self.merge(self.local(), out=out)

    def __del__(self):
        if self.handle:
            self.owner.lib.tk_mxt_destroy(self.handle)


class MXFP8TopKLinear(nn.Module):
    """Forward-only topK of bias-free MXFP8 linear; no sampling or autograd."""
    def __init__(self, *, top_k=16, config=8, raster_group=13, sms=0, permute_n=True):
        super().__init__()
        _validate_options(top_k, config, raster_group, sms, permute_n)
        self.top_k, self.config = top_k, config
        self.raster_group, self.sms, self.permute_n = raster_group, sms, permute_n
        self._states = {}
        self._quant = self._gemm = None

    def forward(self, x, weight, *, quantized_weight=None):
        return self._run(x, weight, quantized_weight=quantized_weight)

    @torch.no_grad()
    def _run(self, x, weight, *, quantized_weight=None, finish=None):
        if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16
                or not x.is_contiguous() or min(x.shape) <= 0):
            raise ValueError("Expected nonempty contiguous CUDA BF16 X[M,H]")
        if (weight.ndim != 2 or weight.shape[1] != x.shape[1] or weight.shape[0] <= 0
                or weight.device != x.device or not weight.is_contiguous()
                or weight.dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError("Expected contiguous FP32/BF16 W[V,H] on X device")
        validate_k(self.top_k, weight.shape[0])
        if x.shape[1] % min(CONFIGS[self.config][2], 128):
            raise ValueError("H must align to selected TMA panels")
        if torch.cuda.get_device_capability(x.device) != (12, 0):
            raise RuntimeError("SM120 required")
        if self._gemm is None:
            self._quant, self._gemm = SiluBwdQuantizeLibrary(), Library()
        with torch.cuda.device(x.device):
            if quantized_weight is None:
                wq, ws = quantize_weight(weight.float())
            else:
                wq, ws = quantized_weight
                if (wq.shape != weight.shape or wq.dtype != torch.float8_e4m3fn
                        or ws.shape != ((weight.shape[0] + 31) // 32, weight.shape[1] // 32)
                        or ws.dtype != torch.uint8 or wq.device != x.device or ws.device != x.device
                        or not wq.is_contiguous() or not ws.is_contiguous()):
                    raise ValueError("Invalid quantized weight cache")
            key = (x.device, tuple(x.shape), tuple(weight.shape), self.top_k, self.config,
                   self.raster_group, self.sms, self.permute_n)
            if key not in self._states:
                xq = self._quant.plan_direct(x)
                plan = self._gemm.plan(xq.row_fp8, wq, xq.row_scale, ws,
                                      top_k=self.top_k, config=self.config,
                                      raster_group=self.raster_group, sms=self.sms, permute_n=self.permute_n)
                self._states[key] = xq, plan
            xq, plan = self._states[key]
            try:
                xq.rebind(x)
                plan.rebind(xq.row_fp8, wq, xq.row_scale, ws)
                xq()
                if finish is not None:
                    return finish(plan)
                out = (torch.empty((len(x), self.top_k), device=x.device, dtype=torch.bfloat16),
                       torch.empty((len(x), self.top_k), device=x.device, dtype=torch.int64))
                return plan(out=out)
            finally:
                xq.x = None
                plan.refs = ()

    def clear_workspace(self):
        self._states.clear()


if __name__ == "__main__":
    build()
