"""Torch adapter for the all-BF16 elementwise SwiGLU backward kernel."""

import ctypes as ct
from pathlib import Path
import subprocess

import torch

from experiments.tk_persistent_gemm import ROOT, FLAGS

OUTPUT = ROOT / "runs/tk_silu_bwd"


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/local/cuda/bin/nvcc",
        *FLAGS,
        "-Xptxas=-v",
        "-shared",
        "-Xcompiler=-fPIC",
        "cuda/silu_bwd_elementwise.cu",
        "-lcuda",
        "-o",
        str(OUTPUT / "libsilu_bwd.so"),
    ]
    with (OUTPUT / "build.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )


class SiluBwdLibrary:
    def __init__(self):
        self.lib = ct.CDLL(str(OUTPUT / "libsilu_bwd.so"))
        self.lib.tk_silu_bwd_create.argtypes = [ct.c_void_p] * 5 + [ct.c_int64]
        self.lib.tk_silu_bwd_create.restype = ct.c_void_p
        self.lib.tk_silu_bwd_launch.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.tk_silu_bwd_launch.restype = ct.c_int
        self.lib.tk_silu_bwd_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.tk_silu_bwd_destroy.argtypes = [ct.c_void_p]
        self.lib.tk_silu_bwd_error.restype = ct.c_char_p

    def plan(self, dy, half_gate, up, dgate=None, dup=None):
        return SiluBwdPlan(self, dy, half_gate, up, dgate, dup)


class SiluBwdPlan:
    def __init__(self, owner, dy, half_gate, up, dgate, dup):
        if dy.shape != half_gate.shape or dy.shape != up.shape:
            raise ValueError("dy, half_gate, and up must have equal shapes")
        if dy.dtype != torch.bfloat16 or half_gate.dtype != dy.dtype or up.dtype != dy.dtype:
            raise ValueError("Expected BF16 inputs")
        if dgate is None:
            dgate = torch.empty_like(dy)
        if dup is None:
            dup = torch.empty_like(dy)
        if dgate.shape != dy.shape or dup.shape != dy.shape:
            raise ValueError("Outputs must have the same shape as dy")
        tensors = [dy, half_gate, up, dgate, dup]
        if not dy.is_cuda or any(
            tensor.dtype != torch.bfloat16
            or tensor.device != dy.device
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous BF16 tensors on one CUDA device")
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            raise ValueError("Tensor pointers must be 16-byte aligned")
        for index, output in enumerate([dgate, dup]):
            lo = output.data_ptr()
            hi = lo + output.numel() * output.element_size()
            if any(
                lo < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                and tensor.data_ptr() < hi
                for tensor in [dy, half_gate, up, *([dgate] if index else [])]
            ):
                raise ValueError("Outputs must not overlap inputs or each other")
        self.owner = owner
        self.dy, self.half_gate, self.up = dy, half_gate, up
        self.dgate, self.dup = dgate, dup
        self.handle = owner.lib.tk_silu_bwd_create(
            dy.data_ptr(),
            half_gate.data_ptr(),
            up.data_ptr(),
            dgate.data_ptr(),
            dup.data_ptr(),
            dy.numel(),
        )
        if not self.handle:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_error().decode())
        info = (ct.c_int * 2)()
        owner.lib.tk_silu_bwd_info(self.handle, info)
        self.info = {"grid": info[0], "threads": info[1]}

    def __call__(self):
        status = self.owner.lib.tk_silu_bwd_launch(
            self.handle,
            torch.cuda.current_stream(self.dy.device).cuda_stream,
        )
        if status:
            raise RuntimeError(self.owner.lib.tk_silu_bwd_error().decode())
        return self.dgate, self.dup

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tk_silu_bwd_destroy(self.handle)
            self.handle = None
