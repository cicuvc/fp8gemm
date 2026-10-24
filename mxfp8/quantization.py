"""Public MXFP8 weight quantization helpers."""

from __future__ import annotations

import torch

from ._core import quantize_weight as _quantize_weight
from ._kernels.gemm import pack_a_scales
from .runtime import ensure_extensions


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an FP32 weight with E4M3 values and 32x32 E8M0 scales."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    ensure_extensions("quantize")
    return _quantize_weight(weight)


__all__ = ["pack_a_scales", "quantize_weight"]
