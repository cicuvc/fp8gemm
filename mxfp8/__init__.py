"""Bias-free, SM120 MXFP8 training modules."""

from .fp8_modules import FP8DownProjection, FP8Linear, FP8MLP, FP8SwiGLU
from ._kernels.fp8_gemm import FP8GemmPlan, fp8_gemm
from .loss import MXFP8FusedLinearCrossEntropy, fused_linear_cross_entropy_loss
from .modules import MXFP8DownProjection, MXFP8Linear, MXFP8MLP, MXFP8SwiGLU
from .norm import MXFP8RMSNorm
from .optim import FusedAdamWMXFP8, bind_optimizer
from .quantization import pack_a_scales, quantize_weight, quantize_weight_rowwise
from .streamk import configure_ffn_streamk
from .runtime import (
    build_extensions,
    ensure_extensions,
    extension_status,
    require_sm120,
)

__version__ = "0.1.0"

__all__ = [
    "FP8GemmPlan",
    "FP8DownProjection",
    "FP8Linear",
    "FP8MLP",
    "FP8SwiGLU",
    "FusedAdamWMXFP8",
    "MXFP8DownProjection",
    "MXFP8FusedLinearCrossEntropy",
    "MXFP8Linear",
    "MXFP8MLP",
    "MXFP8RMSNorm",
    "MXFP8SwiGLU",
    "bind_optimizer",
    "configure_ffn_streamk",
    "build_extensions",
    "ensure_extensions",
    "extension_status",
    "fp8_gemm",
    "fused_linear_cross_entropy_loss",
    "pack_a_scales",
    "quantize_weight",
    "quantize_weight_rowwise",
    "require_sm120",
]
