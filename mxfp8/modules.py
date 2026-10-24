"""Bias-free linear and SwiGLU modules backed by SM120 MXFP8 kernels."""

from __future__ import annotations

import torch

from experiments.tk_mxfp8_down import (
    MXFP8DownProjection as _ExperimentDownProjection,
)
from experiments.tk_mxfp8_linear import MXFP8Linear as _ExperimentLinear
from experiments.tk_mxfp8_swiglu_layer import (
    MXFP8SwiGLU as _ExperimentSwiGLU,
)

from .runtime import ensure_extensions, require_sm120


class MXFP8Linear(_ExperimentLinear):
    """Bias-free linear layer with pure-MXFP8 forward and backward GEMMs.

    Inputs must be contiguous BF16 ``[M, K]`` tensors. Weights are FP32
    masters and the layer cache is refreshed through ``bind_optimizer``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | int | None = None,
        forward_config: int = 8,
        dx_config: int = 12,
        dw_config: int = 12,
        raster_group: int = 13,
    ):
        resolved = require_sm120(device)
        ensure_extensions("linear")
        super().__init__(
            in_features,
            out_features,
            device=resolved,
            forward_config=forward_config,
            dx_config=dx_config,
            dw_config=dw_config,
            raster_group=raster_group,
        )


class MXFP8DownProjection(_ExperimentDownProjection):
    """Bias-free down projection using MXFP8 forward, dX, and dW GEMMs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | int | None = None,
        forward_config: int = 70,
        dx_config: int = 12,
        dw_config: int = 12,
        raster_group: int = 13,
    ):
        resolved = require_sm120(device)
        ensure_extensions("linear")
        super().__init__(
            in_features,
            out_features,
            device=resolved,
            forward_config=forward_config,
            dx_config=dx_config,
            dw_config=dw_config,
            raster_group=raster_group,
        )


class MXFP8SwiGLU(_ExperimentSwiGLU):
    """Bias-free ``down(silu(gate(x)) * up(x))`` training module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        device: torch.device | str | int | None = None,
        forward_config: int = 1,
        dx_config: int = 5,
        wgrad_config: int = 6,
        down_forward_config: int = 70,
        down_dx_config: int = 12,
        down_dw_config: int = 12,
        raster_group: int = 13,
    ):
        resolved = require_sm120(device)
        ensure_extensions("mlp")
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            bias=False,
            device=resolved,
            forward_config=forward_config,
            dx_config=dx_config,
            wgrad_config=wgrad_config,
            down_forward_config=down_forward_config,
            down_dx_config=down_dx_config,
            down_dw_config=down_dw_config,
            raster_group=raster_group,
        )


MXFP8MLP = MXFP8SwiGLU


__all__ = ["MXFP8DownProjection", "MXFP8Linear", "MXFP8MLP", "MXFP8SwiGLU"]
