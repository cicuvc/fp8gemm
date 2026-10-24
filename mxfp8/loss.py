"""Fused MXFP8 language-model loss."""

from __future__ import annotations

import torch

from ._impl.fused_ce import (
    MXFP8FusedLinearCrossEntropy as _ExperimentFusedLinearCrossEntropy,
)

from .runtime import ensure_extensions, require_sm120


class MXFP8FusedLinearCrossEntropy(_ExperimentFusedLinearCrossEntropy):
    """Chunked bias-free MXFP8 LM head followed by cross-entropy."""

    def forward(self, x, target, weight, **kwargs):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        require_sm120(x.device)
        ensure_extensions("loss")
        return super().forward(x, target, weight, **kwargs)


def fused_linear_cross_entropy_loss(
    x,
    target,
    weight,
    *,
    chunk_size=1024,
    forward_config=8,
    dx_config=12,
    dw_config=12,
    raster_group=13,
    quantized_weight=None,
    ignore_index=-100,
    balance=None,
    cu_seqlens=None,
    softcap=None,
    return_nll=False,
):
    """One-shot fused linear cross-entropy with the public module's checks."""
    return MXFP8FusedLinearCrossEntropy(
        chunk_size=chunk_size,
        forward_config=forward_config,
        dx_config=dx_config,
        dw_config=dw_config,
        raster_group=raster_group,
        ignore_index=ignore_index,
        balance=balance,
        softcap=softcap,
    )(
        x,
        target,
        weight,
        quantized_weight=quantized_weight,
        cu_seqlens=cu_seqlens,
        return_nll=return_nll,
    )


__all__ = ["MXFP8FusedLinearCrossEntropy", "fused_linear_cross_entropy_loss"]
