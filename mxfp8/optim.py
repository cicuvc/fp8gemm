"""Optimizer integration for MXFP8 training."""

from torch import nn
from torch.optim import Optimizer

from ._core import bind_fp8_optimizer
from .adamw import FusedAdamWMXFP8


def bind_optimizer(module: nn.Module, optimizer: Optimizer):
    """Keep every nested MXFP8 weight cache synchronized after optimizer steps."""
    return bind_fp8_optimizer(module, optimizer)


__all__ = ["FusedAdamWMXFP8", "bind_optimizer"]
