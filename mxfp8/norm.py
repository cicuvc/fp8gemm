"""Autograd RMSNorm module using the repository's Triton prenorm path."""

from __future__ import annotations

import torch
from torch import nn

from ._impl.rmsnorm import RMSNormBackwardPlan, RMSNormDualQuantPlan
from .runtime import require_sm120


class _ForwardState:
    def __init__(self, x, weight, eps):
        self.plan = RMSNormDualQuantPlan(
            x.contiguous(),
            weight.contiguous(),
            eps=eps,
            write_bf16=True,
        )

    def __call__(self, x, weight):
        self.plan.rebind(x.contiguous(), weight.contiguous())
        return self.plan()


class _BackwardState:
    def __init__(self, dy, x, weight, inverse_rms):
        self.plan = RMSNormBackwardPlan(
            dy.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            inverse_rms,
        )

    def __call__(self, dy, x, weight, inverse_rms):
        self.plan.allocate_workspace()
        self.plan.rebind(
            dy.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            inverse_rms,
        )
        return self.plan()


class _MXFP8RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, module):
        state = module._forward_state(x)
        state(x, weight)
        inverse_rms = state.plan.inverse_rms
        output = state.plan.y_bf16
        ctx.module = module
        ctx.save_for_backward(x, weight, inverse_rms)
        return output

    @staticmethod
    def backward(ctx, dy):
        x, weight, inverse_rms = ctx.saved_tensors
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        state = ctx.module._backward_state(dy, x, weight, inverse_rms)
        dx, weight_grad = state(dy, x, weight, inverse_rms)
        return dx, weight_grad.to(weight.dtype), None


class MXFP8RMSNorm(nn.Module):
    """BF16 RMSNorm prenorm with FP32 reduction and a fused quantize pass.

    The module returns a BF16 activation so it composes like a normal norm.
    It also computes both MXFP8 quantization layouts; those reusable buffers
    remain available through :meth:`quantized`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None):
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if hidden_size % 32:
            raise ValueError("hidden_size must be a multiple of 32")
        if not isinstance(eps, (int, float)) or float(eps) <= 0:
            raise ValueError("eps must be positive")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        )
        self._forward_states = {}
        self._backward_states = {}

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def _forward_state(self, x):
        key = (x.device.index, tuple(x.shape), x.dtype, self.weight.dtype, self.eps)
        state = self._forward_states.get(key)
        if state is None:
            state = _ForwardState(x, self.weight.detach(), self.eps)
            self._forward_states[key] = state
        return state

    def _backward_state(self, dy, x, weight, inverse_rms):
        key = (
            dy.device.index,
            tuple(dy.shape),
            tuple(x.shape),
            weight.dtype,
            self.eps,
        )
        state = self._backward_states.get(key)
        if state is None:
            state = _BackwardState(dy, x, weight.detach(), inverse_rms)
            self._backward_states[key] = state
        return state

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError(
                f"expected input [M,{self.hidden_size}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError("expected contiguous CUDA BF16 input")
        require_sm120(x.device)
        return _MXFP8RMSNormFunction.apply(x.contiguous(), self.weight, self)

    def quantized(self, x):
        """Return the BF16 output and both reusable MXFP8 quantization layouts."""
        output = self.forward(x)
        state = self._forward_state(x.contiguous())
        return (
            output,
            state.plan.row_fp8,
            state.plan.row_scale,
            state.plan.column_fp8,
            state.plan.column_scale,
            state.plan.inverse_rms,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._forward_states.clear()
        self._backward_states.clear()
        return result

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


__all__ = ["MXFP8RMSNorm"]
