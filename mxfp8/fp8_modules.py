"""Bias-free SM120 training with E4M3 and K128 FP32 dequantization scales."""

import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from ._core import bind_fp8_optimizer
from ._kernels.fp8_gemm import FP8GemmPlan, _FP8DualPlan
from ._kernels.fp8_quantize import BlockQuantized, DualQuantized
from .runtime import ensure_extensions, require_sm120


def _features(*sizes):
    if any(type(n) is not int or n <= 0 or n % 128 for n in sizes):
        raise ValueError("Feature dimensions must be positive multiples of 128")


def _input(x, weight):
    if (x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != weight.shape[1]
            or x.dtype != torch.bfloat16 or x.device != weight.device
            or not x.is_contiguous()):
        raise ValueError("Expected contiguous CUDA BF16 input[M,K] with positive M")


def _gemm(a, b, *, transpose_b=False, wgrad=False):
    if wgrad:
        return FP8GemmPlan(a.col, b.col, a.col_scale, b.col_scale,
                           dtype=torch.float32, b_scale_mode="row")
    return FP8GemmPlan(a.row, b.col if transpose_b else b.row, a.row_scale,
                       b.col_scale if transpose_b else b.row_scale)


def _run(plan):
    # Returned tensors and accumulated .grad must survive subsequent calls.
    plan._new_output()
    # Autograd attaches its context to the returned Tensor. Keep that Tensor
    # distinct from the plan's output reference to avoid a state/ctx cycle.
    return plan().detach()


class _Lease:
    """Keep a workspace exclusive for the lifetime of its autograd context.

    In particular, multiple live forwards and retain_graph=True must not overwrite
    saved quantized activations. Each stream has a separate workspace pool.
    """

    def __init__(self, pool, state):
        self.pool, self.state = pool, state

    def __del__(self):
        self.pool.append(self.state)


class _CacheModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache_signature = None
        self._synced_versions = None
        self._pools = {}

    def _weights(self):
        raise NotImplementedError

    @torch.no_grad()
    def refresh_fp8_weights(self):
        weights = self._weights()
        if any(w.dtype != torch.float32 or not w.is_cuda or not w.is_contiguous() for w in weights):
            raise ValueError("FP8 modules require contiguous CUDA FP32 master weights")
        signature = tuple((w.device, w.data_ptr(), tuple(w.shape)) for w in weights)
        if signature != self._cache_signature:
            self._quantized = [BlockQuantized(*w.shape, w.device) for w in weights]
            self._pools = {}
            self._cache_signature = signature
        for q, w in zip(self._quantized, weights):
            q.quantize(w)
        self._synced_versions = tuple(w._version for w in weights)

    def _sync_if_needed(self):
        weights = self._weights()
        signature = tuple((w.device, w.data_ptr(), tuple(w.shape)) for w in weights)
        if (signature != self._cache_signature
                or tuple(w._version for w in weights) != self._synced_versions):
            self.refresh_fp8_weights()

    def mark_fp8_dirty(self):
        self._synced_versions = None

    def bind_optimizer(self, optimizer):
        return bind_fp8_optimizer(self, optimizer)

    def _lease(self, x, factory):
        self._sync_if_needed()
        key = (tuple(x.shape), torch.cuda.current_stream(x.device).cuda_stream)
        pool = self._pools.setdefault(key, [])
        return _Lease(pool, pool.pop() if pool else factory(x, self._quantized))


class _LinearState:
    def __init__(self, x, weights):
        w, = weights
        m, k = x.shape
        n = w.row.shape[0]
        self.x = DualQuantized(m, k, x.device)
        self.dy = DualQuantized(m, n, x.device)
        self.forward = _gemm(self.x, w)
        self.dx = _gemm(self.dy, w, transpose_b=True)
        self.dw = _gemm(self.dy, self.x, wgrad=True)


class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, layer):
        ctx.lease = layer._lease(x, _LinearState)
        state = ctx.lease.state
        # Version checking catches updates between forward and backward.
        ctx.save_for_backward(weight)
        state.x.quantize(x)
        return _run(state.forward)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        _ = ctx.saved_tensors
        state = ctx.lease.state
        state.dy.quantize(dy.contiguous())
        return (_run(state.dx) if ctx.needs_input_grad[0] else None,
                _run(state.dw) if ctx.needs_input_grad[1] else None, None)


class FP8Linear(_CacheModule):
    """FP32 master weights, BF16 inputs/outputs and dX, FP32 dW.

    Activation/gradient quantization is independently rowwise and columnwise
    at granularity 128; weights use 128x128 blocks. Backward uses quantized
    operands as a first-order training approximation (no double backward).
    """

    def __init__(self, in_features, out_features, *, device=None):
        super().__init__()
        _features(in_features, out_features)
        device = require_sm120(device)
        ensure_extensions("fp8_gemm")
        self.in_features, self.out_features = in_features, out_features
        self.master_weight = nn.Parameter(torch.empty(out_features, in_features,
                                                       device=device, dtype=torch.float32))
        self.reset_parameters()

    @property
    def weight(self):
        return self.master_weight

    def _weights(self):
        return (self.master_weight,)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        self.refresh_fp8_weights()

    @property
    def quantized_weight(self):
        self._sync_if_needed()
        q, = self._quantized
        return q.row, q.row_scale

    def forward(self, x):
        _input(x, self.master_weight)
        return _LinearFunction.apply(x, self.master_weight, self)


class _SwiGLUState:
    def __init__(self, x, weights):
        wg, wu = weights
        m, k = x.shape
        h = wg.row.shape[0]
        self.x = DualQuantized(m, k, x.device)
        self.dg = DualQuantized(m, h, x.device)
        self.du = DualQuantized(m, h, x.device)
        self.gate = torch.empty((m, h), device=x.device, dtype=torch.bfloat16)
        self.up = torch.empty_like(self.gate)
        self.forward = _FP8DualPlan(
            self.x.row, wg.row, self.x.row_scale, wg.row_scale,
            a2=self.x.row, b2=wu.row, sa2=self.x.row_scale, sb2=wu.row_scale,
            mode=1, out=torch.empty_like(self.gate), gate=self.gate, up=self.up,
        )
        self.dx = _FP8DualPlan(
            self.dg.row, wg.col, self.dg.row_scale, wg.col_scale,
            a2=self.du.row, b2=wu.col, sa2=self.du.row_scale, sb2=wu.col_scale,
            mode=2, out=torch.empty_like(x),
        )
        self.dwg = _gemm(self.dg, self.x, wgrad=True)
        self.dwu = _gemm(self.du, self.x, wgrad=True)


class _SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, up, layer):
        ctx.lease = layer._lease(x, _SwiGLUState)
        state = ctx.lease.state
        ctx.save_for_backward(gate, up)
        state.x.quantize(x)
        return _run(state.forward)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        _ = ctx.saved_tensors
        state = ctx.lease.state
        state.dg.silu_backward(dy.contiguous(), state.gate, state.up, state.du)
        return (_run(state.dx) if ctx.needs_input_grad[0] else None,
                _run(state.dwg) if ctx.needs_input_grad[1] else None,
                _run(state.dwu) if ctx.needs_input_grad[2] else None, None)


class FP8SwiGLU(_CacheModule):
    """Bias-free down(silu(gate(x)) * up(x)) with K128 FP32 scales.

    Gate/up forward and their summed dX are fused CUDA GEMMs using QMMA.SF.
    The SiLU backward is fused with both gradient quantizations. Gate and up
    projections are rounded to BF16 before SiLU; SiLU and its derivative are
    evaluated in FP32, then activations/gradients are rounded to BF16.
    All feature dimensions must be multiples of 128; M may be any positive size.
    """

    def __init__(self, in_features, hidden_features, out_features, *, device=None):
        super().__init__()
        _features(in_features, hidden_features, out_features)
        device = require_sm120(device)
        ensure_extensions("fp8_gemm")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gate_weight = nn.Parameter(torch.empty(hidden_features, in_features,
                                                     device=device, dtype=torch.float32))
        self.up_weight = nn.Parameter(torch.empty_like(self.gate_weight))
        self.down_proj = FP8Linear(hidden_features, out_features, device=device)
        self.reset_parameters()

    def _weights(self):
        return self.gate_weight, self.up_weight

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        self.down_proj.reset_parameters()
        self.refresh_fp8_weights()

    def refresh_fp8_weights(self):
        super().refresh_fp8_weights()
        self.down_proj.refresh_fp8_weights()

    def mark_fp8_dirty(self):
        super().mark_fp8_dirty()
        self.down_proj.mark_fp8_dirty()

    def forward(self, x):
        _input(x, self.gate_weight)
        hidden = _SwiGLUFunction.apply(x, self.gate_weight, self.up_weight, self)
        return self.down_proj(hidden)


FP8DownProjection = FP8Linear
FP8MLP = FP8SwiGLU
