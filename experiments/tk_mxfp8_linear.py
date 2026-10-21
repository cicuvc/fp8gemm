"""Bias-free linear layer built from the pure-MXFP8 CUDA kernels."""

import math

import torch
from torch import nn

from experiments.mxfp8_core import bind_fp8_optimizer, quantize_weight
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary


class _ForwardState:
    def __init__(
        self,
        x,
        weight_fp8,
        weight_scale,
        forward_config,
        raster_group,
    ):
        self.quantize = SiluBwdQuantizeLibrary().plan_direct(x)
        self.gemm = GemmLibrary().plan(
            self.quantize.row_fp8,
            weight_fp8,
            self.quantize.row_scale,
            weight_scale,
            config=forward_config,
            raster_group=raster_group,
        )

    def __call__(self):
        self.quantize()
        return self.gemm()


class _BackwardState:
    def __init__(
        self,
        dy,
        x_column,
        x_column_scale,
        weight_fp8,
        weight_scale,
        dx_config,
        dw_config,
        raster_group,
    ):
        self.dy_quantize = SiluBwdQuantizeLibrary().plan_direct(
            dy, transpose=True
        )
        self.dx = DgradLibrary().plan(
            self.dy_quantize.row_fp8,
            weight_fp8,
            self.dy_quantize.row_scale,
            weight_scale,
            config=dx_config,
            raster_group=raster_group,
        )
        self.dw = WgradLibrary().plan(
            self.dy_quantize.column_fp8,
            x_column,
            self.dy_quantize.column_scale,
            x_column_scale,
            config=dw_config,
            raster_group=raster_group,
        )

    def __call__(self):
        self.dy_quantize()
        return self.dx(), self.dw()


class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master_weight, module):
        module._sync_if_needed()
        state = module._forward_state(x)
        out = state()
        ctx.module = module
        ctx.save_for_backward(
            x,
            master_weight,
            module._weight_fp8,
            module._weight_scale,
            state.quantize.column_fp8,
            state.quantize.column_scale,
        )
        return out

    @staticmethod
    def backward(ctx, dy):
        x, master_weight, weight_fp8, weight_scale, x_column, x_column_scale = (
            ctx.saved_tensors
        )
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        state = ctx.module._backward_state(
            dy,
            x_column,
            x_column_scale,
            weight_fp8,
            weight_scale,
        )
        dx, dw = state()
        return dx, dw.float(), None


class MXFP8Linear(nn.Module):
    """Bias-free FP32-master linear using pure-MXFP8 forward and backward."""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        device=None,
        forward_config=8,
        dx_config=12,
        dw_config=12,
        raster_group=13,
    ):
        super().__init__()
        if in_features % 64 or out_features % 64:
            raise ValueError("in_features and out_features must be multiples of 64")
        self.in_features = in_features
        self.out_features = out_features
        self.forward_config = forward_config
        self.dx_config = dx_config
        self.dw_config = dw_config
        self.raster_group = raster_group
        self.master_weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), dtype=torch.float32, device=device
            )
        )
        self.register_buffer(
            "_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._synced_version = -1
        self._fp8_dirty = False
        self._forward_states = {}
        self._backward_states = {}
        self._optimizer_hook = None
        self.reset_parameters()
        self.refresh_fp8_weight()

    @property
    def weight(self):
        return self.master_weight

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))

    def refresh_fp8_weight(self):
        if not self.master_weight.is_cuda:
            raise RuntimeError("MXFP8Linear requires CUDA")
        with torch.no_grad():
            self._weight_fp8, self._weight_scale = quantize_weight(
                self.master_weight.detach()
            )
        self._synced_version = self.master_weight._version
        self._fp8_dirty = False
        self._forward_states.clear()
        self._backward_states.clear()
        return self

    def refresh_fp8_weights(self):
        return self.refresh_fp8_weight()

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        return self

    def _sync_if_needed(self):
        if self._fp8_dirty or self._synced_version != self.master_weight._version:
            self.refresh_fp8_weight()

    @property
    def quantized_weight(self):
        self._sync_if_needed()
        return self._weight_fp8, self._weight_scale

    def _forward_state(self, x):
        key = (
            x.device.index,
            x.data_ptr(),
            tuple(x.shape),
            self._weight_fp8.data_ptr(),
            self.forward_config,
            self.raster_group,
        )
        state = self._forward_states.get(key)
        if state is None:
            state = _ForwardState(
                x,
                self._weight_fp8,
                self._weight_scale,
                self.forward_config,
                self.raster_group,
            )
            self._forward_states = {key: state}
        return state

    def _backward_state(
        self,
        dy,
        x_column,
        x_column_scale,
        weight_fp8,
        weight_scale,
    ):
        key = (
            dy.device.index,
            dy.data_ptr(),
            x_column.data_ptr(),
            weight_fp8.data_ptr(),
            self.dx_config,
            self.dw_config,
            self.raster_group,
        )
        state = self._backward_states.get(key)
        if state is None:
            state = _BackwardState(
                dy,
                x_column,
                x_column_scale,
                weight_fp8,
                weight_scale,
                self.dx_config,
                self.dw_config,
                self.raster_group,
            )
            self._backward_states = {key: state}
        return state

    def bind_optimizer(self, optimizer):
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()
        self._optimizer_hook = bind_fp8_optimizer(self, optimizer)
        return self

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input [M,{self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError("expected CUDA BF16 input")
        return _MXFP8LinearFunction.apply(
            x.contiguous(), self.master_weight, self
        )

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, forward_config={self.forward_config}, "
            f"dx_config={self.dx_config}, dw_config={self.dw_config}"
        )
