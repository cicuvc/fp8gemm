"""Complete MXFP8 gate/up SwiGLU layer built from the repository components."""

import math

import torch
from torch import nn

from fp8_linear import FP8Linear
from experiments.tk_mxfp8_dgrad import RowQuantizePlan
from experiments.tk_mxfp8_swiglu import MXFP8SwiGLULibrary
from experiments.tk_mxfp8_swiglu_backward import MXFP8SwiGLUBackward


class _ForwardState:
    def __init__(
        self,
        x,
        weight_gate,
        weight_gate_scale,
        weight_up,
        weight_up_scale,
        config,
        raster_group,
    ):
        self.quantize_x = RowQuantizePlan(x)
        shape = (x.shape[0], weight_gate.shape[0])
        self.gate_linear = torch.empty(shape, device=x.device, dtype=torch.bfloat16)
        self.up_linear = torch.empty_like(self.gate_linear)
        self.forward = MXFP8SwiGLULibrary().plan(
            self.quantize_x.out,
            self.quantize_x.scales,
            weight_gate,
            weight_gate_scale,
            weight_up,
            weight_up_scale,
            config=config,
            raster_group=raster_group,
            gate_out=self.gate_linear,
            up_out=self.up_linear,
        )

    def __call__(self):
        self.quantize_x()
        return self.forward()


class _MXFP8SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, layer):
        layer._sync_if_needed()
        state = layer._forward_state(x)
        out = state()
        ctx.layer = layer
        ctx.save_for_backward(
            x,
            gate_weight,
            up_weight,
            layer._gate_weight_fp8,
            layer._gate_weight_scale,
            layer._up_weight_fp8,
            layer._up_weight_scale,
            state.gate_linear,
            state.up_linear,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            gate_weight,
            up_weight,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            gate_linear,
            up_linear,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)
        layer = ctx.layer
        state = layer._backward_state(
            grad_output,
            gate_linear,
            up_linear,
            x,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
        )
        grad_x, grad_gate, grad_up = state()
        return grad_x, grad_gate.float(), grad_up.float(), None


class MXFP8SwiGLU(nn.Module):
    """FP32-master, BF16-activation gate/up SwiGLU layer.

    ``out = up(x) * silu(gate(x))``. The forward and both weight gradients use
    MXFP8 tensor cores with FP32 accumulation; dX and the public activations
    use BF16. Weight quantization is cached and refreshed from parameter
    versions or eagerly after optimizer steps through ``bind_optimizer``.
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        *,
        bias=True,
        down_backward_backend="cuda",
        device=None,
        forward_config=1,
        dx_config=5,
        wgrad_config=6,
        raster_group=13,
    ):
        super().__init__()
        if in_features % 64 or hidden_features % 64:
            raise ValueError("in_features and hidden_features must be multiples of 64")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.down_backward_backend = down_backward_backend
        self.forward_config = forward_config
        self.dx_config = dx_config
        self.wgrad_config = wgrad_config
        self.raster_group = raster_group
        self.gate_weight = nn.Parameter(
            torch.empty((hidden_features, in_features), dtype=torch.float32, device=device)
        )
        self.up_weight = nn.Parameter(
            torch.empty((hidden_features, in_features), dtype=torch.float32, device=device)
        )
        self.down_proj = FP8Linear(
            hidden_features,
            out_features,
            bias=bias,
            device=device,
            backward_backend=down_backward_backend,
        )
        self.register_buffer(
            "_gate_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_gate_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._synced_versions = None
        self._fp8_dirty = False
        self._forward_states = {}
        self._backward_states = {}
        self._optimizer_hook = None
        self.reset_parameters()
        self.refresh_fp8_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        self.down_proj.reset_parameters()

    def refresh_fp8_weights(self):
        if not self.gate_weight.is_cuda:
            raise RuntimeError("MXFP8SwiGLU requires a CUDA device")
        import mxtest

        with torch.no_grad():
            self._gate_weight_fp8, self._gate_weight_scale = (
                mxtest.launch_weight_quantize(self.gate_weight.detach())
            )
            self._up_weight_fp8, self._up_weight_scale = (
                mxtest.launch_weight_quantize(self.up_weight.detach())
            )
            self.down_proj.refresh_fp8_weight()
        self._synced_versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.down_proj.master_weight._version,
        )
        self._fp8_dirty = False
        self._forward_states.clear()
        self._backward_states.clear()
        return self

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        self.down_proj.mark_fp8_dirty()
        return self

    def _sync_if_needed(self):
        versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.down_proj.master_weight._version,
        )
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    @property
    def quantized_gate_weight(self):
        self._sync_if_needed()
        return self._gate_weight_fp8, self._gate_weight_scale

    @property
    def quantized_up_weight(self):
        self._sync_if_needed()
        return self._up_weight_fp8, self._up_weight_scale

    def _forward_state(self, x):
        key = (
            x.device.index,
            x.data_ptr(),
            tuple(x.shape),
            self._gate_weight_fp8.data_ptr(),
            self._up_weight_fp8.data_ptr(),
            self.forward_config,
            self.raster_group,
        )
        state = self._forward_states.get(key)
        if state is None:
            state = _ForwardState(
                x,
                self._gate_weight_fp8,
                self._gate_weight_scale,
                self._up_weight_fp8,
                self._up_weight_scale,
                self.forward_config,
                self.raster_group,
            )
            self._forward_states = {key: state}
        return state

    def _backward_state(
        self,
        dy,
        gate_linear,
        up_linear,
        x,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
    ):
        key = (
            dy.device.index,
            dy.data_ptr(),
            x.data_ptr(),
            gate_linear.data_ptr(),
            up_linear.data_ptr(),
            gate_fp8.data_ptr(),
            up_fp8.data_ptr(),
            self.dx_config,
            self.wgrad_config,
            self.raster_group,
        )
        state = self._backward_states.get(key)
        if state is None:
            state = MXFP8SwiGLUBackward(
                dy,
                gate_linear,
                up_linear,
                x,
                gate_fp8,
                up_fp8,
                gate_scale,
                up_scale,
                dx_config=self.dx_config,
                wgrad_config=self.wgrad_config,
                raster_group=self.raster_group,
            )
            self._backward_states = {key: state}
        return state

    def bind_optimizer(self, optimizer):
        from fp8_linear import bind_fp8_optimizer

        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()
        self._optimizer_hook = bind_fp8_optimizer(self, optimizer)
        return self

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input [M,{self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError("expected BF16 input")
        if not x.is_cuda:
            raise RuntimeError("MXFP8SwiGLU requires CUDA")
        hidden = _MXFP8SwiGLUFunction.apply(
            x.contiguous(), self.gate_weight, self.up_weight, self
        )
        return self.down_proj(hidden)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, bias={self.down_proj.bias is not None}, "
            f"down_backward_backend={self.down_backward_backend!r}, "
            f"forward_config={self.forward_config}, dx_config={self.dx_config}, "
            f"wgrad_config={self.wgrad_config}"
        )
