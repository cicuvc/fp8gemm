import math

import torch
from torch import nn

import mxtest
from fp8_linear import FP8Linear


class _FP8SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        gate_master,
        up_master,
        down_master,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
        down_fp8,
        down_scale,
        bias,
        use_fp8_wgrad,
    ):
        gate = torch.empty(
            (x.shape[0], gate_fp8.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        up = torch.empty_like(gate)
        hidden = mxtest.launch_fp8_swiglu(
            x,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            gate,
            up,
        )
        out = mxtest.launch_fp8gemm(hidden, down_fp8, down_scale, bias)
        ctx.save_for_backward(
            x,
            gate,
            up,
            hidden,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            down_fp8,
            down_scale,
        )
        ctx.has_bias = bias is not None
        ctx.use_fp8_wgrad = use_fp8_wgrad
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            gate,
            up,
            hidden,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            down_fp8,
            down_scale,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        grad_x, grad_gate, grad_up, grad_down = mxtest.launch_fp8_down_swiglu_backward(
            grad_output,
            hidden,
            gate,
            up,
            down_fp8,
            down_scale,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            x,
            use_fp8_wgrad=ctx.use_fp8_wgrad,
        )
        grad_bias = grad_output.float().sum(dim=0) if ctx.has_bias else None
        return (
            grad_x,
            grad_gate,
            grad_up,
            grad_down,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_bias,
            None,
        )


class FP8SwiGLU(nn.Module):
    """FP8 SwiGLU MLP with FP32 master weights and BF16 activations.

    The gate and up projections are fused, and the down projection reuses
    FP8Linear. Input gradients use FP8 block-scaled GEMMs. Weight gradients
    default to BF16 inputs with FP32 accumulation/output for better transpose
    GEMM throughput; set ``fp8_wgrad=True`` to force the FP8 weight-gradient
    kernels.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        fp8_wgrad: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fp8_wgrad = fp8_wgrad

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
        )

        self.register_buffer(
            "_gate_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "_gate_weight_scale",
            torch.empty(0, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_scale",
            torch.empty(0, dtype=torch.uint8),
            persistent=False,
        )

        self._synced_versions = None
        self._optimizer_hook = None
        self.reset_parameters()
        self.refresh_fp8_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        self.down_proj.reset_parameters()

    def refresh_fp8_weights(self):
        if not self.gate_weight.is_cuda:
            raise RuntimeError("FP8SwiGLU requires a CUDA device")
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
        return self

    def _sync_if_needed(self):
        versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.down_proj.master_weight._version,
        )
        if self._synced_versions != versions:
            self.refresh_fp8_weights()

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        """Refresh all FP8 weights after every optimizer step."""
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()

        def post_step_hook(optimizer, args, kwargs):
            self.refresh_fp8_weights()

        self._optimizer_hook = optimizer.register_step_post_hook(post_step_hook)
        return self

    def forward(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input shape [M, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError(f"expected bfloat16 input, got {x.dtype}")
        if not x.is_cuda:
            raise RuntimeError("FP8SwiGLU requires a CUDA device")

        x = x.contiguous()
        self._sync_if_needed()
        return _FP8SwiGLUFunction.apply(
            x,
            self.gate_weight,
            self.up_weight,
            self.down_proj.master_weight,
            self._gate_weight_fp8,
            self._gate_weight_scale,
            self._up_weight_fp8,
            self._up_weight_scale,
            self.down_proj._weight_fp8,
            self.down_proj._weight_scale,
            self.down_proj.bias,
            self.fp8_wgrad,
        )

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.down_proj.bias is not None}"
        )


if __name__ == "__main__":
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)

    M, K, H, O = 96, 160, 192, 67
    layer = FP8SwiGLU(K, H, O, bias=True, device="cuda:0")
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0", requires_grad=True)

    out = layer(x)
    assert out.shape == (M, O)
    assert out.dtype == torch.bfloat16

    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    assert layer.gate_weight.grad is not None
    assert layer.gate_weight.grad.dtype == torch.float32
    assert layer.up_weight.grad is not None
    assert layer.up_weight.grad.dtype == torch.float32
    assert layer.down_proj.master_weight.grad is not None
    assert layer.down_proj.master_weight.grad.dtype == torch.float32
    assert layer.down_proj.bias.grad is not None
    assert layer.down_proj.bias.grad.dtype == torch.float32

    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    layer.bind_optimizer(optimizer)
    old_gate_fp8 = layer._gate_weight_fp8.clone()
    old_down_fp8 = layer.down_proj._weight_fp8.clone()
    optimizer.step()

    assert not torch.equal(layer._gate_weight_fp8, old_gate_fp8)
    assert not torch.equal(layer.down_proj._weight_fp8, old_down_fp8)

    print(
        "FP8SwiGLU check passed:",
        f"output={tuple(out.shape)}/{out.dtype}",
        f"grad_input={tuple(x.grad.shape)}/{x.grad.dtype}",
        f"gate_grad={layer.gate_weight.grad.dtype}",
        f"up_grad={layer.up_weight.grad.dtype}",
        f"down_grad={layer.down_proj.master_weight.grad.dtype}",
        f"bias_grad={layer.down_proj.bias.grad.dtype}",
    )
