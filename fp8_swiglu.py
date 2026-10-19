import math

import torch
from torch import nn

import mxtest
from fp8_linear import FP8Linear, bind_fp8_optimizer


@torch.library.custom_op("fp8gemm::swiglu_fwd", mutates_args=())
def _fp8_swiglu_fwd(
    x: torch.Tensor,
    gate_master: torch.Tensor,
    up_master: torch.Tensor,
    down_master: torch.Tensor,
    gate_fp8: torch.Tensor,
    gate_scale: torch.Tensor,
    up_fp8: torch.Tensor,
    up_scale: torch.Tensor,
    down_fp8: torch.Tensor,
    down_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_fp8_wgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return out, gate, up, hidden


@_fp8_swiglu_fwd.register_fake
def _fp8_swiglu_fwd_fake(
    x: torch.Tensor,
    gate_master: torch.Tensor,
    up_master: torch.Tensor,
    down_master: torch.Tensor,
    gate_fp8: torch.Tensor,
    gate_scale: torch.Tensor,
    up_fp8: torch.Tensor,
    up_scale: torch.Tensor,
    down_fp8: torch.Tensor,
    down_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_fp8_wgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_features = gate_fp8.shape[0]
    out_features = down_master.shape[0]
    return (
        torch.empty(
            (x.shape[0], out_features),
            dtype=torch.bfloat16,
            device=x.device,
        ),
        torch.empty(
            (x.shape[0], hidden_features),
            dtype=torch.bfloat16,
            device=x.device,
        ),
        torch.empty(
            (x.shape[0], hidden_features),
            dtype=torch.bfloat16,
            device=x.device,
        ),
        torch.empty(
            (x.shape[0], hidden_features),
            dtype=torch.bfloat16,
            device=x.device,
        ),
    )


@torch.library.custom_op("fp8gemm::swiglu_bwd", mutates_args=())
def _fp8_swiglu_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    hidden: torch.Tensor,
    gate_fp8: torch.Tensor,
    gate_scale: torch.Tensor,
    up_fp8: torch.Tensor,
    up_scale: torch.Tensor,
    down_fp8: torch.Tensor,
    down_scale: torch.Tensor,
    use_fp8_wgrad: bool,
    has_bias: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    grad_output = grad_output.contiguous()
    if grad_output.dtype != torch.bfloat16:
        grad_output = grad_output.to(torch.bfloat16)

    grad_x, grad_gate, grad_up, grad_down = (
        mxtest.launch_fp8_down_swiglu_backward(
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
            use_fp8_wgrad=use_fp8_wgrad,
        )
    )
    # Keep the custom-op output schema tensor-only without launching work when
    # the layer has no bias. Autograd discards this empty placeholder.
    grad_bias = (
        grad_output.sum(dim=0, dtype=torch.float32)
        if has_bias else torch.empty(0, device=grad_output.device, dtype=torch.float32)
    )
    return grad_x, grad_gate, grad_up, grad_down, grad_bias


@_fp8_swiglu_bwd.register_fake
def _fp8_swiglu_bwd_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    hidden: torch.Tensor,
    gate_fp8: torch.Tensor,
    gate_scale: torch.Tensor,
    up_fp8: torch.Tensor,
    up_scale: torch.Tensor,
    down_fp8: torch.Tensor,
    down_scale: torch.Tensor,
    use_fp8_wgrad: bool,
    has_bias: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    hidden_features, in_features = gate_fp8.shape
    out_features = down_fp8.shape[0]
    return (
        torch.empty(
            (x.shape[0], in_features),
            dtype=torch.bfloat16,
            device=x.device,
        ),
        torch.empty(
            (hidden_features, in_features),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty(
            (hidden_features, in_features),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty(
            (out_features, hidden_features),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty(
            (out_features if has_bias else 0,),
            dtype=torch.float32,
            device=x.device,
        ),
    )


def _fp8_swiglu_setup_context(ctx, inputs, output):
    (
        x,
        _,
        _,
        _,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
        down_fp8,
        down_scale,
        bias,
        use_fp8_wgrad,
    ) = inputs
    _, gate, up, hidden = output
    # These outputs only carry saved activations; their gradients are unused.
    ctx.mark_non_differentiable(gate, up, hidden)
    ctx.set_materialize_grads(False)
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


def _fp8_swiglu_backward(ctx, *grad_outputs):
    grad_output = grad_outputs[0]
    if grad_output is None:
        return (None,) * 12
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
    grad_x, grad_gate, grad_up, grad_down, grad_bias = _fp8_swiglu_bwd(
        grad_output,
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
        ctx.use_fp8_wgrad,
        ctx.has_bias,
    )
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
        grad_bias if ctx.has_bias else None,
        None,
    )


torch.library.register_autograd(
    _fp8_swiglu_fwd,
    _fp8_swiglu_backward,
    setup_context=_fp8_swiglu_setup_context,
)


class FP8SwiGLU(nn.Module):
    """FP8 SwiGLU MLP with FP32 master weights and BF16 activations.

    The gate and up projections are fused, and the down projection reuses
    FP8Linear. Input gradients use FP8 block-scaled GEMMs. Weight gradients
    default to BF16 inputs with FP32 accumulation/output for better transpose
    GEMM throughput; set ``fp8_wgrad=True`` to force the FP8 weight-gradient
    kernels. Parameter updates are synchronized lazily before forward or when
    a quantized-weight property is accessed. Inside ``torch.compile``, refresh
    after optimizer steps with ``step_fp8_optimizers`` or
    ``bind_fp8_optimizer`` because version checks are skipped in the graph.
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
        self._fp8_dirty = False
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
        self._fp8_dirty = False
        return self

    def mark_fp8_dirty(self):
        """Force the next quantized-weight access to refresh all FP8 caches."""
        self._fp8_dirty = True
        return self

    def _sync_if_needed(self):
        if torch.compiler.is_compiling():
            return
        versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.down_proj.master_weight._version,
        )
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    @property
    def quantized_gate_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_if_needed()
        return self._gate_weight_fp8, self._gate_weight_scale

    @property
    def quantized_up_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_if_needed()
        return self._up_weight_fp8, self._up_weight_scale

    @property
    def quantized_down_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_if_needed()
        return self.down_proj.quantized_weight

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        """Optionally refresh all FP8 weights after every optimizer step."""
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()

        self._optimizer_hook = bind_fp8_optimizer(self, optimizer)
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
        gate_fp8, gate_scale = self.quantized_gate_weight
        up_fp8, up_scale = self.quantized_up_weight
        down_fp8, down_scale = self.quantized_down_weight
        out, _, _, _ = _fp8_swiglu_fwd(
            x,
            self.gate_weight,
            self.up_weight,
            self.down_proj.master_weight,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            down_fp8,
            down_scale,
            self.down_proj.bias,
            self.fp8_wgrad,
        )
        return out

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
    old_gate_fp8 = layer.quantized_gate_weight[0].clone()
    old_down_fp8 = layer.quantized_down_weight[0].clone()
    optimizer.step()

    assert not torch.equal(layer.quantized_gate_weight[0], old_gate_fp8)
    assert not torch.equal(layer.quantized_down_weight[0], old_down_fp8)

    print(
        "FP8SwiGLU check passed:",
        f"output={tuple(out.shape)}/{out.dtype}",
        f"grad_input={tuple(x.grad.shape)}/{x.grad.dtype}",
        f"gate_grad={layer.gate_weight.grad.dtype}",
        f"up_grad={layer.up_weight.grad.dtype}",
        f"down_grad={layer.down_proj.master_weight.grad.dtype}",
        f"bias_grad={layer.down_proj.bias.grad.dtype}",
    )
