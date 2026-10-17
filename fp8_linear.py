import math

import torch
from torch import nn

import mxtest


class _FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master_weight, weight_fp8, weight_scale, bias):
        out = mxtest.launch_fp8gemm(x, weight_fp8, weight_scale, bias)

        ctx.save_for_backward(x, weight_fp8, weight_scale)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_fp8, weight_scale = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        grad_x = mxtest.launch_fp8gemm_backward(grad_output, weight_fp8, weight_scale)
        grad_weight = mxtest.launch_fp8gemm_weight_grad(
            grad_output,
            x,
            out_dtype=torch.float32,
        )
        grad_bias = grad_output.float().sum(dim=0) if ctx.has_bias else None

        return grad_x, grad_weight, None, None, grad_bias


class FP8Linear(nn.Module):
    """Linear layer with FP32 master weights and BF16 activations.

    The forward and input-gradient GEMMs consume one FP8 weight quantized with
    one E8M0 scale per 32x32 block. The weight gradient is accumulated in FP32.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.master_weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_weight_fp8", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer("_weight_scale", torch.empty(0, dtype=torch.uint8), persistent=False)
        self._synced_version = -1
        self._optimizer_hook = None

        self.reset_parameters()
        self.refresh_fp8_weight()

    @property
    def weight(self):
        return self.master_weight

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def refresh_fp8_weight(self):
        if not self.master_weight.is_cuda:
            raise RuntimeError("FP8Linear requires a CUDA device")
        with torch.no_grad():
            self._weight_fp8, self._weight_scale = mxtest.launch_weight_quantize(
                self.master_weight.detach()
            )
            self._synced_version = self.master_weight._version
        return self

    def _sync_if_needed(self):
        if self._synced_version != self.master_weight._version:
            self.refresh_fp8_weight()

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        """Refresh FP8 weights after every optimizer step."""
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()

        def post_step_hook(optimizer, args, kwargs):
            self.refresh_fp8_weight()

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
            raise RuntimeError("FP8Linear requires a CUDA device")

        x = x.contiguous()
        self._sync_if_needed()
        return _FP8LinearFunction.apply(
            x,
            self.master_weight,
            self._weight_fp8,
            self._weight_scale,
            self.bias,
        )

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


if __name__ == "__main__":
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)

    M, N, K = 128, 128, 128
    layer = FP8Linear(K, N, bias=True, device="cuda:0")
    x = torch.randn((M, K), device="cuda:0", dtype=torch.bfloat16, requires_grad=True)

    out = layer(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == (M, N)
    xq, x_scale = mxtest.launch_row_quantize(x.detach())
    expected = (
        mxtest.dequantize_row(xq, x_scale)
        @ mxtest.dequantize_weight(layer._weight_fp8, layer._weight_scale).T
        + layer.bias.detach()[None, :]
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    assert layer.master_weight.grad is not None
    assert layer.master_weight.grad.dtype == torch.float32
    assert layer.bias.grad is not None and layer.bias.grad.dtype == torch.float32

    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
    layer.bind_optimizer(optimizer)
    previous_fp8 = layer._weight_fp8.clone()
    optimizer.step()

    assert not torch.equal(previous_fp8, layer._weight_fp8)
    expected_fp8, expected_scale = mxtest.launch_weight_quantize(layer.master_weight.detach())
    assert torch.equal(layer._weight_fp8, expected_fp8)
    assert torch.equal(layer._weight_scale, expected_scale)

    print(
        "FP8Linear check passed:",
        f"output={tuple(out.shape)}/{out.dtype}",
        f"grad_input={tuple(x.grad.shape)}/{x.grad.dtype}",
        f"master_grad={tuple(layer.master_weight.grad.shape)}/{layer.master_weight.grad.dtype}",
    )
