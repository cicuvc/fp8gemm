"""Tests for fused FP8 linear + residual output."""

import pytest
import torch

from experiments.residual_linear import FP8LinearResidual
from fp8_linear import FP8Linear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="residual FP8 linear tests require CUDA",
)


@pytest.mark.parametrize("bias", [False, True])
def test_residual_linear_forward_backward_matches_separate_add(bias):
    torch.manual_seed(70)
    M, N, K = 48, 96, 128
    residual_layer = FP8LinearResidual(
        K,
        N,
        bias=bias,
        device="cuda",
    )
    reference = FP8Linear(K, N, bias=bias, device="cuda")
    reference.load_state_dict(residual_layer.state_dict())
    reference.refresh_fp8_weight()

    x_fused = torch.randn(
        M,
        K,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    x_reference = x_fused.detach().clone().requires_grad_(True)
    residual_fused = torch.randn(
        M,
        N,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    residual_reference = residual_fused.detach().clone().requires_grad_(True)

    fused_output = residual_layer(x_fused, residual_fused)
    reference_output = reference(x_reference) + residual_reference
    torch.testing.assert_close(
        fused_output,
        reference_output,
        rtol=1e-2,
        atol=1e-2,
    )

    grad_output = torch.randn_like(fused_output)
    fused_output.backward(grad_output)
    reference_output.backward(grad_output)

    torch.testing.assert_close(x_fused.grad, x_reference.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        residual_layer.master_weight.grad,
        reference.master_weight.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        residual_fused.grad,
        grad_output,
        rtol=0,
        atol=0,
    )
    if bias:
        torch.testing.assert_close(
            residual_layer.bias.grad,
            reference.bias.grad,
            rtol=1e-2,
            atol=1e-2,
        )


def test_residual_linear_supports_torch_compile():
    torch.manual_seed(71)
    M, N, K = 32, 64, 96
    layer = FP8LinearResidual(K, N, bias=True, device="cuda")
    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(
        M,
        K,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    residual = torch.randn(
        M,
        N,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = compiled(x, residual)
    output.float().square().mean().backward()

    assert output.shape == (M, N)
    assert output.dtype == torch.bfloat16
    assert x.grad is not None
    assert residual.grad is not None
