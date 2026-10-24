"""GPU integration tests for the public MXFP8 module facade."""

import pytest
import torch

from mxfp8 import MXFP8Linear, MXFP8RMSNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def test_public_linear_builds_and_has_bias_free_parameters():
    layer = MXFP8Linear(
        128,
        64,
        device="cuda",
        forward_config=0,
        dx_config=2,
        dw_config=0,
        raster_group=1,
    )
    assert [name for name, _ in layer.named_parameters()] == ["master_weight"]
    x = torch.randn((137, 128), device="cuda", dtype=torch.bfloat16)
    y = layer(x)
    assert y.shape == (137, 64)
    assert y.dtype == torch.bfloat16


def test_public_rmsnorm_forward_and_backward():
    torch.manual_seed(932)
    hidden_size = 128
    norm = MXFP8RMSNorm(hidden_size, eps=1e-6, device="cuda")
    with torch.no_grad():
        norm.weight.copy_(
            torch.randn(hidden_size, device="cuda").mul_(0.1).add_(1)
        )
    x = torch.randn(
        (33, hidden_size), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    reference_weight = norm.weight.detach().float().requires_grad_()
    reference_x = x.detach().float().requires_grad_()
    reference_inverse = torch.rsqrt(
        reference_x.square().mean(1, keepdim=True) + norm.eps
    )
    reference = (
        reference_x * reference_inverse * reference_weight[None, :]
    ).bfloat16()
    dy = torch.randn_like(reference)
    reference.backward(dy.float())

    output = norm(x)
    dx, dweight = torch.autograd.grad(output, (x, norm.weight), dy)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(dx, reference_x.grad.bfloat16(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        dweight,
        reference_weight.grad.bfloat16(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_public_rmsnorm_exposes_dual_quantized_layouts():
    norm = MXFP8RMSNorm(128, device="cuda")
    x = torch.randn((17, 128), device="cuda", dtype=torch.bfloat16)
    output, row, row_scale, column, column_scale, inverse = norm.quantized(x)
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert row.shape == x.shape and row.dtype == torch.float8_e4m3fn
    assert row_scale.dtype == torch.int32
    assert column.shape == (128, 128)
    assert column_scale.dtype == torch.int32
    assert inverse.shape == (17,) and inverse.dtype == torch.float32
