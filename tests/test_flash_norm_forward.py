"""Forward correctness tests for the FlashNorm linear experiment."""

import pytest
import torch
import torch.nn.functional as F

import mxtest
from experiments.flash_norm import FlashNormLinear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashNorm forward tests require CUDA",
)


def _reference_flash_norm(layer, x):
    effective_weight = (
        layer.master_weight.detach().float()
        * layer.elementwise_factor.detach().float()[None, :]
    ).to(torch.bfloat16)
    weight_fp8, weight_scale = mxtest.launch_weight_quantize(effective_weight)
    x_fp8, x_scale = mxtest.launch_row_quantize(x.detach())
    matmul = (
        mxtest.dequantize_row(x_fp8, x_scale)
        @ mxtest.dequantize_weight(weight_fp8, weight_scale).T
    )
    inverse_rms = torch.rsqrt(
        x.float().square().mean(dim=-1) + layer.eps
    )
    output = matmul * inverse_rms[:, None]
    if layer.bias is not None:
        output = output + layer.bias.detach().float()[None, :]
    return output.to(torch.bfloat16), effective_weight


@pytest.mark.parametrize("bias", [False, True])
def test_flash_norm_forward_matches_dequantized_reference(bias):
    torch.manual_seed(60)
    M, N, K = 37, 80, 96
    layer = FlashNormLinear(K, N, bias=bias, eps=1e-6, device="cuda")
    with torch.no_grad():
        layer.elementwise_factor.copy_(
            torch.randn(K, device="cuda").mul_(0.1).add_(1.0)
        )
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    actual = layer(x)
    expected, _ = _reference_flash_norm(layer, x)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert actual.shape == (M, N)
    assert actual.dtype == torch.bfloat16


def test_flash_norm_weight_cache_tracks_factor_updates():
    torch.manual_seed(61)
    M, N, K = 32, 64, 96
    layer = FlashNormLinear(K, N, bias=True, device="cuda")
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    before = layer(x)

    with torch.no_grad():
        layer.elementwise_factor.mul_(1.3)
    after = layer(x)
    expected, _ = _reference_flash_norm(layer, x)

    assert not torch.equal(before, after)
    torch.testing.assert_close(after, expected, rtol=1e-2, atol=1e-2)


def test_flash_norm_is_close_to_full_precision_expression():
    torch.manual_seed(62)
    M, N, K = 64, 128, 128
    layer = FlashNormLinear(K, N, bias=True, device="cuda")
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    actual = layer(x).float()
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + layer.eps)
    normalized = (
        x.float()
        * inverse_rms[:, None]
        * layer.elementwise_factor.detach().float()[None, :]
    )
    reference = F.linear(
        normalized,
        layer.master_weight.detach().float(),
        layer.bias.detach().float(),
    )

    relative_l2 = (
        torch.linalg.vector_norm(actual - reference)
        / torch.linalg.vector_norm(reference)
    ).item()
    assert relative_l2 < 5e-2


@pytest.mark.parametrize("b_output_bf16", [False, True])
def test_flash_norm_backward_matches_full_precision_reference(b_output_bf16):
    torch.manual_seed(63)
    M, N, K = 64, 96, 128
    layer = FlashNormLinear(
        K,
        N,
        bias=True,
        b_output_bf16=b_output_bf16,
        device="cuda",
    )
    with torch.no_grad():
        layer.elementwise_factor.copy_(
            torch.randn(K, device="cuda").mul_(0.1).add_(1.0)
        )
    x = torch.randn(
        M,
        K,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        M,
        N,
        device="cuda",
        dtype=torch.bfloat16,
    )

    layer(x).backward(grad_output)

    x_reference = x.detach().float().requires_grad_(True)
    weight_reference = (
        layer.master_weight.detach().float().requires_grad_(True)
    )
    factor_reference = (
        layer.elementwise_factor.detach().float().requires_grad_(True)
    )
    bias_reference = layer.bias.detach().float().requires_grad_(True)
    inverse_rms = torch.rsqrt(
        x_reference.square().mean(dim=-1) + layer.eps
    )
    normalized = (
        x_reference
        * inverse_rms[:, None]
        * factor_reference[None, :]
    )
    reference = F.linear(
        normalized,
        weight_reference,
        bias_reference,
    )
    reference.backward(grad_output.float())

    def compare(actual, expected, cosine_threshold, relative_threshold):
        actual = actual.float().flatten()
        expected = expected.float().flatten()
        cosine = F.cosine_similarity(actual, expected, dim=0).item()
        relative_l2 = (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        ).item()
        assert cosine > cosine_threshold
        assert relative_l2 < relative_threshold

    compare(x.grad, x_reference.grad, 0.97, 0.20)
    compare(
        layer.master_weight.grad,
        weight_reference.grad,
        0.98,
        0.15,
    )
    compare(
        layer.elementwise_factor.grad,
        factor_reference.grad,
        0.97,
        0.20,
    )
    compare(layer.bias.grad, bias_reference.grad, 0.999, 0.02)
