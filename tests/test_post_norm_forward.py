"""Forward tests for postnorm residual linear."""

import pytest
import torch

import mxtest
from experiments.flash_norm import launch_inverse_rms
from experiments.post_norm import PostNormResidualLinear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="postnorm forward tests require CUDA",
)


@pytest.mark.parametrize("bias", [False, True])
def test_post_norm_residual_forward_matches_reference(bias):
    torch.manual_seed(80)
    M, N, K = 37, 80, 96
    layer = PostNormResidualLinear(
        K,
        N,
        bias=bias,
        device="cuda",
    )
    z = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    residual_source = torch.randn(
        M,
        N,
        device="cuda",
        dtype=torch.bfloat16,
    )
    factor = torch.randn(N, device="cuda").mul_(0.1).add_(1.0)
    inverse_rms = launch_inverse_rms(residual_source, eps=1e-6)

    actual = layer(
        z,
        residual_source,
        inverse_rms,
        factor,
    )

    z_fp8, z_scale = mxtest.launch_row_quantize(z)
    weight_fp8, weight_scale = layer.quantized_weight
    linear = (
        mxtest.dequantize_row(z_fp8, z_scale)
        @ mxtest.dequantize_weight(weight_fp8, weight_scale).T
    )
    if layer.bias is not None:
        linear += layer.bias.detach().float()[None, :]
    scaled_residual = (
        residual_source.float()
        * inverse_rms[:, None]
        * factor[None, :]
    )
    expected = (linear + scaled_residual).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert actual.shape == (M, N)
    assert actual.dtype == torch.bfloat16


def test_post_norm_residual_tracks_weight_updates():
    torch.manual_seed(81)
    M, N, K = 32, 64, 96
    layer = PostNormResidualLinear(K, N, bias=True, device="cuda")
    z = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    residual_source = torch.randn(
        M,
        N,
        device="cuda",
        dtype=torch.bfloat16,
    )
    factor = torch.ones(N, device="cuda")
    inverse_rms = launch_inverse_rms(residual_source, eps=1e-6)
    before = layer(z, residual_source, inverse_rms, factor)

    with torch.no_grad():
        layer.master_weight.add_(0.01)
    after = layer(z, residual_source, inverse_rms, factor)
    assert not torch.equal(before, after)
