"""Tests for the pair-level postnorm forward/backward."""

import pytest
import torch
import torch.nn.functional as F

from experiments.post_norm_pair import PostNormPair


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="postnorm pair tests require CUDA",
)


@pytest.mark.parametrize("b_output_bf16", [False, True])
def test_post_norm_pair_backward_matches_reference(b_output_bf16):
    torch.manual_seed(90)
    M, D, P, Q = 64, 64, 80, 96
    pair = PostNormPair(
        residual_features=D,
        first_features=P,
        second_input_features=Q,
        bias=True,
        b_output_bf16=b_output_bf16,
        device="cuda",
    )
    with torch.no_grad():
        pair.elementwise_factor.copy_(
            torch.randn(D, device="cuda").mul_(0.1).add_(1.0)
        )
    x = torch.randn(
        M,
        D,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    z = torch.randn(
        M,
        Q,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_first = torch.randn(
        M,
        P,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_second = torch.randn(
        M,
        D,
        device="cuda",
        dtype=torch.bfloat16,
    )

    first_output, second_output = pair(x, z)
    (
        (first_output.float() * grad_first.float()).sum()
        + (second_output.float() * grad_second.float()).sum()
    ).backward()

    x_ref = x.detach().float().requires_grad_(True)
    z_ref = z.detach().float().requires_grad_(True)
    first_weight_ref = pair.first_weight.detach().float().requires_grad_(True)
    second_weight_ref = pair.second_weight.detach().float().requires_grad_(True)
    factor_ref = pair.elementwise_factor.detach().float().requires_grad_(True)
    first_bias_ref = pair.first_bias.detach().float().requires_grad_(True)
    second_bias_ref = pair.second_bias.detach().float().requires_grad_(True)

    inverse_rms = torch.rsqrt(x_ref.square().mean(dim=-1) + pair.eps)
    first_ref = (
        (x_ref @ (first_weight_ref * factor_ref[None, :]).T)
        * inverse_rms[:, None]
        + first_bias_ref
    )
    second_ref = (
        z_ref @ second_weight_ref.T
        + second_bias_ref
        + inverse_rms[:, None] * x_ref * factor_ref[None, :]
    )
    (
        (first_ref * grad_first.float()).sum()
        + (second_ref * grad_second.float()).sum()
    ).backward()

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

    compare(x.grad, x_ref.grad, 0.97, 0.20)
    compare(z.grad, z_ref.grad, 0.98, 0.15)
    compare(pair.first_weight.grad, first_weight_ref.grad, 0.98, 0.15)
    compare(pair.second_weight.grad, second_weight_ref.grad, 0.98, 0.15)
    compare(pair.elementwise_factor.grad, factor_ref.grad, 0.97, 0.20)
    compare(pair.first_bias.grad, first_bias_ref.grad, 0.999, 0.02)
    compare(pair.second_bias.grad, second_bias_ref.grad, 0.999, 0.02)


def test_post_norm_pair_supports_one_unused_output():
    torch.manual_seed(91)
    M, D, P, Q = 32, 64, 80, 96
    pair = PostNormPair(D, P, Q, bias=True, device="cuda")
    x = torch.randn(
        M,
        D,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    z = torch.randn(
        M,
        Q,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    _, second_output = pair(x, z)
    second_output.float().square().mean().backward()
    assert x.grad is not None
    assert z.grad is not None
    assert pair.first_weight.grad is not None
    assert pair.first_weight.grad.abs().sum().item() == 0.0
