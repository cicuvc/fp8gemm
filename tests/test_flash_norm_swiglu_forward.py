"""Forward tests for prenorm RMSNorm fused into FP8 SwiGLU."""

import pytest
import torch

import mxtest
from experiments.flash_norm_swiglu import (
    FlashNormSwiGLU,
    launch_rmsnorm,
)
from fp8_swiglu import FP8SwiGLU


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashNorm SwiGLU tests require CUDA",
)


def test_flash_norm_swiglu_matches_materialized_forward():
    torch.manual_seed(100)
    M, K, H, O = 64, 96, 128, 64
    fused = FlashNormSwiGLU(K, H, O, device="cuda")
    baseline = FP8SwiGLU(
        K,
        H,
        O,
        bias=False,
        device="cuda",
    )
    with torch.no_grad():
        baseline.gate_weight.copy_(fused.gate_weight)
        baseline.up_weight.copy_(fused.up_weight)
        baseline.down_proj.master_weight.copy_(fused.down_proj.master_weight)
        fused.elementwise_factor.copy_(
            torch.randn(K, device="cuda").mul_(0.1).add_(1.0)
        )
    baseline.refresh_fp8_weights()

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    normalized = launch_rmsnorm(x, fused.elementwise_factor, fused.eps)
    baseline_hidden = mxtest.launch_fp8_swiglu(
        normalized,
        baseline._gate_weight_fp8,
        baseline._gate_weight_scale,
        baseline._up_weight_fp8,
        baseline._up_weight_scale,
    )
    down_weight, down_scale = baseline.down_proj.quantized_weight
    expected = mxtest.launch_fp8gemm(
        baseline_hidden,
        down_weight,
        down_scale,
    )

    actual = fused(x)
    x_fp8, x_scale = mxtest.launch_row_quantize(x)
    gate_fp8, gate_scale = fused._gate_weight_fp8, fused._gate_weight_scale
    up_fp8, up_scale = fused._up_weight_fp8, fused._up_weight_scale
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + fused.eps)
    gate = (
        mxtest.dequantize_row(x_fp8, x_scale)
        @ mxtest.dequantize_weight(gate_fp8, gate_scale).T
    ) * inverse_rms[:, None]
    up = (
        mxtest.dequantize_row(x_fp8, x_scale)
        @ mxtest.dequantize_weight(up_fp8, up_scale).T
    ) * inverse_rms[:, None]
    hidden_reference = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    hidden_fp8, hidden_scale = mxtest.launch_row_quantize(hidden_reference)
    down_weight, down_scale = fused.down_proj.quantized_weight
    fused_reference = (
        mxtest.dequantize_row(hidden_fp8, hidden_scale)
        @ mxtest.dequantize_weight(down_weight, down_scale).T
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, fused_reference, rtol=1e-2, atol=1e-2)

    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float())
    ).item()
    assert relative_l2 < 1e-1
    assert actual.shape == (M, O)
    assert actual.dtype == torch.bfloat16


def test_flash_norm_swiglu_tracks_factor_updates():
    torch.manual_seed(101)
    M, K, H, O = 32, 96, 128, 64
    layer = FlashNormSwiGLU(K, H, O, device="cuda")
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    before = layer(x)
    with torch.no_grad():
        layer.elementwise_factor.mul_(1.2)
    after = layer(x)
    assert not torch.equal(before, after)
