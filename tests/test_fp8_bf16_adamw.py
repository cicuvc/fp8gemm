"""Tests for AdamW with reduced-precision optimizer state."""

import pytest
import torch

from fp8_adamw import (
    AdamWStateDtype,
    FusedAdamWMXFP8,
    optimizer_state_dtypes,
    optimizer_state_nbytes,
)
from experiments.mxfp8_core import quantize_weight


def test_bf16_moments_track_fp32_moments():
    torch.manual_seed(20)
    initial = torch.randn(256, 256, dtype=torch.float32)
    gradients = [
        torch.randn_like(initial) * 0.01
        for _ in range(32)
    ]

    parameter_fp32 = initial.clone().requires_grad_(True)
    parameter_bf16 = initial.clone().requires_grad_(True)
    optimizer_fp32 = AdamWStateDtype(
        [parameter_fp32],
        lr=1e-3,
        moment1_dtype=torch.float32,
        moment2_dtype=torch.float32,
    )
    optimizer_bf16 = AdamWStateDtype(
        [parameter_bf16],
        lr=1e-3,
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )

    for gradient in gradients:
        parameter_fp32.grad = gradient.clone()
        parameter_bf16.grad = gradient.clone()
        optimizer_fp32.step()
        optimizer_bf16.step()

    relative_error = (
        torch.linalg.vector_norm(parameter_fp32 - parameter_bf16)
        / torch.linalg.vector_norm(parameter_fp32)
    ).item()
    assert relative_error < 5e-3
    assert optimizer_state_dtypes(optimizer_bf16) == (
        {torch.bfloat16},
        {torch.bfloat16},
    )


def test_bf16_state_uses_half_the_memory():
    parameter_fp32 = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
    parameter_bf16 = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
    optimizer_fp32 = AdamWStateDtype(
        [parameter_fp32],
        moment1_dtype=torch.float32,
        moment2_dtype=torch.float32,
    )
    optimizer_bf16 = AdamWStateDtype(
        [parameter_bf16],
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )

    gradient = torch.ones_like(parameter_fp32)
    parameter_fp32.grad = gradient.clone()
    parameter_bf16.grad = gradient.clone()
    optimizer_fp32.step()
    optimizer_bf16.step()

    fp32_bytes = optimizer_state_nbytes(optimizer_fp32)
    bf16_bytes = optimizer_state_nbytes(optimizer_bf16)
    assert bf16_bytes * 2 == fp32_bytes


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for BF16 parameter updates",
)
def test_bf16_moments_support_bf16_parameters():
    torch.manual_seed(21)
    parameter = torch.nn.Parameter(
        torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    )
    optimizer = AdamWStateDtype(
        [parameter],
        lr=1e-3,
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )
    for _ in range(8):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert parameter.dtype == torch.bfloat16
    assert torch.isfinite(parameter).all()
    assert optimizer_state_dtypes(optimizer) == (
        {torch.bfloat16},
        {torch.bfloat16},
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)
def test_fused_adamw_updates_fp32_master_and_mxfp8_outputs():
    torch.manual_seed(22)
    initial = torch.randn((64, 96), device="cuda") * 0.02
    reference = torch.nn.Parameter(initial.clone())
    fused = torch.nn.Parameter(initial.clone())
    reference_optimizer = AdamWStateDtype(
        [reference],
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.03,
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )
    fused_optimizer = FusedAdamWMXFP8(
        [fused],
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.03,
    )
    weight_fp8, weight_scale = quantize_weight(fused.detach())
    fused_optimizer.bind_fp8_parameter(fused, weight_fp8, weight_scale)

    for _ in range(12):
        gradient = torch.randn_like(fused) * 0.01
        reference.grad = gradient.clone()
        fused.grad = gradient.clone()
        reference_optimizer.step()
        fused_optimizer.step()

    torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-6)
    assert fused.dtype == torch.float32
    assert optimizer_state_dtypes(fused_optimizer) == (
        {torch.bfloat16},
        {torch.bfloat16},
    )
    expected_fp8, expected_scale = quantize_weight(fused.detach())
    assert torch.equal(
        weight_fp8.view(torch.uint8), expected_fp8.view(torch.uint8)
    )
    assert torch.equal(weight_scale, expected_scale)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)
def test_fused_adamw_parameter_groups_and_bf16_parameters():
    torch.manual_seed(23)
    first = torch.nn.Parameter(torch.randn((32, 64), device="cuda") * 0.1)
    second = torch.nn.Parameter(torch.randn((32, 64), device="cuda") * 0.1)
    norm = torch.nn.Parameter(torch.ones(64, device="cuda", dtype=torch.bfloat16))
    reference_first = torch.nn.Parameter(first.detach().clone())
    reference_second = torch.nn.Parameter(second.detach().clone())
    reference_norm = torch.nn.Parameter(norm.detach().clone())

    fused = FusedAdamWMXFP8(
        [
            {
                "params": [first],
                "lr": 1e-3,
                "betas": (0.7, 0.9),
                "eps": 1e-6,
                "weight_decay": 0.01,
            },
            {
                "params": [second, norm],
                "lr": 2e-3,
                "betas": (0.9, 0.99),
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        ]
    )
    output, scales = quantize_weight(first.detach())
    fused.bind_fp8_parameter(first, output, scales)
    reference_first_optimizer = AdamWStateDtype(
        [reference_first],
        lr=1e-3,
        betas=(0.7, 0.9),
        eps=1e-6,
        weight_decay=0.01,
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )
    reference_second_optimizer = AdamWStateDtype(
        [reference_second, reference_norm],
        lr=2e-3,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.1,
        moment1_dtype=torch.bfloat16,
        moment2_dtype=torch.bfloat16,
    )

    for _ in range(5):
        first.grad = torch.randn_like(first)
        second.grad = torch.randn_like(second)
        norm.grad = torch.randn_like(norm) * 0.01
        reference_first.grad = first.grad.clone()
        reference_second.grad = second.grad.clone()
        reference_norm.grad = norm.grad.clone()
        fused.step()
        reference_first_optimizer.step()
        reference_second_optimizer.step()

    torch.testing.assert_close(first, reference_first, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(second, reference_second, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(norm, reference_norm, rtol=2e-2, atol=2e-3)
    assert norm.dtype == torch.bfloat16
