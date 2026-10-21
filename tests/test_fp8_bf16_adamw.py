"""Tests for AdamW with reduced-precision optimizer state."""

import torch

from fp8_adamw import (
    AdamWStateDtype,
    optimizer_state_dtypes,
    optimizer_state_nbytes,
)


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
