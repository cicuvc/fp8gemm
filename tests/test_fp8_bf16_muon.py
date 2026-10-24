"""Tests for Muon with reduced-precision momentum state."""

import pytest
import torch

from fp8_adamw import optimizer_state_nbytes
from fp8_muon import MuonStateDtype, muon_state_dtypes


@pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="torch.optim.Muon is unavailable in this PyTorch build",
)
def test_fp32_muon_state_dtype_matches_torch_muon():
    torch.manual_seed(30)
    initial = torch.randn(32, 64)
    gradients = [
        torch.randn_like(initial) * 0.01
        for _ in range(8)
    ]

    parameter_reference = initial.clone().requires_grad_(True)
    parameter_custom = initial.clone().requires_grad_(True)
    reference = torch.optim.Muon(
        [parameter_reference],
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="original",
    )
    custom = MuonStateDtype(
        [parameter_custom],
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="original",
        momentum_dtype=torch.float32,
    )

    for gradient in gradients:
        parameter_reference.grad = gradient.clone()
        parameter_custom.grad = gradient.clone()
        reference.step()
        custom.step()

    torch.testing.assert_close(parameter_custom, parameter_reference, rtol=0, atol=0)


def test_bf16_momentum_tracks_fp32_momentum():
    torch.manual_seed(31)
    initial = torch.randn(32, 64)
    gradients = [
        torch.randn_like(initial) * 0.01
        for _ in range(16)
    ]

    parameter_fp32 = initial.clone().requires_grad_(True)
    parameter_bf16 = initial.clone().requires_grad_(True)
    optimizer_fp32 = MuonStateDtype(
        [parameter_fp32],
        lr=0.02,
        momentum_dtype=torch.float32,
    )
    optimizer_bf16 = MuonStateDtype(
        [parameter_bf16],
        lr=0.02,
        momentum_dtype=torch.bfloat16,
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
    assert relative_error < 5e-4
    assert muon_state_dtypes(optimizer_bf16) == {torch.bfloat16}


def test_bf16_momentum_uses_half_the_memory():
    parameter_fp32 = torch.nn.Parameter(torch.zeros(32, 64))
    parameter_bf16 = torch.nn.Parameter(torch.zeros(32, 64))
    optimizer_fp32 = MuonStateDtype(
        [parameter_fp32],
        momentum_dtype=torch.float32,
    )
    optimizer_bf16 = MuonStateDtype(
        [parameter_bf16],
        momentum_dtype=torch.bfloat16,
    )

    gradient = torch.ones_like(parameter_fp32)
    parameter_fp32.grad = gradient.clone()
    parameter_bf16.grad = gradient.clone()
    optimizer_fp32.step()
    optimizer_bf16.step()

    fp32_bytes = optimizer_state_nbytes(optimizer_fp32)
    bf16_bytes = optimizer_state_nbytes(optimizer_bf16)
    assert bf16_bytes * 2 == fp32_bytes
