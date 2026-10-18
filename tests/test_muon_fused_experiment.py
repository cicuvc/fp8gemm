"""Tests for the isolated fused wgrad + Muon-momentum experiment."""

import pytest
import torch
import torch.nn.functional as F

import mxtest
from experiments.muon_fused import (
    FusedMuonLinear,
    finalize_muon_step,
    launch_stochastic_round_bf16,
    launch_wgrad_momentum,
    set_accum_first,
)
from fp8_linear import FP8Linear
from fp8_muon import MuonStateDtype


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused Muon experiment requires CUDA",
)


def test_wgrad_kernel_matches_reference_momentum_accumulation():
    torch.manual_seed(50)
    M, N, K = 128, 64, 64
    beta = 0.95
    fused_momentum = torch.zeros(
        (N, K),
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference_momentum = torch.zeros_like(fused_momentum)

    for microstep in range(8):
        dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        accum_first = microstep == 0
        launch_wgrad_momentum(
            dy,
            x,
            fused_momentum,
            beta,
            accum_first,
            stochastic_rounding=False,
            rounding_seed=123,
            rounding_counter=microstep,
        )

        grad = mxtest.launch_fp8gemm_weight_grad(
            dy,
            x,
            out_dtype=torch.float32,
        )
        reference_fp32 = reference_momentum.float()
        if accum_first:
            reference_fp32.mul_(beta).add_(grad, alpha=1.0 - beta)
        else:
            reference_fp32.add_(grad, alpha=1.0 - beta)
        reference_momentum.copy_(reference_fp32)

    relative_error = (
        torch.linalg.vector_norm(fused_momentum.float() - reference_momentum.float())
        / torch.linalg.vector_norm(reference_momentum.float())
    ).item()
    cosine = F.cosine_similarity(
        fused_momentum.float().flatten(),
        reference_momentum.float().flatten(),
        dim=0,
    ).item()
    assert relative_error < 2e-3
    assert cosine > 0.99999


def test_stochastic_rounding_is_unbiased_and_reproducible():
    torch.manual_seed(52)
    count = 1_000_000

    # The low half of the discarded mantissa always rounds down under RNE.
    # Stochastic rounding should recover the missing mass in expectation.
    low = torch.randint(
        0,
        0x8000,
        (count,),
        dtype=torch.int64,
    )
    bits = 0x3F800000 + low
    values = bits.to(torch.int32).view(torch.float32).cuda()

    rounded_once = launch_stochastic_round_bf16(values, seed=1234)
    rounded_twice = launch_stochastic_round_bf16(values, seed=1234)
    deterministic = values.to(torch.bfloat16)

    torch.testing.assert_close(rounded_once, rounded_twice, rtol=0, atol=0)

    values_mean = values.double().mean().item()
    stochastic_mean = rounded_once.double().mean().item()
    deterministic_mean = deterministic.double().mean().item()

    stochastic_error = abs(stochastic_mean - values_mean)
    deterministic_error = abs(deterministic_mean - values_mean)
    assert stochastic_error < deterministic_error * 0.05


@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_fused_module_matches_stable_wgrad_plus_muon(stochastic_rounding):
    torch.manual_seed(51)
    in_features, out_features = 32, 64
    fused = FusedMuonLinear(
        in_features,
        out_features,
        device="cuda",
        muon_momentum=0.95,
        stochastic_rounding=stochastic_rounding,
    )
    reference = FP8Linear(
        in_features,
        out_features,
        bias=False,
        device="cuda",
    )
    with torch.no_grad():
        reference.master_weight.copy_(fused.master_weight)
    reference.refresh_fp8_weight()
    reference_optimizer = MuonStateDtype(
        [reference.master_weight],
        lr=0.02,
        momentum=0.95,
        nesterov=False,
        momentum_dtype=torch.bfloat16,
        adjust_lr_fn="original",
    )

    accumulation_steps = 4
    lr = 0.02
    for group in range(3):
        if group > 0:
            reference.zero_grad(set_to_none=True)

        microbatches = [
            (
                torch.randn(32, in_features, dtype=torch.bfloat16, device="cuda"),
                torch.randn(32, out_features, dtype=torch.bfloat16, device="cuda"),
            )
            for _ in range(accumulation_steps)
        ]
        for microstep, (x, target) in enumerate(microbatches):
            accum_first = microstep == 0
            set_accum_first(fused, accum_first)
            fused_out = fused(x)
            F.mse_loss(fused_out.float(), target.float()).backward()

            reference_out = reference(x)
            F.mse_loss(reference_out.float(), target.float()).backward()

        finalize_muon_step(fused, lr=lr)
        reference_optimizer.step()
        reference.refresh_fp8_weight()

        reference_momentum = reference_optimizer.state[
            reference.master_weight
        ]["momentum_buffer"]
        momentum_error = (
            torch.linalg.vector_norm(
                fused.momentum_buffer.float() - reference_momentum.float()
            )
            / torch.linalg.vector_norm(reference_momentum.float())
        ).item()
        assert momentum_error < 1e-2

    relative_error = (
        torch.linalg.vector_norm(fused.master_weight - reference.master_weight)
        / torch.linalg.vector_norm(reference.master_weight)
    ).item()
    assert relative_error < 2e-3
