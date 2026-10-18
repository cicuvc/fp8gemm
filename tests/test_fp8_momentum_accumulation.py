"""Numerical checks for momentum updates fused with wgrad accumulation."""

import torch

from fp8_muon import _zeropower_via_newtonschulz


def _accumulate_moments(groups, parameter, beta1, beta2, state_dtype):
    moment1 = torch.zeros_like(parameter, dtype=state_dtype)
    moment2 = torch.zeros_like(parameter, dtype=state_dtype)

    for group_index, group in enumerate(groups, start=1):
        for microstep, gradient in enumerate(group):
            moment1_fp32 = moment1.float()
            moment2_fp32 = moment2.float()
            if microstep == 0:
                moment1_fp32.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                moment2_fp32.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )
            else:
                moment1_fp32.add_(gradient, alpha=1.0 - beta1)
                moment2_fp32.addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )
            moment1.copy_(moment1_fp32)
            moment2.copy_(moment2_fp32)

        moment1_hat = moment1.float() / (1.0 - beta1**group_index)
        moment2_hat = moment2.float() / (1.0 - beta2**group_index)
        parameter.addcdiv_(
            moment1_hat,
            moment2_hat.sqrt().add_(1e-8),
            value=-1e-3,
        )

    return moment1, moment2, parameter.clone()


def _accumulate_muon_momentum(groups, parameter, state_dtype):
    momentum = torch.zeros_like(parameter, dtype=state_dtype)
    for group in groups:
        for microstep, gradient in enumerate(group):
            momentum_fp32 = momentum.float()
            if microstep == 0:
                momentum_fp32.mul_(0.95).add_(gradient, alpha=0.05)
            else:
                momentum_fp32.add_(gradient, alpha=0.05)
            momentum.copy_(momentum_fp32)

        update = _zeropower_via_newtonschulz(
            momentum.float(),
            (3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
        )
        adjusted_lr = 0.02 * max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        parameter.add_(update, alpha=-adjusted_lr)
    return momentum, parameter.clone()


def test_bf16_adamw_moments_support_fused_gradient_accumulation():
    torch.manual_seed(40)
    shape = (64, 64)
    initial = torch.randn(shape, dtype=torch.float32) * 0.1
    groups = [
        [torch.randn(shape, dtype=torch.float32) * 0.01 for _ in range(16)]
        for _ in range(20)
    ]

    moment1_fp32, moment2_fp32, parameter_fp32 = _accumulate_moments(
        groups,
        initial.clone(),
        0.9,
        0.999,
        torch.float32,
    )
    moment1_bf16, moment2_bf16, parameter_bf16 = _accumulate_moments(
        groups,
        initial.clone(),
        0.9,
        0.999,
        torch.bfloat16,
    )

    moment1_error = (
        torch.linalg.vector_norm(moment1_fp32 - moment1_bf16)
        / torch.linalg.vector_norm(moment1_fp32)
    ).item()
    moment2_error = (
        torch.linalg.vector_norm(moment2_fp32 - moment2_bf16)
        / torch.linalg.vector_norm(moment2_fp32)
    ).item()
    parameter_error = (
        torch.linalg.vector_norm(parameter_fp32 - parameter_bf16)
        / torch.linalg.vector_norm(parameter_fp32)
    ).item()

    assert moment1_error < 0.02
    assert moment2_error < 0.05
    assert parameter_error < 1e-3


def test_bf16_muon_momentum_supports_fused_gradient_accumulation():
    torch.manual_seed(41)
    shape = (64, 64)
    initial = torch.randn(shape, dtype=torch.float32) * 0.1
    groups = [
        [torch.randn(shape, dtype=torch.float32) * 0.01 for _ in range(16)]
        for _ in range(20)
    ]

    momentum_fp32, parameter_fp32 = _accumulate_muon_momentum(
        groups,
        initial.clone(),
        torch.float32,
    )
    momentum_bf16, parameter_bf16 = _accumulate_muon_momentum(
        groups,
        initial.clone(),
        torch.bfloat16,
    )

    momentum_error = (
        torch.linalg.vector_norm(momentum_fp32 - momentum_bf16)
        / torch.linalg.vector_norm(momentum_fp32)
    ).item()
    parameter_error = (
        torch.linalg.vector_norm(parameter_fp32 - parameter_bf16)
        / torch.linalg.vector_norm(parameter_fp32)
    ).item()

    assert momentum_error < 0.02
    assert parameter_error < 0.01


def test_averaging_momentum_equals_averaging_gradients():
    torch.manual_seed(42)
    ranks = 4
    momentum_old = torch.randn(1024, dtype=torch.float32)
    local_gradients = [
        torch.randn(1024, dtype=torch.float32) * 0.01
        for _ in range(ranks)
    ]
    beta = 0.9

    local_momenta = [
        beta * momentum_old + (1.0 - beta) * gradient
        for gradient in local_gradients
    ]
    averaged_momentum = torch.stack(local_momenta).mean(dim=0)
    momentum_of_averaged_gradient = (
        beta * momentum_old
        + (1.0 - beta) * torch.stack(local_gradients).mean(dim=0)
    )

    torch.testing.assert_close(
        averaged_momentum,
        momentum_of_averaged_gradient,
        rtol=1e-6,
        atol=1e-7,
    )


def test_averaging_second_moment_is_not_averaging_gradients():
    positive = torch.tensor([1.0])
    negative = torch.tensor([-1.0])
    beta2 = 0.999

    averaged_local_second_moment = (
        (1.0 - beta2)
        * (positive.square() + negative.square())
        / 2.0
    )
    second_moment_of_averaged_gradient = (
        (1.0 - beta2)
        * ((positive + negative) / 2.0).square()
    )

    assert averaged_local_second_moment > second_moment_of_averaged_gradient
