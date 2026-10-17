"""Muon with configurable optimizer-state dtype."""

import math

import torch


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError(f"Muon requires a 2D gradient, got shape {grad.shape}")

    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    transposed = grad.shape[0] > grad.shape[1]
    if transposed:
        ortho_grad = ortho_grad.T

    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix,
            gram_matrix,
            gram_matrix,
            beta=b,
            alpha=c,
        )
        ortho_grad = torch.addmm(
            ortho_grad,
            gram_update,
            ortho_grad,
            beta=a,
        )

    return ortho_grad.T if transposed else ortho_grad


def _adjust_lr(
    lr: float,
    adjust_lr_fn: str | None,
    parameter_shape: torch.Size,
) -> float:
    out_features, in_features = parameter_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, out_features / in_features))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(out_features, in_features))
    else:
        raise ValueError(f"unsupported Muon LR adjustment: {adjust_lr_fn}")
    return lr * ratio


class MuonStateDtype(torch.optim.Optimizer):
    """Muon with FP32 master parameters and configurable momentum storage."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "original",
        momentum_dtype: torch.dtype = torch.float32,
    ):
        if momentum_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported momentum dtype: {momentum_dtype}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        self.momentum_dtype = momentum_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2D parameters, got {parameter.shape}"
                    )
                if parameter.dtype != torch.float32:
                    raise TypeError(
                        f"expected FP32 master parameter, got {parameter.dtype}"
                    )
                if parameter.grad.dtype != torch.float32:
                    raise TypeError(
                        f"expected FP32 gradient, got {parameter.grad.dtype}"
                    )

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter,
                        dtype=self.momentum_dtype,
                    )
                state["step"] += 1

                momentum = group["momentum"]
                grad = parameter.grad
                momentum_buffer = state["momentum_buffer"]

                if momentum_buffer.dtype == torch.float32:
                    momentum_buffer.lerp_(grad, 1.0 - momentum)
                    update = (
                        grad.lerp(momentum_buffer, momentum)
                        if group["nesterov"]
                        else momentum_buffer
                    )
                else:
                    momentum_fp32 = momentum_buffer.float()
                    momentum_fp32.lerp_(grad, 1.0 - momentum)
                    momentum_buffer.copy_(momentum_fp32)
                    update = (
                        grad.lerp(momentum_fp32, momentum)
                        if group["nesterov"]
                        else momentum_fp32
                    )

                update = _zeropower_via_newtonschulz(
                    update,
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                )
                adjusted_lr = _adjust_lr(
                    group["lr"],
                    group["adjust_lr_fn"],
                    parameter.shape,
                )

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-adjusted_lr)

        return loss


def muon_state_dtypes(
    optimizer: torch.optim.Optimizer,
) -> set[torch.dtype]:
    return {
        state["momentum_buffer"].dtype
        for state in optimizer.state.values()
        if "momentum_buffer" in state
    }
