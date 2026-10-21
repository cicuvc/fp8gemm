"""AdamW variants with configurable optimizer-state dtypes."""

import torch


class AdamWStateDtype(torch.optim.Optimizer):
    """AdamW with configurable parameter and moment dtypes.

    Updates are computed in FP32. Output parameters and moments may be BF16
    to reduce persistent optimizer and model memory.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        moment1_dtype: torch.dtype = torch.float32,
        moment2_dtype: torch.dtype = torch.float32,
    ):
        if moment1_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported moment1 dtype: {moment1_dtype}")
        if moment2_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported moment2 dtype: {moment2_dtype}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.moment1_dtype = moment1_dtype
        self.moment2_dtype = moment2_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype not in (torch.float32, torch.bfloat16):
                    raise TypeError(
                        f"expected FP32 or BF16 parameter, got {parameter.dtype}"
                    )
                if parameter.grad.dtype != parameter.dtype:
                    raise TypeError(
                        f"expected {parameter.dtype} gradient, got "
                        f"{parameter.grad.dtype}"
                    )
                if parameter.grad.is_sparse:
                    raise RuntimeError("sparse gradients are not supported")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        dtype=self.moment1_dtype,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        dtype=self.moment2_dtype,
                    )

                state["step"] += 1
                step = state["step"]
                grad = parameter.grad.float()

                moment1_fp32 = state["exp_avg"].float()
                moment1_fp32.lerp_(grad, 1.0 - beta1)
                state["exp_avg"].copy_(moment1_fp32)

                moment2_fp32 = state["exp_avg_sq"].float()
                moment2_fp32.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )
                state["exp_avg_sq"].copy_(moment2_fp32)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denominator = (
                    state["exp_avg_sq"].float()
                    .div(bias_correction2)
                    .sqrt()
                    .add_(eps)
                )

                if parameter.dtype == torch.float32:
                    parameter.mul_(1.0 - lr * weight_decay)
                    parameter.addcdiv_(
                        state["exp_avg"].float(),
                        denominator,
                        value=-step_size,
                    )
                else:
                    parameter_fp32 = parameter.float()
                    parameter_fp32.mul_(1.0 - lr * weight_decay)
                    parameter_fp32.addcdiv_(
                        state["exp_avg"].float(),
                        denominator,
                        value=-step_size,
                    )
                    parameter.copy_(parameter_fp32)

        return loss


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    """Return persistent tensor bytes held in an optimizer's state."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def optimizer_state_dtypes(
    optimizer: torch.optim.Optimizer,
) -> tuple[set[torch.dtype], set[torch.dtype]]:
    moment1_dtypes = set()
    moment2_dtypes = set()
    for state in optimizer.state.values():
        if "exp_avg" in state:
            moment1_dtypes.add(state["exp_avg"].dtype)
        if "exp_avg_sq" in state:
            moment2_dtypes.add(state["exp_avg_sq"].dtype)
    return moment1_dtypes, moment2_dtypes
