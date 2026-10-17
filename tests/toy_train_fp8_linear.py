"""Trainability smoke test for FP8Linear on a synthetic affine regression task.

The task has a known solution and uses deterministic, full-batch training.  It
is intentionally small so failures can be attributed to the FP8 layer rather
than to data loading, augmentation, or stochastic optimization.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fp8_linear import FP8Linear


class FP32Linear(nn.Module):
    """Same parameter layout as FP8Linear, with a full-precision control path."""

    def __init__(self, in_features: int, out_features: int, bias: bool, device):
        super().__init__()
        self.master_weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.master_weight.dtype:
            x = x.to(self.master_weight.dtype)
        return F.linear(x, self.master_weight, self.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify FP8Linear trainability with deterministic affine regression."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--M", type=int, default=512, help="number of training rows")
    parser.add_argument("--K", type=int, default=128, help="input features")
    parser.add_argument("--N", type=int, default=128, help="output features")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--block-dynamic-range",
        action="store_true",
        help="scale each contiguous 32-feature input block by a different power of two",
    )
    return parser.parse_args()


def make_data(
    M: int,
    K: int,
    N: int,
    device: torch.device,
    block_dynamic_range: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    if block_dynamic_range:
        if K % 32 != 0:
            raise ValueError("--block-dynamic-range requires K to be a multiple of 32")
        block_exp = torch.arange(K // 32, device=device, dtype=torch.float32) % 5
        block_scale = torch.pow(2.0, -block_exp)
        x = (x.float() * block_scale.repeat_interleave(32)).to(torch.bfloat16)

    teacher_weight = torch.randn((N, K), dtype=torch.float32, device=device)
    teacher_weight /= math.sqrt(K)
    teacher_bias = 0.1 * torch.randn(N, dtype=torch.float32, device=device)
    y = (x.float() @ teacher_weight.T + teacher_bias).to(torch.bfloat16)
    return x, y


def make_models(
    K: int,
    N: int,
    device: torch.device,
) -> tuple[FP8Linear, FP32Linear]:
    fp8_model = FP8Linear(K, N, bias=True, device=device)
    control = FP32Linear(K, N, bias=True, device=device)
    control.load_state_dict(fp8_model.state_dict())
    return fp8_model, control


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(model(x).float(), y.float())


def gradient_comparison(
    fp8_model: FP8Linear,
    control: FP32Linear,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, tuple[float, float]]:
    """Compare gradients at the same initial weights."""
    x_fp8 = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().float().requires_grad_(True)

    compute_loss(fp8_model, x_fp8, y).backward()
    fp8_grads = {
        "grad_x": x_fp8.grad.detach(),
        "grad_weight": fp8_model.master_weight.grad.detach(),
        "grad_bias": fp8_model.bias.grad.detach(),
    }
    fp8_model.zero_grad(set_to_none=True)

    compute_loss(control, x_ref, y).backward()
    control_grads = {
        "grad_x": x_ref.grad.detach(),
        "grad_weight": control.master_weight.grad.detach(),
        "grad_bias": control.bias.grad.detach(),
    }
    control.zero_grad(set_to_none=True)

    result = {}
    for name in fp8_grads:
        lhs = fp8_grads[name].float().flatten()
        rhs = control_grads[name].float().flatten()
        cosine = F.cosine_similarity(lhs, rhs, dim=0).item()
        relative_l2 = (torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(rhs)).item()
        result[name] = (cosine, relative_l2)
    return result


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    log_every: int,
    label: str,
    fp8_model: FP8Linear | None = None,
    log: bool = True,
) -> tuple[list[float], int]:
    losses = []
    quantized_changes = 0
    previous_quantized = None
    if fp8_model is not None:
        weight_fp8, weight_scale = fp8_model.quantized_weight
        previous_quantized = (weight_fp8.clone(), weight_scale.clone())

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label} produced a non-finite loss at step {step}")
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if fp8_model is not None:
            current_quantized = fp8_model.quantized_weight
            if not all(
                torch.equal(previous, current)
                for previous, current in zip(previous_quantized, current_quantized)
            ):
                quantized_changes += 1
            previous_quantized = (
                current_quantized[0].clone(),
                current_quantized[1].clone(),
            )

        if log and (step == 1 or step % log_every == 0):
            print(f"{label} step={step:4d} loss={loss_value:.6f}")

    return losses, quantized_changes


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        loss = compute_loss(model, x, y).item()
    model.train()
    return loss


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP8Linear requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    x, y = make_data(
        args.M,
        args.K,
        args.N,
        device,
        args.block_dynamic_range,
    )
    fp8_model, control = make_models(args.K, args.N, device)

    print(
        "task=affine_regression",
        f"shape=(M={args.M}, K={args.K}, N={args.N})",
        f"device={torch.cuda.get_device_name(device)}",
    )

    initial_grads = gradient_comparison(fp8_model, control, x, y)
    print("initial gradient agreement (FP8 vs FP32):")
    for name, (cosine, relative_l2) in initial_grads.items():
        print(f"  {name:11s} cosine={cosine:.6f} relative_l2={relative_l2:.6f}")

    initial_fp8_loss = evaluate(fp8_model, x, y)
    initial_control_loss = evaluate(control, x, y)

    fp8_optimizer = torch.optim.AdamW(
        fp8_model.parameters(), lr=args.lr, weight_decay=0.0
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=args.lr, weight_decay=0.0
    )
    fp8_model.bind_optimizer(fp8_optimizer)

    fp8_losses, quantized_changes = train(
        fp8_model,
        fp8_optimizer,
        x,
        y,
        args.steps,
        args.log_every,
        "fp8",
        fp8_model=fp8_model,
    )
    control_losses, _ = train(
        control,
        control_optimizer,
        x,
        y,
        args.steps,
        args.log_every,
        "fp32",
    )

    final_fp8_loss = evaluate(fp8_model, x, y)
    final_control_loss = evaluate(control, x, y)
    fp8_reduction = initial_fp8_loss / max(final_fp8_loss, 1e-30)
    control_reduction = initial_control_loss / max(final_control_loss, 1e-30)

    print()
    print(
        f"fp8 loss: initial={initial_fp8_loss:.6f} "
        f"final={final_fp8_loss:.6f} reduction={fp8_reduction:.1f}x"
    )
    print(
        f"fp32 loss: initial={initial_control_loss:.6f} "
        f"final={final_control_loss:.6f} reduction={control_reduction:.1f}x"
    )
    print(
        f"FP8 weight/scale refresh changed on "
        f"{quantized_changes}/{args.steps} optimizer steps"
    )

    weight_cosine = initial_grads["grad_weight"][0]
    weight_relative_l2 = initial_grads["grad_weight"][1]
    checks = {
        "loss fell by at least 10x": fp8_reduction >= 10.0,
        "weight-gradient cosine >= 0.98": weight_cosine >= 0.98,
        "weight-gradient relative L2 <= 0.05": weight_relative_l2 <= 0.05,
        "quantized weights refreshed": quantized_changes > 0,
    }
    print()
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
