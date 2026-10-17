"""Trainability test for FP8SwiGLU on nonlinear two-moons classification.

Both the default BF16-input weight-gradient path and the optional FP8
weight-gradient path can be tested.  A same-initialization FP32 SwiGLU model
is trained on identical data as the control.
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from fp8_swiglu import FP8SwiGLU
from tests.toy_train_fp8_mlp import make_dataset


class FP32DownProj(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device):
        super().__init__()
        self.master_weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, dtype=torch.float32, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.master_weight.dtype:
            x = x.to(self.master_weight.dtype)
        return F.linear(x, self.master_weight, self.bias)


class FP32SwiGLU(nn.Module):
    """Parameter-compatible full-precision control for FP8SwiGLU."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        device: torch.device,
    ):
        super().__init__()
        self.gate_weight = nn.Parameter(
            torch.empty((hidden_features, in_features), dtype=torch.float32, device=device)
        )
        self.up_weight = nn.Parameter(
            torch.empty((hidden_features, in_features), dtype=torch.float32, device=device)
        )
        self.down_proj = FP32DownProj(hidden_features, out_features, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        gate = F.linear(x, self.gate_weight)
        up = F.linear(x, self.up_weight)
        return self.down_proj(F.silu(gate) * up)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify FP8SwiGLU trainability on two-moons classification."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=32)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--fp8-wgrad",
        action="store_true",
        help="use the FP8 weight-gradient kernels instead of BF16-input weight gradients",
    )
    return parser.parse_args()


def make_models(
    in_features: int,
    hidden_features: int,
    num_classes: int,
    device: torch.device,
    fp8_wgrad: bool,
) -> tuple[FP8SwiGLU, FP32SwiGLU]:
    fp8_model = FP8SwiGLU(
        in_features,
        hidden_features,
        num_classes,
        bias=True,
        fp8_wgrad=fp8_wgrad,
        device=device,
    )
    control = FP32SwiGLU(in_features, hidden_features, num_classes, device)
    control.load_state_dict(fp8_model.state_dict())
    return fp8_model, control


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(model(x).float(), y)


def accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x).float()
        loss = F.cross_entropy(logits, y).item()
        correct = (logits.argmax(dim=1) == y).float().mean().item()
    model.train()
    return loss, correct


def compare_gradients(
    fp8_model: FP8SwiGLU,
    control: FP32SwiGLU,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, tuple[float, float]]:
    x_fp8 = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().float().requires_grad_(True)

    compute_loss(fp8_model, x_fp8, y).backward()
    fp8_grads = {
        "grad_x": x_fp8.grad.detach(),
        "gate_weight": fp8_model.gate_weight.grad.detach(),
        "up_weight": fp8_model.up_weight.grad.detach(),
        "down_weight": fp8_model.down_proj.master_weight.grad.detach(),
        "down_bias": fp8_model.down_proj.bias.grad.detach(),
    }
    fp8_model.zero_grad(set_to_none=True)

    compute_loss(control, x_ref, y).backward()
    control_grads = {
        "grad_x": x_ref.grad.detach(),
        "gate_weight": control.gate_weight.grad.detach(),
        "up_weight": control.up_weight.grad.detach(),
        "down_weight": control.down_proj.master_weight.grad.detach(),
        "down_bias": control.down_proj.bias.grad.detach(),
    }
    control.zero_grad(set_to_none=True)

    result = {}
    for name in fp8_grads:
        lhs = fp8_grads[name].float().flatten()
        rhs = control_grads[name].float().flatten()
        cosine = F.cosine_similarity(lhs, rhs, dim=0).item()
        relative_l2 = (
            torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(rhs)
        ).item()
        result[name] = (cosine, relative_l2)
    return result


def quantized_snapshots(
    model: FP8SwiGLU,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "gate": (model._gate_weight_fp8.clone(), model._gate_weight_scale.clone()),
        "up": (model._up_weight_fp8.clone(), model._up_weight_scale.clone()),
        "down": (
            model.down_proj._weight_fp8.clone(),
            model.down_proj._weight_scale.clone(),
        ),
    }


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    log_every: int,
    label: str,
    fp8_model: FP8SwiGLU | None = None,
) -> dict[str, int]:
    snapshots = quantized_snapshots(fp8_model) if fp8_model is not None else {}
    refresh_counts = {name: 0 for name in snapshots}

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label} produced a non-finite loss at step {step}")
        loss.backward()
        optimizer.step()

        if fp8_model is not None:
            current = quantized_snapshots(fp8_model)
            for name in snapshots:
                if not all(
                    torch.equal(previous, latest)
                    for previous, latest in zip(snapshots[name], current[name])
                ):
                    refresh_counts[name] += 1
            snapshots = current

        if step == 1 or step % log_every == 0 or step == steps:
            train_loss, train_accuracy = accuracy(model, x, y)
            print(
                f"{label} step={step:4d} "
                f"loss={train_loss:.5f} train_acc={train_accuracy:.3f}"
            )

    return refresh_counts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP8SwiGLU requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_x, train_y, test_x, test_y = make_dataset(
        args.train_samples,
        args.test_samples,
        args.in_features,
        args.noise,
        device,
    )
    fp8_model, control = make_models(
        args.in_features,
        args.hidden_features,
        2,
        device,
        args.fp8_wgrad,
    )

    wgrad_mode = "fp8" if args.fp8_wgrad else "bf16"
    print(
        "task=two_moons_swiglu",
        f"shape=(M={args.train_samples}, K={args.in_features}, "
        f"H={args.hidden_features}, N=2)",
        f"wgrad={wgrad_mode}",
        f"device={torch.cuda.get_device_name(device)}",
    )

    initial_grads = compare_gradients(fp8_model, control, train_x, train_y)
    print("initial gradient agreement (FP8 vs FP32):")
    for name, (cosine, relative_l2) in initial_grads.items():
        print(f"  {name:11s} cosine={cosine:.6f} relative_l2={relative_l2:.6f}")

    fp8_optimizer = torch.optim.AdamW(
        fp8_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    fp8_model.bind_optimizer(fp8_optimizer)

    refresh_counts = train(
        fp8_model,
        fp8_optimizer,
        train_x,
        train_y,
        args.steps,
        args.log_every,
        "fp8",
        fp8_model=fp8_model,
    )
    train(
        control,
        control_optimizer,
        train_x,
        train_y,
        args.steps,
        args.log_every,
        "fp32",
    )

    fp8_train_loss, fp8_train_accuracy = accuracy(fp8_model, train_x, train_y)
    fp8_test_loss, fp8_test_accuracy = accuracy(fp8_model, test_x, test_y)
    control_train_loss, control_train_accuracy = accuracy(control, train_x, train_y)
    control_test_loss, control_test_accuracy = accuracy(control, test_x, test_y)

    print()
    print(
        f"fp8: train_loss={fp8_train_loss:.5f} train_acc={fp8_train_accuracy:.3f} "
        f"test_loss={fp8_test_loss:.5f} test_acc={fp8_test_accuracy:.3f}"
    )
    print(
        f"fp32: train_loss={control_train_loss:.5f} "
        f"train_acc={control_train_accuracy:.3f} "
        f"test_loss={control_test_loss:.5f} test_acc={control_test_accuracy:.3f}"
    )
    print(
        "FP8 weight/scale refresh steps: "
        + ", ".join(
            f"{name}={count}/{args.steps}"
            for name, count in refresh_counts.items()
        )
    )

    weight_names = ("gate_weight", "up_weight", "down_weight")
    min_weight_cosine = min(initial_grads[name][0] for name in weight_names)
    max_weight_relative_l2 = max(initial_grads[name][1] for name in weight_names)
    checks = {
        "FP8 test accuracy >= 0.90": fp8_test_accuracy >= 0.90,
        "test accuracy within 3 points of FP32": (
            fp8_test_accuracy >= control_test_accuracy - 0.03
        ),
        "FP8/FP32 input-gradient cosine >= 0.99": (
            initial_grads["grad_x"][0] >= 0.99
        ),
        "FP8/FP32 input-gradient relative L2 <= 0.10": (
            initial_grads["grad_x"][1] <= 0.10
        ),
        "FP8/FP32 weight-gradient cosine >= 0.99": min_weight_cosine >= 0.99,
        "FP8/FP32 weight-gradient relative L2 <= 0.10": (
            max_weight_relative_l2 <= 0.10
        ),
        "gate/up/down quantized weights refreshed": all(
            count > 0 for count in refresh_counts.values()
        ),
    }
    print()
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
