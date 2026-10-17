"""Train a small FP8 MLP on a nonlinear two-moons classification task.

The 2-D data is lifted to 32 dimensions with a fixed orthonormal projection so
the problem exercises real FP8 GEMM shapes while retaining a nonlinear decision
boundary.  A same-initialization FP32 model is trained as a control.
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

from fp8_linear import FP8Linear, bind_fp8_optimizer


class FP32Linear(nn.Module):
    """Same parameter layout as FP8Linear, with a full-precision GEMM."""

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


class FP8TwoLayerMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.fc1 = FP8Linear(in_features, hidden_features, bias=True, device=device)
        self.fc2 = FP8Linear(hidden_features, num_classes, bias=True, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def fp8_layers(self) -> list[FP8Linear]:
        return [self.fc1, self.fc2]


class FP32TwoLayerMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.fc1 = FP32Linear(in_features, hidden_features, device)
        self.fc2 = FP32Linear(hidden_features, num_classes, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a two-layer FP8 MLP on noisy two-moons classification."
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
    return parser.parse_args()


def make_moons(
    count: int,
    noise: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count % 2 != 0:
        raise ValueError("sample count must be even")

    half = count // 2
    theta = math.pi * torch.rand(half, dtype=torch.float32, device=device)
    outer = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    inner = torch.stack((1.0 - torch.cos(theta), 0.5 - torch.sin(theta)), dim=1)
    points = torch.cat((outer, inner), dim=0)
    points += noise * torch.randn_like(points)
    labels = torch.cat(
        (
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones(half, dtype=torch.long, device=device),
        )
    )
    permutation = torch.randperm(count, device=device)
    return points[permutation], labels[permutation]


def make_dataset(
    train_samples: int,
    test_samples: int,
    in_features: int,
    noise: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points, labels = make_moons(train_samples + test_samples, noise, device)

    if in_features < 2 or in_features % 32 != 0:
        raise ValueError("--in-features must be a multiple of 32 and at least 32")
    projection, _ = torch.linalg.qr(
        torch.randn(in_features, 2, dtype=torch.float32, device=device)
    )
    features = (points @ projection.T).to(torch.bfloat16)

    train_x, test_x = features[:train_samples], features[train_samples:]
    train_y, test_y = labels[:train_samples], labels[train_samples:]
    return train_x, train_y, test_x, test_y


def make_models(
    in_features: int,
    hidden_features: int,
    num_classes: int,
    device: torch.device,
) -> tuple[FP8TwoLayerMLP, FP32TwoLayerMLP]:
    fp8_model = FP8TwoLayerMLP(in_features, hidden_features, num_classes, device)
    control = FP32TwoLayerMLP(in_features, hidden_features, num_classes, device)
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
    fp8_model: FP8TwoLayerMLP,
    control: FP32TwoLayerMLP,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, tuple[float, float]]:
    x_fp8 = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().float().requires_grad_(True)

    compute_loss(fp8_model, x_fp8, y).backward()
    fp8_grads = {
        "grad_x": x_fp8.grad.detach(),
        "fc1.weight": fp8_model.fc1.master_weight.grad.detach(),
        "fc1.bias": fp8_model.fc1.bias.grad.detach(),
        "fc2.weight": fp8_model.fc2.master_weight.grad.detach(),
        "fc2.bias": fp8_model.fc2.bias.grad.detach(),
    }
    fp8_model.zero_grad(set_to_none=True)

    compute_loss(control, x_ref, y).backward()
    control_grads = {
        "grad_x": x_ref.grad.detach(),
        "fc1.weight": control.fc1.master_weight.grad.detach(),
        "fc1.bias": control.fc1.bias.grad.detach(),
        "fc2.weight": control.fc2.master_weight.grad.detach(),
        "fc2.bias": control.fc2.bias.grad.detach(),
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
    model: FP8TwoLayerMLP,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    snapshots = {}
    for name, layer in zip(("fc1", "fc2"), model.fp8_layers()):
        weight_fp8, weight_scale = layer.quantized_weight
        snapshots[name] = (weight_fp8.clone(), weight_scale.clone())
    return snapshots


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    log_every: int,
    label: str,
    fp8_model: FP8TwoLayerMLP | None = None,
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
        raise RuntimeError("FP8Linear requires CUDA")

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
    )

    print(
        "task=two_moons_mlp",
        f"shape=(M={args.train_samples}, K={args.in_features}, "
        f"H={args.hidden_features}, N=2)",
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
    bind_fp8_optimizer(fp8_model, fp8_optimizer)

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

    min_weight_cosine = min(
        initial_grads[name][0] for name in ("fc1.weight", "fc2.weight")
    )
    max_weight_relative_l2 = max(
        initial_grads[name][1] for name in ("fc1.weight", "fc2.weight")
    )
    checks = {
        "FP8 test accuracy >= 0.90": fp8_test_accuracy >= 0.90,
        "test accuracy within 3 points of FP32": (
            fp8_test_accuracy >= control_test_accuracy - 0.03
        ),
        "FP8/FP32 weight-gradient cosine >= 0.98": min_weight_cosine >= 0.98,
        "FP8/FP32 weight-gradient relative L2 <= 0.05": (
            max_weight_relative_l2 <= 0.05
        ),
        "both FP8 layers refreshed": all(
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
