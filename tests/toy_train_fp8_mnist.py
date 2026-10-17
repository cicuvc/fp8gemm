"""Train a three-layer FP8Linear MLP on MNIST.

The 28x28 image is flattened to 784 features and padded to 800 so every FP8
GEMM contraction dimension is a multiple of the 32-element quantization block.
A same-initialization FP32 model consumes identical mini-batches as a control.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import MNIST

from fp8_linear import FP8Linear
from tests.toy_train_fp8_mlp import FP32Linear


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class FP8MNISTMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.fc1 = FP8Linear(in_features, hidden1, bias=True, device=device)
        self.fc2 = FP8Linear(hidden1, hidden2, bias=True, device=device)
        self.fc3 = FP8Linear(hidden2, num_classes, bias=True, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def fp8_layers(self) -> list[FP8Linear]:
        return [self.fc1, self.fc2, self.fc3]


class FP32MNISTMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.fc1 = FP32Linear(in_features, hidden1, device)
        self.fc2 = FP32Linear(hidden1, hidden2, device)
        self.fc3 = FP32Linear(hidden2, num_classes, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a three-layer FP8Linear MLP on MNIST."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / ".cache" / "fp8gemm" / "mnist",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading MNIST when the cache is missing",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="random training subset size; 0 uses all 60000 examples",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="random test subset size; 0 uses all 10000 examples",
    )
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def subset_indices(
    count: int,
    requested: int,
    seed: int,
) -> torch.Tensor:
    if requested <= 0 or requested >= count:
        size = count
    else:
        size = requested
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator)[:size]


def load_split(
    dataset: MNIST,
    requested: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = subset_indices(len(dataset), requested, seed)
    images = dataset.data[indices].to(device=device, dtype=torch.float32)
    images = images.reshape(images.shape[0], -1) / 255.0
    images = (images - MNIST_MEAN) / MNIST_STD

    pad_features = (-images.shape[1]) % 32
    if pad_features:
        images = F.pad(images, (0, pad_features))
    images = images.to(torch.bfloat16)

    labels = dataset.targets[indices].to(device=device, dtype=torch.long)
    return images, labels


def load_mnist(
    data_root: Path,
    download: bool,
    train_samples: int,
    test_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_dataset = MNIST(
        root=str(data_root),
        train=True,
        download=download,
        transform=None,
    )
    test_dataset = MNIST(
        root=str(data_root),
        train=False,
        download=download,
        transform=None,
    )
    train_x, train_y = load_split(train_dataset, train_samples, seed, device)
    test_x, test_y = load_split(test_dataset, test_samples, seed + 1, device)
    return train_x, train_y, test_x, test_y


def make_models(
    in_features: int,
    hidden1: int,
    hidden2: int,
    num_classes: int,
    device: torch.device,
) -> tuple[FP8MNISTMLP, FP32MNISTMLP]:
    fp8_model = FP8MNISTMLP(
        in_features,
        hidden1,
        hidden2,
        num_classes,
        device,
    )
    control = FP32MNISTMLP(
        in_features,
        hidden1,
        hidden2,
        num_classes,
        device,
    )
    control.load_state_dict(fp8_model.state_dict())
    return fp8_model, control


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(model(x).float(), y)


def compare_gradients(
    fp8_model: FP8MNISTMLP,
    control: FP32MNISTMLP,
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
        "fc3.weight": fp8_model.fc3.master_weight.grad.detach(),
        "fc3.bias": fp8_model.fc3.bias.grad.detach(),
    }
    fp8_model.zero_grad(set_to_none=True)

    compute_loss(control, x_ref, y).backward()
    control_grads = {
        "grad_x": x_ref.grad.detach(),
        "fc1.weight": control.fc1.master_weight.grad.detach(),
        "fc1.bias": control.fc1.bias.grad.detach(),
        "fc2.weight": control.fc2.master_weight.grad.detach(),
        "fc2.bias": control.fc2.bias.grad.detach(),
        "fc3.weight": control.fc3.master_weight.grad.detach(),
        "fc3.bias": control.fc3.bias.grad.detach(),
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
    model: FP8MNISTMLP,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    snapshots = {}
    for name, layer in zip(("fc1", "fc2", "fc3"), model.fp8_layers()):
        snapshots[name] = (
            layer._weight_fp8.clone(),
            layer._weight_scale.clone(),
        )
    return snapshots


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb).float()
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
    model.train()
    return total_loss / x.shape[0], total_correct / x.shape[0]


def train_models(
    fp8_model: FP8MNISTMLP,
    control: FP32MNISTMLP,
    fp8_optimizer: torch.optim.Optimizer,
    control_optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
    epochs: int,
    seed: int,
    log_every: int,
) -> dict[str, int]:
    snapshots = quantized_snapshots(fp8_model)
    refresh_counts = {name: 0 for name in snapshots}
    global_step = 0

    for epoch in range(1, epochs + 1):
        order_generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(train_x.shape[0], generator=order_generator).to(
            train_x.device
        )
        fp8_loss_sum = 0.0
        control_loss_sum = 0.0
        fp8_correct = 0
        control_correct = 0
        seen = 0

        for start in range(0, train_x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            xb = train_x[indices]
            yb = train_y[indices]

            fp8_optimizer.zero_grad(set_to_none=True)
            fp8_logits = fp8_model(xb).float()
            fp8_loss = F.cross_entropy(fp8_logits, yb)
            if not torch.isfinite(fp8_loss):
                raise RuntimeError(f"FP8 produced a non-finite loss at step {global_step}")
            fp8_loss.backward()
            fp8_optimizer.step()

            current = quantized_snapshots(fp8_model)
            for name in snapshots:
                if not all(
                    torch.equal(previous, latest)
                    for previous, latest in zip(snapshots[name], current[name])
                ):
                    refresh_counts[name] += 1
            snapshots = current

            control_optimizer.zero_grad(set_to_none=True)
            control_logits = control(xb).float()
            control_loss = F.cross_entropy(control_logits, yb)
            control_loss.backward()
            control_optimizer.step()

            count = yb.shape[0]
            seen += count
            fp8_loss_sum += fp8_loss.item() * count
            control_loss_sum += control_loss.item() * count
            fp8_correct += (fp8_logits.argmax(dim=1) == yb).sum().item()
            control_correct += (control_logits.argmax(dim=1) == yb).sum().item()
            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                print(
                    f"step={global_step:5d} "
                    f"fp8_loss={fp8_loss_sum / seen:.4f} "
                    f"fp8_acc={fp8_correct / seen:.4f} "
                    f"fp32_loss={control_loss_sum / seen:.4f} "
                    f"fp32_acc={control_correct / seen:.4f}"
                )

        fp8_test_loss, fp8_test_accuracy = evaluate(
            fp8_model,
            test_x,
            test_y,
        )
        control_test_loss, control_test_accuracy = evaluate(
            control,
            test_x,
            test_y,
        )
        print(
            f"epoch={epoch} "
            f"fp8_train_loss={fp8_loss_sum / seen:.4f} "
            f"fp8_train_acc={fp8_correct / seen:.4f} "
            f"fp8_test_loss={fp8_test_loss:.4f} "
            f"fp8_test_acc={fp8_test_accuracy:.4f} | "
            f"fp32_train_loss={control_loss_sum / seen:.4f} "
            f"fp32_train_acc={control_correct / seen:.4f} "
            f"fp32_test_loss={control_test_loss:.4f} "
            f"fp32_test_acc={control_test_accuracy:.4f}"
        )

    return refresh_counts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP8Linear requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_x, train_y, test_x, test_y = load_mnist(
        args.data_root.expanduser(),
        not args.no_download,
        args.train_samples,
        args.test_samples,
        args.seed,
        device,
    )
    fp8_model, control = make_models(
        train_x.shape[1],
        args.hidden1,
        args.hidden2,
        10,
        device,
    )

    print(
        "task=mnist_mlp",
        f"shape=(M={train_x.shape[0]}, K={train_x.shape[1]}, "
        f"H1={args.hidden1}, H2={args.hidden2}, N=10)",
        f"batch_size={args.batch_size}",
        f"device={torch.cuda.get_device_name(device)}",
    )

    initial_grads = compare_gradients(
        fp8_model,
        control,
        train_x[: args.batch_size],
        train_y[: args.batch_size],
    )
    print("initial gradient agreement (FP8 vs FP32):")
    for name, (cosine, relative_l2) in initial_grads.items():
        print(f"  {name:11s} cosine={cosine:.6f} relative_l2={relative_l2:.6f}")

    fp8_optimizer = torch.optim.AdamW(
        fp8_model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )
    for layer in fp8_model.fp8_layers():
        layer.bind_optimizer(fp8_optimizer)

    refresh_counts = train_models(
        fp8_model,
        control,
        fp8_optimizer,
        control_optimizer,
        train_x,
        train_y,
        test_x,
        test_y,
        args.batch_size,
        args.epochs,
        args.seed,
        args.log_every,
    )

    fp8_test_loss, fp8_test_accuracy = evaluate(fp8_model, test_x, test_y)
    control_test_loss, control_test_accuracy = evaluate(control, test_x, test_y)
    print()
    print(
        f"final fp8: test_loss={fp8_test_loss:.4f} "
        f"test_acc={fp8_test_accuracy:.4f}"
    )
    print(
        f"final fp32: test_loss={control_test_loss:.4f} "
        f"test_acc={control_test_accuracy:.4f}"
    )
    print(
        "FP8 weight/scale refresh steps: "
        + ", ".join(
            f"{name}={count}"
            for name, count in refresh_counts.items()
        )
    )

    weight_names = ("fc1.weight", "fc2.weight", "fc3.weight")
    min_weight_cosine = min(initial_grads[name][0] for name in weight_names)
    max_weight_relative_l2 = max(initial_grads[name][1] for name in weight_names)
    # Error compounds through three quantized linear layers and two ReLUs.
    # Cosine still catches a wrong gradient direction, while the relative-L2
    # bounds allow for the larger approximation floor of this deeper stack.
    checks = {
        "FP8 test accuracy >= 0.97": fp8_test_accuracy >= 0.97,
        "test accuracy within 1 point of FP32": (
            fp8_test_accuracy >= control_test_accuracy - 0.01
        ),
        "FP8/FP32 input-gradient cosine >= 0.95": (
            initial_grads["grad_x"][0] >= 0.95
        ),
        "FP8/FP32 input-gradient relative L2 <= 0.30": (
            initial_grads["grad_x"][1] <= 0.30
        ),
        "FP8/FP32 weight-gradient cosine >= 0.98": min_weight_cosine >= 0.98,
        "FP8/FP32 weight-gradient relative L2 <= 0.20": (
            max_weight_relative_l2 <= 0.20
        ),
        "all FP8 layers refreshed": all(
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
