"""Tests for automatic FP8 weight-cache synchronization."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fp8_linear import (
    FP8Linear,
    bind_fp8_optimizer,
    iter_fp8_modules,
    step_fp8_optimizers,
)
from fp8_swiglu import FP8SwiGLU


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 synchronization tests require CUDA",
)


def _quantized_tensors(module: nn.Module) -> list[torch.Tensor]:
    tensors = []
    for fp8_module in iter_fp8_modules(module):
        for name in (
            "quantized_weight",
            "quantized_gate_weight",
            "quantized_up_weight",
            "quantized_down_weight",
        ):
            value = getattr(fp8_module, name, None)
            if value is not None:
                tensors.extend(tensor.clone() for tensor in value)
    return tensors


def test_linear_lazy_sync_without_binding():
    torch.manual_seed(0)
    layer = FP8Linear(32, 64, bias=True, device="cuda")
    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")

    layer(x).float().square().mean().backward()
    old_fp8 = layer._weight_fp8.clone()
    old_version = layer._synced_version

    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
    optimizer.step()

    assert layer._synced_version == old_version
    assert torch.equal(layer._weight_fp8, old_fp8)
    assert layer.master_weight._version != layer._synced_version

    new_fp8, _ = layer.quantized_weight
    assert not torch.equal(new_fp8, old_fp8)
    assert layer.master_weight._version == layer._synced_version


def test_mark_dirty_handles_data_mutation():
    torch.manual_seed(1)
    layer = FP8Linear(32, 64, bias=True, device="cuda")
    old_fp8 = layer._weight_fp8.clone()
    old_version = layer.master_weight._version

    with torch.no_grad():
        layer.master_weight.data.add_(0.25)

    assert layer.master_weight._version == old_version
    layer.mark_fp8_dirty()
    new_fp8, _ = layer.quantized_weight
    assert not torch.equal(new_fp8, old_fp8)
    assert not layer._fp8_dirty


def test_model_level_eager_binding():
    torch.manual_seed(2)
    model = nn.Sequential(
        FP8Linear(32, 64, bias=True, device="cuda"),
        nn.ReLU(),
        FP8Linear(64, 10, bias=True, device="cuda"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    bind_fp8_optimizer(model, optimizer)
    old_tensors = _quantized_tensors(model)

    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    F.cross_entropy(model(x).float(), torch.randint(0, 10, (16,), device="cuda")).backward()
    optimizer.step()

    for layer in iter_fp8_modules(model):
        assert layer._synced_version == layer.master_weight._version
    assert any(
        not torch.equal(old, new)
        for old, new in zip(old_tensors, _quantized_tensors(model))
    )


def test_multiple_optimizers_refresh_once_after_both_steps():
    torch.manual_seed(3)
    model = nn.Sequential(
        FP8Linear(32, 64, bias=True, device="cuda"),
        nn.ReLU(),
        FP8Linear(64, 10, bias=True, device="cuda"),
    )
    weights = [p for p in model.parameters() if p.ndim == 2]
    biases = [p for p in model.parameters() if p.ndim != 2]
    optimizers = [
        torch.optim.Muon(weights, lr=0.02, weight_decay=0.0),
        torch.optim.AdamW(biases, lr=1e-3, weight_decay=0.0),
    ]

    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    y = torch.randint(0, 10, (16,), device="cuda")
    F.cross_entropy(model(x).float(), y).backward()
    step_fp8_optimizers(model, optimizers)

    for layer in iter_fp8_modules(model):
        assert layer._synced_version == layer.master_weight._version
        assert not layer._fp8_dirty


def test_swiglu_lazy_sync_without_binding():
    torch.manual_seed(4)
    layer = FP8SwiGLU(32, 64, 10, bias=True, device="cuda")
    old_gate = layer._gate_weight_fp8.clone()
    old_up = layer._up_weight_fp8.clone()
    old_down = layer.down_proj._weight_fp8.clone()

    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    F.cross_entropy(layer(x).float(), torch.randint(0, 10, (16,), device="cuda")).backward()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    optimizer.step()

    assert torch.equal(layer._gate_weight_fp8, old_gate)
    assert torch.equal(layer._up_weight_fp8, old_up)
    assert torch.equal(layer.down_proj._weight_fp8, old_down)

    gate_fp8, _ = layer.quantized_gate_weight
    up_fp8, _ = layer.quantized_up_weight
    down_fp8, _ = layer.quantized_down_weight
    assert not torch.equal(gate_fp8, old_gate)
    assert not torch.equal(up_fp8, old_up)
    assert not torch.equal(down_fp8, old_down)
