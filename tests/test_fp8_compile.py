"""torch.compile tests for the FP8 custom operators."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fp8_linear import FP8Linear, bind_fp8_optimizer, iter_fp8_modules
from fp8_swiglu import FP8SwiGLU


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 compile tests require CUDA",
)


def _reset_dynamo():
    torch._dynamo.reset()


def test_linear_fullgraph_forward_backward():
    _reset_dynamo()
    torch.manual_seed(10)
    layer = FP8Linear(32, 64, bias=True, device="cuda")
    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    target = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(layer, fullgraph=True)

    out = compiled(x)
    F.mse_loss(out.float(), target.float()).backward()
    with torch.no_grad():
        eager_out = layer(x)

    torch.testing.assert_close(out, eager_out, rtol=0, atol=0)
    assert layer.master_weight.grad is not None
    assert layer.bias.grad is not None
    _reset_dynamo()


def test_mlp_fullgraph_training_step():
    _reset_dynamo()
    torch.manual_seed(11)
    model = nn.Sequential(
        FP8Linear(32, 64, bias=True, device="cuda"),
        nn.ReLU(),
        FP8Linear(64, 10, bias=True, device="cuda"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    bind_fp8_optimizer(model, optimizer)
    compiled = torch.compile(model, fullgraph=True)
    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    y = torch.randint(0, 10, (16,), device="cuda")

    out = compiled(x)
    F.cross_entropy(out.float(), y).backward()
    optimizer.step()

    for layer in iter_fp8_modules(model):
        assert layer._synced_version == layer.master_weight._version

    out_after_step = compiled(x)
    with torch.no_grad():
        eager_after_step = model(x)
    torch.testing.assert_close(out_after_step, eager_after_step, rtol=0, atol=0)
    _reset_dynamo()


def test_mlp_fullgraph_dynamic_batch():
    _reset_dynamo()
    torch.manual_seed(12)
    model = nn.Sequential(
        FP8Linear(32, 64, bias=True, device="cuda"),
        nn.ReLU(),
        FP8Linear(64, 10, bias=True, device="cuda"),
    )
    compiled = torch.compile(model, fullgraph=True, dynamic=True)

    for batch_size in (16, 32, 64):
        x = torch.randn(
            batch_size,
            32,
            dtype=torch.bfloat16,
            device="cuda",
        )
        out = compiled(x)
        assert out.shape == (batch_size, 10)
    _reset_dynamo()


@pytest.mark.parametrize("fp8_wgrad", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_swiglu_fullgraph_forward_backward(fp8_wgrad, bias):
    _reset_dynamo()
    torch.manual_seed(13)
    layer = FP8SwiGLU(
        32,
        64,
        10,
        bias=bias,
        fp8_wgrad=fp8_wgrad,
        device="cuda",
    )
    x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.randn(16, 10, dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(layer, fullgraph=True)

    out = compiled(x)
    F.mse_loss(out.float(), target.float()).backward()
    targets = (x, *layer.parameters())
    compiled_grads = [tensor.grad.clone() for tensor in targets]
    for tensor in targets:
        tensor.grad = None
    eager_out = layer(x)
    F.mse_loss(eager_out.float(), target.float()).backward()
    for tensor, compiled_grad in zip(targets, compiled_grads):
        torch.testing.assert_close(tensor.grad, compiled_grad, rtol=0, atol=0)

    torch.testing.assert_close(out, eager_out, rtol=0, atol=0)
    assert layer.gate_weight.grad is not None
    assert layer.up_weight.grad is not None
    assert layer.down_proj.master_weight.grad is not None
    if bias:
        assert layer.down_proj.bias.grad is not None
    _reset_dynamo()
