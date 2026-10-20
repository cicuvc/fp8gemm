"""Forward and autograd integration tests for the complete MXFP8 SwiGLU."""

import pytest
import torch

from fp8_linear import FP8Linear
from experiments.tk_mxfp8_dgrad import RowQuantizePlan
from experiments.tk_mxfp8_swiglu import MXFP8SwiGLULibrary
from experiments.tk_mxfp8_swiglu_backward import MXFP8SwiGLUBackward
from experiments.tk_mxfp8_swiglu_layer import MXFP8SwiGLU

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def test_forward_and_backward_match_explicit_component_chain():
    torch.manual_seed(404)
    layer = MXFP8SwiGLU(128, 128, 96, device="cuda")
    assert layer.down_proj.backward_backend == "cuda"
    x = torch.randn((137, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn((137, 96), device="cuda", dtype=torch.bfloat16)

    out = layer(x)
    torch.cuda.synchronize()
    gate_fp8 = layer._gate_weight_fp8
    gate_scale = layer._gate_weight_scale
    up_fp8 = layer._up_weight_fp8
    up_scale = layer._up_weight_scale
    quantized_x = RowQuantizePlan(x.detach())
    gate_linear = torch.empty((137, 128), device="cuda", dtype=torch.bfloat16)
    up_linear = torch.empty_like(gate_linear)
    forward = MXFP8SwiGLULibrary().plan(
        quantized_x.out,
        quantized_x.scales,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
        config=1,
        raster_group=13,
        gate_out=gate_linear,
        up_out=up_linear,
    )
    quantized_x()
    forward()
    reference_down = FP8Linear(
        128,
        96,
        bias=True,
        device="cuda",
        backward_backend="triton",
    )
    with torch.no_grad():
        reference_down.master_weight.copy_(layer.down_proj.master_weight)
        reference_down.bias.copy_(layer.down_proj.bias)
    reference_down.refresh_fp8_weight()
    expected_out = reference_down(forward.out)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)

    grads = torch.autograd.grad(
        out,
        (
            x,
            layer.gate_weight,
            layer.up_weight,
            layer.down_proj.master_weight,
            layer.down_proj.bias,
        ),
        dy,
    )
    hidden = forward.out.detach().requires_grad_()
    down_out = reference_down(hidden)
    down_grads = torch.autograd.grad(
        down_out,
        (hidden, reference_down.master_weight, reference_down.bias),
        dy,
    )
    backward = MXFP8SwiGLUBackward(
        down_grads[0],
        gate_linear,
        up_linear,
        x.detach(),
        gate_fp8,
        up_fp8,
        gate_scale,
        up_scale,
        dx_config=5,
        wgrad_config=6,
        raster_group=13,
    )
    expected_dx, expected_gate, expected_up = backward()
    torch.testing.assert_close(grads[0], expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(grads[1], expected_gate.float(), rtol=0, atol=0)
    torch.testing.assert_close(grads[2], expected_up.float(), rtol=0, atol=0)
    torch.testing.assert_close(grads[3], down_grads[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[4], down_grads[2], rtol=0, atol=0)


def test_weight_refresh_and_optimizer_hook():
    torch.manual_seed(811)
    layer = MXFP8SwiGLU(64, 64, 64, device="cuda")
    before = layer._gate_weight_fp8.clone()
    with torch.no_grad():
        layer.gate_weight.add_(1)
    layer.refresh_fp8_weights()
    assert not torch.equal(before, layer._gate_weight_fp8)

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer.bind_optimizer(optimizer)
    old_scale_pointer = layer._up_weight_scale.data_ptr()
    with torch.no_grad():
        layer.up_weight.add_(0.25)
    optimizer.step()
    assert layer._synced_versions == (
        layer.gate_weight._version,
        layer.up_weight._version,
        layer.down_proj.master_weight._version,
    )
    assert isinstance(old_scale_pointer, int)
