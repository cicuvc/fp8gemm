"""Backward regressions: auxiliary outputs, optional bias and FP8 prequantization."""
import pytest
import torch

import mxtest
from experiments import swiglu_prequant
from fp8_swiglu import FP8SwiGLU, _fp8_swiglu_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def forward_with_saved(layer, x):
    gw, gs = layer.quantized_gate_weight
    uw, us = layer.quantized_up_weight
    dw, ds = layer.quantized_down_weight
    return _fp8_swiglu_fwd(
        x, layer.gate_weight, layer.up_weight, layer.down_proj.master_weight,
        gw, gs, uw, us, dw, ds, layer.down_proj.bias, layer.fp8_wgrad,
    ), (dw, ds, gw, gs, uw, us)


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('fp8_wgrad', [False, True])
def test_autograd_matches_explicit_backward(bias, fp8_wgrad):
    torch.manual_seed(93)
    layer = FP8SwiGLU(65, 97, 33, bias=bias, fp8_wgrad=fp8_wgrad, device='cuda')
    x = torch.randn(17, 65, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    (y, gate, up, hidden), weights = forward_with_saved(layer, x)
    assert y.requires_grad
    assert not any(t.requires_grad for t in (gate, up, hidden))
    # Exercise grad_output.contiguous() with a non-contiguous incoming gradient.
    dy = torch.randn(33, 17, dtype=torch.bfloat16, device='cuda').T
    targets = (x, layer.gate_weight, layer.up_weight, layer.down_proj.master_weight)
    if bias:
        targets += (layer.down_proj.bias,)
    grads = torch.autograd.grad(y, targets, dy, retain_graph=True)
    expected = mxtest.launch_fp8_down_swiglu_backward(
        dy.contiguous(), hidden, gate, up, *weights, x, use_fp8_wgrad=fp8_wgrad,
    )
    if bias:
        expected += (dy.float().sum(0),)
    for actual, reference in zip(grads, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Retained graphs must remain valid with materialize_grads disabled.
    again = torch.autograd.grad(y, targets, dy)
    for actual, reference in zip(again, grads):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(1, 31, 33, 17), (17, 65, 97, 33), (128, 256, 512, 192), (256, 512, 1024, 512)])
def test_prequantized_backward_matches_online(shape):
    m, k, h, o = shape
    torch.manual_seed(94)
    layer = FP8SwiGLU(k, h, o, bias=False, device='cuda')
    x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        (_, gate, up, hidden), weights = forward_with_saved(layer, x)
    dy = torch.randn(m, o, dtype=torch.bfloat16, device='cuda')
    inputs = (dy, hidden, gate, up, *weights, x)
    online = mxtest.launch_fp8_down_swiglu_backward(*inputs, return_intermediate_grads=True)
    prequantized = swiglu_prequant.launch_fp8_down_swiglu_backward(*inputs, prequantize=True, return_intermediate_grads=True)
    for actual, reference in zip(prequantized, online):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
