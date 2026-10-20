"""Integration tests for the MXFP8 SwiGLU backward composition."""

import pytest
import torch

from experiments.tk_mxfp8_gemm import pack_a_scales
from experiments.tk_mxfp8_swiglu_backward import MXFP8SwiGLUBackward, build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    from experiments.tk_mxfp8_swiglu_dgrad import OUTPUT as DX_OUTPUT
    from experiments.tk_mxfp8_wgrad import OUTPUT as WGRAD_OUTPUT
    from experiments.tk_silu_bwd import OUTPUT as ELEMENTWISE_OUTPUT

    binaries = [
        DX_OUTPUT / "libswd.so",
        WGRAD_OUTPUT / "libwgrad.so",
        ELEMENTWISE_OUTPUT / "libsilu_bwd_quantize.so",
    ]
    if any(not binary.exists() for binary in binaries):
        build()


def inputs(m, n, k):
    torch.manual_seed(909)
    dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    half_gate = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight_gate = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    weight_up = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.full(
        (n // 32, k // 32), 127, device="cuda", dtype=torch.uint8
    )
    return (
        dy,
        half_gate,
        up,
        x,
        weight_gate,
        weight_up,
        weight_scale,
        weight_scale.clone(),
    )


def test_shapes_reuse_and_graph():
    m, n, k = 137, 128, 128
    plan = MXFP8SwiGLUBackward(*inputs(m, n, k))
    dx, gate_grad, up_grad = plan()
    torch.cuda.synchronize()
    assert dx.shape == (m, k) and dx.dtype == torch.bfloat16
    assert gate_grad.shape == (n, k) and gate_grad.dtype == torch.bfloat16
    assert up_grad.shape == (n, k) and up_grad.dtype == torch.bfloat16
    assert torch.isfinite(dx).all()
    assert torch.isfinite(gate_grad).all()
    assert torch.isfinite(up_grad).all()
    baseline = (dx.clone(), gate_grad.clone(), up_grad.clone())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    saved_gate = plan.quantize.half_gate.clone()
    saved_up = plan.quantize.up.clone()
    saved_x = plan.x.clone()
    plan.quantize.half_gate.zero_()
    plan.quantize.up.zero_()
    plan.x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(plan.dx_plan.out) == 0
    assert torch.count_nonzero(plan.gate_weight_grad.out) == 0
    assert torch.count_nonzero(plan.up_weight_grad.out) == 0
    plan.quantize.half_gate.copy_(saved_gate)
    plan.quantize.up.copy_(saved_up)
    plan.x.copy_(saved_x)
    graph.replay()
    for actual, expected in zip(plan(), baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wgrad_outputs_match_existing_kernel_chain():
    from experiments.tk_mxfp8_wgrad import ColumnQuantizePlan, Library
    from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

    m, n, k = 137, 128, 128
    values = inputs(m, n, k)
    plan = MXFP8SwiGLUBackward(*values)
    dx, gate_grad, up_grad = plan()
    torch.cuda.synchronize()

    quantized = SiluBwdQuantizeLibrary().plan(values[0], values[1], values[2])
    quantized()
    xq = ColumnQuantizePlan(values[3])
    xq()
    library = Library()
    reference_gate = library.plan(
        quantized.gate_col,
        xq.out,
        quantized.gate_col_scale,
        xq.scales,
        config=12,
        raster_group=13,
    )()
    reference_up = library.plan(
        quantized.up_col,
        xq.out,
        quantized.up_col_scale,
        xq.scales,
        config=12,
        raster_group=13,
    )()
    torch.testing.assert_close(gate_grad, reference_gate, rtol=0, atol=0)
    torch.testing.assert_close(up_grad, reference_up, rtol=0, atol=0)
