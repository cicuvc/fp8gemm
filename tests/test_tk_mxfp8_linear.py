"""Autograd integration for the bias-free pure-MXFP8 linear layer."""

import pytest
import torch

from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_linear import MXFP8Linear
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def test_forward_backward_matches_component_chain():
    torch.manual_seed(407)
    layer = MXFP8Linear(
        128,
        64,
        device="cuda",
        forward_config=0,
        dx_config=2,
        dw_config=0,
        raster_group=1,
    )
    x = torch.randn((137, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn((137, 64), device="cuda", dtype=torch.bfloat16)

    out = layer(x)
    dx, dw = torch.autograd.grad(out, (x, layer.master_weight), dy)
    torch.cuda.synchronize()

    xq = SiluBwdQuantizeLibrary().plan_direct(x.detach())
    weight, weight_scale = layer.quantized_weight
    forward = GemmLibrary().plan(
        xq.row_fp8,
        weight,
        xq.row_scale,
        weight_scale,
        config=0,
        raster_group=1,
    )
    xq()
    torch.testing.assert_close(out, forward(), rtol=0, atol=0)

    dyq = SiluBwdQuantizeLibrary().plan_direct(dy, transpose=True)
    dx_plan = DgradLibrary().plan(
        dyq.row_fp8,
        weight,
        dyq.row_scale,
        weight_scale,
        config=2,
        raster_group=1,
    )
    dw_plan = WgradLibrary().plan(
        dyq.column_fp8,
        xq.column_fp8,
        dyq.column_scale,
        xq.column_scale,
        config=0,
        raster_group=1,
    )
    dyq()
    torch.testing.assert_close(dx, dx_plan(), rtol=0, atol=0)
    torch.testing.assert_close(dw, dw_plan().float(), rtol=0, atol=0)


def test_weight_cache_refresh_and_optimizer_hook():
    layer = MXFP8Linear(
        64,
        64,
        device="cuda",
        forward_config=0,
        dx_config=2,
        dw_config=0,
    )
    before = layer._weight_fp8.clone()
    with torch.no_grad():
        layer.master_weight.add_(0.25)
    layer.refresh_fp8_weight()
    assert not torch.equal(before, layer._weight_fp8)

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer.bind_optimizer(optimizer)
    with torch.no_grad():
        layer.master_weight.add_(0.25)
    optimizer.step()
    assert layer._synced_version == layer.master_weight._version
