"""Pure-MXFP8 down projection forward/backward integration tests."""

import pytest
import torch

from experiments.tk_mxfp8_dgrad import Library as DgradLibrary, RowQuantizePlan
from experiments.tk_mxfp8_down import MXFP8DownProjection
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_wgrad import ColumnQuantizePlan, Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def test_forward_and_backward_match_explicit_component_chain():
    torch.manual_seed(145)
    m, hidden_features, out_features = 137, 128, 96
    module = MXFP8DownProjection(
        hidden_features,
        out_features,
        device="cuda",
        forward_config=0,
        dx_config=2,
        dw_config=0,
        raster_group=1,
    )
    hidden = torch.randn(
        (m, hidden_features), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    dy = torch.randn((m, out_features), device="cuda", dtype=torch.bfloat16)
    out = module(hidden)
    dx, dw = torch.autograd.grad(out, (hidden, module.master_weight), dy)
    torch.cuda.synchronize()

    quantized_hidden = SiluBwdQuantizeLibrary().plan_direct(hidden.detach())
    weight_fp8, weight_scale = module.quantized_weight
    forward = GemmLibrary().plan(
        quantized_hidden.row_fp8,
        weight_fp8,
        quantized_hidden.row_scale,
        weight_scale,
        config=0,
        raster_group=1,
    )
    quantized_hidden()
    expected_out = forward()
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)

    dy_row = RowQuantizePlan(dy)
    dy_column = ColumnQuantizePlan(dy, transpose=True)
    dy_row()
    dy_column()
    dx_plan = DgradLibrary().plan(
        dy_row.out,
        weight_fp8,
        dy_row.scales,
        weight_scale,
        config=2,
        raster_group=1,
    )
    dw_plan = WgradLibrary().plan(
        dy_column.out,
        quantized_hidden.column_fp8,
        dy_column.scales,
        quantized_hidden.column_scale,
        config=0,
        raster_group=1,
    )
    torch.testing.assert_close(dx, dx_plan(), rtol=0, atol=0)
    torch.testing.assert_close(dw, dw_plan().float(), rtol=0, atol=0)


def test_weight_cache_refresh():
    module = MXFP8DownProjection(
        64,
        64,
        device="cuda",
        forward_config=0,
        dx_config=2,
        dw_config=0,
    )
    before = module._weight_fp8.clone()
    with torch.no_grad():
        module.master_weight.add_(0.25)
    module.refresh_fp8_weight()
    assert not torch.equal(before, module._weight_fp8)
