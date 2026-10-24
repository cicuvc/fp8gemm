"""E4M3 uses the upper safe E8M0 exponent range."""

import pytest
import torch

from mxfp8._core import quantize_weight
from mxfp8._impl.rmsnorm import RMSNormDualQuantPlan
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def unpack_scale_words(words):
    return words.view(torch.uint8).reshape(words.shape[0], -1)


def test_weight_quantizer_uses_seven_exponent_offset():
    weight = torch.ones((32, 128), device="cuda", dtype=torch.float32)
    quantized, scales = quantize_weight(weight)

    assert torch.all(scales == 120)
    assert quantized.float().abs().max().item() == 128


def test_dual_quantizer_uses_seven_exponent_offset():
    values = torch.ones((32, 128), device="cuda", dtype=torch.bfloat16)
    plan = SiluBwdQuantizeLibrary().plan_direct(values)
    row, row_scale, column, column_scale = plan()

    assert torch.all(unpack_scale_words(row_scale) == 120)
    assert torch.all((column_scale & 0xFF) == 120)
    assert row.float().abs().max().item() == 128
    assert column.float().abs().max().item() == 128


def test_dual_quantizer_clamps_zero_blocks():
    values = torch.zeros((32, 128), device="cuda", dtype=torch.bfloat16)
    plan = SiluBwdQuantizeLibrary().plan_direct(values)
    row, row_scale, column, column_scale = plan()

    assert torch.count_nonzero(row) == 0
    assert torch.count_nonzero(column) == 0
    assert torch.count_nonzero(row_scale) == 0
    assert torch.count_nonzero(column_scale) == 0


def test_rmsnorm_quantizer_uses_seven_exponent_offset():
    values = torch.ones((32, 128), device="cuda", dtype=torch.bfloat16)
    factor = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    plan = RMSNormDualQuantPlan(values, factor)
    row, row_scale, column, column_scale, _ = plan()

    assert torch.all(unpack_scale_words(row_scale) == 120)
    assert torch.all((column_scale & 0xFF) == 120)
    assert row.float().abs().max().item() == 128
    assert column.float().abs().max().item() == 128


def test_rmsnorm_quantizer_clamps_zero_blocks():
    values = torch.zeros((32, 128), device="cuda", dtype=torch.bfloat16)
    factor = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    plan = RMSNormDualQuantPlan(values, factor)
    row, row_scale, column, column_scale, _ = plan()

    assert torch.count_nonzero(row) == 0
    assert torch.count_nonzero(column) == 0
    assert torch.count_nonzero(row_scale) == 0
    assert torch.count_nonzero(column_scale) == 0
