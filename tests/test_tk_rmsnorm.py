"""Correctness and replay tests for RMSNorm dual-direction FP8 prenorm."""

import pytest
import torch

from experiments.tk_mxfp8_dgrad import RowQuantizePlan
from experiments.tk_mxfp8_wgrad import ColumnQuantizePlan
from experiments.tk_rmsnorm import RMSNormDualQuantPlan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton CUDA test",
)


@pytest.mark.parametrize("shape", [(1, 32), (17, 128), (137, 192)])
def test_matches_materialized_rmsnorm_and_quantizers(shape):
    m, k = shape
    torch.manual_seed(72)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    factor = torch.randn(k, device="cuda").mul_(0.1).add_(1.0)
    eps = 1e-6
    plan = RMSNormDualQuantPlan(x, factor, eps=eps)
    row, row_scale, column, column_scale, inverse = plan()
    torch.cuda.synchronize()

    reference_inverse = torch.rsqrt(x.float().square().mean(1) + eps)
    normalized = (
        x.float() * inverse[:, None] * factor[None, :]
    ).bfloat16()
    expected_row, expected_row_scale = RowQuantizePlan(normalized)()
    expected_column, expected_column_scale = ColumnQuantizePlan(normalized)()
    torch.testing.assert_close(inverse, reference_inverse, rtol=2e-7, atol=2e-7)
    torch.testing.assert_close(
        row.view(torch.uint8), expected_row.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(row_scale, expected_row_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        column.view(torch.uint8),
        expected_column.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(column_scale, expected_column_scale, rtol=0, atol=0)


def test_repeated_inputs_clear_column_scale_and_graph_replay():
    m, k = 137, 128
    torch.manual_seed(811)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    factor = torch.randn(k, device="cuda").mul_(0.1).add_(1.0)
    plan = RMSNormDualQuantPlan(x, factor)
    first = [output.clone() for output in plan()]
    torch.cuda.synchronize()
    second = [output.clone() for output in plan()]
    for actual, expected in zip(second, first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    saved_x = x.clone()
    x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(plan.row_fp8) == 0
    assert torch.count_nonzero(plan.column_fp8) == 0
    assert torch.count_nonzero(plan.row_scale) == 0
    assert torch.count_nonzero(plan.column_scale) == 0
    x.copy_(saved_x)
    graph.replay()
    for actual, expected in zip(plan(), first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
