"""Grouped task mapping: complete/partial groups, TMA tails, graph replay."""
import pytest
import torch

from experiments.tk_linear_backward import LinearBackwardPlan, libraries
from experiments.benchmark_tk_dual_online_gemm import exact_input, quantized_a
import mxtest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120")


@pytest.fixture(scope="module")
def operands():
    torch.set_num_threads(4)
    # 28 R16 output rows / 14 R32 rows: group13 has both full and short
    # groups; 192 columns also exercise the C128 output boundary.
    m, n, k = 193, 3457, 192
    x = exact_input(m, k, 73).cuda()
    dy = exact_input(m, n, 29).cuda()
    w, ws = mxtest.launch_weight_quantize(exact_input(n, k, 17).float().cuda())
    p = LinearBackwardPlan(dy, x, w, ws, dw_raster_group=1)
    p.prepare_dy()
    p.quantize_x()
    # Pad the reduction dimension before the independent dyadic FP64 oracle.
    dt = torch.nn.functional.pad(dy.cpu().T.contiguous(), (0, (-m) % 32))
    xt = torch.nn.functional.pad(x.cpu().T.contiguous(), (0, (-m) % 32))
    ref = (quantized_a(dt).double() @ quantized_a(xt).double().T).float()
    return p, ref


@pytest.mark.parametrize("config", range(8))
@pytest.mark.parametrize("swap", [False, True])
def test_grouped_native_dw(operands, config, swap):
    p, ref = operands
    out = torch.full_like(p.dwp, float("nan"))
    plan = p.dw_plan.owner.plan_native_dy(
        p.dyp, p.xq, p.xs, config, out=out, sms=3, swap_ab=swap, raster_group=13)
    torch.testing.assert_close(plan()[:p.n, :p.k].cpu(), ref, rtol=0, atol=0)
    # Ensure changing task order still overwrites all padded output elements.
    assert torch.isfinite(out).all().item()


@pytest.mark.parametrize("size", [1, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 64])
def test_grouped_reuse_stream_graph(operands, size):
    p, ref = operands
    q = p.xq.clone()
    out = torch.full_like(p.dwp, float("nan"))
    plan = p.dw_plan.owner.plan_native_dy(
        p.dyp, q, p.xs, out=out, sms=3, raster_group=size)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(out[:p.n, :p.k].cpu(), ref, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    q.view(torch.uint8).zero_()
    graph.replay()
    assert torch.count_nonzero(out).item() == 0
    q.copy_(p.xq)
    graph.replay()
    torch.testing.assert_close(out[:p.n, :p.k].cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_generic_gemm(online, dtype):
    _, _, lib = libraries()
    a = exact_input(1793, 128, 73).cuda()
    b = exact_input(192, 128, 29).cuda()
    bq, bs = mxtest.launch_row_quantize(b)
    kwargs = dict(output_dtype=dtype, b_scales=None if online else bs, sms=3)
    initial = torch.full((a.shape[0], b.shape[0]), float("nan"), device="cuda", dtype=dtype)
    baseline = lib.plan(a, b if online else bq, out=initial, **kwargs)
    ref = baseline().clone()
    out = torch.full_like(ref, float("nan"))
    grouped = lib.plan(a, b if online else bq, out=out, raster_group=13, **kwargs)
    torch.testing.assert_close(grouped(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, -1, 1.5, True, 2**31])
def test_invalid_group(operands, size):
    p, _ = operands
    with pytest.raises(ValueError, match="positive int32"):
        p.dw_plan.owner.plan_native_dy(p.dyp, p.xq, p.xs, raster_group=size)
