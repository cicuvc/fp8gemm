"""Grouped forward raster and K32 register prefetch: tails, slot reuse, streams."""
import pytest
import torch

from experiments.tk_persistent_gemm import GemmLibrary, build, DEFAULT_OUTPUT, ROOT
from experiments.benchmark_tk_persistent_gemm import exact_inputs, quantized_a

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120")


@pytest.fixture(scope="module", autouse=True)
def compiled_library():
    binary = DEFAULT_OUTPUT / "libgemm.so"
    sources = [ROOT/"cuda/persistent_fp8_gemm.cu", ROOT/"cuda/tma_ring_pipe.cuh",
               *(ROOT/"third_party/thunderkittens/include").rglob("*.cuh")]
    if not binary.exists() or any(p.stat().st_mtime > binary.stat().st_mtime for p in sources):
        build()


@pytest.fixture(scope="module", params=[(1, 64, 64), (1793, 192, 192), (777, 320, 320)])
def operands(request):
    torch.set_num_threads(4)
    m, n, k = request.param
    a, b = exact_inputs(m, n, k)
    s = (torch.arange(((n+31)//32)*((k+31)//32)).reshape((n+31)//32, (k+31)//32) % 5 + 125).byte()
    scale = torch.ldexp(torch.ones_like(s, dtype=torch.float64), s.int()-127)
    dequant = b.double() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
    ref = (quantized_a(a).double() @ dequant.T).bfloat16()
    unit_ref = (quantized_a(a).double() @ b.double().T).bfloat16()
    return GemmLibrary(), a.cuda(), b.cuda(), s.cuda(), ref, unit_ref


@pytest.mark.parametrize("config", range(10))
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("prefetch,group", [(False, 13), (True, 1), (True, 13)])
def test_forward_exact(operands, config, scaled, prefetch, group):
    lib, a, b, s, ref, unit_ref = operands
    out = torch.full((a.shape[0], b.shape[0]), float("nan"), device="cuda", dtype=torch.bfloat16)
    plan = lib.plan(a, b, config, out=out, sms=3, scales=s if scaled else None,
                    raster_group=group, prefetch=prefetch)
    torch.testing.assert_close(plan().cpu(), ref if scaled else unit_ref, rtol=0, atol=0)


@pytest.mark.parametrize("config", [0, 2, 4, 6])
@pytest.mark.parametrize("k", [32, 96, 160])
def test_odd_k32_fragments(config, k):
    a, b = exact_inputs(1031, 192, k)
    ref = (quantized_a(a).double() @ b.double().T).bfloat16()
    p = GemmLibrary().plan(a.cuda(), b.cuda(), config, sms=3, raster_group=4, prefetch=True)
    torch.testing.assert_close(p().cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("group", [1, 4, 8, 13, 16])
def test_prefetch_tail_reuse(group):
    # Three BK64 shared packets and six K32 fragments per task, with a short
    # final raster group; three persistent CTAs reuse the same ring many times.
    a, b = exact_inputs(1793, 192, 192)
    a, b = a.cuda(), b.cuda()
    saved = a.clone()
    scales = torch.full((6, 6), 127, device="cuda", dtype=torch.uint8)
    p = GemmLibrary().plan(a, b, 8, scales=scales, sms=3, raster_group=group, prefetch=True)
    ref = p().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        p()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(p.out, ref, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p()
    a.zero_()
    graph.replay()
    assert torch.count_nonzero(p.out).item() == 0
    a.copy_(saved)
    graph.replay()
    torch.testing.assert_close(p.out, ref, rtol=0, atol=0)
    scales.fill_(126)
    graph.replay()
    expected = (quantized_a(saved.cpu()).double() @ b.cpu().double().T * .5).bfloat16()
    torch.testing.assert_close(p.out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("group", [0, -1, True, 1.5, 2**31])
def test_invalid_group(group):
    with pytest.raises(ValueError, match="positive int32"):
        GemmLibrary().plan(None, None, raster_group=group)
