"""Exact BF16 topK, TMA N permutation + scales, tails, reuse, and graphs."""

import pytest
import torch

from experiments.mxfp8_core import quantize_weight
from experiments.tk_mxfp8_topk import Library, MXFP8TopKLinear, CONFIGS, ROOT, OUTPUT, build
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary
from tests.test_tk_mxfp8_gemm import exact_operands

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled_topk():
    torch.set_num_threads(4)
    sources = [ROOT / "cuda" / p for p in (
        "persistent_mxfp8_topk.cu", "mxfp8_topk.cuh", "mxfp8_topk_configs.inc",
        "fp8_shared_load.cuh", "tma_ring_pipe.cuh")]
    binary = OUTPUT / "libtopk.so"
    if not binary.exists() or any(p.stat().st_mtime > binary.stat().st_mtime for p in sources):
        build()


def inputs(m, h, v, seed=592):
    torch.manual_seed(seed)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(v, h, device="cuda") * .05
    # Different scale groups expose stale/unpermuted B scale metadata.
    w *= (2. ** ((torch.arange(v, device="cuda") // 32) % 7 - 3))[:, None]
    return x, w


def quantized(x, w):
    xq = SiluBwdQuantizeLibrary().plan_direct(x); xq()
    wq, ws = quantize_weight(w.float())
    return xq, (xq.row_fp8, wq, xq.row_scale, ws)


def plain_logits(args, config):
    a, b, sa, sb = args
    v, h = b.shape
    # Ordinary BF16-output GEMM needs N aligned to its output TMA panel.
    # Padding prequantized rows preserves every original dot product and scale.
    vp = (v + 63) // 64 * 64
    if vp != v:
        bpad = torch.zeros((vp, h), device=b.device, dtype=b.dtype)
        spad = torch.zeros((vp // 32, h // 32), device=b.device, dtype=sb.dtype)
        bpad[:v].copy_(b); spad[:len(sb)].copy_(sb)
        b, sb = bpad, spad
    return GemmLibrary().plan(a, b, sa, sb, config=config)()[:, :v].contiguous()


def reference(z, k):
    ids = torch.argsort(z.float(), dim=-1, descending=True, stable=True)[:, :k]
    return z.gather(1, ids), ids


def assert_exact(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.fixture(scope="module", params=[(0, 1, 96, 64), (8, 137, 192, 320),
                                        (70, 257, 256, 32000), (71, 129, 64, 320)])
def operands(request):
    config, m, h, v = request.param
    x, w = inputs(m, h, v)
    xq, args = quantized(x, w)
    expected = reference(plain_logits(args, config), 16)
    return config, xq, args, expected


@pytest.mark.parametrize("k", range(1, 17))
@pytest.mark.parametrize("permute", [False, True])
def test_all_k_and_configs(operands, k, permute):
    config, xq, args, expected = operands
    plan = Library().plan(*args, top_k=k, config=config, permute_n=permute, sms=3)
    assert_exact(plan(), tuple(t[:, :k] for t in expected))
    assert plan.local_keys.shape == (len(args[0]), plan.parts, plan.bucket)
    assert plan.info["initial_regs"] >= 168


@pytest.mark.parametrize("config", [0, 8, 71])
@pytest.mark.parametrize("v", [1, 7, 17, 31, 33, 127, 129, 255, 257, 511])
def test_vocab_tails(config, v):
    x, w = inputs(7, 128, v)
    xq, args = quantized(x, w)
    k = min(v, 16)
    expected = reference(plain_logits(args, config), k)
    plan = Library().plan(*args, top_k=k, config=config)
    assert_exact(plan(), expected)
    assert ((plan.merge.indices >= 0) & (plan.merge.indices < v)).all()


@pytest.mark.parametrize("config", [0, 8, 70, 71])
def test_exact_scale_mapping(config):
    args, z = exact_operands(133, 512, 384)
    expected = reference(z, 16)
    for permute in (False, True):
        plan = Library().plan(*args, config=config, permute_n=permute, top_k=16, sms=2)
        assert_exact(plan(), expected)


@pytest.mark.parametrize("config", [0, 8, 70, 71])
@pytest.mark.parametrize("kind", ["zero", "negative", "ties"])
def test_ties_and_all_negative(config, kind):
    x = torch.ones(17, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(273, 128, device="cuda")
    if kind == "negative":
        w.fill_(-.25)
    elif kind == "ties":
        # Top scores repeat across lanes and across complete / partial tiles.
        w[::17] = .5
        w[::19] = -.25
    xq, args = quantized(x, w)
    expected = reference(plain_logits(args, config), 16)
    for permute in (False, True):
        assert_exact(Library().plan(*args, top_k=16, config=config, permute_n=permute)(), expected)


def unpack(keys):
    order = keys >> 32
    bits = torch.where((order & 0x8000) != 0, order ^ 0x8000, (~order) & 0xffff)
    values = (bits << 16).to(torch.int32).view(torch.float32).bfloat16()
    ids = 0xffffffff - (keys & 0xffffffff)
    return values, ids


@pytest.mark.parametrize("config", [0, 8, 70, 71])
@pytest.mark.parametrize("k", [3, 16])
def test_local_candidates_and_rebind(config, k):
    c = CONFIGS[config][1]
    x, w = inputs(137, 128, c * 2 + 5)
    xq, args = quantized(x, w)
    plan = Library().plan(*args, top_k=k, config=config, sms=1)
    for seed in (1, 2):
        x, w = inputs(137, 128, c * 2 + 5, seed=seed)
        xq, args = quantized(x, w)
        plan.rebind(*args)
        plan.local_keys.fill_(-1)
        local = plan.local()
        z = plain_logits(args, config)
        vals, ids = unpack(local)
        for part in range(plan.parts):
            begin, end = part * c, min((part + 1) * c, w.shape[0])
            count = min(plan.bucket, end - begin)
            rv, ri = reference(z[:, begin:end], count)
            assert_exact((vals[:, part, :count], ids[:, part, :count]), (rv, ri + begin))
            assert torch.count_nonzero(local[:, part, count:]).item() == 0
        assert_exact(plan.merge(local), reference(z, k))


@pytest.mark.parametrize("k", [1, 7, 16])
@pytest.mark.parametrize("permute", [False, True])
def test_module_graph_replay_and_output_lifetime(k, permute):
    x, w = inputs(137, 128, 513)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight(w.detach())
    module = MXFP8TopKLinear(top_k=k, permute_n=permute)
    def step():
        return module(x, w, quantized_weight=(wq, ws))
    saved = step()
    copies = tuple(t.clone() for t in saved)
    assert not saved[0].requires_grad and saved[1].dtype == torch.int64
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed in (12, 13):
        nx, nw = inputs(137, 128, 513, seed=seed)
        with torch.no_grad():
            x.copy_(nx); w.copy_(nw)
            nq, ns = quantize_weight(w)
            wq.copy_(nq); ws.copy_(ns)
        expected = step()
        graph.replay()
        assert_exact(captured, expected)
        xq, args = quantized(x.detach(), w.detach())
        assert_exact(captured, reference(plain_logits(args, 8), k))
    assert_exact(saved, copies)
    assert all(xq.x is None and not p.refs for xq, p in module._states.values())
    module.clear_workspace()
    assert_exact(saved, copies)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_uncached_weight(dtype):
    x, w = inputs(17, 128, 259)
    w = w.to(dtype)
    xq, args = quantized(x, w)
    result = MXFP8TopKLinear(top_k=11)(x, w)
    assert_exact(result, reference(plain_logits(args, 8), 11))


def test_large_token_ids_and_four_merge_levels():
    v = 131137
    x = torch.ones(3, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(v, 128, device="cuda")
    w[-100:] = .5
    module = MXFP8TopKLinear(top_k=16)
    values, ids = module(x, w)
    expected_ids = torch.arange(v - 100, v - 84, device="cuda").expand(3, -1)
    assert_exact((values, ids), (torch.full_like(values, 64), expected_ids))
    _, plan = next(iter(module._states.values()))
    assert plan.parts == 1025 and len(plan.merge.levels) == 3


def test_input_contract():
    for k in (0, 17, True, 1.5):
        with pytest.raises(ValueError, match="top_k"):
            MXFP8TopKLinear(top_k=k)
    x, w = inputs(3, 128, 256)
    module = MXFP8TopKLinear()
    with pytest.raises(ValueError, match="BF16"):
        module(x.float(), w)
    with pytest.raises(ValueError, match="top_k"):
        module(x, w[:7])
    with pytest.raises(ValueError, match="panels"):
        module(x[:, :96].contiguous(), w[:, :96].contiguous())
    xq, args = quantized(x, w)
    plan = Library().plan(*args)
    with pytest.raises(ValueError, match="Expected"):
        plan.rebind(args[0][:1], *args[1:])
    with pytest.raises(ValueError, match="quantized weight"):
        module(x, w, quantized_weight=(args[1], args[3].int()))
    with pytest.raises(ValueError, match="overlap"):
        plan.merge(plan.local(), out=(plan.merge.values, plan.local_keys.view(-1)[:48].reshape(3, 16)))
