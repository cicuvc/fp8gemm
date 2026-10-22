"""LM-head LSE, fused CE quantization, and chunked first-order autograd."""

import pytest
import torch
import torch.nn.functional as F

from experiments.mxfp8_core import quantize_weight
from experiments.mxfp8_ce import CEPlan, merge_lse
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_lse import Library as LSELibrary, CONFIGS
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary
from experiments.tk_mxfp8_fused_ce import MXFP8FusedLinearCrossEntropy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def inputs(m, h, v, seed=921):
    torch.manual_seed(seed)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(v, h, device="cuda", dtype=torch.float32) * .05
    y = torch.randint(v, (m,), device="cuda")
    return x, w, y


@pytest.mark.parametrize("config,m,h,v", [
    (0, 1, 64, 64), (8, 137, 128, 320), (70, 257, 256, 32000),
    (71, 129, 128, 192),
])
def test_gemm_partial_lse_and_rebind(config, m, h, v):
    x, w, y = inputs(m, h, v)
    xq = SiluBwdQuantizeLibrary().plan_direct(x)
    wq, ws = quantize_weight(w)
    xq()
    normal = GemmLibrary().plan(xq.row_fp8, wq, xq.row_scale, ws, config=config)
    plan = LSELibrary().plan(xq.row_fp8, wq, xq.row_scale, ws, config=config)
    z, partial = plan()
    torch.testing.assert_close(z, normal(), rtol=0, atol=0)
    block = CONFIGS[config][1]
    expected = torch.stack([t.float().logsumexp(-1) for t in z.split(block, dim=-1)], dim=-1)
    torch.testing.assert_close(partial, expected, atol=2e-6, rtol=2e-6)
    lse, loss = merge_lse(z, partial, y)
    torch.testing.assert_close(lse, z.float().logsumexp(-1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(loss, F.cross_entropy(z.float(), y, reduction="none"), atol=2e-6, rtol=2e-6)
    old_z = z.clone()
    x.mul_(2)
    xq()
    out, lp = torch.empty_like(z), torch.empty_like(partial)
    plan.rebind(xq.row_fp8, wq, xq.row_scale, ws, out=out, partial_lse=lp)
    plan()
    torch.testing.assert_close(out, normal(), rtol=0, atol=0)
    torch.testing.assert_close(z, old_z, rtol=0, atol=0)
    expected = torch.stack([t.float().logsumexp(-1) for t in out.split(block, dim=-1)], dim=-1)
    torch.testing.assert_close(lp, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("m,v,uniform", [(1, 64, True), (33, 128, True), (137, 320, False), (256, 32000, False)])
def test_ce_dual_quantization_matches_cuda(m, v, uniform):
    torch.manual_seed(29)
    z = torch.zeros((m, v), device="cuda", dtype=torch.bfloat16) if uniform else torch.randn((m, v), device="cuda", dtype=torch.bfloat16)
    y = torch.arange(m, device="cuda") % v
    lse = z.float().logsumexp(-1)
    g = torch.exp(z.float() - lse[:, None])
    g[torch.arange(m, device="cuda"), y] -= 1
    g = g.bfloat16()
    reference = SiluBwdQuantizeLibrary().plan_direct(g, transpose=True)
    reference()
    plan = CEPlan(m, v, z.device)
    saved_z = z.clone()
    for _ in range(2):
        plan(z, lse, y)
        for actual, expected in ((plan.row_scale, reference.row_scale), (plan.column_scale, reference.column_scale)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # exp implementations can straddle a BF16 rounding midpoint very rarely.
        for actual, expected in ((plan.row_fp8, reference.row_fp8), (plan.column_fp8, reference.column_fp8)):
            delta = (actual.float() - expected.float()).abs()
            if uniform:
                assert delta.max().item() == 0
            else:
                assert (delta != 0).float().mean().item() < 1e-5
                torch.testing.assert_close(actual.float(), expected.float(), rtol=.125, atol=0)
        assert torch.count_nonzero(plan.column_fp8[:, m:].float()).item() == 0
    torch.testing.assert_close(z, saved_z, rtol=0, atol=0)


def component_reference(x, w, y, chunk_size, configs=(8, 12, 12)):
    """Independent chain: plain GEMM -> PyTorch CE math -> CUDA quantizer."""
    wq, ws = quantize_weight(w.float())
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w, dtype=torch.float32)
    losses = []
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        xq = SiluBwdQuantizeLibrary().plan_direct(x[start:end])
        xq()
        z = GemmLibrary().plan(xq.row_fp8, wq, xq.row_scale, ws, config=configs[0])()
        losses.append(F.cross_entropy(z.float(), y[start:end], reduction="sum"))
        g = torch.exp(z.float() - z.float().logsumexp(-1)[:, None])
        g[torch.arange(end-start, device=x.device), y[start:end]] -= 1
        gq = SiluBwdQuantizeLibrary().plan_direct(g.bfloat16(), transpose=True)
        gq()
        dx[start:end] = DgradLibrary().plan(gq.row_fp8, wq, gq.row_scale, ws, config=configs[1])()
        dw.add_(WgradLibrary().plan(gq.column_fp8, xq.column_fp8, gq.column_scale, xq.column_scale, config=configs[2])())
    return torch.stack(losses).sum(), dx, dw.to(w.dtype)


@pytest.mark.parametrize("m,h,v,chunk,configs,dtype", [
    (1, 64, 64, 128, (0, 0, 0), torch.float32),
    (257, 128, 256, 128, (8, 12, 12), torch.float32),
    (137, 128, 32000, 128, (70, 12, 12), torch.bfloat16),
])
def test_chunked_loss_and_gradients(m, h, v, chunk, configs, dtype):
    x, w, y = inputs(m, h, v)
    w = w.to(dtype)
    expected_loss, expected_dx, expected_dw = component_reference(x, w, y, chunk, configs)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=chunk, forward_config=configs[0], dx_config=configs[1], dw_config=configs[2])
    loss = criterion(x, y, w)
    dx, dw = torch.autograd.grad(loss, (x, w), retain_graph=True)
    torch.testing.assert_close(loss, expected_loss, atol=2e-4, rtol=2e-6)
    for actual, expected in ((dx, expected_dx), (dw, expected_dw)):
        relative = (actual.float()-expected.float()).norm() / expected.float().norm()
        assert relative.item() < 2e-3
    # Reuse workspace before consuming the first saved gradients again.
    criterion(x * 2, y, w)
    for scale in (.125, -2., 0.):
        scaled = torch.autograd.grad(loss, (x, w), grad_outputs=torch.tensor(scale, device=x.device), retain_graph=True)
        for actual, expected in zip(scaled, (dx, dw)):
            torch.testing.assert_close(actual, expected * scale, atol=0, rtol=0)
    # Workspaces don't retain input activations after forward.
    for state in criterion._states.values():
        assert state.xq.x is None and not state.gemm.refs and not state.dx.refs


def test_bf16_reference_accuracy_and_tied_weight():
    x, w, y = inputs(257, 128, 512)
    w = w.bfloat16().requires_grad_()
    x.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)
    loss = criterion(x, y, w) / len(x)
    dx, dw = torch.autograd.grad(loss, (x, w))
    ref = F.cross_entropy(F.linear(x, w).float(), y)
    rx, rw = torch.autograd.grad(ref, (x, w))
    assert abs(loss.item() - ref.item()) < .01
    for a, b in ((dx, rx), (dw, rw)):
        assert ((a.float()-b.float()).norm()/b.float().norm()).item() < .08
    # The same BF16 parameter can also receive an embedding lookup gradient.
    embedded = F.embedding(y, w)
    both = criterion(embedded, y, w)
    both.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_cuda_graph_replay_with_new_inputs_and_weight_cache():
    x, w, y = inputs(137, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight(w.detach())
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)
    def step():
        loss = criterion(x, y, w, quantized_weight=(wq, ws))
        dx, dw = torch.autograd.grad(loss / len(x), (x, w))
        return loss, dx, dw
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for i in range(2):
        with torch.no_grad():
            x.add_(.125)
            y.copy_((y + 17) % w.shape[0])
            w.add_(.001)
            new_q, new_s = quantize_weight(w.detach())
            wq.copy_(new_q); ws.copy_(new_s)
        expected = step()
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_input_contract():
    x, w, y = inputs(3, 128, 256)
    criterion = MXFP8FusedLinearCrossEntropy()
    with pytest.raises(ValueError, match="chunk_size"):
        MXFP8FusedLinearCrossEntropy(chunk_size=33)
    with pytest.raises(ValueError, match="BF16"):
        criterion(x.float(), y, w)
    with pytest.raises(ValueError, match="int64"):
        criterion(x, y.int(), w)
    with pytest.raises(ValueError, match="multiples of 64"):
        criterion(x, y, w[:255])


def test_saved_gradients_and_chunk_workspace_bound():
    x, w, y = inputs(513, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)
    saved = []
    def pack(t):
        saved.append((tuple(t.shape), t.dtype))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = criterion(x, y, w)
    assert saved == [((513, 128), torch.bfloat16), ((256, 128), torch.float32)]
    assert sorted(state.ce.m for state in criterion._states.values()) == [1, 128]
    # Clearing scratch must not invalidate the independently saved gradients.
    criterion.clear_workspace()
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
