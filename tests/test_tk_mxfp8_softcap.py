"""Softcapped LSE, chain rule before quantization, and balanced chunked CE."""

import pytest
import torch
import torch.nn.functional as F

from mxfp8._impl.ce import CEPlan, merge_lse
from mxfp8 import quantize_weight_rowwise
from tests.mxfp8_reference import rowwise_logits
from mxfp8._core import quantize_weight
from mxfp8._kernels.lse import Library as LSELibrary, CONFIGS
from mxfp8._kernels.gemm import Library as GemmLibrary
from mxfp8._kernels.wgrad import Library as WgradLibrary
from mxfp8._impl.fused_ce import MXFP8FusedLinearCrossEntropy, fused_linear_cross_entropy_loss
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary
from tests.test_tk_mxfp8_fused_ce import inputs
from tests.test_tk_mxfp8_balance import reference_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.mark.parametrize("config,m,h,v", [(0, 1, 64, 64), (8, 137, 128, 320),
                                          (70, 257, 128, 32000), (71, 129, 128, 320)])
@pytest.mark.parametrize("softcap", [.3, 30.])
def test_softcap_partial_lse_and_rebind(config, m, h, v, softcap):
    x, w, y = inputs(m, h, v)
    xq = SiluBwdQuantizeLibrary().plan_direct(x); xq()
    wq, ws = quantize_weight(w)
    args = (xq.row_fp8, wq, xq.row_scale, ws)
    plain = GemmLibrary().plan(*args, config=config)
    plan = LSELibrary().plan(*args, config=config, softcap=softcap)
    for _ in range(2):
        z, partial = plan()
        # Raw logits are retained, even when tanh saturates.
        torch.testing.assert_close(z, plain(), rtol=0, atol=0)
        capped = softcap * torch.tanh(z.float() / softcap)
        block = CONFIGS[config][1]
        expected = torch.stack([t.logsumexp(-1) for t in capped.split(block, -1)], -1)
        torch.testing.assert_close(partial, expected, atol=2e-6, rtol=2e-6)
        lse, loss = merge_lse(z, partial, y, softcap=softcap)
        torch.testing.assert_close(lse, capped.logsumexp(-1), atol=2e-6, rtol=2e-6)
        # Hardware tanh has absolute approximation error, amplified by s.
        torch.testing.assert_close(loss, F.cross_entropy(capped, y, reduction="none"),
                                   atol=2e-6 + softcap * 5e-7, rtol=2e-6)
        x.mul_(2); xq()
        plan.rebind(*args, out=torch.empty_like(z), partial_lse=torch.empty_like(partial))


@pytest.mark.parametrize("softcap", [.1, 2., 30.])
@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_softcap_ce_dual_quantization(softcap, ignore_index):
    m, v = 137, 320
    torch.manual_seed(119)
    z = (torch.randn(m, v, device="cuda") * 3).bfloat16()
    # Exactly zero and deeply saturated rows test both extremes of the derivative.
    z[0] = 0
    z[1] = 1000
    z[2] = -1000
    saved = z.clone()
    y = torch.arange(m, device="cuda") % v
    y[32:64] = ignore_index
    y[-1] = ignore_index
    row_weight = torch.linspace(.01, 1., m, device="cuda")
    raw = z.float().requires_grad_()
    capped = softcap * torch.tanh(raw / softcap)
    losses = F.cross_entropy(capped, y, reduction="none", ignore_index=ignore_index) * row_weight
    g, = torch.autograd.grad(losses.sum(), raw)
    reference = SiluBwdQuantizeLibrary().plan_direct(g.bfloat16(), transpose=True); reference()
    partial = torch.stack([t.logsumexp(-1) for t in capped.detach().split(128, -1)], -1)
    lse, loss = merge_lse(z, partial, y, softcap=softcap, row_weight=row_weight, ignore_index=ignore_index)
    torch.testing.assert_close(loss, losses, atol=2e-6, rtol=2e-6)
    plan = CEPlan(m, v, z.device)
    plan(z, lse, y, softcap=softcap, row_weight=row_weight, ignore_index=ignore_index)
    # Compare represented gradients, including scales. At tanh saturation the
    # approximation may round a tiny residual to zero, changing a whole group's
    # exponent even though its absolute gradient error is negligible.
    def dequant(q, scale):
        exponent = scale.view(torch.uint8).reshape(scale.shape[0], scale.shape[1], 4)
        exponent = exponent.permute(1, 0, 2).reshape(scale.shape[1], -1)
        exponent = exponent.repeat_interleave(32, 1)[:q.shape[0], :q.shape[1]].int() - 127
        return torch.ldexp(q.float(), exponent)
    for direction in ("row", "column"):
        actual = dequant(getattr(plan, direction + "_fp8"), getattr(plan, direction + "_scale"))
        expected = dequant(getattr(reference, direction + "_fp8"), getattr(reference, direction + "_scale"))
        assert ((actual - expected).norm() / expected.norm()).item() < .002
    for rows in (y == ignore_index, torch.arange(m, device="cuda") == 1,
                 torch.arange(m, device="cuda") == 2):
        assert torch.count_nonzero(plan.row_fp8.float()[rows]).item() == 0
        assert torch.count_nonzero(plan.column_fp8.float()[:, :m][:, rows]).item() == 0
    assert torch.count_nonzero(plan.column_fp8.float()[:, m:]).item() == 0
    torch.testing.assert_close(z, saved, rtol=0, atol=0)


def component_reference(x, w, y, softcap, weights, denominator):
    wq, ws = quantize_weight_rowwise(w.float())
    dx, dw = torch.empty_like(x), torch.zeros_like(w, dtype=torch.float32)
    losses = []
    for start in range(0, len(x), 128):
        end = min(start + 128, len(x))
        xq = SiluBwdQuantizeLibrary().plan_direct(x[start:end]); xq()
        z = rowwise_logits(xq.row_fp8, wq, xq.row_scale, ws)
        raw = z.float().requires_grad_()
        capped = softcap * torch.tanh(raw / softcap)
        loss = (F.cross_entropy(capped, y[start:end], reduction="none") * weights[start:end]).sum()
        g, = torch.autograd.grad(loss, raw)
        losses.append(loss.detach())
        gq = SiluBwdQuantizeLibrary().plan_direct(g.bfloat16(), transpose=True); gq()
        dx[start:end] = g.bfloat16() @ w.bfloat16()
        dw.add_(WgradLibrary().plan(gq.column_fp8, xq.column_fp8, gq.column_scale, xq.column_scale, config=12)())
    inv = denominator.clamp_min(1).float().reciprocal()
    return torch.stack(losses).sum() * inv, dx * inv, dw.to(w.dtype) * inv


@pytest.mark.parametrize("balance", [None, "token", "document"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softcap_chunked_balanced_gradients(balance, dtype):
    x, w, y = inputs(385, 128, 256)
    w = w.to(dtype)
    y[::7] = -100
    y[1:31] = -100
    boundaries = [0, 0, 1, 31, 320, 384, 385, 385]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    weights, counts = reference_weights(y, boundaries)
    if balance != "document":
        weights = torch.ones_like(weights)
    denominator = ((counts > 0).sum() if balance == "document" else
                   (y != -100).sum() if balance == "token" else y.new_tensor(1))
    reference = component_reference(x, w, y, .7, weights, denominator)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=.7)
    loss = criterion(x, y, w, cu_seqlens=cu)
    dx, dw = torch.autograd.grad(loss, (x, w), retain_graph=True)
    torch.testing.assert_close(loss, reference[0], atol=2e-4, rtol=2e-6)
    for actual, expected in zip((dx, dw), reference[1:]):
        assert ((actual.float() - expected.float()).norm() / expected.float().norm()).item() < .004
    # Changes to the criterion cannot alter gradients precomputed by this forward.
    criterion.softcap = 30.
    criterion(x, y, w, cu_seqlens=cu)
    scaled = torch.autograd.grad(loss * -.5, (x, w))
    for actual, expected in zip(scaled, (dx, dw)):
        torch.testing.assert_close(actual, expected * -.5, atol=0, rtol=0)
    functional = fused_linear_cross_entropy_loss(x, y, w, chunk_size=128, balance=balance, softcap=.7, cu_seqlens=cu)
    torch.testing.assert_close(functional, loss, atol=0, rtol=0)


@pytest.mark.parametrize("balance", [None, "token", "document"])
def test_softcap_graph_replay_and_all_ignored(balance):
    x, w, y = inputs(137, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight_rowwise(w.detach())
    cu = torch.tensor([0, 31, 137], device="cuda", dtype=torch.int32)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=1.3)
    def step():
        loss = criterion(x, y, w, cu_seqlens=cu, quantized_weight=(wq, ws))
        return loss, *torch.autograd.grad(loss, (x, w))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for ignored in (False, True, False):
        with torch.no_grad():
            x.mul_(1.5)
            y.copy_(torch.arange(len(y), device="cuda") % 256)
            y[::3] = -100
            if ignored:
                y.fill_(-100)
            cu[1] = 129
        expected = step()
        graph.replay()
        for actual, ref in zip(captured, expected):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
            if ignored:
                assert torch.count_nonzero(actual).item() == 0


def test_softcap_validation():
    for invalid in (0, -1., True, float('nan'), float('inf'), 1e100, 1e-100, "30", torch.tensor(30.)):
        with pytest.raises(ValueError, match="softcap"):
            MXFP8FusedLinearCrossEntropy(softcap=invalid)
    assert MXFP8FusedLinearCrossEntropy(softcap=30).softcap == 30.
