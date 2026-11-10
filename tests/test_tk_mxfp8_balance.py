"""Balanced losses, varlen metadata, and their quantized gradients."""

import pytest
import torch
import torch.nn.functional as F

from mxfp8._impl.balance import document_weights, reduce_balanced_loss
from mxfp8 import quantize_weight_rowwise
from tests.mxfp8_reference import rowwise_logits
from mxfp8._impl.fused_ce import MXFP8FusedLinearCrossEntropy, fused_linear_cross_entropy_loss
from mxfp8._kernels.wgrad import Library as WgradLibrary
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary
from tests.test_tk_mxfp8_fused_ce import inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def reference_weights(y, boundaries, ignore_index=-100):
    weights = torch.zeros_like(y, dtype=torch.float32)
    counts = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        valid = y[begin:end] != ignore_index
        count = valid.sum()
        counts.append(count)
        weights[begin:end] = valid.float() / count.clamp_min(1)
    return weights, torch.stack(counts)


@pytest.mark.parametrize("ignore_index", [-100, 0, 2**40])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_document_metadata_long_and_empty_segments(ignore_index, all_ignored):
    boundaries = [0, 0, 1, 32, 65, 1090, 10091, 10091]
    y = torch.arange(boundaries[-1], device="cuda") % 64
    if all_ignored:
        y.fill_(ignore_index)
    else:
        y[::5] = ignore_index
        y[32:65] = ignore_index
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int64)
    weights, counts = document_weights(y, cu, ignore_index=ignore_index)
    rw, rc = reference_weights(y, boundaries, ignore_index)
    torch.testing.assert_close(weights, rw, atol=0, rtol=2e-7)
    torch.testing.assert_close(counts, rc, atol=0, rtol=0)
    losses = torch.linspace(0., 11., len(y), device="cuda")
    losses[y == ignore_index] = 0
    for mode in ("token", "document"):
        weighted = losses if mode == "token" else losses * weights
        total = (y != ignore_index).sum() if mode == "token" else (counts > 0).sum()
        out, inv = reduce_balanced_loss(weighted, y, balance=mode, doc_counts=counts, ignore_index=ignore_index)
        torch.testing.assert_close(out, weighted.sum() / total.clamp_min(1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(inv, total.clamp_min(1).float().reciprocal(), atol=0, rtol=2e-7)


def test_more_documents_than_tokens_and_multistage_document_count():
    y = torch.arange(65, device="cuda")
    # Most documents are empty; D > 8192 exercises the multi-stage denominator.
    cu = torch.cat((torch.zeros(9000, device="cuda", dtype=torch.int32),
                    torch.tensor([31, 65], device="cuda", dtype=torch.int32)))
    weights, counts = document_weights(y, cu)
    out, inv = reduce_balanced_loss(weights * 3, y, balance="document", doc_counts=counts)
    torch.testing.assert_close(out, torch.tensor(3., device="cuda"))
    assert inv.item() == .5
    assert counts.sum().item() == 65 and (counts > 0).sum().item() == 2


def component_reference(x, w, y, weights, denominator, chunk_size):
    """Plain GEMM + PyTorch weighted CE + existing CUDA dual quantizer."""
    wq, ws = quantize_weight_rowwise(w.float())
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w, dtype=torch.float32)
    losses = []
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        xq = SiluBwdQuantizeLibrary().plan_direct(x[start:end]); xq()
        z = rowwise_logits(xq.row_fp8, wq, xq.row_scale, ws)
        labels = y[start:end]
        valid = labels != -100
        row_weight = weights[start:end]
        losses.append((F.cross_entropy(z.float(), labels, reduction="none") * row_weight).sum())
        g = (z.float() - z.float().logsumexp(-1)[:, None]).exp()
        g[torch.arange(end-start, device=x.device)[valid], labels[valid]] -= 1
        g[~valid] = 0
        g = (g * row_weight[:, None]).bfloat16()
        gq = SiluBwdQuantizeLibrary().plan_direct(g, transpose=True); gq()
        dx[start:end] = g.bfloat16() @ w.bfloat16()
        dw.add_(WgradLibrary().plan(gq.column_fp8, xq.column_fp8, gq.column_scale, xq.column_scale, config=12)())
    inv = denominator.clamp_min(1).float().reciprocal()
    return torch.stack(losses).sum() * inv, dx * inv, dw.to(w.dtype) * inv


@pytest.mark.parametrize("mode", ["token", "document"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_balanced_chunked_loss_and_gradients(mode, dtype, all_ignored):
    x, w, y = inputs(385, 128, 256)
    w = w.to(dtype)
    # A long document crosses both 128-token chunk and 32-token quantization boundaries.
    boundaries = [0, 0, 1, 31, 320, 384, 385, 385]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    y[1:31] = -100
    y[::7] = -100
    if all_ignored:
        y.fill_(-100)
    doc_weight, counts = reference_weights(y, boundaries)
    weights = torch.ones_like(doc_weight) if mode == "token" else doc_weight
    denominator = (y != -100).sum() if mode == "token" else (counts > 0).sum()
    reference = component_reference(x, w, y, weights, denominator, 128)
    x.requires_grad_(); w.requires_grad_()
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=mode)
    loss = criterion(x, y, w, cu_seqlens=cu)
    dx, dw = torch.autograd.grad(loss, (x, w), retain_graph=True)
    torch.testing.assert_close(loss, reference[0], atol=2e-6, rtol=2e-6)
    for actual, expected in zip((dx, dw), reference[1:]):
        if all_ignored:
            assert torch.count_nonzero(actual).item() == 0
        else:
            assert ((actual.float()-expected.float()).norm()/expected.float().norm()).item() < .004
    assert torch.count_nonzero(dx[y == -100]).item() == 0
    if all_ignored:
        assert loss.item() == 0
    # Saved normalizer must survive workspace reuse with a different valid count.
    criterion(x, torch.zeros_like(y), w, cu_seqlens=cu)
    scaled = torch.autograd.grad(loss * -.5, (x, w))
    for actual, expected in zip(scaled, (dx, dw)):
        torch.testing.assert_close(actual, expected * -.5, atol=0, rtol=0)
    one_shot = fused_linear_cross_entropy_loss(x, y, w, chunk_size=128, balance=mode, cu_seqlens=cu)
    torch.testing.assert_close(one_shot, loss, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["token", "document"])
def test_balanced_bf16_accuracy(mode):
    x, w, y = inputs(513, 128, 512)
    boundaries = [0, 1, 3, 34, 91, 513]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int64)
    y[::11] = -100
    row_weight, counts = reference_weights(y, boundaries)
    if mode == "token":
        row_weight = (y != -100).float()
        denominator = row_weight.sum()
    else:
        denominator = (counts > 0).sum()
    w = w.bfloat16().requires_grad_(); x.requires_grad_()
    ref = (F.cross_entropy(F.linear(x, w).float(), y, reduction="none") * row_weight).sum() / denominator
    rx, rw = torch.autograd.grad(ref, (x, w))
    actual = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=mode)(x, y, w, cu_seqlens=cu)
    dx, dw = torch.autograd.grad(actual, (x, w))
    assert abs(actual.item() - ref.item()) < .015
    for a, b in ((dx, rx), (dw, rw)):
        assert ((a.float()-b.float()).norm()/b.float().norm()).item() < .10


@pytest.mark.parametrize("mode", ["token", "document"])
def test_balance_graph_replay_with_changed_boundaries(mode):
    x, w, y = inputs(257, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight_rowwise(w.detach())
    cu = torch.tensor([0, 1, 31, 257], device="cuda", dtype=torch.int32)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=mode)
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
    for boundaries, all_ignored in (([0, 0, 129, 257], False), ([0, 7, 256, 257], True), ([0, 33, 33, 257], False)):
        with torch.no_grad():
            cu.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
            y.copy_(torch.arange(len(y), device="cuda") % 256)
            y[::3] = -100
            if all_ignored:
                y.fill_(-100)
            x.mul_(.9)
        expected = step()
        graph.replay()
        for actual, ref in zip(captured, expected):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
            if all_ignored:
                assert torch.count_nonzero(actual).item() == 0


def test_balance_api_validation():
    x, w, y = inputs(3, 128, 256)
    with pytest.raises(ValueError, match="balance"):
        MXFP8FusedLinearCrossEntropy(balance="invalid")
    criterion = MXFP8FusedLinearCrossEntropy(balance="document")
    bad = [None, torch.tensor([0, 3]), torch.tensor([0., 3.], device="cuda"),
           torch.tensor([0], device="cuda", dtype=torch.int32),
           torch.zeros((2, 2), device="cuda", dtype=torch.int32)]
    for cu in bad:
        with pytest.raises(ValueError, match="cu_seqlens"):
            criterion(x, y, w, cu_seqlens=cu)
