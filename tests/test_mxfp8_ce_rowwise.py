"""Rowwise LM-head weights and the BF16-dX/MXFP8-dW contract."""

import pytest
import torch
import torch.nn.functional as F

from mxfp8 import MXFP8FusedLinearCrossEntropy, quantize_weight_rowwise
from mxfp8._core import quantize_weight
from mxfp8._impl.ce import CEPlan, merge_lse
from mxfp8._kernels.lse import CONFIGS, Library as LSELibrary
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary
from tests.mxfp8_reference import dequant_rows, rowwise_logits


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.mark.parametrize("shape", [(35, 77), (64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_quantization_reference_and_embedding_independence(shape, dtype):
    torch.manual_seed(193)
    n, k = shape
    w = torch.randn(shape, device="cuda")
    w *= torch.exp2((torch.arange(n, device="cuda") % 21 - 10).float())[:, None]
    w[0] = 0
    w[1, :32] = 1
    w[2, :32] = 1024
    w = w.to(dtype)
    q, scales = quantize_weight_rowwise(w)
    assert q.shape == w.shape and scales.shape == (n, (k + 31) // 32)
    assert q.dtype == torch.float8_e4m3fn and scales.dtype == torch.uint8
    rounded = w.bfloat16().float()
    padded = F.pad(rounded, (0, (-k) % 32)).reshape(n, -1, 32)
    maximum = padded.abs().amax(-1)
    _, exponent = torch.frexp(maximum)
    expected_scales = torch.where(maximum == 0, 0, (exponent + 119).clamp_min(0))
    expected_q = torch.ldexp(
        rounded, (127 - expected_scales).repeat_interleave(32, 1)[:, :k]
    ).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales.int(), expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    assert scales[1, 0].item() == 120 and scales[2, 0].item() == 130
    # A large outlier in one embedding cannot change any other embedding.
    w[2] *= 1024
    changed_q, changed_scales = quantize_weight_rowwise(w)
    keep = torch.arange(n, device="cuda") != 2
    torch.testing.assert_close(changed_scales[keep], scales[keep], rtol=0, atol=0)
    torch.testing.assert_close(
        changed_q.view(torch.uint8)[keep], q.view(torch.uint8)[keep], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "config,m,h,v",
    [
        (0, 17, 64, 64),
        (8, 137, 192, 320),
        (70, 33, 384, 320),
        (71, 129, 128, 320),
    ],
)
@pytest.mark.parametrize("softcap", [None, 0.7])
def test_rowwise_lse_reference_and_rebind(config, m, h, v, softcap):
    torch.manual_seed(781)
    x = torch.randn((m, h), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((v, h), device="cuda") * 0.03
    w *= torch.exp2((torch.arange(v, device="cuda") % 9 - 4).float())[:, None]
    w *= torch.exp2((torch.arange(h, device="cuda") // 32 % 5 - 2).float())[None, :]
    xq = SiluBwdQuantizeLibrary().plan_direct(x)
    xq()
    wq, ws = quantize_weight_rowwise(w)
    plan = LSELibrary().plan(
        xq.row_fp8,
        wq,
        xq.row_scale,
        ws,
        config=config,
        softcap=softcap,
        rowwise_weight=True,
    )
    for _ in range(2):
        z, partial = plan()
        expected_z = rowwise_logits(xq.row_fp8, wq, xq.row_scale, ws)
        torch.testing.assert_close(z, expected_z, rtol=0, atol=0)
        capped = (
            z.float() if softcap is None else softcap * torch.tanh(z.float() / softcap)
        )
        expected_lse = torch.stack(
            [t.logsumexp(-1) for t in capped.split(CONFIGS[config][1], -1)], -1
        )
        torch.testing.assert_close(partial, expected_lse, atol=2e-6, rtol=2e-6)
        # New pointers, new scales, and new output buffers exercise rebind.
        wq, ws = quantize_weight_rowwise(w * 1.7)
        plan.rebind(
            xq.row_fp8,
            wq,
            xq.row_scale,
            ws,
            out=torch.empty_like(z),
            partial_lse=torch.empty_like(partial),
        )
    with pytest.raises(ValueError):
        plan.rebind(xq.row_fp8, wq, xq.row_scale, ws[::32].contiguous())


@pytest.mark.parametrize("m,v", [(1, 64), (137, 320)])
@pytest.mark.parametrize("softcap", [None, 0.3, 30.0])
def test_ce_bf16_dz_preserves_column_quantization(m, v, softcap):
    torch.manual_seed(231)
    z = (torch.randn((m, v), device="cuda") * 3).bfloat16()
    original = z.clone()
    y = torch.arange(m, device="cuda") % v
    weight = torch.linspace(0.01, 1.0, m, device="cuda")
    mixed = CEPlan(m, v, z.device, bf16_dgrad=True)
    dual = CEPlan(m, v, z.device)
    assert mixed.row_fp8 is None and mixed.row_scale is None
    for ignored in (False, True, False):
        labels = y.clone()
        labels[::3] = -100
        if ignored:
            labels.fill_(-100)
        raw = z.float().requires_grad_()
        capped = raw if softcap is None else softcap * torch.tanh(raw / softcap)
        losses = F.cross_entropy(capped, labels, reduction="none") * weight
        (expected,) = torch.autograd.grad(losses.sum(), raw)
        partial = capped.detach().logsumexp(-1)[:, None].contiguous()
        lse, _ = merge_lse(z, partial, labels, softcap=softcap)
        for plan in (mixed, dual):
            plan(z, lse, labels, row_weight=weight, softcap=softcap)
        torch.testing.assert_close(
            mixed.row_bf16.float(), expected.bfloat16().float(), rtol=0.01, atol=2e-6
        )
        for name in ("column_fp8", "column_scale"):
            torch.testing.assert_close(
                getattr(mixed, name).view(torch.uint8),
                getattr(dual, name).view(torch.uint8),
                rtol=0,
                atol=0,
            )
        assert torch.count_nonzero(mixed.row_bf16[labels == -100]).item() == 0
        assert torch.count_nonzero(mixed.column_fp8.float()[:, m:]).item() == 0
    torch.testing.assert_close(z, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dx_uses_original_bf16_weight_and_cache_contract(dtype):
    torch.manual_seed(761)
    x = torch.randn((33, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn((256, 128), device="cuda") * 0.05).to(dtype).requires_grad_()
    y = torch.arange(len(x), device="cuda")
    wq, ws = quantize_weight_rowwise(w)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)
    # Exercise autocast too: BF16 dX must not become an FP16 GEMM.
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(x, y, w, quantized_weight=(wq, ws))
    (dx,) = torch.autograd.grad(loss, x)
    (state,) = criterion._states.values()
    assert state.ce.inplace
    assert state.ce.row_bf16 is state.gemm.out
    expected = state.ce.row_bf16 @ w.detach().bfloat16()
    torch.testing.assert_close(dx, expected, atol=0, rtol=0)
    quantized_dx = state.ce.row_bf16 @ dequant_rows(wq, ws).bfloat16()
    assert not torch.equal(dx, quantized_dx)
    assert state.ce.row_bf16.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="rowwise quantized weight cache"):
        criterion(x, y, w, quantized_weight=quantize_weight(w.detach().float()))


@pytest.mark.parametrize("m,v", [(1, 64), (137, 77), (137, 320), (33, 32000)])
@pytest.mark.parametrize("softcap", [None, 0.3, 30.0])
def test_inplace_dz_matches_separate_buffer_on_reuse(m, v, softcap):
    torch.manual_seed(312)
    separate = CEPlan(m, v, "cuda", bf16_dgrad=True)
    inplace = CEPlan(m, v, "cuda", bf16_dgrad=True, inplace=True)
    assert inplace.row_bf16 is None  # Construction allocates no BF16 row buffer.
    assert inplace.row_fp8 is None and inplace.row_scale is None
    logits = torch.empty((m, v), device="cuda", dtype=torch.bfloat16)
    for mode in ("active", "mixed", "ignored", "active"):
        # Reuse storage, then rebind to new storage; each launch must overwrite
        # even ignored rows, and must use raw logits for softcap's derivative.
        if mode == "ignored":
            previous_logits = logits
            previous_dz = logits.clone()
            logits = torch.empty_like(logits)
        logits.normal_(0, 3)
        if m > 1:
            logits[1] = 1000
        labels = torch.arange(m, device="cuda") % v
        if mode == "mixed":
            labels[::3] = -100
        elif mode == "ignored":
            labels.fill_(-100)
        weights = torch.linspace(0.01, 1, m, device="cuda")
        saved = logits.clone()
        capped = (
            logits.float()
            if softcap is None
            else softcap * torch.tanh(logits.float() / softcap)
        )
        partial = capped.logsumexp(-1)[:, None].contiguous()
        nll = torch.empty(m, device="cuda", dtype=torch.float32)
        lse, loss = merge_lse(
            logits, partial, labels, nll=nll, row_weight=weights, softcap=softcap
        )
        saved_loss, saved_lse, saved_nll = loss.clone(), lse.clone(), nll.clone()
        expected, _, cq, cs = separate(
            saved, lse, labels, row_weight=weights, softcap=softcap
        )
        actual, _, iq, ins = inplace(
            logits, lse, labels, row_weight=weights, softcap=softcap
        )
        assert actual is logits and inplace.row_bf16 is logits
        for a, b in (
            (actual, expected),
            (iq.view(torch.uint8), cq.view(torch.uint8)),
            (ins, cs),
            (loss, saved_loss),
            (lse, saved_lse),
            (nll, saved_nll),
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if mode == "ignored":
            torch.testing.assert_close(previous_logits, previous_dz, rtol=0, atol=0)
            assert torch.count_nonzero(logits).item() == 0


def test_inplace_requires_bf16_dgrad():
    with pytest.raises(ValueError, match="inplace requires"):
        CEPlan(1, 64, "cuda", inplace=True)


def test_graph_requantizes_uncached_weight_for_forward_and_bf16_dx():
    torch.manual_seed(908)
    x = torch.randn((33, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn((256, 128), device="cuda") * 0.05).requires_grad_()
    y = torch.arange(len(x), device="cuda")
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)

    def step():
        loss = criterion(x, y, w)
        grads = torch.autograd.grad(loss, (x, w))
        # This replay test only needs values, not a retained autograd graph.
        return loss.detach(), *grads

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        with torch.no_grad():
            w.mul_(1.25)
            x.add_(0.25)
            y.add_(1)
        expected = step()
        graph.replay()
        for actual, ref in zip(captured, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
