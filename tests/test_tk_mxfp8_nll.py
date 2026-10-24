"""Per-position evaluation NLL and optional auxiliary output during training."""

import pytest
import torch
import torch.nn.functional as F

import mxfp8._impl.fused_ce as ce_module
from mxfp8._core import quantize_weight
from mxfp8._impl.fused_ce import MXFP8FusedLinearCrossEntropy, fused_linear_cross_entropy_loss
from mxfp8._kernels.gemm import Library as GemmLibrary
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary
from tests.test_tk_mxfp8_fused_ce import inputs
from tests.test_tk_mxfp8_balance import reference_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def reference_nll(x, w, y, softcap, ignore_index):
    wq, ws = quantize_weight(w.float())
    losses = []
    for start in range(0, len(x), 128):
        xq = SiluBwdQuantizeLibrary().plan_direct(x[start:start + 128])
        xq()
        z = GemmLibrary().plan(xq.row_fp8, wq, xq.row_scale, ws, config=8)().float()
        if softcap is not None:
            z = softcap * torch.tanh(z / softcap)
        losses.append(F.cross_entropy(z, y[start:start + 128], reduction="none",
                                      ignore_index=ignore_index))
    return torch.cat(losses)


@pytest.mark.parametrize("balance", [None, "token", "document"])
@pytest.mark.parametrize("softcap", [None, .7])
@pytest.mark.parametrize("ignore_index", [-100, 0, 2**40])
def test_nll_matches_plain_gemm_and_unweighted_ce(balance, softcap, ignore_index):
    x, w, y = inputs(385, 128, 320)
    boundaries = [0, 0, 1, 31, 320, 384, 385, 385]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    y[1:31] = ignore_index
    y[::7] = ignore_index
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance,
                                            softcap=softcap, ignore_index=ignore_index)
    for all_ignored in (False, True):
        if all_ignored:
            y.fill_(ignore_index)
        expected = reference_nll(x, w, y, softcap, ignore_index)
        loss, nll = criterion(x, y, w, cu_seqlens=cu, return_nll=True)
        assert nll.shape == y.shape and nll.dtype == torch.float32
        assert not nll.requires_grad and nll.grad_fn is None
        torch.testing.assert_close(nll, expected, atol=3e-6, rtol=2e-6)
        weights, counts = reference_weights(y, boundaries, ignore_index)
        scalar = (expected * weights).sum() / (counts > 0).sum().clamp_min(1) if balance == "document" else (
            expected.sum() / (y != ignore_index).sum().clamp_min(1) if balance == "token" else expected.sum())
        torch.testing.assert_close(loss, scalar, atol=2e-4, rtol=2e-6)
        torch.testing.assert_close(criterion(x, y, w, cu_seqlens=cu), loss, atol=0, rtol=0)
        saved = nll.clone()
        criterion(x * 2, torch.ones_like(y), w, cu_seqlens=cu, return_nll=True)
        criterion.clear_workspace()
        torch.testing.assert_close(nll, saved, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode", "no_requires_grad"])
@pytest.mark.parametrize("return_nll", [False, True])
def test_evaluation_does_not_prepare_or_compute_gradients(monkeypatch, mode, return_nll):
    x, w, y = inputs(137, 128, 256)
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not prepare backward")
    for name in ("DgradLibrary", "WgradLibrary", "CEPlan"):
        monkeypatch.setattr(ce_module, name, forbidden)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128)
    if mode != "no_requires_grad":
        x.requires_grad_(); w.requires_grad_()
    context = torch.inference_mode if mode == "inference_mode" else torch.no_grad if mode == "no_grad" else torch.enable_grad
    with context():
        result = criterion(x, y, w, return_nll=return_nll)
    loss = result[0] if return_nll else result
    assert not loss.requires_grad
    assert criterion._dgrad is None and criterion._wgrad is None
    for state in criterion._states.values():
        assert state.ce is None and state.dx is None and state.dw is None
        assert state.xq.x is None and not state.gemm.refs


@pytest.mark.parametrize("balance", [None, "token", "document"])
def test_auxiliary_nll_preserves_scalar_backward_and_eval_train_switch(balance):
    x, w, y = inputs(137, 128, 256)
    x.requires_grad_(); w.requires_grad_()
    y[::3] = -100
    cu = torch.tensor([0, 0, 13, 137], device="cuda", dtype=torch.int64)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=.3)
    # Inference workspaces must not interfere with subsequent training or no_grad.
    with torch.inference_mode():
        eval_loss, eval_nll = criterion(x, y, w, cu_seqlens=cu, return_nll=True)
    criterion.eval()  # eval alone must still permit gradients.
    scalar = criterion(x, y, w, cu_seqlens=cu)
    expected_grads = torch.autograd.grad(scalar, (x, w))
    loss, nll = criterion(x, y, w, cu_seqlens=cu, return_nll=True)
    assert loss.requires_grad and not nll.requires_grad and nll.grad_fn is None
    torch.testing.assert_close(loss, eval_loss, atol=0, rtol=0)
    torch.testing.assert_close(nll, eval_nll, atol=0, rtol=0)
    nll.zero_()  # NLL is not a saved tensor and cannot affect scalar backward.
    with torch.no_grad():
        criterion(x * 2, y, w, cu_seqlens=cu, return_nll=True)
    criterion.clear_workspace()
    grads = torch.autograd.grad(loss * -.5, (x, w))
    for actual, expected in zip(grads, expected_grads):
        torch.testing.assert_close(actual, expected * -.5, atol=0, rtol=0)


@pytest.mark.parametrize("balance", [None, "document"])
def test_nll_evaluation_cuda_graph_replay(balance):
    x, w, y = inputs(137, 128, 320)
    x.requires_grad_(); w.requires_grad_()
    wq, ws = quantize_weight(w.detach())
    cu = torch.tensor([0, 0, 32, 137], device="cuda", dtype=torch.int32)
    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=128, balance=balance, softcap=.7)
    @torch.no_grad()
    def step():
        return criterion(x, y, w, cu_seqlens=cu, quantized_weight=(wq, ws), return_nll=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for all_ignored in (True, False):
        with torch.no_grad():
            x.add_(.25)
            w.add_(.01)
            q, s = quantize_weight(w.detach())
            wq.copy_(q); ws.copy_(s)
            y.copy_(torch.arange(len(y), device=y.device) % w.shape[0])
            y[::3] = -100
            if all_ignored:
                y.fill_(-100)
            cu[2] = 130
        expected = step()
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_functional_nll_and_validation():
    x, w, y = inputs(1, 128, 320)
    loss, nll = fused_linear_cross_entropy_loss(x, y, w, return_nll=True, softcap=.5)
    torch.testing.assert_close(loss, nll[0], atol=0, rtol=0)
    assert isinstance(fused_linear_cross_entropy_loss(x, y, w), torch.Tensor)
    with pytest.raises(ValueError, match="return_nll"):
        fused_linear_cross_entropy_loss(x, y, w, return_nll=1)
