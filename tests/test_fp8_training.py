"""K128 FP32-scale training against independently dequantized Torch references."""

import gc

import pytest
import torch
import torch.nn.functional as F

from mxfp8 import FP8GemmPlan, FP8Linear, FP8SwiGLU, FusedAdamWMXFP8, bind_optimizer
from mxfp8._kernels.fp8_gemm import _FP8DualPlan
from mxfp8._kernels.fp8_quantize import BlockQuantized, DualQuantized

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def row_quant(x):
    m, k = x.shape
    v = x.float().reshape(m, k // 128, 128)
    maximum = v.abs().amax(-1)
    s = torch.where(maximum == 0, 1., (maximum / 448).clamp_min(torch.finfo(torch.float32).tiny))
    q = (v / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(m, k)
    return q, s


def block_quant(w):
    n, k = w.shape
    v = w.float().reshape(n // 128, 128, k // 128, 128)
    maximum = v.abs().amax((1, 3))
    s = torch.where(maximum == 0, 1., (maximum / 448).clamp_min(torch.finfo(torch.float32).tiny))
    q = (v / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(n, k)
    return q, s


def dequant(q, s, block=False):
    if block:
        s = s.repeat_interleave(128, 0)
    return q.double() * s.double().repeat_interleave(128, 1)


def row_value(x):
    return dequant(*row_quant(x))


def col_value(x):
    xp = F.pad(x.float(), (0, 0, 0, (-x.shape[0]) % 128))
    return row_value(xp.T.contiguous())


def relative(actual, expected, limit=.01):
    error = (actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-20)
    assert error.item() < limit, error.item()


@pytest.mark.parametrize("m,k", [(1, 128), (17, 256), (128, 128), (133, 384), (257, 256)])
def test_dual_quantization(m, k):
    x = rand(m, k)
    x[:, :128] = 0
    q = DualQuantized(m, k, x.device).quantize(x)
    qr, sr = row_quant(x)
    xc = F.pad(x, (0, 0, 0, (-m) % 128)).T.contiguous()
    qc, sc = row_quant(xc)
    torch.testing.assert_close(q.row.float(), qr.float(), rtol=0, atol=0)
    torch.testing.assert_close(q.col.float(), qc.float(), rtol=0, atol=0)
    torch.testing.assert_close(q.row_scale, sr)
    torch.testing.assert_close(q.col_scale, sc)


@pytest.mark.parametrize("n,k", [(128, 128), (256, 384)])
def test_weight_quantization_and_transpose(n, k):
    w = rand(n, k, dtype=torch.float32)
    w[:128, :128] = 0
    cache = BlockQuantized(n, k, w.device).quantize(w)
    q, s = block_quant(w)
    torch.testing.assert_close(cache.row.float(), q.float(), atol=0, rtol=0)
    torch.testing.assert_close(cache.row_scale, s)
    torch.testing.assert_close(cache.col.float(), cache.row.T.float(), atol=0, rtol=0)
    torch.testing.assert_close(cache.col_scale, cache.row_scale.T, atol=0, rtol=0)


@pytest.mark.parametrize("m,k,n", [(1, 128, 128), (133, 256, 384), (257, 384, 256),
                                    (257, 1536, 1536)])
def test_wgrad_rowwise_scales(m, k, n):
    a = DualQuantized(m, n, "cuda").quantize(rand(m, n))
    b = DualQuantized(m, k, "cuda").quantize(rand(m, k))
    plan = FP8GemmPlan(a.col, b.col, a.col_scale, b.col_scale,
                       dtype=torch.float32, b_scale_mode="row")
    reference = dequant(a.col, a.col_scale) @ dequant(b.col, b.col_scale).T
    torch.testing.assert_close(plan().double(), reference, rtol=3e-5, atol=3e-4)
    assert plan.info["local_bytes"] == 0


@pytest.mark.parametrize("m,k,n", [(1, 128, 128), (133, 256, 384), (257, 384, 256),
                                    (1025, 1536, 1536)])
def test_fused_forward_and_dx(m, k, n):
    x = DualQuantized(m, k, "cuda").quantize(rand(m, k))
    wg = BlockQuantized(n, k, "cuda").quantize(rand(n, k, dtype=torch.float32) * .02)
    wu = BlockQuantized(n, k, "cuda").quantize(rand(n, k, dtype=torch.float32) * .03)
    gate, up, out = [torch.empty((m, n), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    fwd = _FP8DualPlan(x.row, wg.row, x.row_scale, wg.row_scale,
                      a2=x.row, b2=wu.row, sa2=x.row_scale, sb2=wu.row_scale,
                      mode=1, out=out, gate=gate, up=up)
    fwd()
    xr = dequant(x.row, x.row_scale)
    gr = (xr @ dequant(wg.row, wg.row_scale, True).T).bfloat16()
    ur = (xr @ dequant(wu.row, wu.row_scale, True).T).bfloat16()
    torch.testing.assert_close(gate, gr, rtol=.008, atol=.0005)
    torch.testing.assert_close(up, ur, rtol=.008, atol=.0005)
    # The activation is evaluated from the saved BF16 projections.
    torch.testing.assert_close(out, (F.silu(gate.float()) * up.float()).bfloat16(), rtol=.008, atol=1e-5)
    dg = DualQuantized(m, n, "cuda").quantize(rand(m, n))
    du = DualQuantized(m, n, "cuda").quantize(rand(m, n))
    dx = _FP8DualPlan(dg.row, wg.col, dg.row_scale, wg.col_scale,
                     a2=du.row, b2=wu.col, sa2=du.row_scale, sb2=wu.col_scale,
                     mode=2, out=torch.empty((m, k), device="cuda", dtype=torch.bfloat16))
    ref = (dequant(dg.row, dg.row_scale) @ dequant(wg.row, wg.row_scale, True)
           + dequant(du.row, du.row_scale) @ dequant(wu.row, wu.row_scale, True))
    torch.testing.assert_close(dx().double(), ref.bfloat16().double(), rtol=.008, atol=.0005)
    assert fwd.info["local_bytes"] == dx.info["local_bytes"] == 0


def ffn_reference(x, wg, wu, wd, dy):
    qg = dequant(*block_quant(wg), block=True)
    qu = dequant(*block_quant(wu), block=True)
    qd = dequant(*block_quant(wd), block=True)
    xr = row_value(x)
    gate, up = (xr @ qg.T).bfloat16(), (xr @ qu.T).bfloat16()
    h = (F.silu(gate.float()) * up.float()).bfloat16()
    out = (row_value(h) @ qd.T).bfloat16()
    dh = (row_value(dy) @ qd).bfloat16()
    g, u = gate.float(), up.float()
    sig = torch.sigmoid(g)
    dg = (dh.float() * u * sig * (1 + g * (1 - sig))).bfloat16()
    du = (dh.float() * g * sig).bfloat16()
    dx = (row_value(dg) @ qg + row_value(du) @ qu).bfloat16()
    return out, dx, col_value(dg) @ col_value(x).T, col_value(du) @ col_value(x).T, col_value(dy) @ col_value(h).T


@pytest.mark.parametrize("m,k,h,n", [(1, 128, 128, 128), (129, 256, 384, 128), (257, 128, 256, 256)])
def test_ffn_forward_backward_reference(m, k, h, n):
    torch.manual_seed(123)
    layer = FP8SwiGLU(k, h, n)
    x = rand(m, k).requires_grad_()
    dy = rand(m, n)
    ref = ffn_reference(x.detach(), layer.gate_weight.detach(), layer.up_weight.detach(),
                        layer.down_proj.weight.detach(), dy)
    y = layer(x)
    y.backward(dy)
    actual = (y, x.grad, layer.gate_weight.grad, layer.up_weight.grad, layer.down_proj.weight.grad)
    for a, b in zip(actual, ref):
        relative(a, b, .012)
    assert x.grad.dtype == y.dtype == torch.bfloat16
    assert all(p.grad.dtype == torch.float32 for p in layer.parameters())


@pytest.mark.parametrize("kind", [FP8Linear, FP8SwiGLU])
def test_multiple_live_forwards_and_retained_graph(kind):
    args = (128, 256) if kind is FP8Linear else (128, 256, 128)
    a, b = kind(*args), kind(*args)
    b.load_state_dict(a.state_dict())
    x1, x2 = rand(17, 128).requires_grad_(), rand(17, 128).requires_grad_()
    r1, r2 = x1.detach().clone().requires_grad_(), x2.detach().clone().requires_grad_()
    y1, y2 = a(x1), a(x2)
    y1.sum().backward(retain_graph=True)
    y2.sum().backward()
    # A repeated backward on the retained first forward must preserve its inputs.
    y1.sum().backward()
    b(r1).sum().backward()
    b(r2).sum().backward()
    b(r1).sum().backward()
    torch.testing.assert_close(x1.grad, r1.grad, atol=0, rtol=0)
    torch.testing.assert_close(x2.grad, r2.grad, atol=0, rtol=0)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    old = y1.detach().clone()
    with torch.no_grad():
        for _ in range(3):
            a(x2)
    torch.testing.assert_close(y1, old, atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_optimizer_cache_refresh_and_training(fused):
    torch.manual_seed(1)
    layer = FP8SwiGLU(128, 256, 128)
    optimizer = (FusedAdamWMXFP8(layer.parameters(), lr=.001) if fused
                 else torch.optim.SGD(layer.parameters(), lr=.5))
    hook = bind_optimizer(layer, optimizer)
    x = rand(33, 128)
    pointers = [q.row.data_ptr() for q in layer._quantized]
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = layer(x).float().square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        for module in (layer, layer.down_proj):
            for w, q in zip(module._weights(), module._quantized):
                qr, sr = block_quant(w)
                torch.testing.assert_close(q.row.float(), qr.float(), atol=0, rtol=0)
                torch.testing.assert_close(q.row_scale, sr)
    assert losses[-1] < losses[0]
    assert pointers == [q.row.data_ptr() for q in layer._quantized]
    hook.remove()


def test_cuda_graph_training_replay():
    layer = FP8SwiGLU(128, 256, 128)
    x = rand(19, 128).requires_grad_()
    optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
    hook = bind_optimizer(layer, optimizer)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = layer(x)
        y.float().square().mean().backward()
        optimizer.step()
        return y

    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
        gc.collect()
        reference = FP8SwiGLU(128, 256, 128)
        reference.load_state_dict(layer.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = step()
    torch.cuda.current_stream().wait_stream(stream)
    before = layer.gate_weight.detach().clone()
    for _ in range(3):
        x.detach().copy_(rand(19, 128))
        graph.replay()
        reference_optimizer.zero_grad(set_to_none=True)
        reference(x.detach()).float().square().mean().backward()
        reference_optimizer.step()
    torch.cuda.synchronize()
    for p, q in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert torch.isfinite(y).all()
    assert not torch.equal(before, layer.gate_weight)
    for module in (layer, layer.down_proj):
        for w, q in zip(module._weights(), module._quantized):
            qr, sr = block_quant(w)
            torch.testing.assert_close(q.row.float(), qr.float(), atol=0, rtol=0)
            torch.testing.assert_close(q.row_scale, sr)
    hook.remove()


def test_parameter_version_check():
    layer = FP8SwiGLU(128, 128, 128)
    y = layer(rand(17, 128))
    with torch.no_grad():
        layer.gate_weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


def test_validation():
    with pytest.raises(ValueError, match="multiples of 128"):
        FP8SwiGLU(128, 192, 128)
    layer = FP8Linear(128, 128)
    with pytest.raises(ValueError, match="BF16"):
        layer(rand(8, 128, dtype=torch.float32))


@pytest.mark.parametrize("train_x,train_weight", [(True, False), (False, True)])
def test_linear_partial_gradients(train_x, train_weight):
    layer = FP8Linear(128, 256)
    layer.weight.requires_grad_(train_weight)
    x = rand(7, 128).requires_grad_(train_x)
    dy = rand(7, 256)
    y = layer(x)
    y.backward(dy)
    assert (x.grad is not None) == train_x
    assert (layer.weight.grad is not None) == train_weight
    if train_x:
        w = dequant(*block_quant(layer.weight), block=True)
        torch.testing.assert_close(x.grad, (row_value(dy) @ w).bfloat16(), rtol=.008, atol=.001)
    if train_weight:
        torch.testing.assert_close(layer.weight.grad.double(), col_value(dy) @ col_value(x).T,
                                   rtol=3e-5, atol=3e-4)
