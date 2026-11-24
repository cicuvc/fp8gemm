"""Stream-K coverage, FP64 oracle, mutable graphs and autograd integration."""
import pytest
import torch

from mxfp8 import MXFP8SwiGLU, configure_ffn_streamk, bind_optimizer
from mxfp8._kernels.gemm import pack_a_scales
from mxfp8._kernels.streamk import Library
from mxfp8._kernels.streamk_wgrad import Library as Wgrad
from mxfp8._kernels.streamk_schedule import schedule

GPU = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
                        reason='SM120 required')


def fp8(m, n):
    return (torch.randn(m, n, device='cuda') * .125).to(torch.float8_e4m3fn)


def packed(m, k):
    return pack_a_scales(torch.randint(124, 130, (m, k//32), device='cuda', dtype=torch.uint8))


def blocks(n, k):
    return torch.randint(124, 130, ((n+31)//32, k//32), device='cuda', dtype=torch.uint8)


def row_value(a, s):
    m, k = a.shape
    e = s.view(torch.uint8).reshape((k+127)//128, -1, 4).permute(1, 0, 2).reshape(-1, (k+127)//128*4)
    return a.double() * torch.exp2(e[:m, :k//32].double()-127).repeat_interleave(32, 1)


def block_value(a, s):
    return a.double() * torch.exp2(s.double()-127).repeat_interleave(32, 0).repeat_interleave(32, 1)[:a.shape[0], :a.shape[1]]


def operands(f, m, n, k):
    if f == 'gemm': return [fp8(m,k), fp8(n,k), packed(m,k), blocks(n,k)]
    if f == 'dgrad': return [fp8(m,k), fp8(k,n), packed(m,k), blocks(k,n)]
    if f == 'swiglu': return [fp8(m,k), packed(m,k), fp8(n,k), blocks(n,k), fp8(n,k), blocks(n,k)]
    if f == 'wgrad': return [fp8(m,k), fp8(k,n), packed(m,k), packed(n,k)]
    return [fp8(m,k), fp8(m,k), fp8(k,n), fp8(k,n), packed(m,k), packed(m,k), blocks(k,n), blocks(k,n)]


def oracle(f, x):
    if f == 'gemm': return [row_value(x[0],x[2]) @ block_value(x[1],x[3]).T]
    if f == 'dgrad': return [row_value(x[0],x[2]) @ block_value(x[1],x[3])]
    if f == 'wgrad': return [row_value(x[0],x[2]) @ row_value(x[1].T.contiguous(),x[3]).T]
    if f == 'swiglu':
        a = row_value(x[0],x[1])
        g = (a*.5) @ block_value(x[2],x[3]).T
        u = a @ block_value(x[4],x[5]).T
        return [g*(1+g.tanh())*u, g, u]
    return [row_value(x[0],x[4]) @ block_value(x[2],x[6]) + row_value(x[1],x[5]) @ block_value(x[3],x[7])]


def check(actual, expected, tol=.003):
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        error = (a.double()-b.double()).norm() / b.double().norm().clamp_min(1e-30)
        assert error.item() <= tol, error.item()


@pytest.mark.parametrize('cols,block', [(64,64),(64,128),(128,64),(128,128)])
def test_schedule(cols, block):
    for n in (1,128,129,1024):
        for k in (128,384,1024):
            for m in (128,640,8192):
                for sms in (1,2,7,170):
                    for waves in (0,1,2):
                        s = schedule(n,k,m,sms,tail_waves=waves,tile_cols=cols,block_k=block)
                        assert {(w[0],w[1]) for w in s['work']} == {(r,c) for r in range(0,n,128) for c in range(0,k,cols)}
                        assert max(s['info']['cta_steps'])-min(s['info']['cta_steps']) <= 1


@GPU
@pytest.mark.parametrize('family', ['gemm','dgrad','swiglu','swd','wgrad'])
@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
@pytest.mark.parametrize('m,n,k,sms,waves', [(1,384,640,7,0), (129,384,640,7,1), (513,384,640,7,0), (256,128,128,2,1)])
def test_oracle_graph_rebind(family, reduction, m, n, k, sms, waves):
    torch.manual_seed(99)
    lib = Wgrad(reduction=reduction) if family == 'wgrad' else Library(family,reduction=reduction)
    inputs = operands(family,m,n,k)
    kw = {} if family == 'wgrad' else dict(policy='streamk')
    plan = lib.plan(*inputs, sms=sms, tail_waves=waves, **kw)
    def outputs():
        return [plan.out,plan.gate_out,plan.up_out] if family == 'swiglu' else [plan.out]
    def poison():
        plan.workspace.fill_(float('nan'))
        for t in outputs(): t.fill_(float('nan'))
    plan()
    check(outputs(),oracle(family,inputs))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): plan()
    for t, new in zip(inputs, operands(family,m,n,k)): t.copy_(new)
    expected = oracle(family,inputs)
    plan()
    exact = [t.clone() for t in outputs()]
    for _ in range(3):
        poison(); graph.replay()
        check(outputs(),expected)
        if reduction == 'deterministic':
            for a,b in zip(outputs(),exact): torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Release/rebind after allocator churn exercises the driver-encoded TMA maps.
    new_inputs = [t.clone() for t in inputs]
    new_out = torch.empty_like(plan.out)
    plan.release()
    plan.rebind(*new_inputs,out=new_out)
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2): plan()
    graph2.replay(); check(outputs(),expected)
    torch.cuda.synchronize()


@GPU
@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
@pytest.mark.parametrize('down', ['persistent','streamk'])
def test_ffn_autograd_and_optimizer(reduction,down):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        original = MXFP8SwiGLU(128,384,128,device='cuda')
        model = MXFP8SwiGLU(128,384,128,device='cuda')
        model.load_state_dict(original.state_dict())
        model.refresh_fp8_weights()
        ids = [id(p) for p in model.parameters()]
        report = configure_ffn_streamk(model,reduction,down_forward=down)
        assert configure_ffn_streamk(model,reduction,down_forward=down) is report
        assert ids == [id(p) for p in model.parameters()]
        x = torch.randn(513,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy = torch.randn_like(x)*.1
        def step(layer):
            layer.zero_grad(set_to_none=True); x.grad=None
            y=layer(x); y.backward(dy)
            return [y,x.grad,*[p.grad for p in layer.parameters()]]
        ref = [t.clone() for t in step(original)]
        for _ in range(3): check(step(model),ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): captured=step(model)
        x.detach().normal_(); dy.normal_()
        expected = [t.clone() for t in step(model)]
        for _ in range(3): graph.replay(); check(captured,expected)
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        handle = bind_optimizer(model,optimizer)
        step(model)
        before = model.gate_weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, model.gate_weight)
        from mxfp8 import quantize_weight
        for master, q, scale in model.fp8_parameter_bindings():
            expected_q, expected_scale = quantize_weight(master.detach())
            torch.testing.assert_close(q.float(), expected_q.float(), atol=0, rtol=0)
            torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
        assert all(torch.isfinite(t).all() for t in step(model))
        handle.remove()
        assert report['scratch_used_bytes'] <= report['scratch_capacity_bytes']
        if down == 'persistent':
            assert all(p['selected']['backend']=='persistent' for p in report['plans'] if p['family']=='gemm')
    stream.synchronize()
