"""CUDA linear backward: actual transposes/padding, FP32 dW, and autograd."""
import pytest
import torch
import torch.nn.functional as F
import mxtest
from fp8_linear import FP8Linear
from experiments.tk_linear_backward import LinearBackwardPlan
from experiments.benchmark_tk_dual_online_gemm import exact_input
from experiments.benchmark_tk_persistent_gemm import quantized_a

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),reason='requires SM120')


def quant_rows(x):
    k=x.shape[1]
    return quantized_a(F.pad(x,(0,(-k)%32)))[:,:k]


def initialized_plan(*args,**kwargs):
    p=LinearBackwardPlan(*args,**kwargs)
    # TMA output coverage check; also avoids initcheck's TMA-store tracking
    # limitation when subsequently copying results. Not part of the real API.
    p.dxp.fill_(float('nan'));p.dwp.fill_(float('nan'))
    return p


@pytest.mark.parametrize('shape',[(1,31,33),(17,97,65),(137,192,128),(257,320,192)])
@pytest.mark.parametrize('bias',[False,True])
def test_exact_backward(shape,bias):
    m,n,k=shape
    xc=exact_input(m,k,73);dc=exact_input(m,n,29)
    wc=exact_input(n,k,17).float()
    w,ws=mxtest.launch_weight_quantize(wc.cuda())
    # Distinct neighboring 32x32 scales exercise their transpose for dX.
    wf=w.cpu().float()*torch.exp2(ws.cpu().float()-127).repeat_interleave(32,0).repeat_interleave(32,1)[:n,:k]
    dx=(quant_rows(dc).double() @ wf.double()).bfloat16()
    dw=(quant_rows(dc.T.contiguous()).double() @ quant_rows(xc.T.contiguous()).double().T).float()
    db=dc.float().sum(0)
    # Both operands strided, including their last dimension.
    x=xc.cuda().T.contiguous().T
    dy=dc.cuda().T.contiguous().T
    p=initialized_plan(dy,x,w,ws,bias)
    actual=p()
    torch.testing.assert_close(actual[0].cpu(),dx,rtol=0,atol=0)
    torch.testing.assert_close(actual[1].cpu(),dw,rtol=0,atol=0)
    if bias:torch.testing.assert_close(actual[2].cpu(),db,rtol=0,atol=0)
    else:assert actual[2].numel()==0
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    assert actual[0].dtype==torch.bfloat16 and actual[1].dtype==torch.float32
    # Exercise both prequantized B refreshes through unchanged descriptors.
    x.zero_();w.view(torch.uint8).zero_()
    p()
    assert torch.count_nonzero(p.dx).item()==0 and torch.count_nonzero(p.dw).item()==0
    x.copy_(xc);w_new,ws_new=mxtest.launch_weight_quantize((wc*.5).cuda())
    w.copy_(w_new);ws.copy_(ws_new)
    p()
    torch.testing.assert_close(p.dx.cpu(),dx*.5,rtol=0,atol=0)
    torch.testing.assert_close(p.dw.cpu(),dw,rtol=0,atol=0)


def test_stream_and_graph():
    torch.manual_seed(7)
    x=torch.randn(137,65,device='cuda',dtype=torch.bfloat16)
    dy=torch.randn(137,97,device='cuda',dtype=torch.bfloat16)
    w,ws=mxtest.launch_weight_quantize(torch.randn(97,65,device='cuda'))
    p=initialized_plan(dy,x,w,ws)
    p();torch.cuda.synchronize()
    expected=tuple(t.clone() for t in (p.dx,p.dw,p.db))
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):p()
    torch.cuda.current_stream().wait_stream(s)
    for got,ref in zip((p.dx,p.dw,p.db),expected):torch.testing.assert_close(got,ref,rtol=0,atol=0)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):p()
    dy.zero_();g.replay()
    for t in (p.dx,p.dw,p.db):assert torch.count_nonzero(t).item()==0


@pytest.mark.parametrize('bias',[False,True])
def test_autograd_compile_and_weight_refresh(bias):
    torch.manual_seed(8)
    layer=FP8Linear(65,97,bias=bias,device='cuda',backward_backend='cuda')
    ref=FP8Linear(65,97,bias=bias,device='cuda')
    ref.load_state_dict(layer.state_dict());ref.refresh_fp8_weight()
    x=torch.randn(17,65,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn(97,17,device='cuda',dtype=torch.float32).T
    # Explicit FP32 upstream gradient is rounded to BF16 by both backends.
    targets=(x,layer.master_weight)+((layer.bias,) if bias else ())
    y=layer(x)
    grads=torch.autograd.grad(y,targets,dy,retain_graph=True)
    again=torch.autograd.grad(y,targets,dy)
    for a,b in zip(grads,again):torch.testing.assert_close(a,b,rtol=0,atol=0)
    ref_targets=(x,ref.master_weight)+((ref.bias,) if bias else ())
    ref_grads=torch.autograd.grad(ref(x),ref_targets,dy)
    for a,b in zip(grads,ref_grads):
        rel=((a.float()-b.float()).norm()/b.float().norm()).item()
        assert rel<5e-4,rel
    compiled=torch.compile(layer,fullgraph=True)
    compiled_grads=torch.autograd.grad(compiled(x),targets,dy)
    for a,b in zip(compiled_grads,grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Accumulation plus lazy weight refresh after a real optimizer step.
    layer(x).backward(dy);layer(x).backward(dy)
    torch.testing.assert_close(layer.master_weight.grad,grads[1]*2,rtol=0,atol=0)
    optimizer=torch.optim.SGD(layer.parameters(),lr=.01)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    ref.load_state_dict(layer.state_dict());ref.refresh_fp8_weight()
    after=torch.autograd.grad(layer(x),targets,dy)
    ref_after=torch.autograd.grad(ref(x),ref_targets,dy)
    assert layer._synced_version==layer.master_weight._version
    for a,b in zip(after,ref_after):
        assert ((a.float()-b.float()).norm()/b.float().norm()).item()<5e-4
    torch._dynamo.reset()
