"""K128 FP32 scales: Stream-K reduction and training contracts."""
import pytest
import torch
import torch.nn.functional as F

from mxfp8 import FP8GemmPlan, FP8SwiGLU, bind_optimizer
from mxfp8._kernels.fp8_gemm import _FP8DualPlan
from mxfp8._kernels.fp8_quantize import BlockQuantized, DualQuantized
from tests.test_fp8_training import dequant, ffn_reference, relative, rand

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12,0), reason='SM120 required')


@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
@pytest.mark.parametrize('dtype,row_b', [(torch.bfloat16,False),(torch.float32,False),(torch.float32,True)])
@pytest.mark.parametrize('m,n,k,sms,waves', [(1,128,1024,7,0),(129,384,640,7,1),(513,384,640,7,0),(256,128,128,2,1)])
def test_scaled_gemm_graph(reduction,dtype,row_b,m,n,k,sms,waves):
    a,b = rand(m,k).to(torch.float8_e4m3fn), rand(n,k).to(torch.float8_e4m3fn)
    sa = torch.rand(m,k//128,device='cuda')+.01
    sb = torch.rand(n if row_b else n//128,k//128,device='cuda')+.01
    plan = FP8GemmPlan(a,b,sa,sb,dtype=dtype,streamk=reduction,sms=sms,
                       b_scale_mode='row' if row_b else 'block',tail_waves=waves)
    def reference(): return dequant(a,sa) @ dequant(b,sb,not row_b).T
    def check():
        ref=reference()
        if dtype==torch.float32: torch.testing.assert_close(plan.out.double(),ref,rtol=4e-5,atol=2e-4)
        else: relative(plan.out,ref,.003)
    plan();check()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):plan()
    a.copy_(rand(m,k).to(a.dtype));b.copy_(rand(n,k).to(b.dtype));sa.uniform_(.01,2);sb.uniform_(.01,2)
    plan();expected=plan.out.clone()
    for _ in range(3):
        plan.workspace.fill_(float('nan'));plan.out.fill_(float('nan'));graph.replay();check()
        if reduction=='deterministic':torch.testing.assert_close(plan.out,expected,atol=0,rtol=0)
    old=plan.out; snapshot=old.clone();plan._new_output();plan();check()
    torch.testing.assert_close(old,snapshot,atol=0,rtol=0)
    assert plan.info['local_bytes']==0 and plan.info['shared_bytes']<=101376
    torch.cuda.synchronize()


@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
@pytest.mark.parametrize('waves', [0,1])
def test_fused_gate_up_and_dx(reduction,waves):
    m,k,n=129,640,384
    x=DualQuantized(m,k,'cuda').quantize(rand(m,k))
    wg=BlockQuantized(n,k,'cuda').quantize(rand(n,k,dtype=torch.float32)*.02)
    wu=BlockQuantized(n,k,'cuda').quantize(rand(n,k,dtype=torch.float32)*.03)
    gate,up,out=[torch.empty((m,n),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    fwd=_FP8DualPlan(x.row,wg.row,x.row_scale,wg.row_scale,a2=x.row,b2=wu.row,
                    sa2=x.row_scale,sb2=wu.row_scale,mode=1,out=out,gate=gate,up=up,
                    streamk=reduction,sms=7,tail_waves=waves)
    dg=DualQuantized(m,n,'cuda').quantize(rand(m,n));du=DualQuantized(m,n,'cuda').quantize(rand(m,n))
    dx=_FP8DualPlan(dg.row,wg.col,dg.row_scale,wg.col_scale,a2=du.row,b2=wu.col,
                   sa2=du.row_scale,sb2=wu.col_scale,mode=2,out=torch.empty((m,k),device='cuda',dtype=torch.bfloat16),
                   streamk=reduction,sms=7,tail_waves=waves)
    def check():
        xr=dequant(x.row,x.row_scale);gr=(xr@dequant(wg.row,wg.row_scale,True).T).bfloat16()
        ur=(xr@dequant(wu.row,wu.row_scale,True).T).bfloat16()
        relative(gate,gr,.003);relative(up,ur,.003)
        relative(out,(F.silu(gr.float())*ur.float()).bfloat16(),.004)
        ref=dequant(dg.row,dg.row_scale)@dequant(wg.row,wg.row_scale,True)+dequant(du.row,du.row_scale)@dequant(wu.row,wu.row_scale,True)
        relative(dx.out,ref,.003)
    fwd();dx();check()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fwd();dx()
    x.quantize(rand(m,k));wg.quantize(rand(n,k,dtype=torch.float32)*.03);wu.quantize(rand(n,k,dtype=torch.float32)*.01)
    dg.quantize(rand(m,n));du.quantize(rand(m,n))
    fwd();dx();expected=[t.clone() for t in (out,gate,up,dx.out)]
    for _ in range(3):
        fwd.workspace.fill_(float('nan'));dx.workspace.fill_(float('nan'));graph.replay();check()
        if reduction=='deterministic':
            for a,b in zip((out,gate,up,dx.out),expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert dx.info['stages']==1 and fwd.info['local_bytes']==dx.info['local_bytes']==0
    torch.cuda.synchronize()


@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
def test_ffn_training_graph(reduction):
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        layer=FP8SwiGLU(256,384,128,streamk=reduction)
        x=rand(129,256).requires_grad_();dy=rand(129,128)
        ref=ffn_reference(x.detach(),layer.gate_weight.detach(),layer.up_weight.detach(),layer.down_proj.weight.detach(),dy)
        y=layer(x);y.backward(dy)
        for a,b in zip((y,x.grad,layer.gate_weight.grad,layer.up_weight.grad,layer.down_proj.weight.grad),ref):relative(a,b,.012)
        del y
        optimizer=torch.optim.SGD(layer.parameters(),lr=.1)
        handle=bind_optimizer(layer,optimizer)
        def step():
            optimizer.zero_grad(set_to_none=True);x.grad=None
            y=layer(x);y.float().square().mean().backward();optimizer.step()
            return y
        for _ in range(3):step()
        comparison=FP8SwiGLU(256,384,128,streamk=reduction)
        comparison.load_state_dict(layer.state_dict())
        opt2=torch.optim.SGD(comparison.parameters(),lr=.1)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=step()
        for _ in range(3):
            x.detach().copy_(rand(129,256));graph.replay()
            opt2.zero_grad(set_to_none=True)
            comparison(x.detach()).float().square().mean().backward();opt2.step()
        for a,b in zip(layer.parameters(),comparison.parameters()):torch.testing.assert_close(a,b,atol=3e-7,rtol=3e-5)
        assert torch.isfinite(captured).all()
        handle.remove()
    stream.synchronize()


@pytest.mark.parametrize('reduction', ['atomic','deterministic'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16])
def test_subnormal_partials(reduction,dtype):
    a=torch.zeros((1,256),device='cuda');a[:,0]=1;a[:,128]=1
    b=a.repeat(128,1).to(torch.float8_e4m3fn);a=a.to(b.dtype)
    sa=torch.full((1,2),2.**-67,device='cuda');sb=sa.clone()
    plan=FP8GemmPlan(a,b,sa,sb,streamk=reduction,sms=4,dtype=dtype)
    assert plan.info['fixup_tiles']>0
    torch.testing.assert_close(plan(),torch.full((1,128),2.**-133,device='cuda',dtype=dtype),atol=0,rtol=0)
