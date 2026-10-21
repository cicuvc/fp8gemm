"""MXFP8 AB dW: per-feature scales, TMA tails, persistent reuse and preparation."""
import pytest
import torch
from experiments.tk_mxfp8_wgrad import Library,build,OUTPUT,ROOT,CONFIGS
from experiments.tk_mxfp8_gemm import pack_a_scales
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),
                            reason='SM120 required')


@pytest.fixture(scope='module',autouse=True)
def compiled():
    torch.set_num_threads(4)
    sources=[ROOT/'cuda/persistent_mxfp8_wgrad.cu',ROOT/'cuda/mxfp8_wgrad_configs.inc',
             ROOT/'cuda/tma_ring_pipe.cuh',ROOT/'cuda/fp8_shared_load.cuh',
             *(ROOT/'third_party/thunderkittens/include').rglob('*.cuh')]
    binary=OUTPUT/'libwgrad.so'
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):build()


def exact_inputs(n,m,k):
    ai=torch.arange(n*m).reshape(n,m);bi=torch.arange(m*k).reshape(m,k)
    a=((ai*13%17-8).float()/4).to(torch.float8_e4m3fn)
    b=((bi*7%17-8).float()/4).to(torch.float8_e4m3fn)
    # Vary every feature independently: block32 weight scales would be wrong.
    sa=((torch.arange(n)[:,None]*3+torch.arange(m//32)[None,:]*7)%5+125).byte()
    sb=((torch.arange(k)[:,None]*7+torch.arange(m//32)[None,:]*3)%5+125).byte()
    af=torch.ldexp(a.double(),(sa.int()-127).repeat_interleave(32,1))
    bf=torch.ldexp(b.double(),(sb.T.int()-127).repeat_interleave(32,0))
    return [a.cuda(),b.cuda(),pack_a_scales(sa.cuda()),pack_a_scales(sb.cuda())],(af@bf).bfloat16().cuda()


@pytest.fixture(scope='module',params=[(1,128,128),(137,128,384),(1031,1152,384)])
def operands(request):
    return exact_inputs(*request.param)


@pytest.mark.parametrize('config',range(len(CONFIGS)))
@pytest.mark.parametrize('group',[1,13])
def test_exact_tiles(operands,config,group):
    args,ref=operands
    out=torch.full_like(ref,float('nan'))
    p=Library().plan(*args,config=config,out=out,sms=3,raster_group=group)
    assert p() is out
    torch.testing.assert_close(out,ref,rtol=0,atol=0)


@pytest.mark.parametrize('config',[0,1,3,4,5,15])
@pytest.mark.parametrize('m',[32,96,160])
def test_partial_scale_word(config,m):
    args,ref=exact_inputs(137,m,192)
    p=Library().plan(*args,config=config,sms=3,raster_group=5)
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)


@pytest.mark.parametrize('config',[0,7,12,14,17])
def test_graph_reuse(config):
    args,ref=exact_inputs(777,256,384)
    p=Library().plan(*args,config=config,sms=3,raster_group=13)
    p();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):p()
    saved=args[0].clone();args[0].zero_();p.out.fill_(float('nan'))
    graph.replay();assert torch.count_nonzero(p.out)==0
    args[0].copy_(saved);graph.replay()
    torch.testing.assert_close(p.out,ref,rtol=0,atol=0)
    # Scale buffers are live inputs too, not copied into a descriptor at setup.
    old=args[3].clone();args[3].fill_(0x7f7f7f7f)
    graph.replay();assert not torch.equal(p.out,ref)
    args[3].copy_(old)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):p()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(p.out,ref,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(1,32),(137,192),(1031,384)])
@pytest.mark.parametrize('transpose',[False,True])
def test_quantizer(shape,transpose):
    m,f=shape;torch.manual_seed(821)
    x=torch.randn((m,f),device='cuda',dtype=torch.bfloat16)
    normal=SiluBwdQuantizeLibrary().plan_direct(x)
    oriented=SiluBwdQuantizeLibrary().plan_direct(x,transpose=transpose)
    nq=normal();oriented_outputs=oriented();q=oriented_outputs[2]
    expected=nq[2].T.contiguous() if transpose else nq[2]
    torch.testing.assert_close(q.view(torch.uint8),expected.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(oriented.column_scale,nq[3],rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):oriented()
    x.zero_();q.view(torch.uint8).fill_(255);oriented.column_scale.fill_(-1);graph.replay()
    assert torch.count_nonzero(q.view(torch.uint8))==0
    assert torch.count_nonzero(oriented.column_scale)==0


def test_quantized_chain():
    # Quantization remains a separate operation, including in graph capture.
    torch.manual_seed(9)
    dy=torch.randn((137,37),device='cuda',dtype=torch.bfloat16)
    x=torch.randn((137,128),device='cuda',dtype=torch.bfloat16)
    dy=torch.nn.functional.pad(dy,(0,27))
    qa=SiluBwdQuantizeLibrary().plan_direct(dy,transpose=True)
    qb=SiluBwdQuantizeLibrary().plan_direct(x)
    qa();qb()
    p=Library().plan(
        qa.column_fp8,
        qb.column_fp8,
        qa.column_scale,
        qb.column_scale,
        config=12,
        sms=3,
    )
    def dequant(q,s,transpose):
        f=q.shape[0] if transpose else q.shape[1]
        scale=((s[:,:f].to(torch.int64)[:,:,None]>>(8*torch.arange(4,device='cuda')))&255)
        scale=scale.permute(1,0,2).reshape(f,-1).repeat_interleave(32,1)
        values=q.double() if transpose else q.double().T
        return torch.ldexp(values,scale.int()-127)
    ref=(
        dequant(qa.column_fp8,qa.column_scale,True)
        @ dequant(qb.column_fp8,qb.column_scale,False).T
    ).bfloat16()
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):qa();qb();p()
    dy.zero_();graph.replay();assert torch.count_nonzero(p.out)==0


def test_validation():
    args,ref=exact_inputs(16,128,128);lib=Library()
    with pytest.raises(ValueError,match='BF16 output'):lib.plan(*args,out=ref.float())
    with pytest.raises(ValueError,match='scale shapes'):lib.plan(*args[:3],args[3][:,:16])
    with pytest.raises(ValueError,match='int32'):lib.plan(args[0],args[1],args[2].byte(),args[3])
    with pytest.raises(ValueError,match=r'A\[N,M\]'):lib.plan(args[0],args[1][:-1],*args[2:])
    with pytest.raises(ValueError,match='config'):lib.plan(*args,config=-1)
    with pytest.raises(ValueError,match='raster'):lib.plan(*args,raster_group=0)
    alias=args[1].view(torch.bfloat16)
    alias=alias.as_strided(ref.shape,(128,1))
    with pytest.raises(ValueError,match='aliases'):lib.plan(*args,out=alias)
