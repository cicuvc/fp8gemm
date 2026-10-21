"""MXFP8 swapAB dX: operand layouts/scales, TMA tails, persistent reuse."""
import pytest
import torch
from experiments.tk_mxfp8_dgrad import Library,build,OUTPUT,ROOT,CONFIGS
from experiments.tk_mxfp8_gemm import pack_a_scales

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),
                            reason='SM120 required')


@pytest.fixture(scope='module',autouse=True)
def compiled():
    torch.set_num_threads(4)
    sources=[ROOT/'cuda/persistent_mxfp8_dgrad.cu',ROOT/'cuda/mxfp8_dgrad_configs.inc',
             ROOT/'cuda/tma_ring_pipe.cuh',ROOT/'cuda/fp8_shared_load.cuh',
             *(ROOT/'third_party/thunderkittens/include').rglob('*.cuh')]
    binary=OUTPUT/'libdgrad.so'
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):build()


def exact_inputs(m,n,k):
    ai=torch.arange(m*n).reshape(m,n);bi=torch.arange(n*k).reshape(n,k)
    a=((ai*13%17-8).float()/4).to(torch.float8_e4m3fn)
    b=((bi*7%17-8).float()/4).to(torch.float8_e4m3fn)
    # dY scales vary per token/N32, W scales per N32/K32.
    sa=((torch.arange(m)[:,None]*3+torch.arange(n//32)[None,:]*7)%5+125).byte()
    sb=((torch.arange(n//32)[:,None]*7+torch.arange(k//32)[None,:]*3)%5+125).byte()
    af=torch.ldexp(a.double(),(sa.int()-127).repeat_interleave(32,1))
    bf=torch.ldexp(b.double(),(sb.int()-127).repeat_interleave(32,0).repeat_interleave(32,1))
    return [a.cuda(),b.cuda(),pack_a_scales(sa.cuda()),sb.cuda()],(af@bf).bfloat16().cuda()


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
@pytest.mark.parametrize('n',[32,96,160])
def test_partial_scale_word(config,n):
    args,ref=exact_inputs(137,n,192)
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
    old=args[3].clone();args[3].fill_(127)
    graph.replay();assert not torch.equal(p.out,ref)
    args[3].copy_(old)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):p()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(p.out,ref,rtol=0,atol=0)


def test_validation():
    args,ref=exact_inputs(16,128,128);lib=Library()
    with pytest.raises(ValueError,match='BF16 output'):lib.plan(*args,out=ref.float())
    with pytest.raises(ValueError,match='scale shapes'):lib.plan(*args[:3],args[3][:,:1])
    with pytest.raises(ValueError,match='int32'):lib.plan(args[0],args[1],args[2].byte(),args[3])
    with pytest.raises(ValueError,match=r'dY\[M,N\]'):lib.plan(args[0],args[1][:-1],*args[2:])
    with pytest.raises(ValueError,match='config'):lib.plan(*args,config=-1)
    with pytest.raises(ValueError,match='raster'):lib.plan(*args,raster_group=0)
    alias=args[1].view(torch.bfloat16)
    alias=alias.as_strided(ref.shape,(128,1))
    with pytest.raises(ValueError,match='aliases'):lib.plan(*args,out=alias)
