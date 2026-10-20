"""Prequantized GEMM scale layout, TMA tails, persistent reuse and graph safety."""
import pytest
import torch
import mxtest
from experiments.tk_mxfp8_gemm import Library, build, OUTPUT, ROOT, pack_a_scales
from experiments.tk_persistent_gemm import GemmLibrary

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),reason='SM120 required')
@pytest.fixture(scope='module',autouse=True)
def compiled():
    torch.set_num_threads(4)
    binary=OUTPUT/'libgemm.so'
    sources=[ROOT/'cuda/persistent_mxfp8_gemm.cu',ROOT/'cuda/tma_ring_pipe.cuh',ROOT/'cuda/linear_tma_buffer.cuh',
             *(ROOT/'third_party/thunderkittens/include').rglob('*.cuh')]
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):build()

@pytest.fixture(scope='module',params=[(1,192,64),(137,192,192),(260,192,192),(1031,320,320)])
def operands(request):
    m,n,k=request.param
    i=torch.arange(m*k).reshape(m,k);j=torch.arange(n*k).reshape(n,k)
    a=((i*13%17-8).float()/4).to(torch.float8_e4m3fn)
    b=((j*7%17-8).float()/4).to(torch.float8_e4m3fn)
    sa=((torch.arange(m*(k//32)).reshape(k//32,m)*7)%5+125).byte()
    sb=((torch.arange(((n+31)//32)*(k//32)).reshape((n+31)//32,k//32)*3)%5+125).byte()
    af=torch.ldexp(a.double(),(sa.T.int()-127).repeat_interleave(32,1))
    bf=torch.ldexp(b.double(),(sb.int()-127).repeat_interleave(32,0).repeat_interleave(32,1)[:n])
    return [a.cuda(),b.cuda(),pack_a_scales(sa.T.cuda()),sb.cuda()],(af@bf.T).bfloat16().cuda()

@pytest.mark.parametrize('config',range(10))
@pytest.mark.parametrize('group',[1,3,13])
def test_exact(operands,config,group):
    args,ref=operands
    p=Library().plan(*args,config=config,sms=3,raster_group=group)
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)

@pytest.mark.parametrize('config',[0,2,4,6])
@pytest.mark.parametrize('k',[32,96,160])
def test_odd_k(config,k):
    a=torch.ones((777,k),device='cuda').to(torch.float8_e4m3fn)
    b=torch.ones((192,k),device='cuda').to(torch.float8_e4m3fn)
    sa=pack_a_scales(torch.full((777,k//32),127,device='cuda',dtype=torch.uint8))
    sb=torch.full((6,k//32),127,device='cuda',dtype=torch.uint8)
    p=Library().plan(a,b,sa,sb,config=config,sms=3,raster_group=3)
    torch.testing.assert_close(p(),torch.full((777,192),k,device='cuda',dtype=torch.bfloat16),rtol=0,atol=0)

@pytest.mark.parametrize('config',[0,2,7,8])
@pytest.mark.parametrize('m',[1793,1796])
def test_online_equivalence_and_reuse(config,m):
    torch.manual_seed(123)
    a=torch.randn((m,192),device='cuda',dtype=torch.bfloat16)
    b,sb=mxtest.launch_weight_quantize(torch.randn((320,192),device='cuda'))
    aq,sa=mxtest.launch_row_quantize(a);sa=pack_a_scales(sa)
    p=Library().plan(aq,b,sa,sb,config=config,sms=3,raster_group=13)
    ref=GemmLibrary().plan(a,b,config=config,scales=sb,sms=3,raster_group=13)().clone()
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):p()
    saved=aq.clone();aq.zero_();graph.replay();assert torch.count_nonzero(p.out)==0
    aq.copy_(saved);graph.replay();torch.testing.assert_close(p.out,ref,rtol=0,atol=0)
    sa.sub_(0x01010101);graph.replay();torch.testing.assert_close(p.out,ref*.5,rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):p()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(p.out,ref*.5,rtol=0,atol=0)
