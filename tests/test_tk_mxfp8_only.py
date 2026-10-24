"""FP8-only epilogue: exact dual-output equivalence and persistent buffer reuse."""
import os
import pytest
import torch
from mxfp8._core import quantize_weight
from mxfp8._kernels.gemm import Library, build, OUTPUT, ROOT, CONFIGS, pack_a_scales
from mxfp8._kernels.silu import SiluBwdQuantizeLibrary
from tests.test_tk_mxfp8_gemm import exact_operands

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),
                            reason='SM120 required')
VARIANTS=['native','factored','movm','factored_movm']


@pytest.fixture(scope='module',params=os.environ.get('FP8_ONLY_VARIANTS',','.join(VARIANTS)).split(','))
def library(request):
    torch.set_num_threads(4)
    name=request.param
    assert name in VARIANTS
    path=OUTPUT if name=='native' else ROOT/'runs/tk_mxfp8_only'/name
    sources=[ROOT/'cuda/persistent_mxfp8_gemm.cu',ROOT/'cuda/mxfp8_epilogue.cuh',
             ROOT/'cuda/shared_store.cuh']
    binary=path/'libgemm.so'
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):
        build(path,movmatrix='movm' in name,factored_stores='factored' in name)
    return Library(path)


@pytest.fixture(scope='module',params=[(13,384,128),(1031,384,1152)])
def operands(request):
    return exact_operands(*request.param)


@pytest.mark.parametrize('config',[i for i,c in enumerate(CONFIGS) if c[1]%32==0])
def test_tiles(library,operands,config):
    args,ref=operands
    dual=library.plan(*args,config=config,sms=3,raster_group=13,save_quantized=True)
    only=library.plan(*args,config=config,sms=3,raster_group=13,save_quantized=True,save_bf16=False)
    dual()
    torch.testing.assert_close(dual.out,ref,rtol=0,atol=0)
    only.qout.view(torch.uint8).fill_(255);only.qscales.fill_(-1)
    assert only() is only.qout and only.out is None
    assert all(t.dtype!=torch.bfloat16 for t in only.refs)
    torch.testing.assert_close(only.qout.view(torch.uint8),dual.qout.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(only.qscales,dual.qscales,rtol=0,atol=0)
    assert only.info['shared_bytes']<dual.info['shared_bytes']


@pytest.mark.parametrize('config',[43,70,106])
def test_graph_reuse(library,config):
    torch.manual_seed(91)
    xq=SiluBwdQuantizeLibrary().plan_direct(
        torch.randn((777,256),device='cuda',dtype=torch.bfloat16)
    )
    xq()
    b,sb=quantize_weight(torch.randn((384,256),device='cuda'))
    args=[xq.row_fp8,b,xq.row_scale,sb]
    dual=library.plan(*args,config=config,sms=3,save_quantized=True)
    q=torch.empty((777,384),device='cuda',dtype=xq.row_fp8.dtype)
    qs=torch.empty((3,784),device='cuda',dtype=torch.int32)
    only=library.plan(*args,config=config,sms=3,save_quantized=True,save_bf16=False,
                      quantized_out=q,quantized_scales=qs)
    assert only.qout is q and only.qscales is qs
    dual();only()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):only()
    saved=xq.row_fp8.clone();xq.row_fp8.zero_();q.view(torch.uint8).fill_(255);qs.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(q.view(torch.uint8))==0 and torch.count_nonzero(qs)==0
    xq.row_fp8.copy_(saved);graph.replay()
    torch.testing.assert_close(q.view(torch.uint8),dual.qout.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(qs,dual.qscales,rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):only()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(q.view(torch.uint8),dual.qout.view(torch.uint8),rtol=0,atol=0)
    # The returned FP8 and packed scales feed the next GEMM directly.
    w,sw=quantize_weight(torch.randn((128,384),device='cuda'))
    actual=library.plan(q,w,qs,sw,config=70)()
    expected=library.plan(dual.qout,w,dual.qscales,sw,config=70)()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_validation(library):
    args,_=exact_operands(16,128,128)
    with pytest.raises(ValueError,match='requires save_quantized'):
        library.plan(*args,save_bf16=False)
    with pytest.raises(ValueError,match='out requires'):
        library.plan(*args,save_bf16=False,save_quantized=True,
                     out=torch.empty((16,128),device='cuda',dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='aliases'):
        library.plan(*args,save_bf16=False,save_quantized=True,quantized_out=args[0])


@pytest.mark.parametrize('config',[0,1,43,106])
def test_partial_scale_word(library,config):
    args,_=exact_operands(137,192,128)
    dual=library.plan(*args,config=config,sms=3,save_quantized=True)
    only=library.plan(*args,config=config,sms=3,save_quantized=True,save_bf16=False)
    dual();only.qscales.fill_(-1);only()
    torch.testing.assert_close(only.qout.view(torch.uint8),dual.qout.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(only.qscales,dual.qscales,rtol=0,atol=0)
