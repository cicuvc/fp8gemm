"""Dual-output GEMM: bitwise quantization, padding, reuse and next-layer use."""
import ctypes as ct
import subprocess

import pytest
import torch

import mxtest
from experiments.tk_mxfp8_gemm import Library, OUTPUT, ROOT, CONFIGS, pack_a_scales
from experiments.tk_persistent_gemm import FLAGS
from tests.test_tk_mxfp8_gemm import compiled, exact_operands

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,0),
                            reason='SM120 required')


def check_quantized(p):
    q,sc=mxtest.launch_row_quantize(p.out)
    torch.testing.assert_close(p.qout.view(torch.uint8),q.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(p.qscales,pack_a_scales(sc),rtol=0,atol=0)


@pytest.fixture(scope='module',params=[(13,384,128),(1031,640,1152)])
def operands(request):
    return exact_operands(*request.param)


@pytest.mark.parametrize('config',[i for i,c in enumerate(CONFIGS) if c[1]%32==0])
def test_tiles(operands,config):
    args,ref=operands
    p=Library().plan(*args,config=config,sms=3,raster_group=13,save_quantized=True)
    p.qout.view(torch.uint8).fill_(255);p.qscales.fill_(-1)
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)
    check_quantized(p)


@pytest.mark.parametrize('config',[43,70,106])
def test_graph_reuse(config):
    torch.manual_seed(851)
    a,sa=mxtest.launch_row_quantize(torch.randn((777,256),device='cuda',dtype=torch.bfloat16))
    b,sb=mxtest.launch_weight_quantize(torch.randn((384,256),device='cuda'))
    args=[a,b,pack_a_scales(sa),sb]
    out=torch.empty((777,384),device='cuda',dtype=torch.bfloat16)
    q=torch.empty_like(out,dtype=torch.float8_e4m3fn)
    qs=torch.empty((3,784),device='cuda',dtype=torch.int32)
    p=Library().plan(*args,config=config,sms=3,raster_group=13,save_quantized=True,
                     out=out,quantized_out=q,quantized_scales=qs)
    assert p.qout is q and p.qscales is qs and p.out is out
    ref=Library().plan(*args,config=config,sms=3,raster_group=13)().clone()
    p();check_quantized(p)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):p()
    saved=a.clone();a.zero_();q.view(torch.uint8).fill_(255);qs.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(out)==0
    assert torch.count_nonzero(q.view(torch.uint8))==0
    assert torch.count_nonzero(qs)==0
    a.copy_(saved);graph.replay()
    torch.testing.assert_close(out,ref,rtol=0,atol=0);check_quantized(p)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):p()
    torch.cuda.current_stream().wait_stream(stream)
    check_quantized(p)


@pytest.mark.parametrize('config',[0,1,43,106])
def test_partial_scale_word(config):
    args,ref=exact_operands(137,192,128)
    p=Library().plan(*args,config=config,sms=3,save_quantized=True)
    p.qscales.fill_(-1)
    torch.testing.assert_close(p(),ref,rtol=0,atol=0)
    check_quantized(p)


def test_next_layer():
    args,ref=exact_operands(137,256,128)
    lib=Library()
    p=lib.plan(*args,config=70,save_quantized=True);p()
    weight,sw=mxtest.launch_weight_quantize(torch.randn((192,256),device='cuda'))
    next_plan=lib.plan(p.qout,weight,p.qscales,sw,config=70)
    q,sc=mxtest.launch_row_quantize(ref)
    expected=lib.plan(q,weight,pack_a_scales(sc),sw,config=70)().clone()
    torch.testing.assert_close(next_plan(),expected,rtol=0,atol=0)


def test_validation():
    args,_=exact_operands(16,128,128)
    lib=Library()
    with pytest.raises(ValueError,match='C divisible'):
        lib.plan(*args,config=5,save_quantized=True)
    with pytest.raises(ValueError,match='aliases'):
        lib.plan(*args,save_quantized=True,quantized_out=args[0])
    with pytest.raises(ValueError,match='save_quantized'):
        lib.plan(*args,quantized_out=args[0])
    with pytest.raises(ValueError,match='scales'):
        lib.plan(*args,save_quantized=True,quantized_scales=args[3])
    tail_args,_=exact_operands(16,192,128)
    with pytest.raises(ValueError,match='swizzle panels'):
        lib.plan(*tail_args,config=70,save_quantized=True)


def test_stmatrix_m16n8_layout():
    directory=ROOT/'runs/tk_mxfp8_epilogue'
    directory.mkdir(parents=True,exist_ok=True)
    binary=directory/'libprobe.so';source=ROOT/'cuda/probe_stmatrix_m16n8.cu'
    if not binary.exists() or source.stat().st_mtime>binary.stat().st_mtime:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-shared','-Xcompiler=-fPIC',
                        str(source),'-o',str(binary)],cwd=ROOT,check=True,capture_output=True)
    lib=ct.CDLL(str(binary))
    lib.stmatrix_probe.argtypes=[ct.c_void_p,ct.c_void_p]
    out=torch.empty((128,),device='cuda',dtype=torch.uint8)
    assert lib.stmatrix_probe(out.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
    logical=torch.arange(128,dtype=torch.uint8).reshape(16,8)
    torch.testing.assert_close(out.cpu(),logical.T.contiguous().flatten(),rtol=0,atol=0)
    assert not torch.equal(out.cpu(),logical.flatten())
