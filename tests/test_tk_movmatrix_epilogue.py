"""MOVM BF16 transpose -> FP8 conversion -> STMATRIX m16n8 row-major output."""
import ctypes as ct
import subprocess

import pytest
import torch

import mxtest
from experiments.tk_mxfp8_gemm import Library, ROOT, build, pack_a_scales
from experiments.tk_persistent_gemm import FLAGS
from tests import test_tk_mxfp8_epilogue as reference_tests
from tests.test_tk_mxfp8_epilogue import (
    operands, test_tiles, test_graph_reuse, test_next_layer,
    test_validation, test_partial_scale_word,
)

OUTPUT=ROOT/'runs/tk_movmatrix_epilogue'
pytestmark=reference_tests.pytestmark


@pytest.fixture(scope='module',autouse=True)
def compiled_movmatrix():
    torch.set_num_threads(4)
    binary=OUTPUT/'libgemm.so'
    sources=[ROOT/'cuda/persistent_mxfp8_gemm.cu',ROOT/'cuda/mxfp8_epilogue.cuh',ROOT/'cuda/shared_store.cuh']
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):
        build(OUTPUT,movmatrix=True)
    probe=OUTPUT/'libprobe.so';source=ROOT/'cuda/probe_movmatrix_epilogue.cu'
    if not probe.exists() or any(p.stat().st_mtime>probe.stat().st_mtime for p in [source,*sources]):
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        str(source),'-o',str(probe)],cwd=ROOT,check=True,capture_output=True)


@pytest.fixture(autouse=True)
def use_movmatrix_library(monkeypatch):
    # Run the exact same persistent/graph/padding/next-layer contract suite.
    monkeypatch.setattr(reference_tests,'Library',lambda:Library(OUTPUT))


@pytest.mark.parametrize('c',[32,64,128,256])
@pytest.mark.parametrize('kind',['rows','zeros','rounding'])
def test_probe(c,kind):
    torch.manual_seed(41)
    x=torch.randn((16,c),device='cuda',dtype=torch.bfloat16)
    if kind=='rows':
        x=(x.float()*torch.exp2(torch.arange(16,device='cuda')*4-30)[:,None]).bfloat16()
    elif kind=='zeros':x.zero_()
    else:
        bits=torch.randint(0,256,(16,c),device='cuda',dtype=torch.int16)+0x3f00
        x=bits.view(torch.bfloat16)
    expected,sc=mxtest.launch_row_quantize(x)
    lib=ct.CDLL(str(OUTPUT/'libprobe.so'))
    fn=lib.probe_movmatrix
    fn.argtypes=[ct.c_void_p]*3+[ct.c_int]*2+[ct.c_void_p]
    for mode in [0,1]:
        q=torch.empty_like(expected);s=pack_a_scales(sc);s.fill_(-1)
        assert fn(x.data_ptr(),q.data_ptr(),s.data_ptr(),c,mode,
                  torch.cuda.current_stream().cuda_stream)==0
        torch.testing.assert_close(q.view(torch.uint8),expected.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(s,pack_a_scales(sc),rtol=0,atol=0)
