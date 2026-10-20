"""ctypes cuBLASLt benchmark adapter; inputs X[M,K], W[N,K], output BF16."""
import ctypes as ct
from pathlib import Path
import subprocess
import torch
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/cublaslt_gemm'
def build():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-shared','-Xcompiler=-fPIC',
                        'cuda/cublaslt_benchmark.cu','-lcublasLt','-o',str(OUTPUT/'liblt.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
class Library:
    def __init__(self):
        self.lib=ct.CDLL(str(OUTPUT/'liblt.so'))
        self.lib.lt_create.argtypes=[ct.c_void_p]*6+[ct.c_size_t]+[ct.c_int]*6
        self.lib.lt_create.restype=ct.c_void_p
        self.lib.lt_error.restype=ct.c_char_p
        self.lib.lt_count.argtypes=[ct.c_void_p]
        self.lib.lt_launch.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
        self.lib.lt_info.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int),ct.POINTER(ct.c_size_t)]
        self.lib.lt_destroy.argtypes=[ct.c_void_p]
        self.lib.lt_version.restype=ct.c_size_t
    def plan(self,x,w,sx,sw,out,workspace,mode,fast=0):
        return Plan(self,x,w,sx,sw,out,workspace,mode,fast)
class Plan:
    def __init__(self,owner,x,w,sx,sw,out,workspace,mode,fast):
        self.owner=owner;self.refs=(x,w,sx,sw,out,workspace);self.handle=None
        self.handle=owner.lib.lt_create(*[t.data_ptr() if t is not None else None for t in self.refs],
            workspace.numel(),x.shape[0],w.shape[0],x.shape[1],mode,fast,64)
        if not self.handle:raise RuntimeError(owner.lib.lt_error().decode())
        self.count=owner.lib.lt_count(self.handle)
    def launch(self,i):
        if self.owner.lib.lt_launch(self.handle,i,torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(self.owner.lib.lt_error().decode())
    def info(self,i):
        a=(ct.c_int*7)();w=ct.c_size_t()
        self.owner.lib.lt_info(self.handle,i,a,ct.byref(w))
        return dict(zip(['id','tile','stages','split_k','reduction','numerical_flags','numerical_flags_status'],a))|{'workspace_bytes':w.value}
    def __del__(self):
        if self.handle:self.owner.lib.lt_destroy(self.handle)
