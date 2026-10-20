"""Prequantized WASP GEMM. A scales pack four K32 E8M0 bytes into int32 [ceil(K/128),roundup(M,16)]; B scales [ceil(N/32),K/32]."""
import ctypes as ct
from pathlib import Path
import subprocess
import torch
from experiments.tk_persistent_gemm import ROOT, FLAGS, CONFIGS
OUTPUT=ROOT/'runs/tk_mxfp8_gemm'
def build():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_mxfp8_gemm.cu','-lcuda','-o',str(OUTPUT/'libgemm.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
def pack_a_scales(scales):
    """Convert [M,ceil(K/32)] uint8 exponents to TMA-friendly four-byte words."""
    m,k=scales.shape
    padded=torch.zeros(((m+15)//16*16,(k+3)//4*4),device=scales.device,dtype=torch.int32)
    padded[:m,:k]=scales
    words=(padded.reshape(padded.shape[0],-1,4) << torch.tensor([0,8,16,24],device=scales.device,dtype=torch.int32)).sum(-1,dtype=torch.int32)
    return words.T.contiguous()

class Library:
    def __init__(self):
        self.lib=ct.CDLL(str(OUTPUT/'libgemm.so'))
        self.lib.tk_mx_create.argtypes=[ct.c_void_p]*5+[ct.c_int]*6
        self.lib.tk_mx_create.restype=ct.c_void_p
        self.lib.tk_mx_error.restype=ct.c_char_p
        self.lib.tk_mx_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_mx_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_mx_destroy.argtypes=[ct.c_void_p]
    def plan(self,a,b,as_,bs,config=8,out=None,sms=0,raster_group=1):
        return Plan(self,a,b,as_,bs,config,out,sms,raster_group)
class Plan:
    def __init__(self,owner,a,b,as_,bs,config,out,sms,group):
        self.handle=None
        if type(group)!=int or not 0<group<2**31:raise ValueError('Invalid raster group')
        if config not in range(len(CONFIGS)):raise ValueError('Invalid config')
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('Invalid matrix shapes')
        m,k=a.shape;n=b.shape[0]
        if min(m,n,k)<=0 or k%32:raise ValueError('Invalid dimensions')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:raise ValueError('Expected E4M3 operands')
        if as_.shape!=((k+127)//128,(m+15)//16*16) or bs.shape!=((n+31)//32,k//32):raise ValueError('Invalid scale shapes')
        if as_.dtype!=torch.int32 or bs.dtype!=torch.uint8:raise ValueError('Expected packed int32 A scales and uint8 B scales')
        if out is None:out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
        if out.shape!=(m,n) or out.dtype!=torch.bfloat16:raise ValueError('Invalid output')
        tensors=[a,b,as_,bs,out]
        if not a.is_cuda or any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if any(t.data_ptr()%16 for t in [a,b,as_,out]):raise ValueError('Unaligned TMA pointer')
        lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
        if any(lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi for t in tensors[:-1]):
            raise ValueError('Output aliases input')
        self.owner=owner;self.refs=tensors;self.out=out
        with torch.cuda.device(a.device):
            self.handle=owner.lib.tk_mx_create(*[t.data_ptr() for t in tensors],m,n,k,config,sms,group)
        if not self.handle:raise RuntimeError(owner.lib.tk_mx_error().decode())
        info=(ct.c_int*7)();owner.lib.tk_mx_info(self.handle,info)
        self.info=dict(zip(['R','C','BK','stages','shared_bytes','initial_regs','grid'],info))
        self.info.update(raster_group=min(group,(m+8*CONFIGS[config][0]-1)//(8*CONFIGS[config][0])))
    def __call__(self):
        with torch.cuda.device(self.out.device):
            if self.owner.lib.tk_mx_launch(self.handle,torch.cuda.current_stream(self.out.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mx_error().decode())
        return self.out
    def __del__(self):
        if self.handle:self.owner.lib.tk_mx_destroy(self.handle)
