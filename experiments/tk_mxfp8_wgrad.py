"""Prequantized MXFP8 dW: A[N,M] @ B[M,K] -> BF16 [N,K].

Both operands use independent per-feature/M32 E8M0 scales, packed as int32
[ceil(M/128),roundup(features,16)]. B stays in physical [M,K] row-major order.
The standalone Triton quantizer is a preparation/reference tool; the CUDA
GEMM has no online quantization. These entry points do not change FP8Linear.
"""
import ctypes as ct
from pathlib import Path
import re
import subprocess
import torch
import triton
import triton.language as tl
import mxtest
from experiments.tk_persistent_gemm import ROOT,FLAGS

OUTPUT=ROOT/'runs/tk_mxfp8_wgrad'
CONFIGS=[tuple(map(int,m[1:])) for m in re.findall(
    r'^CFG\((\d+),(\d+),(\d+),(\d+),(\d+)\);',
    (ROOT/'cuda/mxfp8_wgrad_configs.inc').read_text(),re.M)]


def build(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-Xptxas=-v','-shared','-Xcompiler=-fPIC',
                        'cuda/persistent_mxfp8_wgrad.cu','-lcuda','-o',str(output/'libwgrad.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)


@triton.jit
def _column_quantize(X,Q,S,M:tl.constexpr,F:tl.constexpr,MP:tl.constexpr,SP:tl.constexpr,
                     S0:tl.constexpr,S1:tl.constexpr,TRANSPOSE:tl.constexpr):
    token=tl.program_id(0)*128+tl.arange(0,128)
    feature=tl.program_id(1)*32+tl.arange(0,32)
    x=tl.load(X+token[:,None]*S0+feature[None,:]*S1,
              (token[:,None]<M)&(feature[None,:]<F),0)
    q,scale=mxtest.online_scale(tl.trans(x))
    if TRANSPOSE:
        tl.store(Q+feature[:,None]*MP+token[None,:],q,feature[:,None]<F)
    else:
        tl.store(Q+token[:,None]*F+feature[None,:],tl.trans(q),feature[None,:]<F)
    words=tl.sum(scale.to(tl.uint32)<<(8*tl.arange(0,4))[None,:],axis=1)
    tl.store(S+tl.program_id(0)*SP+feature,words,feature<SP)


class ColumnQuantizePlan:
    """BF16 x[M,F] -> FP8 quantized along M32, optionally physically transposed.

    Token rows are zero-padded to 128; feature dimension is unchanged.
    Scales use the same block-major packed layout in either data orientation.
    Finite-BF16 online_scale semantics, including its exponent-254 limitation.
    """
    def __init__(self,x,*,transpose=False):
        if x.ndim!=2 or min(x.shape)<=0 or x.dtype!=torch.bfloat16 or not x.is_cuda:
            raise ValueError('Expected positive CUDA BF16 [M,F]')
        self.x=x;self.transpose=bool(transpose)
        m,f=x.shape;self.mp=mp=triton.cdiv(m,128)*128;self.sp=sp=triton.cdiv(f,16)*16
        self.out=torch.empty((f,mp) if transpose else (mp,f),device=x.device,dtype=torch.float8_e4m3fn)
        self.scales=torch.empty((mp//128,sp),device=x.device,dtype=torch.int32)

    def __call__(self):
        m,f=self.x.shape
        with torch.cuda.device(self.x.device):
            _column_quantize[(self.mp//128,triton.cdiv(f,32))](
                self.x,self.out,self.scales,m,f,self.mp,self.sp,*self.x.stride(),self.transpose,
                num_warps=4)
        return self.out,self.scales


class Library:
    def __init__(self,output=OUTPUT):
        self.lib=ct.CDLL(str(Path(output)/'libwgrad.so'))
        self.lib.tk_mxw_create.argtypes=[ct.c_void_p]*5+[ct.c_int]*6
        self.lib.tk_mxw_create.restype=ct.c_void_p
        self.lib.tk_mxw_launch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.tk_mxw_error.restype=ct.c_char_p
        self.lib.tk_mxw_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_mxw_destroy.argtypes=[ct.c_void_p]

    def plan(self,a,b,as_,bs,config=7,out=None,*,sms=0,raster_group=1):
        return Plan(self,a,b,as_,bs,config,out,sms,raster_group)


class Plan:
    """Reusable fixed-buffer GEMM; call again after refreshing operand contents.

    A[N,M], B[M,K] are contiguous E4M3. Both scale grids are packed int32.
    N tails are TMA-masked. M/K must align to the selected TMA swizzle panels.
    FP32 internal accumulation, BF16 writeback; no implicit transpose or padding.
    """
    def __init__(self,owner,a,b,as_,bs,config,out,sms,group):
        self.handle=None
        if type(config)!=int or not 0<=config<len(CONFIGS):raise ValueError('Invalid config')
        if type(group)!=int or not 0<group<2**31 or type(sms)!=int or not 0<=sms<2**31:
            raise ValueError('Invalid raster group / SM count')
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or min(*a.shape,*b.shape)<=0:
            raise ValueError('Expected A[N,M] and B[M,K]')
        n,m=a.shape;k=b.shape[1];r,c,bm,stages=CONFIGS[config]
        if m%min(128,bm) or k%min(128,c):
            raise ValueError('M/K must align to TMA swizzle panels')
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=a.dtype:raise ValueError('Expected E4M3 operands')
        if as_.shape!=((m+127)//128,(n+15)//16*16) or bs.shape!=((m+127)//128,(k+15)//16*16):
            raise ValueError('Invalid packed scale shapes')
        if as_.dtype!=torch.int32 or bs.dtype!=torch.int32:raise ValueError('Expected packed int32 scales')
        if out is None:out=torch.empty((n,k),device=a.device,dtype=torch.bfloat16)
        if out.shape!=(n,k) or out.dtype!=torch.bfloat16:raise ValueError('Expected BF16 output [N,K]')
        tensors=[a,b,as_,bs,out]
        if not a.is_cuda or any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Expected contiguous tensors on one CUDA device')
        if torch.cuda.get_device_capability(a.device)!=(12,0):raise RuntimeError('SM120 required')
        if any(t.data_ptr()%16 for t in tensors):raise ValueError('Unaligned buffer')
        lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
        if any(lo<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<hi for t in tensors[:4]):
            raise ValueError('Output aliases an input')
        self.owner=owner;self.refs=tensors;self.out=out
        with torch.cuda.device(a.device):
            self.handle=owner.lib.tk_mxw_create(*[t.data_ptr() for t in tensors],n,k,m,config,sms,group)
        if not self.handle:raise RuntimeError(owner.lib.tk_mxw_error().decode())
        info=(ct.c_int*7)();owner.lib.tk_mxw_info(self.handle,info)
        self.info=dict(zip(['R','C','BM','stages','shared_bytes','initial_regs','grid'],info))
        self.info.update(raster_group=min(group,(n+8*r-1)//(8*r)),
                         shape_nmk=[n,m,k],bf16_output=True)

    def __call__(self):
        with torch.cuda.device(self.out.device):
            if self.owner.lib.tk_mxw_launch(self.handle,torch.cuda.current_stream(self.out.device).cuda_stream):
                raise RuntimeError(self.owner.lib.tk_mxw_error().decode())
        return self.out

    def __del__(self):
        if self.handle:self.owner.lib.tk_mxw_destroy(self.handle)
