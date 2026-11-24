"""Hybrid Stream-K FFN GEMMs; atomic by default, fixed-order partials optional.

GEMM: down forward; dgrad: down dX; swiglu: fused gate/up with saved auxiliaries;
swd: fused gate/up dX. Weight gradients use tk_mxfp8_streamk_wgrad.Library.
Quantization and elementwise backward retain the existing component kernels.
"""
import ctypes as ct
from pathlib import Path
import torch
from .streamk_build import build_shared
from .._core import ROOT, FLAGS
BUILD_ROOT = ROOT / "runs"
from .streamk_schedule import schedule
from .gemm import Library as Gemm
from .dgrad import Library as Dgrad
from .swiglu import MXFP8SwiGLULibrary as SwiGLU
from .swiglu_dgrad import SwiGLUDgradLibrary as SWD

OUTPUT=BUILD_ROOT/'streamk_ffn'
FAMILIES=dict(gemm=(Gemm,70,128,128,1),dgrad=(Dgrad,12,128,128,1),
              swiglu=(SwiGLU,1,64,64,2),swd=(SWD,5,64,64,1))


def build():
    from .streamk_wgrad import build as build_dw
    build_dw()
    for family in FAMILIES:
        for atomic in (False,True):
            build_shared(ROOT/f'cuda/streamk_{family}.cu',
                OUTPUT/family/('atomic' if atomic else 'multi')/'libstreamk.so',
                flags=[*FLAGS,f'-DSTREAMK_ATOMIC={int(atomic)}'])



class Library:
    def __init__(self,family='gemm',*,reduction='atomic',output=OUTPUT):
        if family not in FAMILIES:raise ValueError('Unknown Stream-K family')
        if reduction not in ('atomic','deterministic'):raise ValueError('Unknown reduction mode')
        from ..runtime import ensure_extensions
        ensure_extensions("mlp", "streamk")
        self.family=family;self.atomic=reduction=='atomic'
        cls,self.config,self.cols,self.block,self.components=FAMILIES[family]
        self.baseline=cls()
        self.lib=ct.CDLL(str(Path(output)/family/('atomic' if self.atomic else 'multi')/'libstreamk.so'))
        self.lib.tk_ffsk_error.restype=ct.c_char_p
        self.lib.tk_ffsk_create.argtypes=[ct.POINTER(ct.c_void_p)]+[ct.c_int]*6
        self.lib.tk_ffsk_create.restype=ct.c_void_p
        self.lib.tk_ffsk_launch.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
        self.lib.tk_ffsk_launch.restype=ct.c_int
        self.lib.tk_ffsk_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_ffsk_destroy.argtypes=[ct.c_void_p]

    def plan(self,*inputs,out=None,gate_out=None,up_out=None,sms=0,raster_group=13,tail_waves=None,policy='auto'):
        """Auto keeps persistent GEMMs when their equal-tile wave efficiency is >=92%.

        Use policy='streamk' to force splitting, or 'persistent' as a control.
        Both reduction modes preserve the original single-CTA path in auto.
        """
        if policy not in ('auto','streamk','persistent'):raise ValueError('Unknown scheduling policy')
        if tail_waves is None:tail_waves=0 if self.atomic else 1
        return Plan(self,inputs,out,gate_out,up_out,sms,raster_group,tail_waves,policy)


class Plan:
    """Fixed-shape single-stream plan, including workspace clear in each replay.

    Keep the plan alive until graphs using it finish. Rebind is host-side and
    changes subsequent captures/eager calls, not pointers in an existing graph.
    """
    def __init__(self,owner,inputs,out,gate,up,sms,group,tail_waves,policy):
        self.handle=None;self.owner=owner
        self.signature=[(t.shape,t.dtype,t.device) for t in inputs]
        kwargs=dict(out=out,config=owner.config,sms=sms,raster_group=group)
        if owner.family=='swiglu':
            if (gate is None)!=(up is None):raise ValueError('Supply both auxiliary outputs')
            shape=(inputs[0].shape[0],inputs[2].shape[0])
            if gate is None:
                gate=torch.empty(shape,device=inputs[0].device,dtype=torch.bfloat16)
                up=torch.empty_like(gate)
            kwargs.update(gate_out=gate,up_out=up)
        elif gate is not None or up is not None:raise ValueError('Auxiliary outputs require SwiGLU')
        self.checked=owner.baseline.plan(*inputs,**kwargs)
        self.out=self.checked.out;self.gate_out=gate;self.up_out=up
        m,n=self.out.shape;k=inputs[0].shape[1]
        if k%owner.block or n%owner.cols:raise ValueError('Stream-K requires complete reduction/column panels')
        self.shape=(m,n,k)
        cap=torch.cuda.get_device_properties(inputs[0].device).multi_processor_count
        tiles=((m+127)//128)*(n//owner.cols)
        grid=min(sms or cap,cap,tiles)
        efficiency=tiles/(grid*((tiles+grid-1)//grid))
        self.fallback=policy=='persistent' or (policy=='auto' and efficiency>=.92)
        if self.fallback:
            self.info=dict(self.checked.info,family=owner.family,backend='persistent',atomic=owner.atomic,
                           workspace_bytes=0,workspace_slots=0,partial_slots=0,fixup_tiles=0)
            self.workspace=torch.empty(0,device=inputs[0].device)
            self.refs=list(inputs);self.metadata=[]
            return
        self.schedule=schedule(m,n,k,min(sms or cap,cap),group,tail_waves,
                               tile_cols=owner.cols,block_k=owner.block)
        s=self.schedule;self.info=s['info']
        slots=self.info['partial_slots']
        if owner.atomic:
            dest={(f[0],f[1]):i for i,f in enumerate(s['fixups'])}
            for w in s['work']:
                if w[4]>=0:w[4]=dest[(w[0],w[1])]
            slots=len(dest)
        self.info.update(family=owner.family,backend='streamk',atomic=owner.atomic,workspace_slots=slots,
                         workspace_bytes=max(1,slots)*owner.components*128*owner.cols*4)
        self.workspace=torch.empty((max(1,slots),owner.components,128,owner.cols),
                                   device=inputs[0].device,dtype=torch.float32)
        self.metadata=[torch.tensor(s[key] or [0],device=inputs[0].device,dtype=torch.int32)
                       for key in ('work','offsets','fixups','contributors')]
        self._bind_native(inputs)

    def _bind_native(self,inputs):
        refs=[*inputs,*([None]*(8-len(inputs))),self.out,self.gate_out,self.up_out,
              self.workspace,*self.metadata]
        ptrs=(ct.c_void_p*16)(*[0 if t is None else t.data_ptr() for t in refs])
        with torch.cuda.device(self.out.device):
            handle=self.owner.lib.tk_ffsk_create(ptrs,*self.shape,self.info['grid'],
                self.info['workspace_slots'],self.info['fixup_tiles'])
        if not handle:raise RuntimeError(self.owner.lib.tk_ffsk_error().decode())
        if self.handle:self.owner.lib.tk_ffsk_destroy(self.handle)
        self.handle=handle;self.refs=refs
        info=(ct.c_int*2)();self.owner.lib.tk_ffsk_info(handle,info)
        self.info.update(shared_bytes=info[0],initial_regs=info[1])

    def __call__(self,phase=0):
        if phase not in (0,1,2,3):raise ValueError('phase: 0=all, 1=clear+main, 2=fixup, 3=clear')
        if self.out is None:raise RuntimeError('Rebind a released plan before launch')
        if self.fallback:
            if phase in (0,1):self.checked()
            return self.out
        with torch.cuda.device(self.out.device):
            status=self.owner.lib.tk_ffsk_launch(self.handle,torch.cuda.current_stream().cuda_stream,phase)
        if status:raise RuntimeError(self.owner.lib.tk_ffsk_error().decode())
        stream = torch.cuda.current_stream(self.out.device)
        for tensor in self.refs:
            if isinstance(tensor, torch.Tensor): tensor.record_stream(stream)
        return self.out

    def rebind(self,*inputs,out=None,gate_out=None,up_out=None):
        if [(t.shape,t.dtype,t.device) for t in inputs]!=self.signature:
            raise ValueError('Rebind must preserve input shapes, dtypes and device')
        kwargs=dict(out=self.out if out is None else out)
        if self.owner.family=='swiglu':
            kwargs.update(gate_out=self.gate_out if gate_out is None else gate_out,
                          up_out=self.up_out if up_out is None else up_out)
        with torch.cuda.device(inputs[0].device):
            self.checked.rebind(*inputs,**kwargs)
        self.out=self.checked.out
        if self.owner.family=='swiglu':self.gate_out,self.up_out=self.checked.gate_out,self.checked.up_out
        if not self.fallback:self._bind_native(inputs)
        else:self.refs=list(inputs)
        return self

    def release(self):
        self.refs=[];self.out=None
        if hasattr(self.checked,'release'):self.checked.release()
        else:self.checked.release_inputs()

    def release_inputs(self):
        """Drop forward-only inputs while preserving the reusable GEMM output."""
        self.refs=[]
        if hasattr(self.checked,'release_inputs'):
            self.checked.release_inputs()
        else:
            self.checked.release()

    def __del__(self):
        if self.handle:self.owner.lib.tk_ffsk_destroy(self.handle)
