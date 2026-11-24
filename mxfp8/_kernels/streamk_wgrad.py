"""Config-12 dW hybrid Stream-K: immutable host schedule and ordered fixup."""
import ctypes as ct
from pathlib import Path

import torch
from .streamk_build import build_shared
from .._core import ROOT, FLAGS
BUILD_ROOT = ROOT / "runs"
from .streamk_schedule import schedule
from .wgrad import Library as BaselineLibrary

OUTPUT = BUILD_ROOT/'streamk_wgrad/multi'


def build():
    build_shared(ROOT/'cuda/streamk_wgrad.cu', OUTPUT/'libstreamk.so',
                 flags=[*FLAGS, f'-I{ROOT / "cuda"}'])
    build_shared(ROOT/'cuda/streamk_wgrad.cu', OUTPUT.parent/'atomic/libstreamk.so',
                 flags=[*FLAGS, f'-I{ROOT / "cuda"}', '-DSTREAMK_ATOMIC=1'])




class Library:
    def __init__(self, output=None, *, atomic=True, reduction=None):
        if reduction is not None:
            if reduction not in ('atomic','deterministic'):raise ValueError('Unknown reduction mode')
            atomic=reduction=='atomic'
        from ..runtime import ensure_extensions
        ensure_extensions("mlp", "streamk")
        self.atomic=atomic
        output=output or (OUTPUT.parent/'atomic' if atomic else OUTPUT)
        self.lib=ct.CDLL(str(Path(output)/'libstreamk.so'))
        self.lib.tk_sk_error.restype=ct.c_char_p
        self.lib.tk_sk_create.argtypes=[ct.c_void_p]*10+[ct.c_int]*6
        self.lib.tk_sk_create.restype=ct.c_void_p
        self.lib.tk_sk_launch.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
        self.lib.tk_sk_launch.restype=ct.c_int
        self.lib.tk_sk_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.tk_sk_destroy.argtypes=[ct.c_void_p]
        self.baseline=BaselineLibrary()

    def plan(self,a,b,as_,bs,*,out=None,raster_group=13,sms=0,tail_waves=None):
        if tail_waves is None:tail_waves=0 if self.atomic else 1
        return Plan(self,a,b,as_,bs,raster_group,sms,tail_waves,out)


class Plan:
    """Fixed-buffer, single-stream plan; keep alive through graph execution.

    Refresh input contents between ordered launches. Workspace/output belong to
    the plan and must not be shared by simultaneous executions on other streams.
    Metadata and allocation are prepared on the host before graph capture.
    """
    def __init__(self,owner,a,b,as_,bs,group,sms,tail_waves,out=None):
        self.handle=None
        # Reuse the production shape/device/alias contract, without launching it.
        checked=owner.baseline.plan(a,b,as_,bs,config=12,raster_group=group,sms=sms,out=out)
        self.checked=checked
        self.signature=[(t.shape,t.dtype,t.device) for t in (a,b,as_,bs)]
        self.owner=owner;self.out=checked.out
        self.inputs=[a,b,as_,bs]
        n,m=a.shape;k=b.shape[1]
        prop=torch.cuda.get_device_properties(a.device)
        sched=schedule(n,k,m,min(sms or prop.multi_processor_count,prop.multi_processor_count),
                       group,tail_waves)
        self.schedule=sched;self.info=sched['info']
        slots=self.info['partial_slots']
        if owner.atomic:
            # One shared FP32 destination per split output tile, all contributors
            # use TMA reduce-add. Keep exactly the same CTA/K work distribution.
            destinations={(f[0],f[1]):i for i,f in enumerate(sched['fixups'])}
            for w in sched['work']:
                if w[4]>=0:w[4]=destinations[(w[0],w[1])]
            slots=len(destinations)
        self.info.update(family='wgrad',backend='streamk',atomic=owner.atomic,workspace_slots=slots,
                         workspace_bytes=max(1,slots)*128*128*4)
        self.workspace=torch.empty((max(1,slots),128,128),
                                   device=a.device,dtype=torch.float32)
        # All allocations/uploads happen before capture. No counters or clears
        # are needed for multi-slot; atomic launch includes a stream-ordered clear.
        self.metadata=[torch.tensor(sched[key] or [0],device=a.device,dtype=torch.int32)
                       for key in ('work','offsets','fixups','contributors')]
        self.shape=(n,k,m)
        self._bind_native()

    def _bind_native(self):
        owner=self.owner
        self.refs=[*self.inputs,self.out,self.workspace,*self.metadata]
        with torch.cuda.device(self.out.device):
            handle=owner.lib.tk_sk_create(*[t.data_ptr() for t in self.refs],
                *self.shape,self.info['grid'],self.info['workspace_slots'],self.info['fixup_tiles'])
        if not handle:
            raise RuntimeError(owner.lib.tk_sk_error().decode())
        if self.handle:owner.lib.tk_sk_destroy(self.handle)
        self.handle=handle
        info=(ct.c_int*2)();owner.lib.tk_sk_info(self.handle,info)
        self.info.update(shared_bytes=info[0],initial_regs=info[1],shape_nkm=list(self.shape))

    def rebind(self,a,b,as_,bs,*,out=None):
        inputs=[a,b,as_,bs]
        if [(t.shape,t.dtype,t.device) for t in inputs]!=self.signature:
            raise ValueError('Rebind must preserve shape, dtype and device')
        with torch.cuda.device(a.device):
            self.checked.rebind(a,b,as_,bs,out=self.out if out is None else out)
        self.out=self.checked.out;self.inputs=inputs;self._bind_native()
        return self

    def release(self):
        self.checked.release();self.inputs=[];self.refs=[];self.out=None

    def __call__(self,phase=0):
        if self.out is None:raise RuntimeError('Rebind a released plan before launch')
        if phase not in (0,1,2,3):
            raise ValueError('phase: 0=all, 1=clear+main, 2=fixup, 3=clear only')
        with torch.cuda.device(self.out.device):
            status=self.owner.lib.tk_sk_launch(self.handle,torch.cuda.current_stream().cuda_stream,phase)
        if status:
            raise RuntimeError(self.owner.lib.tk_sk_error().decode())
        stream = torch.cuda.current_stream(self.out.device)
        for tensor in self.refs: tensor.record_stream(stream)
        return self.out

    def __del__(self):
        if self.handle:
            self.owner.lib.tk_sk_destroy(self.handle)
