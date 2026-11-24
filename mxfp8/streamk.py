"""Opt-in FFN Stream-K execution for ordered, single-stream model execution.

Preserves Parameter leaves, autograd functions, quantizers and optimizer wiring.
The fixed shared scratch must not be used by concurrent model executions.
"""
import types
import torch
from ._impl.swiglu import MXFP8SwiGLU
from ._kernels.streamk import Library
from ._kernels.streamk_wgrad import Library as Wgrad


def configure_ffn_streamk(model, reduction='atomic', *, down_forward='persistent'):
    if reduction not in ('atomic','deterministic'):
        raise ValueError('Unknown FFN Stream-K reduction')
    if down_forward not in ('streamk','persistent'):
        raise ValueError('Unknown FFN down forward backend')
    previous=getattr(model,'_mxfp8_streamk_runtime',None)
    if previous is not None:
        if previous['reduction']!=reduction or previous.get('down_forward','streamk')!=down_forward:
            raise ValueError('Cannot change FFN routing after installation')
        return previous
    layers=[m for m in model.modules() if isinstance(m,MXFP8SwiGLU)]
    if not layers:raise ValueError('No MXFP8 FFN layers')
    if any(getattr(m,"activation_checkpointing",False) for m in layers):
        raise ValueError('FFN Stream-K requires no activation checkpointing')
    if any(m._forward_states or m._backward_states or m.down_proj._forward_states or m.down_proj._backward_states for m in layers):
        raise ValueError('Configure FFN Stream-K before the first model forward')
    device=layers[0].gate_weight.device
    if device.type!='cuda' or any(m.gate_weight.device!=device for m in layers):
        raise ValueError('FFN Stream-K requires one CUDA device per model')
    sms=torch.cuda.get_device_properties(device).multi_processor_count
    # Schedule invariants bound partial slots by twice the CTA count. All
    # supported FFN tiles have <=128*128 FP32 elements including components.
    scratch=torch.empty(2*sms*128*128,device=device,dtype=torch.float32)
    libraries={f:Library(f,reduction=reduction) for f in ('gemm','dgrad','swiglu','swd')
               if f!='gemm' or down_forward=='streamk'}
    dw=Wgrad(reduction=reduction)
    report=dict(reduction=reduction,down_forward=down_forward,layers=len(layers),scratch_capacity_bytes=scratch.numel()*4,
                scratch_used_bytes=0,plans=[],plans_created=0)
    recorded=set()

    def convert(old,family):
        if family=='gemm' and down_forward=='persistent':
            key=(family,tuple(old.out.shape),tuple(old.refs[0].shape),'persistent')
            if key not in recorded and len(recorded)<128:
                recorded.add(key)
                report['plans'].append(dict(family=family,shape=list(old.out.shape),
                    original=dict(old.info),selected=dict(old.info,backend='persistent',
                        reason='explicit_down_forward_policy')))
            return old
        if family=='swd':
            inputs=[getattr(old,k) for k in ('gate','up','weight_gate','weight_up',
                    'gate_scale','up_scale','weight_gate_scale','weight_up_scale')]
        elif family=='swiglu':
            inputs=[getattr(old,k) for k in ('x','x_scale','gate','gate_scale','up','up_scale')]
        else:inputs=old.refs[:4]
        cols = 128 if family in ('gemm', 'dgrad', 'wgrad') else 64
        block = 128 if family in ('gemm', 'dgrad', 'wgrad') else 64
        if old.out.shape[1] % cols or inputs[0].shape[1] % block:
            # Stream-K has fixed tiles; retain supported original tail shapes.
            report['plans'].append(dict(family=family, shape=list(old.out.shape),
                original=dict(old.info), selected=dict(old.info, backend='persistent',
                    reason='unsupported_streamk_panels')))
            return old
        kw=dict(out=old.out,raster_group=old.info['raster_group'])
        if family=='swiglu':kw.update(gate_out=old.gate_out,up_out=old.up_out)
        p=(dw if family=='wgrad' else libraries[family]).plan(*inputs,**kw)
        report['plans_created']+=1
        key=(family,tuple(p.out.shape),tuple(inputs[0].shape),p.info['backend'])
        # Eval may create transient plans for every call. Keep a bounded shape
        # manifest, not an ever-growing list of discarded tensor plans.
        if key not in recorded and len(recorded)<128:
            recorded.add(key)
            selected={k:v for k,v in p.info.items() if k!='cta_steps'}
            report['plans'].append(dict(family=family,shape=list(p.out.shape),original=dict(old.info),selected=selected))
        if getattr(p,'fallback',False):
            return old
        n=p.workspace.numel()
        if n>scratch.numel():raise RuntimeError('Stream-K scratch bound exceeded')
        p.workspace=scratch[:n].view(p.workspace.shape)
        report['scratch_used_bytes']=max(report['scratch_used_bytes'],n*4)
        if family=='wgrad':p._bind_native()
        else:p._bind_native(inputs)
        return p

    def hook(module,method,fields):
        original=getattr(module,method)
        def wrapped(self,*args,**kwargs):
            state=original(*args,**kwargs)
            if not getattr(state,'_streamk_execution',False):
                for attr,family in fields:
                    setattr(state,attr,convert(getattr(state,attr),family))
                state._streamk_execution=True
            return state
        setattr(module,method,types.MethodType(wrapped,module))

    for layer in layers:
        hook(layer,'_forward_state',[('forward','swiglu')])
        hook(layer,'_backward_state',[('dx_plan','swd'),('gate_weight_grad','wgrad'),('up_weight_grad','wgrad')])
        hook(layer.down_proj,'_forward_state',[('gemm','gemm')])
        hook(layer.down_proj,'_backward_state',[('dx','dgrad'),('dw','wgrad')])
    # Closures retain scratch/libraries; the caller records the mutable route
    # manifest after warmup has populated the original shape caches.
    model._mxfp8_streamk_runtime=report
    return report
