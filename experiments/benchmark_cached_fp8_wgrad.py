"""Includes forward cache construction and all backward packing costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import triton
import mxtest
from fp8_swiglu import FP8SwiGLU
from experiments.cached_fp8_wgrad import CachedWgradSwiGLU,pack_columns,packed_wgrad
from experiments.evaluate_fp8_wgrad import metrics
from benchmark_vs_fla import DEFAULT_CASES,params,TorchSwiGLU,GatedMLP
from profile_swiglu_backward import benchmark


def run_case(m,k,h,args):
    torch.manual_seed(23)
    layers={
        'bf16_dw':FP8SwiGLU(k,h,k,bias=False,device='cuda'),
        'online_fp8_dw':FP8SwiGLU(k,h,k,bias=False,fp8_wgrad=True,device='cuda'),
        'cached_fp8_dw':CachedWgradSwiGLU(k,h,k,bias=False,device='cuda'),
    }
    if args.fusions:
        layers.update({
            'cached_fused_hidden':CachedWgradSwiGLU(k,h,k,bias=False,device='cuda',fuse_hidden_pack=True),
            'cached_fused_grads':CachedWgradSwiGLU(k,h,k,bias=False,device='cuda',fuse_grad_pack=True),
            'cached_fused_activations':CachedWgradSwiGLU(k,h,k,bias=False,device='cuda',fuse_hidden_pack=True,fuse_input_pack=True),
            'cached_fused_both':CachedWgradSwiGLU(k,h,k,bias=False,device='cuda',fuse_hidden_pack=True,fuse_grad_pack=True),
        })
    for name,layer in layers.items():
        if name!='bf16_dw':
            layer.load_state_dict(layers['bf16_dw'].state_dict())
            layer.refresh_fp8_weights()
    if args.baselines:
        layers['fla_bf16']=GatedMLP(hidden_size=k,intermediate_size=h,
                                  hidden_act='swish',fuse_swiglu=True).cuda().bfloat16()
        layers['torch_bf16']=TorchSwiGLU(k,h)
        with torch.no_grad():
            for name in ('fla_bf16','torch_bf16'):
                for dest,source_weight in zip(params(layers[name]),params(layers['bf16_dw'])):
                    dest.copy_(source_weight)
    source=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    inputs={name:source.clone().requires_grad_() for name in layers}
    dy=torch.randn_like(source)
    outputs={}
    for name,layer in layers.items():
        x=inputs[name]
        y=layer(x)
        grads=torch.autograd.grad(y,(x,)+params(layer),dy)
        outputs[name]=(y.detach(),*grads)
        del y,grads
    errors={}
    for name in layers:
        if not name.startswith('cached'):
            continue
        errors[name]={}
        for ref in ('online_fp8_dw','bf16_dw'):
            errors[name][ref]={label:metrics(a,b) for label,a,b in zip(
                ('output','dx','dgate','dup','ddown'),outputs[name],outputs[ref])}
        for a,b in zip(outputs[name],outputs['online_fp8_dw']):
            torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-4)
    del outputs
    times={name:{p:[] for p in ('training_forward','backward','forward_backward')} for name in layers}
    for repeat in range(args.repeats):
        names=list(layers)
        names=names[repeat%len(names):]+names[:repeat%len(names)]
        for name in names:
            layer,x=layers[name],inputs[name]
            targets=(x,)+params(layer)
            def forward():
                return layer(x)
            def train():
                return torch.autograd.grad(layer(x),targets,dy)
            times[name]['training_forward'].append(benchmark(forward,args.rep,1)['median_ms'])
            times[name]['forward_backward'].append(benchmark(train,args.rep,1)['median_ms'])
            holder=[]
            def setup():
                holder.clear()
                holder.append(layer(x))
            def backward():
                return torch.autograd.grad(holder[0],targets,dy,retain_graph=True)
            times[name]['backward'].append(benchmark(backward,args.rep,1,setup)['median_ms'])
            holder.clear()
    times={name:{p:{'median_ms':statistics.median(v),'round_ms':v} for p,v in t.items()} for name,t in times.items()}
    result={'shape':[m,k,h],'errors_vs':errors,'timings':times}
    print((m,k,h),{name:{p:round(v['median_ms']*1000,2) for p,v in t.items()} for name,t in times.items()},flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusions',action='store_true')
    parser.add_argument('--baselines',action='store_true',help='Include FLA and torch BF16 with the same timing method')
    parser.add_argument('--rep',type=int,default=30)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--case',type=int,choices=(0,1,2))
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/cached_fp8_wgrad.json')
    args=parser.parse_args()
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':triton.__version__,
            'baseline_commit':os.environ.get('FP8GEMM_BASELINE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'settings':{'rep_ms':args.rep,'repeats':args.repeats,'fusions':args.fusions,'baselines':args.baselines},
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('experiments/cached_fp8_wgrad.py','experiments/cached_fp8_wgrad_fused.py')},
            'method':'CUDA graph 16 calls/replay; alternating providers, training forward includes activation packing; backward includes all gradient packing; no optimizer. Identical FP32 master weights for FP8 variants; BF16 baselines use rounded copies and BF16 dW. BF16 inputs; no bias.',
            'cases':[]}
    for shape in (DEFAULT_CASES if args.case is None else [DEFAULT_CASES[args.case]]):
        result['cases'].append(run_case(*shape,args))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
