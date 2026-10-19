"""Same-process A/B against a git revision, including gradients and graph timing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import triton
from fp8_swiglu import FP8SwiGLU
from profile_swiglu_backward import benchmark
from benchmark_vs_fla import DEFAULT_CASES


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='HEAD')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/swiglu_optimization_ab.json')
    parser.add_argument('--rep', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    commit = subprocess.check_output(['git','rev-parse',args.baseline_ref],cwd=ROOT,text=True).strip()
    directory = ROOT/'runs/swiglu_ab_baseline'/commit
    directory.mkdir(parents=True, exist_ok=True)
    for source,target in [('mxtest.py','mxtest_baseline.py'),('fp8_swiglu.py','fp8_swiglu_baseline.py')]:
        content = subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=ROOT,text=True)
        if source == 'fp8_swiglu.py':
            content = content.replace('import mxtest\n','import mxtest_baseline as mxtest\n')
            content = content.replace('fp8gemm::','fp8gemm_ab_baseline::')
        (directory/target).write_text(content)
    import_file('mxtest_baseline',directory/'mxtest_baseline.py')
    baseline = import_file('fp8_swiglu_baseline',directory/'fp8_swiglu_baseline.py')
    result = {'baseline_commit':commit, 'gpu':torch.cuda.get_device_name(),
              'torch':torch.__version__,'triton':triton.__version__,
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ('mxtest.py','fp8_swiglu.py')},
              'method':'Identical inputs and FP32 weights; bias=False; alternating provider order; CUDA graph with 16 calls/replay; median of round medians; no optimizer or weight refresh',
              'rep_ms':args.rep,'repeats':args.repeats,'cases':[]}
    for m,k,h in DEFAULT_CASES:
        torch.manual_seed(23)
        layers={'baseline':baseline.FP8SwiGLU(k,h,k,bias=False,device='cuda'),
                'optimized':FP8SwiGLU(k,h,k,bias=False,device='cuda')}
        layers['optimized'].load_state_dict(layers['baseline'].state_dict())
        layers['optimized'].refresh_fp8_weights()
        inputs={name:torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True) for name in layers}
        with torch.no_grad():
            inputs['optimized'].copy_(inputs['baseline'])
        dy=torch.randn_like(inputs['baseline'])
        tensors={}
        for name,layer in layers.items():
            y=layer(inputs[name])
            grads=torch.autograd.grad(y,(inputs[name],*layer.parameters()),dy)
            tensors[name]=(y.detach(),*grads)
            del y,grads
        errors={}
        for label,a,b in zip(('output','dx','dgate','dup','ddown'),tensors['optimized'],tensors['baseline']):
            errors[label]={'relative_l2':((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item(),
                           'max_abs':(a.float()-b.float()).abs().max().item()}
            torch.testing.assert_close(a,b,rtol=0.016,atol=0.002)
        del tensors
        timings={name:{phase:[] for phase in ('forward','backward','forward_backward')} for name in layers}
        for repeat in range(args.repeats):
            names=list(layers) if repeat%2==0 else list(reversed(layers))
            for name in names:
                layer,x=layers[name],inputs[name]
                targets=(x,*layer.parameters())
                def forward():
                    with torch.no_grad():
                        return layer(x)
                def train():
                    return torch.autograd.grad(layer(x),targets,dy)
                for phase,fn in [('forward',forward),('forward_backward',train)]:
                    timings[name][phase].append(benchmark(fn,args.rep,1)['median_ms'])
                holder=[]
                def setup():
                    holder.clear()
                    holder.append(layer(x))
                def backward():
                    return torch.autograd.grad(holder[0],targets,dy,retain_graph=True)
                timings[name]['backward'].append(benchmark(backward,args.rep,1,setup)['median_ms'])
                holder.clear()
        summary={name:{phase:{'median_ms':statistics.median(v),'round_ms':v} for phase,v in phases.items()}
                 for name,phases in timings.items()}
        speedups={phase:summary['baseline'][phase]['median_ms']/summary['optimized'][phase]['median_ms']
                  for phase in summary['baseline']}
        case={'shape':[m,k,h],'timings':summary,'speedups':speedups,'errors_vs_baseline':errors}
        result['cases'].append(case)
        print(json.dumps(case),flush=True)
        args.output.parent.mkdir(exist_ok=True,parents=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        del layer,layers,inputs,x,targets
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
