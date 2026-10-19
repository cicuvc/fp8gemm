"""Separate extra FP8 dW quantization error from the existing FP8 MLP error."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import triton
import mxtest
from fp8_swiglu import FP8SwiGLU, _fp8_swiglu_fwd
from benchmark_vs_fla import DEFAULT_CASES, TorchSwiGLU, params
from profile_swiglu_backward import benchmark


def metrics(actual, reference):
    a, b = actual.float(), reference.float()
    delta = a-b
    norm_a, norm_b = a.norm(), b.norm()
    return {'relative_l2':(delta.norm()/norm_b.clamp_min(1e-30)).item(),
            'cosine':((a*b).sum()/(norm_a*norm_b).clamp_min(1e-30)).item(),
            'max_abs':delta.abs().max().item(),
            'reference_rms':b.square().mean().sqrt().item()}


def run_case(shape, seeds, rep):
    m,k,h=shape
    measurements=[]
    times=None
    for seed in seeds:
        torch.manual_seed(seed)
        layer=FP8SwiGLU(k,h,k,bias=False,device='cuda')
        reference=TorchSwiGLU(k,h)
        with torch.no_grad():
            for weight,ref in zip(params(layer),params(reference)):
                weight.copy_(ref)
        layer.refresh_fp8_weights()
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn_like(x)
        gw,gs=layer.quantized_gate_weight
        uw,us=layer.quantized_up_weight
        dw,ds=layer.quantized_down_weight
        with torch.no_grad():
            y,gate,up,hidden=_fp8_swiglu_fwd(x,*params(layer),gw,gs,uw,us,dw,ds,None,False)
        inputs=(dy,hidden,gate,up,dw,ds,gw,gs,uw,us,x)
        bf16_dw=mxtest.launch_fp8_down_swiglu_backward(*inputs,use_fp8_wgrad=False,return_intermediate_grads=True)
        fp8_dw=mxtest.launch_fp8_down_swiglu_backward(*inputs,use_fp8_wgrad=True)
        torch.testing.assert_close(fp8_dw[0],bf16_dw[0],rtol=0,atol=0)
        ref_y=reference(x)
        ref_grads=torch.autograd.grad(ref_y,(x,)+params(reference),dy)
        entry={'seed':seed,'output_error_vs_bf16':metrics(y,ref_y),'gradients':{}}
        for name,old,new,ref in zip(('dx','dgate','dup','ddown'),bf16_dw[:4],fp8_dw,ref_grads):
            entry['gradients'][name]={
                'additional_error_vs_bf16_wgrad':metrics(new,old),
                'bf16_wgrad_error_vs_bf16_mlp':metrics(old,ref),
                'fp8_wgrad_error_vs_bf16_mlp':metrics(new,ref),
            }
        measurements.append(entry)
        print(shape,'seed',seed,{name:{'extra_l2':v['additional_error_vs_bf16_wgrad']['relative_l2'],
                                      'total_l2':v['fp8_wgrad_error_vs_bf16_mlp']['relative_l2']}
                                for name,v in entry['gradients'].items()},flush=True)
        if times is None:
            dg,du=bf16_dw[-2:]
            with torch.no_grad():
                functions={
                    'gate_up_dw_bf16':lambda:mxtest.launch_bf16_weight_grad_dual(dg,du,x),
                    'gate_up_dw_fp8':lambda:mxtest.launch_fp8gemm_weight_grad_dual(dg,du,x),
                    'down_dw_bf16':lambda:mxtest.launch_bf16_weight_grad(dy,hidden),
                    'down_dw_fp8':lambda:mxtest.launch_fp8gemm_weight_grad(dy,hidden,out_dtype=torch.float32),
                    'backward_bf16_wgrad':lambda:mxtest.launch_fp8_down_swiglu_backward(*inputs,use_fp8_wgrad=False),
                    'backward_fp8_wgrad':lambda:mxtest.launch_fp8_down_swiglu_backward(*inputs,use_fp8_wgrad=True),
                }
                times={name:benchmark(fn,rep,3) for name,fn in functions.items()}
            print('TIMING',shape,{n:round(v['median_ms']*1000,2) for n,v in times.items()},flush=True)
        del layer,reference,bf16_dw,fp8_dw,ref_grads,ref_y,gate,up,hidden,y
    return {'shape':shape,'measurements':measurements,'graph_timings':times}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[23,24,25])
    parser.add_argument('--rep',type=int,default=30)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/fp8_wgrad_accuracy.json')
    args=parser.parse_args()
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':triton.__version__,
            'seeds':args.seeds,'rep_ms':args.rep,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('mxtest.py','fp8_swiglu.py')},
            'method':'Gaussian BF16 x and dy; identical BF16-representable weights; compare current FP8 forward/dX with BF16 vs FP8 dW using exactly the same intermediates; also compare with full BF16 torch MLP. No optimizer/training trajectory. Graph timing includes online quantization; no bias.',
            'cases':[]}
    for shape in DEFAULT_CASES:
        result['cases'].append(run_case(shape,args.seeds,args.rep))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
