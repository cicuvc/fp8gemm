"""Compare backward prequantization and tune BF16 wgrad with fixed inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton

import mxtest
from experiments import swiglu_prequant
from fp8_swiglu import FP8SwiGLU, _fp8_swiglu_fwd
from benchmark_vs_fla import DEFAULT_CASES, params
from profile_swiglu_backward import benchmark


def run_case(m, k, h, args):
    torch.manual_seed(23)
    layer = FP8SwiGLU(k, h, k, bias=False, device='cuda')
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gw, gs = layer.quantized_gate_weight
    uw, us = layer.quantized_up_weight
    dw, ds = layer.quantized_down_weight
    with torch.no_grad():
        _, gate, up, hidden = _fp8_swiglu_fwd(x, *params(layer), gw, gs, uw, us, dw, ds, None, False)
    inputs = (dy, hidden, gate, up, dw, ds, gw, gs, uw, us, x)
    old = swiglu_prequant.launch_fp8_down_swiglu_backward(*inputs, return_intermediate_grads=True, prequantize=False)
    new = swiglu_prequant.launch_fp8_down_swiglu_backward(*inputs, return_intermediate_grads=True, prequantize=True)
    errors = []
    for a,b in zip(new,old):
        errors.append({'relative_l2': ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item(),
                       'max_abs': (a.float()-b.float()).abs().max().item()})
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    timings = {}
    for mode in (False,True):
        fn = lambda: swiglu_prequant.launch_fp8_down_swiglu_backward(*inputs, prequantize=mode)
        timings[f'prequantize_{mode}'] = benchmark(fn,args.rep,3)
        print((m,k,h), mode, timings[f'prequantize_{mode}'],flush=True)
    result = {'shape':[m,k,h], 'errors_vs_online':errors,'timings':timings}
    if args.tune_dx:
        dg, du = old[-2:]
        dx = torch.empty_like(x)
        dq, dqs = mxtest.launch_row_quantize(dg)
        uq, uqs = mxtest.launch_row_quantize(du)
        configs = list(mxtest.DUAL_GEMM_CONFIGS)
        for bm,bn,bk in ((32,128,128),(32,256,64),(64,128,64),(64,128,128),
                          (64,256,64),(128,64,64),(128,128,64),(128,64,128)):
            for warps in (4,8):
                for stages in (2,3):
                    config=triton.Config({'BLOCK_M':bm,'BLOCK_N':bn,'BLOCK_K':bk},num_warps=warps,num_stages=stages)
                    if str(config) not in {str(c) for c in configs}:
                        configs.append(config)
        result['dx_tuning']={}
        for prequantized in (False,True):
            entries=[]
            for index,config in enumerate(configs):
                if index % 12 == 0:
                    print((m,k,h),'dx',prequantized,f'config {index}/{len(configs)}',flush=True)
                bm,bn,bk=(config.kwargs[v] for v in ('BLOCK_M','BLOCK_N','BLOCK_K'))
                def fn():
                    return swiglu_prequant.fp8_swiglu_bwd_dx.fn[(triton.cdiv(m,bm)*triton.cdiv(k,bk),)](
                        dq if prequantized else dg,uq if prequantized else du,gw,gs,uw,us,dx,
                        m,h,k,dg.stride(0),du.stride(0),gw.stride(0),gs.stride(0),uw.stride(0),us.stride(0),dx.stride(0),
                        PREQUANTIZED=prequantized,dg_scale=dqs if prequantized else None,du_scale=uqs if prequantized else None,
                        **config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages)
                try:
                    kernel=fn()
                    # Different reduction tiles may change final BF16 rounding.
                    err=((dx.float()-old[0].float()).norm()/old[0].float().norm()).item()
                    torch.testing.assert_close(dx,old[0],rtol=0.016,atol=0.002)
                    times=benchmark(fn,10,1)
                    entries.append({'config':str(config),'kwargs':config.kwargs,'warps':config.num_warps,
                                    'stages':config.num_stages,'ms':times['median_ms'],
                                    'n_regs':kernel.n_regs,'n_spills':kernel.n_spills,'relative_l2':err})
                except triton.OutOfResources as exc:
                    entries.append({'config':str(config),'error':str(exc)})
            result['dx_tuning'][str(prequantized)]=sorted(entries,key=lambda r:r.get('ms',float('inf')))
            print((m,k,h),'dx',prequantized,'BEST',result['dx_tuning'][str(prequantized)][:4],flush=True)
    if args.tune:
        dg,du=old[-2:]
        wg,wu=torch.empty_like(layer.gate_weight),torch.empty_like(layer.up_weight)
        wd=torch.empty_like(layer.down_proj.master_weight)
        configs = list(mxtest.BF16_WGRAD_CONFIGS)
        for bm in (32,64,128):
            for bn,bk in ((64,64),(64,128),(128,64),(128,128)):
                for warps in (4,8):
                    for stages in (2,3,4):
                        config=triton.Config({'BLOCK_M':bm,'BLOCK_N':bn,'BLOCK_K':bk},num_warps=warps,num_stages=stages)
                        if str(config) not in {str(c) for c in configs}:
                            configs.append(config)
        tuned = {}
        for name in ('dual','down'):
            entries=[]
            for index,config in enumerate(configs):
                if index % 12 == 0:
                    print((m,k,h), name, f"config {index}/{len(configs)}", flush=True)
                bm,bn,bk = (config.kwargs[v] for v in ('BLOCK_M','BLOCK_N','BLOCK_K'))
                options={**config.kwargs,'num_warps':config.num_warps,'num_stages':config.num_stages}
                if name=='dual':
                    def fn():
                        return mxtest.bf16_wgrad_dual.fn[(triton.cdiv(h,bn)*triton.cdiv(k,bk),)](
                            dg,du,x,wg,wu,m,h,k,dg.stride(0),du.stride(0),x.stride(0),wg.stride(0),wu.stride(0),**options)
                else:
                    def fn():
                        return mxtest.bf16_wgrad.fn[(triton.cdiv(k,bn)*triton.cdiv(h,bk),)](
                            dy,hidden,wd,m,k,h,dy.stride(0),hidden.stride(0),wd.stride(0),**options)
                try:
                    kernel=fn()
                    if name=='dual':
                        torch.testing.assert_close(wg,old[1],rtol=1e-4,atol=1e-5)
                        torch.testing.assert_close(wu,old[2],rtol=1e-4,atol=1e-5)
                    else:
                        torch.testing.assert_close(wd,old[3],rtol=1e-4,atol=1e-5)
                    time=benchmark(fn,10,1)
                    entries.append({'config':str(config),'kwargs':config.kwargs,'warps':config.num_warps,
                                    'stages':config.num_stages,'ms':time['median_ms'],
                                    'n_regs':kernel.n_regs,'n_spills':kernel.n_spills})
                except triton.OutOfResources as exc:
                    entries.append({'config':str(config),'error':str(exc)})
            tuned[name]=sorted(entries,key=lambda r:r.get('ms',float('inf')))
            print((m,k,h),name,'BEST',tuned[name][:5],flush=True)
        result['tuning']=tuned
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune',action='store_true')
    parser.add_argument('--tune-dx',action='store_true')
    parser.add_argument('--rep',type=int,default=30)
    parser.add_argument('--case',type=int,choices=(0,1,2))
    parser.add_argument('--output',type=Path,default=Path('benchmarks/swiglu_prequant_experiment.json'))
    args=parser.parse_args()
    result={'cases':[]}
    for case in (DEFAULT_CASES if args.case is None else [DEFAULT_CASES[args.case]]):
        result['cases'].append(run_case(*case,args))
        args.output.parent.mkdir(exist_ok=True,parents=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
