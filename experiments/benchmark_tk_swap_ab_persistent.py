"""Paired CUDA dX/backward comparison and tile sweep for persistent swapAB."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_linear_backward import LinearBackwardPlan
from experiments.tk_persistent_gemm import CONFIGS, GemmLibrary
from experiments.benchmark_tk_persistent_gemm import graph_benchmark

OUTPUT=ROOT/'runs/tk_swap_ab_persistent'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=9)
    parser.add_argument('--repeats',type=int,default=64)
    parser.add_argument('--skip-sweep',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result={
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'rounds':args.rounds,'repeats':args.repeats,
        'timing':'CUDA graphs, preallocated plans, rotated order, warm buffers, no locked clocks. Full backward includes dY preparation, any W preparation, X quantization, both GEMMs, output cropping and bias reduction.',
        'peak_fp8a32_tflops':419.0, 'benchmarks':[]}
    for m,n,k in [(256,2048,512),(1024,8192,1024),(2048,16384,2048),
                  (4096,4096,4096),(8192,8192,4096),(137,97,65)]:
        torch.manual_seed(123)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        dy=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
        w,ws=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        plans={
            'transpose':LinearBackwardPlan(dy,x,w,ws,transpose_weight=True),
            'direct':LinearBackwardPlan(dy,x,w,ws,swap_ab=False),
            'swap':LinearBackwardPlan(dy,x,w,ws)}
        for plan in plans.values():plan()
        for name in ['direct','swap']:
            for a,b in zip(plans['transpose'](),plans[name]()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        functions={}
        for name,plan in plans.items():
            functions[f'{name}_dx']=plan.dx_plan
            def weight_and_dx(p=plan):
                p.prepare_weight();p.dx_plan()
            functions[f'{name}_weight_and_dx']=weight_and_dx
            functions[f'{name}_backward']=plan
        timings=graph_benchmark(functions,args.rounds,args.repeats)
        swap=plans['swap']
        swap()
        ref=swap.dx_plan.out.clone()
        lib=GemmLibrary()
        sweep={}
        for config in ([] if args.skip_sweep else [0,1,2,3,4,6,7,8,9]):
            if swap.dx_stride%CONFIGS[config][1]:
                continue
            p=lib.plan_ab(swap.dyp,swap.w_direct,ws,config,
                          out=swap.dxp,swap_ab=True)
            torch.testing.assert_close(p(),ref,rtol=0,atol=0)
            sweep[str(config)]=p
        sweep_times=graph_benchmark(sweep,args.rounds,args.repeats) if sweep else {}
        best=min(sweep_times,key=lambda name:sweep_times[name]['median_us']) if sweep_times else None
        row={
            'shape_mnk':[m,n,k],'plans':{name:p.info for name,p in plans.items()},
            'bitwise_equal_vs_transpose':True,
            'timings':timings,
            'swap_tile_sweep':{name:{'info':sweep[name].info,**v} for name,v in sweep_times.items()},
            'best_swap_config':int(best) if best is not None else None,
            'dx_mfu':{name:2*m*n*k/(timings[f'{name}_dx']['median_us']*1e-6)/419e12 for name in plans},
            'backward_mfu':{name:4*m*n*k/(timings[f'{name}_backward']['median_us']*1e-6)/419e12 for name in plans}}
        result['benchmarks'].append(row)
        (OUTPUT/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m,n,k}: dX us '+', '.join(f'{name}={timings[f"{name}_dx"]["median_us"]:.3f}' for name in plans)+
              '; backward us '+', '.join(f'{name}={timings[f"{name}_backward"]["median_us"]:.3f}' for name in plans)+
              (f'; best swap config{best}={sweep_times[best]["median_us"]:.3f}' if best is not None else ''),
              flush=True)
    files=[Path(__file__),ROOT/'cuda/persistent_fp8_gemm.cu',
           ROOT/'cuda/tma_ring_pipe.cuh',ROOT/'cuda/linear_backward_prepare.cu',
           ROOT/'experiments/tk_persistent_gemm.py',ROOT/'experiments/tk_linear_backward.py',
           ROOT/'tests/test_tk_linear_backward.py']
    files+=sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
