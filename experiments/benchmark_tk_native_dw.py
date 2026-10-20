"""Compare cached-X dW with global vs LDSM dY transpose, including swapAB."""
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
from experiments.benchmark_tk_persistent_gemm import graph_benchmark

OUTPUT=ROOT/'runs/tk_native_dw'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds',type=int,default=9)
    p.add_argument('--repeats',type=int,default=64)
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'rounds':args.rounds,'repeats':args.repeats,'peak_fp8a32_tflops':419.,
            'timing':'CUDA graphs, preallocated plans, rotated order, warm buffers, unlocked clocks; dX uses the committed swapAB defaults in all modes; bias enabled.',
            'benchmarks':[]}
    for m,n,k in [(256,2048,512),(1024,8192,1024),(2048,16384,2048),
                  (4096,4096,4096),(8192,8192,4096),(137,97,65)]:
        torch.manual_seed(123)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        dy=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
        w,ws=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        plans={
            'baseline':LinearBackwardPlan(dy,x,w,ws,dw_native_dy=False),
            'native':LinearBackwardPlan(dy,x,w,ws,dw_native_dy=True),
            'swap':LinearBackwardPlan(dy,x,w,ws,dw_native_dy=True,dw_swap_ab=True)}
        ref=tuple(t.clone() for t in plans['baseline']())
        for plan in plans.values():
            for got,expected in zip(plan(),ref):
                torch.testing.assert_close(got,expected,rtol=0,atol=0)
        functions={}
        for name,plan in plans.items():
            functions[name+'_prepare_dy']=plan.prepare_dy
            functions[name+'_dw']=plan.dw_plan
            def prep_dw(plan=plan):
                plan.prepare_dy();plan.dw_plan()
            functions[name+'_prepare_and_dw']=prep_dw
            functions[name+'_backward']=plan
        timings=graph_benchmark(functions,args.rounds,args.repeats)
        row={'shape_mnk':[m,n,k],'plans':{name:plan.info for name,plan in plans.items()},
             'bitwise_equal_vs_baseline':True,'timings':timings,
             'removed_dyt_bytes':plans['baseline'].dyt.numel()*2,
             'dw_mfu':{name:2*m*n*k/(timings[name+'_dw']['median_us']*1e-6)/419e12 for name in plans}}
        result['benchmarks'].append(row)
        (OUTPUT/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m,n,k}: dW '+', '.join(f'{name}={timings[name+"_dw"]["median_us"]:.3f}' for name in plans)+
              '; prep+dW '+', '.join(f'{name}={timings[name+"_prepare_and_dw"]["median_us"]:.3f}' for name in plans)+
              '; backward '+', '.join(f'{name}={timings[name+"_backward"]["median_us"]:.3f}' for name in plans),flush=True)
    files=[Path(__file__),ROOT/'cuda/persistent_dual_online_gemm.cu',
           ROOT/'cuda/linear_backward_prepare.cu',ROOT/'experiments/tk_linear_backward.py',
           ROOT/'experiments/tk_dual_online_gemm.py',ROOT/'cuda/tma_ring_pipe.cuh']
    result['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
