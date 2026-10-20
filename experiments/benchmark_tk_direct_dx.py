"""Paired dX and full-backward measurements with/without weight transpose."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=9)
    parser.add_argument('--repeats',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(4)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'base_commit':'ee5fa5b',
            'rounds':args.rounds,'repeats':args.repeats,
            'timing':'CUDA graphs; preallocated plans; rotated order; warm buffers; no locked clocks; paired full backward includes all preparation/bias; no per-shape tuning',
            'benchmarks':[]}
    for m,n,k in [(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096),(137,97,65)]:
        torch.manual_seed(123)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        dy=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
        w,ws=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        old=LinearBackwardPlan(dy,x,w,ws,transpose_weight=True)
        new=LinearBackwardPlan(dy,x,w,ws,swap_ab=False)
        old();new()
        for a,b in zip((old.dx,old.dw,old.db),(new.dx,new.dw,new.db)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        def old_dx():old.prepare_weight();old.dx_plan()
        def new_dx():new.prepare_weight();new.dx_plan()
        times=graph_benchmark({'old_weight_transpose':old.prepare_weight,'old_dx_gemm':old.dx_plan,
                              'new_dx_gemm':new.dx_plan,'old_weight_and_dx':old_dx,'new_weight_and_dx':new_dx,
                              'old_backward':old,'new_backward':new},args.rounds,args.repeats)
        row={'shape_mnk':[m,n,k],'old':old.info,'new':new.info,'exact_vs_old':True,'timings':times,
             'backward_speedup':times['old_backward']['median_us']/times['new_backward']['median_us'],
             'dx_with_weight_prepare_speedup':times['old_weight_and_dx']['median_us']/times['new_weight_and_dx']['median_us']}
        result['benchmarks'].append(row)
        Path('runs/tk_linear_backward/direct_timings.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m,n,k}: dX {times["old_dx_gemm"]["median_us"]:.3f} -> {times["new_dx_gemm"]["median_us"]:.3f} us; '
              f'weight+dX {times["old_weight_and_dx"]["median_us"]:.3f} -> {times["new_weight_and_dx"]["median_us"]:.3f}; '
              f'backward {times["old_backward"]["median_us"]:.3f} -> {times["new_backward"]["median_us"]:.3f}',flush=True)
    files=[Path(__file__),ROOT/'cuda/persistent_fp8_gemm.cu',ROOT/'cuda/linear_backward_prepare.cu',
           ROOT/'experiments/tk_persistent_gemm.py',ROOT/'experiments/tk_linear_backward.py',
           *sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))]
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    Path('runs/tk_linear_backward/direct_result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
