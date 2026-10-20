"""Sweep pure MXFP8 forward warp tiles, BK and TMA ring depth on one GPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_mxfp8_gemm import CONFIGS, Library, OUTPUT, pack_a_scales
from experiments.benchmark_tk_forward_pipeline import measure


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_tuning_rtx5090.json')
    ap.add_argument('--baseline',type=Path,default=ROOT/'runs/tk_mxfp8_tuning/baseline')
    ap.add_argument('--shape',nargs=3,type=int,action='append')
    ap.add_argument('--rounds',type=int,default=5)
    ap.add_argument('--repeats',type=int,default=16)
    ap.add_argument('--retest-rounds',type=int,default=15)
    ap.add_argument('--retest-repeats',type=int,default=128)
    args=ap.parse_args()
    torch.set_num_threads(4)
    old,new=Library(args.baseline),Library()
    result={
        'gpu':torch.cuda.get_device_name(),'configs':CONFIGS,
        'method':'Pure forward only; prequantized operands; G=1, same register budget and 8 consumer warps. '
                 'Paired shuffled/rotated warm CUDA graphs, unlocked clocks. '
                 'Sweep all legal configurations; independently retest top 6 plus baseline and best of every stage/BK.',
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'binary_sha256':{n:hashlib.sha256((p/'libgemm.so').read_bytes()).hexdigest()
                         for n,p in [('old',args.baseline),('new',OUTPUT)]},
        'source_sha256':{str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'cuda/persistent_mxfp8_gemm.cu','cuda/mxfp8_configs.inc','cuda/fp8_shared_load.cuh',
            'cuda/tma_ring_pipe.cuh','experiments/tk_mxfp8_gemm.py',
            'experiments/benchmark_tk_mxfp8_tuning.py']},
        'nvidia_smi_before':subprocess.check_output(['nvidia-smi'],text=True),
        'shapes':[],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    shapes=args.shape or [(256,2048,512),(1024,8192,1024),(2048,16384,2048),
                          (4096,4096,4096),(8192,8192,4096)]+[
                              (16384,n,k) for n in [1024,2048,4096] for k in [1024,2048,4096]]
    for m,n,k in shapes:
        torch.manual_seed(123)
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        sa=pack_a_scales(sa)
        b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        baseline=7 if ((m+127)//128)*((n+127)//128)<torch.cuda.get_device_properties(0).multi_processor_count else 8
        plans={'baseline':old.plan(a,b,sa,sb,baseline,out=out)}
        ref=plans['baseline']().clone()
        skipped={}
        for c in range(len(CONFIGS)):
            try:
                p=new.plan(a,b,sa,sb,c,out=out)
            except RuntimeError as e:
                if str(e) not in ['Unexpected register pool or local memory',
                                  'K/N must align to TMA swizzle panels']:raise
                skipped[str(c)]=str(e)
                continue
            assert torch.equal(p(),ref),(m,n,k,c,CONFIGS[c])
            plans[f'c{c}']=p
        del ref
        row={'shape_mnk':[m,n,k],'baseline_config':baseline,'bitwise_equal':True,
             'plans':{n:p.info for n,p in plans.items()},'skipped':skipped}
        result['shapes'].append(row);save()
        print((m,n,k),f'{len(plans)-1} configs validated; {len(skipped)} skipped',flush=True)
        row['sweep']=measure(plans,'baseline',args.rounds,args.repeats,123)
        ranked=sorted((n for n in plans if n!='baseline'),key=lambda n:row['sweep'][n]['median_us'])
        shortlist=set(ranked[:6])|{'baseline',f'c{baseline}'}
        for dim in ['stages','BK']:
            for v in sorted({p.info[dim] for p in plans.values()}):
                shortlist.add(next(n for n in ranked if plans[n].info[dim]==v))
        # Compare the winning tile at each feasible ring depth directly.
        winner=plans[ranked[0]].info
        shortlist.update(n for n,p in plans.items()
                         if all(p.info[d]==winner[d] for d in ['R','C','BK']))
        save()
        selected={n:plans[n] for n in sorted(shortlist)}
        row['retest']=measure(selected,'baseline',args.retest_rounds,args.retest_repeats,456)
        for t in row['retest'].values():
            t['tflops']=2*m*n*k/(t['median_us']*1e6)
        row['best']=min(row['retest'],key=lambda n:row['retest'][n]['median_us'])
        best=row['retest'][row['best']]
        save()
        print((m,n,k),row['best'],row['plans'][row['best']],
              f'{best["median_us"]:.3f}us {best["tflops"]:.1f}TF '
              f'{best["paired_speedup"]:.3f}x baseline={row["retest"]["baseline"]["median_us"]:.3f}us',
              flush=True)
        del plans,selected,p,a,b,sa,sb,out
        gc.collect();torch.cuda.empty_cache()
    result['nvidia_smi_after']=subprocess.check_output(['nvidia-smi'],text=True)
    save()


if __name__=='__main__':main()
