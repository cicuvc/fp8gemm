"""Independent tuning shortlist retest with current-binary default controls."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_mxfp8_gemm import Library, OUTPUT, pack_a_scales


def measure(functions,baseline,rounds,repeats,seed):
    """Randomize each round; warm each variant to reduce predecessor-clock bias."""
    graphs={};warm_replays={}
    for name,fn in functions.items():
        for _ in range(3):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):fn()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        graph.replay();start.record();graph.replay();end.record();end.synchronize()
        warm_replays[name]=max(1,math.ceil(50/start.elapsed_time(end)))
        graphs[name]=graph
    rng=random.Random(seed);samples={name:[] for name in functions}
    for _ in range(rounds):
        names=list(functions);rng.shuffle(names)
        for name in names:
            for _ in range(warm_replays[name]):graphs[name].replay()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/repeats)
    return {name:{'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v),
                  'round_us':v,'paired_speedup':statistics.median(
                      a/b for a,b in zip(samples[baseline],v))}
            for name,v in samples.items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'benchmarks/tk_mxfp8_tuning_rtx5090.json')
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_tuning_retest_rtx5090.json')
    ap.add_argument('--rounds',type=int,default=21)
    ap.add_argument('--repeats',type=int,default=128)
    args=ap.parse_args()
    torch.set_num_threads(4)
    source=json.loads(args.input.read_text())
    lib=Library();old=Library(ROOT/'runs/tk_mxfp8_tuning/baseline')
    result={'gpu':torch.cuda.get_device_name(),'rounds':args.rounds,'repeats':args.repeats,
            'method':'Independent top-5 shortlist retest. Baseline uses the current binary and original default config. '
                     'Two independent plans of that exact kernel plus the saved binary are measured as controls. '
                     'CUDA graphs; independently randomized order each round; at least approximately 50ms '
                     'untimed warmup of each variant before its timed sample. '
                     'G=1; quantization excluded; unlocked clocks.',
            'sweep_file':str(args.input),'sweep_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'source_sha256':source['source_sha256']|{
                str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'binary_sha256':hashlib.sha256((OUTPUT/'libgemm.so').read_bytes()).hexdigest(),'shapes':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        with args.output.with_suffix('.csv').open('w') as f:
            w=csv.writer(f)
            w.writerow(['M','N','K','config','R','C','BK','stages','shared_bytes',
                        'baseline_us','best_us','tflops','paired_speedup'])
            for row in result['shapes']:
                p=row['plans'][row['best']];t=row['timing'][row['best']]
                w.writerow([*row['shape_mnk'],row['best'],*[p[d] for d in [
                    'R','C','BK','stages','shared_bytes']],row['timing'][row['baseline']]['median_us'],
                    t['median_us'],t['tflops'],t['paired_speedup']])
    for row in source['shapes']:
        m,n,k=row['shape_mnk'];default=row['baseline_config']
        best=sorted((name for name in row['retest'] if name!='baseline'),
                    key=lambda name:row['retest'][name]['median_us'])[:5]
        candidates={int(name[1:]) for name in best}
        # Keep both depths of the R32 C64 BK64 tile to isolate stage effects.
        candidates.update([104,105,106,default])
        torch.manual_seed(123)
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        sa=pack_a_scales(sa)
        b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        plans={f'c{c}':lib.plan(a,b,sa,sb,c,out=out) for c in sorted(candidates)}
        plans['default_copy']=lib.plan(a,b,sa,sb,default,out=out)
        plans['saved_default']=old.plan(a,b,sa,sb,default,out=out)
        baseline=f'c{default}'
        ref=plans[baseline]().clone()
        assert all(torch.equal(p(),ref) for p in plans.values())
        del ref
        timing=measure(plans,baseline,args.rounds,args.repeats,789)
        for t in timing.values():t['tflops']=2*m*n*k/(t['median_us']*1e6)
        winner=min((name for name in timing if name.startswith('c')),
                   key=lambda name:timing[name]['median_us'])
        data={'shape_mnk':[m,n,k],'baseline':baseline,'best':winner,
              'plans':{name:p.info for name,p in plans.items()},'timing':timing,'bitwise_equal':True}
        # Matched stages: ratios use paired rounds, not ratios of medians.
        data['r32_c64_bk64_stage3_vs_stage2']=statistics.median(
            a/b for a,b in zip(timing['c105']['round_us'],timing['c106']['round_us']))
        result['shapes'].append(data);save()
        print((m,n,k),winner,plans[winner].info,
              f'{timing[winner]["median_us"]:.3f}us {timing[winner]["tflops"]:.1f}TF '
              f'{timing[winner]["paired_speedup"]:.3f}x '
              f'default={timing[baseline]["median_us"]:.3f}us '
              f'copy={timing["default_copy"]["median_us"]:.3f}us '
              f'saved={timing["saved_default"]["median_us"]:.3f}us',flush=True)
        del plans,a,b,sa,sb,out
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
