"""Compare fused BF16+MXFP8 output against BF16 GEMM plus row quantization."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_mxfp8_gemm import Library, OUTPUT, CONFIGS, pack_a_scales
from experiments.benchmark_tk_mxfp8 import quantize_a, audit_sass
from experiments.retest_tk_mxfp8_tuning import measure


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_epilogue_rtx5090.json')
    ap.add_argument('--baseline',type=Path,default=ROOT/'runs/tk_mxfp8_epilogue/baseline')
    ap.add_argument('--rounds',type=int,default=11)
    ap.add_argument('--repeats',type=int,default=64)
    args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123)
    audit=audit_sass()
    raw=(OUTPUT/'kernels.sass').read_text()
    q_audit={}
    for part in raw.split('Function : ')[1:]:
        name=part.splitlines()[0]
        if 'Lb1E' not in name or 'Lb1ELb1E' in name:continue
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',part))
        assert not any(o.startswith(('LDL','STL','CALL')) for o in ops),name
        assert any(o.startswith('STG') for o in ops),name
        q_audit[name]=dict(ops)
        if 'ILi16ELi128ELi128ELi2ELb1E' in name:
            (args.output.parent/'tk_mxfp8_epilogue_c70.sass').write_text('Function : '+part)
    assert len(q_audit)==sum(c[1]%32==0 for c in CONFIGS)
    files=['cuda/persistent_mxfp8_gemm.cu','cuda/mxfp8_epilogue.cuh','cuda/mxfp8_configs.inc',
           'experiments/tk_mxfp8_gemm.py','experiments/benchmark_tk_mxfp8_epilogue.py',
           'third_party/thunderkittens/include/ops/group/register/tile/online_scale.cuh']
    result={'gpu':torch.cuda.get_device_name(),'rounds':args.rounds,'repeats':args.repeats,
            'method':'CUDA graphs, randomized variant order each round, approximately 50ms '
                     'untimed warmup per variant before each sample, unlocked clocks. '
                     'Prepared outputs reused; separate path includes one Triton row quantization '
                     'kernel writing E4M3 and packed E8M0 scales. G=1.',
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
            'binary_sha256':hashlib.sha256((OUTPUT/'libgemm.so').read_bytes()).hexdigest(),
            'baseline_binary_sha256':hashlib.sha256((args.baseline/'libgemm.so').read_bytes()).hexdigest(),
            'sass':{'bf16_only_kernels_checked':len(audit),'quantized_kernels':q_audit},
            'shapes':[]}
    old=Library(args.baseline);lib=Library()
    shapes=[(256,2048,512,43),(4096,4096,4096,70),
            (16384,4096,1024,70),(16384,1024,2048,106),
            (8192,8192,4096,70),(16384,4096,4096,70)]
    for m,n,k,cfg in shapes:
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        sa=pack_a_scales(sa)
        b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        inputs=[a,b,sa,sb]
        baseline=old.plan(*inputs,config=cfg)
        plain=lib.plan(*inputs,config=cfg)
        fused=lib.plan(*inputs,config=cfg,save_quantized=True)
        q=torch.empty_like(fused.qout);qs=torch.empty_like(fused.qscales)
        def separate():
            plain()
            quantize_a[((m+31)//32,(n+127)//128)](plain.out,q,qs,m,n,num_warps=4)
        baseline();fused();separate()
        torch.testing.assert_close(fused.out,baseline.out,rtol=0,atol=0)
        torch.testing.assert_close(plain.out,baseline.out,rtol=0,atol=0)
        torch.testing.assert_close(fused.qout.view(torch.uint8),q.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(fused.qscales,qs,rtol=0,atol=0)
        timings=measure({'saved_bf16':baseline,'bf16':plain,'fused':fused,'separate':separate},
                        'separate',args.rounds,args.repeats,123)
        for t in timings.values():t['gemm_tflops']=2*m*n*k/t['median_us']/1e6
        bf=timings['bf16']['median_us'];fu=timings['fused']['median_us'];se=timings['separate']['median_us']
        row={'shape_mnk':[m,n,k],'config':cfg,'info':fused.info,'timing':timings,
             'fused_extra_us':fu-bf,'fused_overhead_pct':100*(fu/bf-1),
             'fused_vs_separate_speedup':se/fu,'bitwise_equivalence':True}
        result['shapes'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m=} {n=} {k=} c{cfg}: bf16={bf:.3f}, fused={fu:.3f}, separate={se:.3f} us; '
              f'overhead={row["fused_overhead_pct"]:.1f}%, speedup={se/fu:.3f}x',flush=True)
        del baseline,plain,fused,a,b,sa,sb,inputs,q,qs
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
