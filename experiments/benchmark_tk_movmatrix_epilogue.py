"""Compare the SHFL/M8N8 epilogue against BF16 MOVM/M16N8 epilogue."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.mxfp8_core import quantize_weight
from experiments.tk_mxfp8_gemm import Library, OUTPUT as BASELINE
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary

OUTPUT=ROOT/'runs/tk_movmatrix_epilogue'


def measure(functions,reference,rounds,repeats,seed):
    del reference
    torch.manual_seed(seed)
    result={}
    for name,function in functions.items():
        for _ in range(3):function()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):function()
        values=[]
        for _ in range(rounds):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();graph.replay();end.record();end.synchronize()
            values.append(start.elapsed_time(end)*1000/repeats)
        result[name]={'median_us':statistics.median(values),'min_us':min(values),'max_us':max(values)}
    return result


def audit(binary,destination):
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
    result={}
    for part in raw.split('Function : ')[1:]:
        name=part.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',part))
        assert not any(o.startswith(('LDL','STL','CALL')) for o in ops),name
        result[name]=dict(ops)
        if 'ILi16ELi128ELi128ELi2ELb1E' in name and 'Lb1ELb1E' not in name:
            destination.write_text('Function : '+part)
    return result,raw


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=15)
    ap.add_argument('--repeats',type=int,default=128)
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_movmatrix_epilogue_rtx5090.json')
    args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(123)
    old=Library(BASELINE);new=Library(OUTPUT)
    probe,probe_sass=audit(OUTPUT/'libprobe.so',args.output.with_suffix('.unused'))
    (args.output.parent/'tk_movmatrix_epilogue_probe.sass').write_text(probe_sass)
    kernels,raw=audit(OUTPUT/'libgemm.so',args.output.parent/'tk_movmatrix_epilogue_c70.sass')
    qkernels={name:ops for name,ops in kernels.items() if 'Lb1E' in name and 'Lb1ELb1E' not in name}
    assert len(qkernels)==81
    for name,ops in qkernels.items():
        assert ops.get('MOVM.16.MT88',0)>0 and ops.get('STSM.8.MT168.4',0)>0,name
    old_raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                     str(BASELINE/'libgemm.so')],text=True)
    def instructions(raw):
        return {re.search(r'ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E',p.splitlines()[0]).groups():re.findall(
            r'/\*[0-9a-f]+\*/\s+((?:@!?U?P\d+\s+)?[A-Z][^;]+;)',p)
            for p in raw.split('Function : ')[1:] if 'Lb1E' not in p.splitlines()[0]}
    before,after=instructions(old_raw),instructions(raw)
    assert len(before)==121 and before==after
    sources=['cuda/persistent_mxfp8_gemm.cu','cuda/mxfp8_epilogue.cuh',
             'cuda/probe_movmatrix_epilogue.cu','experiments/benchmark_tk_movmatrix_epilogue.py']
    result={'gpu':torch.cuda.get_device_name(),'rounds':args.rounds,'repeats':args.repeats,
            'method':'Same CUDA graph measurement, randomized order, approximately 50ms warmup '
                     'before every sample. Same tiles, stages, G=1. All outputs preallocated. Unlocked clocks.',
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
            'binary_sha256':{label:hashlib.sha256((p/'libgemm.so').read_bytes()).hexdigest()
                             for label,p in [('baseline',BASELINE),('movmatrix',OUTPUT)]},
            'sass':{'probe':probe,'quantized_kernels':qkernels,
                    'bf16_only_instruction_identical':len(before)},'shapes':[]}
    shapes=[(256,2048,512,43),(4096,4096,4096,70),
            (16384,4096,1024,70),(16384,1024,2048,106),
            (8192,8192,4096,70),(16384,4096,4096,70)]
    for m,n,k,cfg in shapes:
        q=SiluBwdQuantizeLibrary().plan_direct(
            torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        )
        q()
        b,sb=quantize_weight(torch.randn((n,k),device='cuda'))
        inputs=[q.row_fp8,b,q.row_scale,sb]
        baseline=old.plan(*inputs,config=cfg,save_quantized=True)
        movm=new.plan(*inputs,config=cfg,save_quantized=True)
        plain=old.plan(*inputs,config=cfg)
        baseline();movm();plain()
        torch.testing.assert_close(baseline.out,plain.out,rtol=0,atol=0)
        torch.testing.assert_close(movm.out,plain.out,rtol=0,atol=0)
        torch.testing.assert_close(movm.qout.view(torch.uint8),baseline.qout.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(movm.qscales,baseline.qscales,rtol=0,atol=0)
        timing=measure({'shfl':baseline,'movmatrix':movm,'bf16':plain},'shfl',
                       args.rounds,args.repeats,123)
        base=timing['shfl']['median_us'];new_us=timing['movmatrix']['median_us']
        result['shapes'].append({'shape_mnk':[m,n,k],'config':cfg,'info':movm.info,
                                 'bitwise_equal':True,'timing':timing,
                                 'speedup':base/new_us,'time_reduction_pct':100*(1-new_us/base)})
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m=} {n=} {k=} c{cfg}: SHFL={base:.3f}, MOVM={new_us:.3f}, '
              f'BF16={timing["bf16"]["median_us"]:.3f} us; speedup={base/new_us:.4f}x',flush=True)
        del baseline,movm,plain,q,b,sb,inputs
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
