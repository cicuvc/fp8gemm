"""Compare dual BF16+FP8 output with FP8-only output on identical GEMMs."""
import argparse
from collections import Counter
import csv
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
from experiments.tk_mxfp8_gemm import Library,OUTPUT,pack_a_scales
from experiments.retest_tk_mxfp8_tuning import measure


def inspect(path,name):
    sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                  str(path/'libgemm.so')],text=True)
    result={}
    count=0
    for block in sass.split('Function : ')[1:]:
        symbol=block.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',block))
        assert not any(op.startswith(('LDL','STL','CALL')) for op in ops),symbol
        count+=1
        if 'ILi16ELi128ELi128ELi2E' not in symbol:continue
        assert 'WARPSYNC.COLLECTIVE' not in ops,symbol
        mode='only' if 'Lb1ELb1E' in symbol else 'dual' if 'Lb1ELb0E' in symbol else 'bf16'
        output=ROOT/f'benchmarks/tk_mxfp8_only_c70_{name}_{mode}.sass'
        output.write_text('Function : '+block)
        result[mode]={'opcodes':dict(ops),'sass':str(output.relative_to(ROOT))}
        stores=sum(v for k,v in ops.items() if k.startswith('STSM'))
        assert stores=={'only':4,'dual':12,'bf16':8}[mode],(name,mode,stores)
        assert ops['UTMASTG.5D']==(2 if mode=='dual' else 1),(name,mode,ops)
    assert count==283 and len(result)==3
    return {'kernels_without_spills':count,'c70':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=15)
    parser.add_argument('--repeats',type=int,default=128)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_only_rtx5090.json')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123)
    paths={'native':OUTPUT,**{s:ROOT/'runs/tk_mxfp8_only'/s for s in ['factored','movm','factored_movm']}}
    libraries={name:Library(path) for name,path in paths.items()}
    sources=['cuda/persistent_mxfp8_gemm.cu','cuda/mxfp8_epilogue.cuh','cuda/shared_store.cuh',
             'experiments/tk_mxfp8_gemm.py','experiments/benchmark_tk_mxfp8_only.py']
    result={'gpu':torch.cuda.get_device_name(),'rounds':args.rounds,'repeats':args.repeats,
            'method':'Shuffled warm CUDA graphs, ~50ms warmup per sample; G=1; unlocked clocks. '
                     'Identical input and output addresses; fixed config per shape. No quantization input cost. '
                     'FP8-only retains BF16 rounding before quantization but removes BF16 allocation, '
                     'shared stores, TMA stores and associated synchronization.',
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            'binary_sha256':{n:hashlib.sha256((p/'libgemm.so').read_bytes()).hexdigest() for n,p in paths.items()},
            'sass':{n:inspect(p,n) for n,p in paths.items()},'shapes':[]}
    shapes=[(256,2048,512,43),(4096,4096,4096,70),(16384,4096,1024,70),
            (16384,1024,2048,106),(8192,8192,4096,70),(16384,4096,4096,70)]
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        with args.output.with_suffix('.csv').open('w') as f:
            writer=csv.writer(f)
            writer.writerow(['M','N','K','config','variant','dual_us','only_us','latency_reduction_pct',
                             'only_tflops','dual_smem','only_smem','bf16_output_bytes_saved'])
            for row in result['shapes']:
                for name in paths:
                    dual=row['timing'][name+'_dual']['median_us']
                    only=row['timing'][name+'_only']['median_us']
                    writer.writerow([*row['shape_mnk'],row['config'],name,dual,only,100*(1-only/dual),
                                     row['timing'][name+'_only']['tflops'],
                                     row['plans'][name+'_dual']['shared_bytes'],
                                     row['plans'][name+'_only']['shared_bytes'],row['bf16_output_bytes_saved']])
    for m,n,k,c in shapes:
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        sa=pack_a_scales(sa)
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        q=torch.empty_like(out,dtype=torch.float8_e4m3fn)
        qs=torch.empty(((n+127)//128,(m+15)//16*16),device='cuda',dtype=torch.int32)
        plans={'bf16':libraries['native'].plan(a,b,sa,sb,config=c,out=out)}
        ref=plans['bf16']().clone()
        refq,refs=mxtest.launch_row_quantize(ref);refs=pack_a_scales(refs)
        for name,lib in libraries.items():
            for only in [False,True]:
                label=name+('_only' if only else '_dual')
                plans[label]=lib.plan(a,b,sa,sb,config=c,out=None if only else out,
                                      save_quantized=True,save_bf16=not only,
                                      quantized_out=q,quantized_scales=qs)
                # A sentinel also verifies that FP8-only never writes the old BF16 allocation.
                if only:out.fill_(13)
                plans[label]()
                if only:assert torch.all(out==13)
                else:torch.testing.assert_close(out,ref,rtol=0,atol=0)
                torch.testing.assert_close(q.view(torch.uint8),refq.view(torch.uint8),rtol=0,atol=0)
                torch.testing.assert_close(qs,refs,rtol=0,atol=0)
        timing=measure(plans,'native_dual',args.rounds,args.repeats,123)
        for t in timing.values():t['tflops']=2*m*n*k/t['median_us']/1e6
        row={'shape_mnk':[m,n,k],'config':c,'bitwise_equal':True,
             'bf16_output_bytes_saved':2*m*n,'plans':{name:p.info for name,p in plans.items()},
             'timing':timing,'speedups':{name:timing[name+'_dual']['median_us']/timing[name+'_only']['median_us']
                                       for name in paths}}
        result['shapes'].append(row);save()
        print((m,n,k,c),{name:round(t['median_us'],3) for name,t in timing.items()},
              'speedups',row['speedups'],flush=True)
        del plans,a,b,sa,sb,out,q,qs,ref,refq,refs
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
