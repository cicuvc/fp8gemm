"""Measure factored STMATRIX addresses in persistent MXFP8 GEMM."""
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
from experiments.tk_mxfp8_gemm import Library,OUTPUT,pack_a_scales
from experiments.retest_tk_mxfp8_tuning import measure
from experiments.benchmark_tk_mxfp8 import audit_sass

RUN=ROOT/'runs/tk_stmatrix_address'


def inspect(directory,label,destination):
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                 str(directory/'libgemm.so')],text=True)
    result={}
    for part in raw.split('Function : ')[1:]:
        name=part.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',part))
        assert not any(o.startswith(('LDL','STL','CALL')) for o in ops),name
        if 'ILi16ELi128ELi128ELi2E' in name and 'Lb1ELb1E' not in name:
            mode='quantized' if 'Lb1E' in name else 'bf16'
            path=destination/f'tk_stmatrix_c70_{label}_{mode}.sass'
            path.write_text('Function : '+part)
            result[mode]={'opcodes':dict(ops),'sass':str(path.relative_to(ROOT)),
                          'stsm':[line.strip().split(';')[0]+';' for line in part.splitlines()
                                  if re.search(r'\sSTSM\.',line)]}
    return {'kernels_no_spills':len(raw.split('Function : '))-1,'c70':result}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=15)
    ap.add_argument('--repeats',type=int,default=128)
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_stmatrix_gemm_rtx5090.json')
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(123)
    dirs={'old':RUN/'baseline','old_movm':RUN/'baseline_movm',
          'new':RUN/'factored','new_movm':RUN/'factored_movm'}
    libs={name:Library(path) for name,path in dirs.items()}
    pure=audit_sass(dirs['new'])
    sources=['cuda/shared_store.cuh','cuda/persistent_mxfp8_gemm.cu','cuda/mxfp8_epilogue.cuh',
             'experiments/benchmark_tk_stmatrix.py','experiments/tk_mxfp8_gemm.py']
    result={'gpu':torch.cuda.get_device_name(),'rounds':args.rounds,'repeats':args.repeats,
            'method':'CUDA graphs; random order; ~50ms warmup before each sample. '
                     'All variants reuse the same output addresses. G=1. Unlocked clocks.',
            'binary_sha256':{k:hashlib.sha256((p/'libgemm.so').read_bytes()).hexdigest() for k,p in dirs.items()},
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            'sass':{label:inspect(path,label,args.output.parent) for label,path in dirs.items()},
            'pure_gemm_audit_count':len(pure),'shapes':[]}
    shapes=[(256,2048,512,43),(4096,4096,4096,70),(16384,4096,1024,70),
            (16384,1024,2048,106),(8192,8192,4096,70),(16384,4096,4096,70)]
    for m,n,k,c in shapes:
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        sa=pack_a_scales(sa);b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        q=torch.empty((m,n),device='cuda',dtype=torch.float8_e4m3fn)
        qs=torch.empty(((n+127)//128,(m+15)//16*16),device='cuda',dtype=torch.int32)
        plans={}
        for name,library in libs.items():
            if not name.endswith('_movm'):
                plans[name+'_bf16']=library.plan(a,b,sa,sb,config=c,out=out)
            plans[name+'_q']=library.plan(a,b,sa,sb,config=c,out=out,save_quantized=True,
                                         quantized_out=q,quantized_scales=qs)
        plans['old_q']();ref=out.clone();refq=q.view(torch.uint8).clone();refs=qs.clone()
        for name,plan in plans.items():
            plan();torch.testing.assert_close(out,ref,rtol=0,atol=0)
            if name.endswith('_q'):
                torch.testing.assert_close(q.view(torch.uint8),refq,rtol=0,atol=0)
                torch.testing.assert_close(qs,refs,rtol=0,atol=0)
        timing=measure(plans,'old_q',args.rounds,args.repeats,123)
        pairs=[('bf16','old_bf16','new_bf16'),('fp8','old_q','new_q'),('movmatrix','old_movm_q','new_movm_q')]
        speedups={label:timing[old]['median_us']/timing[new]['median_us'] for label,old,new in pairs}
        result['shapes'].append({'shape_mnk':[m,n,k],'config':c,'info':plans['new_q'].info,
                                'bitwise_equal':True,'timing':timing,'speedups':speedups})
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print((m,n,k,c),{name:round(t['median_us'],3) for name,t in timing.items()},
              {name:round(v,4) for name,v in speedups.items()},flush=True)
        del plans,plan,a,b,sa,sb,out,q,qs,ref,refq,refs
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
