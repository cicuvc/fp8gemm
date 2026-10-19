"""Same-session A/B of FMA versus add/multiply SiLU in the native fused GEMM."""
import argparse
from collections import Counter
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
from experiments.tk_persistent_swiglu import SwiGLULibrary,build
from experiments.benchmark_tk_persistent_gemm import graph_benchmark
from experiments.benchmark_tk_persistent_swiglu import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    directories={name:ROOT/'runs'/f'tk_swiglu_epilogue_{name}' for name in ['fma','unfused']}
    libraries={}
    result={'timing':'CUDA graphs, 32 launches, 9 rotated rounds, same inputs and tile, unlocked clocks',
            'gpu':torch.cuda.get_device_name(),'benchmarks':[],'sass':{}}
    for name,directory in directories.items():
        if not args.skip_build:build(directory,fused_silu=name=='fma')
        libraries[name]=SwiGLULibrary(directory)
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(directory/'libswiglu.so')],text=True)
        for section in sass.split('Function : ')[1:]:
            fn=section.splitlines()[0]
            # Count the unrolled activation tail for the default R16 C128 BK64.
            if 'ILi16ELi128ELi64ELi2ELb0E' in fn:
                ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',section))
                result['sass'][name]={op:n for op,n in ops.items() if op.startswith(('MUFU','FADD','FMUL','FFMA'))}
    for m,n,k,config in [(256,2048,512,0),(1024,8192,1024,3),(4096,4096,4096,3),(8192,8192,4096,3)]:
        torch.manual_seed(751)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        g,gs=mxtest.launch_weight_quantize(torch.randn(n,k,device='cuda')/k**.5)
        u,us=mxtest.launch_weight_quantize(torch.randn(n,k,device='cuda')/k**.5)
        for aux in [False,True]:
            plans={}
            for name,lib in libraries.items():
                go=torch.empty((m,n),device='cuda',dtype=torch.bfloat16) if aux else None
                uo=torch.empty_like(go) if aux else None
                plans[name]=lib.plan(x,g,gs,u,us,config=config,gate_out=go,up_out=uo)
            a=plans['fma']();b=plans['unfused']()
            error=metrics(a,b)
            # Same-signed finite BF16 numbers have monotonically ordered bits.
            # Handle signed zero as equal when reporting numerical ULP changes.
            ulps=(a.view(torch.int16).int()-b.view(torch.int16).int()).abs()
            ulps=torch.where(a==b,0,ulps)
            error['max_bf16_ulp_difference']=ulps.max().item()
            error['different_fraction']=error['mismatches']/(m*n)
            assert torch.isfinite(a).all() and error['max_bf16_ulp_difference']<=1,error
            assert error['relative_l2']<1e-4,error
            if aux:
                assert torch.equal(plans['fma'].gate_out,plans['unfused'].gate_out)
                assert torch.equal(plans['fma'].up_out,plans['unfused'].up_out)
            times=graph_benchmark(plans,rounds=9,repeats=32)
            row={'shape_mnk':[m,n,k],'config':config,'store_aux':aux,'errors':error,'timings':times,
                 'speedup':times['unfused']['median_us']/times['fma']['median_us']}
            result['benchmarks'].append(row)
            print(f'{m,n,k} aux={aux}: FMA {times["fma"]["median_us"]:.3f} us, unfused {times["unfused"]["median_us"]:.3f} us, '
                  f'speedup {row["speedup"]:.4f}, changed {error["mismatches"]}, rel_L2 {error["relative_l2"]:.3g}, '
                  f'ULP {error["max_bf16_ulp_difference"]}',flush=True)
            del plans
    result['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ['cuda/persistent_fp8_swiglu.cu','experiments/tk_persistent_swiglu.py','experiments/compare_tk_swiglu_epilogue.py']}
    (ROOT/'runs/tk_persistent_swiglu/epilogue_ab.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
