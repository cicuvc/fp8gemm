"""Matched online-A versus prequantized-A WASP GEMM comparison."""
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
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_mxfp8_gemm import Library,OUTPUT,pack_a_scales
from experiments.tk_persistent_gemm import GemmLibrary,DEFAULT_OUTPUT
from experiments.benchmark_tk_forward_pipeline import measure

@triton.jit
def quantize_a(x,q,s,M:tl.constexpr,K:tl.constexpr):
    r=tl.program_id(0)*32+tl.arange(0,32)
    k=tl.program_id(1)*128+tl.arange(0,128)
    a=tl.load(x+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),other=0)
    v,sc=mxtest.online_scale(a)
    tl.store(q+r[:,None]*K+k[None,:],v,(r[:,None]<M)&(k[None,:]<K))
    word=tl.sum(sc.to(tl.uint32) << (8*tl.arange(0,4))[None,:],axis=1)
    mp:tl.constexpr=tl.cdiv(M,16)*16
    tl.store(s+tl.program_id(1)*mp+r,word,r<mp)

def audit_sass():
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(OUTPUT/'libgemm.so')],text=True)
    (OUTPUT/'kernels.sass').write_text(raw)
    result={}
    for part in raw.split('Function : ')[1:]:
        name=part.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',part))
        assert not any(o.startswith(('LDL','STL','STG','CALL')) for o in ops),name
        alloc=part.index('USETMAXREG.TRY_ALLOC')
        loads=list(re.finditer(r'\sLDG\.[^;]+;',part))
        assert loads and all(x.start()<alloc for x in loads),name
        assert any(o.startswith('QMMA.SF') for o in ops),name
        # Verify B register pairs come directly from one native LDSM tuple.
        definitions={};qmma=direct=0
        for line in part.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',line)
            if not m:continue
            pc,op,body=m.groups();operands=body.split(',')
            if op.startswith('QMMA'):
                reg=int(re.match(r'R(\d+)',operands[2].strip())[1])
                origins=[definitions.get(reg+i) for i in [0,1]]
                direct+=int(all(o and o[0].startswith('LDSM') for o in origins) and origins[0][1]==origins[1][1])
                qmma+=1
            dest=operands[1].strip() if op.startswith('SHFL') else operands[0].strip()
            if re.fullmatch(r'R\d+',dest):
                count=4 if op.startswith(('LDSM','QMMA')) or '.128' in op else 2 if '.64' in op or op.startswith(('IMAD.WIDE','LEA.WIDE')) else 1
                for i in range(count):definitions[int(dest[1:])+i]=(op,pc)
        assert direct==qmma and qmma>0,(name,direct,qmma)
        result[name]={'opcodes':dict(ops),'qmma':qmma,'b_pairs_direct_from_ldsm':direct,'consumer_global_loads':0,'local_memory_ops':0}
    assert len(result)==10
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=11)
    ap.add_argument('--repeats',type=int,default=64)
    # The block-scaled instruction has twice the measured issue rate of plain
    # FP8/F32 on this RTX5090 (cuda/probe_mxfp8_rate.cu); 419 is the plain rate.
    ap.add_argument('--peak-tflops',type=float,nargs='+',default=[838])
    ap.add_argument('--shape',type=int,nargs=3,action='append')
    args=ap.parse_args();torch.set_num_threads(4)
    online=GemmLibrary();pure=Library()
    result={'gpu':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,
            'peak_tflops':args.peak_tflops,'rounds':args.rounds,'repeats':args.repeats,
            'peak_reference_note':'Nominal reference, not normalized to measured clocks. RTX5090 MXFP8 default 838 is inferred from 419 plain-FP8 nominal peak and the measured 2x block-scale issue rate; PRO6000 range is supplied explicitly.',
            'method':'Paired, shuffled/rotated warm CUDA graphs, unlocked clocks. Identical R/C/BK/stages/grid/register allocation, G=1, no register prefetch. Online BF16 A versus prequantized E4M3 A; same B and B scales. A scales pack four consecutive K32 exponents per uint32, stored [ceil(K/128),roundup(M,16)] and loaded with TMA. Pure GEMM excludes A quantization; e2e includes one fused quantization/scale-layout kernel. B quantization excluded from all variants.',
            'sass':audit_sass(),'nvidia_smi_before':subprocess.check_output(['nvidia-smi'],text=True),
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                'cuda/persistent_mxfp8_gemm.cu','cuda/persistent_fp8_gemm.cu','cuda/tma_ring_pipe.cuh','cuda/linear_tma_buffer.cuh',
                'experiments/tk_mxfp8_gemm.py','experiments/benchmark_tk_mxfp8.py']},
            'binary_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUTPUT/'libgemm.so',DEFAULT_OUTPUT/'libgemm.so']},'shapes':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    shapes=args.shape or [(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096)]+[(16384,n,k) for n in [1024,2048,4096] for k in [1024,2048,4096]]
    for m,n,k in shapes:
        torch.manual_seed(123)
        a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        b,bs=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        aq,sa_ref=mxtest.launch_row_quantize(a);sa=pack_a_scales(sa_ref)
        saved=aq.clone()
        q=lambda:quantize_a[(triton.cdiv(m,32),triton.cdiv(k,128))](a,aq,sa,m,k,num_warps=4)
        q();assert torch.equal(aq.view(torch.uint8),saved.view(torch.uint8)) and torch.equal(sa,pack_a_scales(sa_ref))
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        cfg=7 if triton.cdiv(m,128)*triton.cdiv(n,128)<result['sm_count'] else 8
        po=online.plan(a,b,cfg,out=out,scales=bs)
        pp=pure.plan(aq,b,sa,bs,cfg,out=out)
        ref=po().clone();assert torch.equal(pp(),ref)
        def e2e():q();pp()
        e2e();assert torch.equal(out,ref)
        del ref,saved,sa_ref
        funcs={'online':po,'pure':pp,'quantize':q,'quantize_plus_pure':e2e}
        timing=measure(funcs,'online',args.rounds,args.repeats,456)
        row={'shape_mnk':[m,n,k],'config':cfg,'online_plan':po.info,'pure_plan':pp.info,
             'bitwise_equal':True,'extra_cached_a_bytes':aq.numel()*aq.element_size()+sa.numel()*sa.element_size(),'timing':timing}
        for name,v in timing.items():
            if name!='quantize':
                v['tflops']=2*m*n*k/(v['median_us']*1e6)
                v['mfu_percent_range']=[100*v['tflops']/max(args.peak_tflops),100*v['tflops']/min(args.peak_tflops)]
        row['online_time_removable_percent']=100*(1-1/timing['pure']['paired_speedup'])
        result['shapes'].append(row);save()
        print((m,n,k),' '.join(f'{name}={v["median_us"]:.3f}us' for name,v in timing.items()),
              f'pure speedup={timing["pure"]["paired_speedup"]:.3f}x, removable={row["online_time_removable_percent"]:.1f}%',flush=True)
        del funcs,po,pp,a,b,bs,aq,sa,out
        gc.collect();torch.cuda.empty_cache()
    result['nvidia_smi_after']=subprocess.check_output(['nvidia-smi'],text=True);save()
if __name__=='__main__':main()
