"""Local pure MXFP8 AB dW versus matched BF16-output A-online ABt GEMM."""
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
from experiments.tk_mxfp8_wgrad import Library,CONFIGS,OUTPUT,ColumnQuantizePlan
from experiments.tk_dual_online_gemm import DualOnlineLibrary,CONFIGS as MIXED_CONFIGS
from experiments.retest_tk_mxfp8_tuning import measure
from experiments.benchmark_tk_persistent_gemm import graph_benchmark
from experiments.tk_linear_backward import _quantize_xt


def audit():
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                 str(OUTPUT/'libwgrad.so')],text=True)
    (OUTPUT/'kernels.sass').write_text(raw)
    result={}
    for block in raw.split('Function : ')[1:]:
        name=block.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',block))
        assert not any(o.startswith(('LDL','STL','STG','CALL','HMUL2')) for o in ops),name
        assert not any(o.startswith('F2FP') and 'E4M3' in o for o in ops),name
        for prefix in ['UTMALDG','UTMASTG','LDSM.8.MT1616.2','QMMA.SF','STSM']:
            assert any(o.startswith(prefix) for o in ops),(name,prefix)
        loads=list(re.finditer(r'\sLDG\.[^;]+;',block))
        assert loads and all(x.start()>block.index('USETMAXREG.TRY_ALLOC') for x in loads),name
        assert all(re.match(r'\sLDG\.E(?:\.CONSTANT)?\s',x[0]) for x in loads),name
        definitions={};qmma=direct_b=0;mov_b=set()
        for line in block.splitlines():
            match=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',line)
            if not match:continue
            pc,op,body=match.groups();operands=body.split(',')
            if op.startswith('QMMA'):
                qmma+=1
                b=int(re.match(r'R(\d+)',operands[2].strip())[1])
                origins=[definitions.get(b+i) for i in range(2)]
                direct_b+=int(all(o and o[0].startswith('LDSM') for o in origins) and origins[0][1]==origins[1][1])
                mov_b.update(o[1] for o in origins if o and o[0]=='MOV')
            dest=operands[1].strip() if op.startswith('SHFL') else operands[0].strip()
            if re.fullmatch(r'R\d+',dest):
                size=4 if op.startswith(('LDSM','QMMA')) or '.128' in op else 2 if '.64' in op or op.startswith(('IMAD.WIDE','LEA.WIDE')) else 1
                for i in range(size):definitions[int(dest[1:])+i]=(op,pc)
        assert qmma and direct_b<qmma and mov_b,name
        result[name]={'opcodes':dict(ops),'qmma':qmma,'b_pairs_direct_from_ldsm':direct_b,
                      'distinct_mov_definitions_feeding_b':len(mov_b)}
        if 'ILi16ELi128ELi128ELi2E' in name:
            path=ROOT/'benchmarks/tk_mxfp8_wgrad_c12_rtx5090.sass'
            path.write_text('\n'.join(x.rstrip() for x in ('Function : '+block).splitlines()).rstrip()+'\n')
    assert len(result)==len(CONFIGS)
    return result


def unpack_scales(packed,features,m):
    shifts=8*torch.arange(4,device=packed.device,dtype=torch.int64)
    return ((packed[:,:features].to(torch.int64)[:,:,None]>>shifts)&255).permute(1,0,2).reshape(features,-1)[:,:m//32].to(torch.uint8).contiguous()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_wgrad_rtx5090.json')
    parser.add_argument('--rounds',type=int,default=11)
    parser.add_argument('--repeats',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123)
    lib=Library();mixed=DualOnlineLibrary()
    sources=['cuda/persistent_mxfp8_wgrad.cu','cuda/mxfp8_wgrad_configs.inc','cuda/fp8_shared_load.cuh',
             'experiments/tk_mxfp8_wgrad.py','experiments/benchmark_tk_mxfp8_wgrad.py']
    result={'gpu':torch.cuda.get_device_name(),'shape_convention':'M tokens, N output features, K input features; dW[N,K]',
            'method':'Both outputs BF16. Pure uses A[N,M], B[M,K], prequantized M32 per feature. '
                     'Mixed uses BF16 dY.T[N,M], cached FP8 X.T[K,M], A online quantization. '
                     'GEMM-only excludes all preparation. Sweep 18 pure and 8 mixed tiles with groups 1/13; '
                     'independent top-3 retest with shuffled warm CUDA graphs, unlocked clocks. '
                     'E2E diagnostics below include all current-value operand preparation for each path.',
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            'binary_sha256':hashlib.sha256((OUTPUT/'libwgrad.so').read_bytes()).hexdigest(),
            'sass':audit(),'shapes':[]}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        with args.output.with_suffix('.csv').open('w') as f:
            w=csv.writer(f,lineterminator='\n')
            w.writerow(['M','N','K','pure_config','pure_group','pure_us','pure_tflops','mixed_us','speedup'])
            for r in result['shapes']:
                t=r['retest'];p=t[r['best_pure']];b=t[r['best_mixed']]
                w.writerow([*r['shape_mnk'],r['best_pure'],r['plans'][r['best_pure']]['raster_group'],
                            p['median_us'],p['tflops'],b['median_us'],b['median_us']/p['median_us']])
    for m,n,k in [(2048,1024,1024),(16384,1024,1024),(16384,4096,1024),
                  (16384,1024,4096),(16384,4096,4096)]:
        dy=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        qa=ColumnQuantizePlan(dy,transpose=True);qb=ColumnQuantizePlan(x)
        qa();qb()
        # Same X quantization values, physically transposed for the ABt control.
        qbt=ColumnQuantizePlan(x,transpose=True);qbt()
        sb=unpack_scales(qbt.scales,k,qbt.mp)
        dyt=dy.T.contiguous()
        out=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
        plans={}
        for group in [1,13]:
            for c in range(len(CONFIGS)):
                plans[f'pure_c{c}_g{group}']=lib.plan(qa.out,qb.out,qa.scales,qb.scales,c,out,raster_group=group)
            for c in range(len(MIXED_CONFIGS)):
                plans[f'mixed_c{c}_g{group}']=mixed.plan(dyt,qbt.out,c,out=out,b_scales=sb,raster_group=group)
        ref=plans['mixed_c3_g1']().clone()
        for name,p in plans.items():
            torch.testing.assert_close(p(),ref,rtol=0,atol=0,msg=lambda msg:name+': '+msg)
        sweep=graph_benchmark(plans,rounds=3,repeats=16)
        selected=[]
        for prefix in ['pure','mixed']:
            selected+=sorted((name for name in plans if name.startswith(prefix)),key=lambda name:sweep[name]['median_us'])[:3]
        baseline=next(name for name in selected if name.startswith('mixed'))
        retest=measure({name:plans[name] for name in selected},baseline,args.rounds,args.repeats,987)
        for t in retest.values():t['tflops']=2*m*n*k/t['median_us']/1e6
        best={prefix:min((name for name in retest if name.startswith(prefix)),key=lambda name:retest[name]['median_us'])
              for prefix in ['pure','mixed']}
        pp=plans[best['pure']];bp=plans[best['mixed']]
        def pure_e2e():qa();qb();pp()
        def mixed_prepare():
            dyt.copy_(dy.T)
            _quantize_xt[(qbt.mp//64,k//64)](x,qbt.out,sb,m,k,qbt.mp,*x.stride(),num_warps=4)
        def mixed_e2e():mixed_prepare();bp()
        # Measure the complete paths directly, never by adding standalone times.
        e2e=measure({'pure_gemm':pp,'mixed_gemm':bp,'pure_prepare':lambda:(qa(),qb()),
                     'pure_e2e':pure_e2e,'mixed_prepare':mixed_prepare,'mixed_e2e':mixed_e2e},
                    'mixed_e2e',args.rounds,args.repeats,654)
        row={'shape_mnk':[m,n,k],'bitwise_equal':True,'plans':{name:p.info for name,p in plans.items()},
             'sweep':sweep,'retest':retest,'best_pure':best['pure'],'best_mixed':best['mixed'],
             'e2e_diagnostic':e2e,
             'e2e_note':'Mixed preparation includes Torch dY transpose copy plus the existing Triton '
                        'X-transpose quantizer; pure preparation uses two Triton M32 quantizers. '
                        'This compares these standalone BF16-output paths, not FP8Linear training.'}
        result['shapes'].append(row);save()
        print((m,n,k),best,{key:round(retest[name]['median_us'],3) for key,name in best.items()},
              'pure TFLOPS',round(retest[best['pure']]['tflops'],2),flush=True)
        del plans,pp,bp,p,dy,x,dyt,qa,qb,qbt,sb,out,ref
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
