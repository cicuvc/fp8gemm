"""Local prequantized MXFP8 dX versus existing BF16-online swapAB dX."""
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
from experiments.tk_mxfp8_dgrad import Library,CONFIGS,OUTPUT,RowQuantizePlan
from experiments.tk_persistent_gemm import GemmLibrary,DEFAULT_OUTPUT
from experiments.retest_tk_mxfp8_tuning import measure
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def audit():
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                 str(OUTPUT/'libdgrad.so')],text=True)
    (OUTPUT/'kernels.sass').write_text(raw)
    result={}
    for block in raw.split('Function : ')[1:]:
        name=block.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',block))
        assert not any(o.startswith(('LDL','STL','STG','CALL','HMUL2')) for o in ops),name
        assert not any(o.startswith('F2FP') and 'E4M3' in o for o in ops),name
        for prefix in ['UTMALDG','UTMASTG','LDSM.8.MT1616.2','LDSM.16.M88.4','QMMA.SF','STSM.16.MT88.4']:
            assert any(o.startswith(prefix) for o in ops),(name,prefix)
        loads=list(re.finditer(r'\sLDG\.[^;]+;',block))
        assert loads and all(x.start()>block.index('USETMAXREG.TRY_ALLOC') for x in loads),name
        # Only 32-bit packed dY scales and byte weight scales read from global.
        assert all(re.match(r'\sLDG\.E(?:\.U8)?(?:\.CONSTANT)?\s',x[0]) for x in loads),name
        definitions={};qmma=direct_a=direct_b=0
        # Lexical def-use audit of the emitted unrolled body, not a CFG analysis.
        for line in block.splitlines():
            match=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',line)
            if not match:continue
            pc,op,body=match.groups();operands=body.split(',')
            if op.startswith('QMMA'):
                qmma+=1
                a=int(re.match(r'R(\d+)',operands[1].strip())[1])
                b=int(re.match(r'R(\d+)',operands[2].strip())[1])
                da=[definitions.get(a+i) for i in range(4)]
                db=[definitions.get(b+i) for i in range(2)]
                direct_a+=int(all(o and o[0]=='LDSM.8.MT1616.2' for o in da) and len({o[1] for o in da})==1)
                direct_b+=int(all(o and o[0]=='LDSM.16.M88.4' for o in db) and len({o[1] for o in db})==1)
            dest=operands[1].strip() if op.startswith('SHFL') else operands[0].strip()
            if re.fullmatch(r'R\d+',dest):
                size=4 if op.startswith(('LDSM','QMMA')) or '.128' in op else 2 if '.64' in op or op.startswith(('IMAD.WIDE','LEA.WIDE')) else 1
                for i in range(size):definitions[int(dest[1:])+i]=(op,pc)
        assert qmma and direct_a==qmma and direct_b==qmma,(name,qmma,direct_a,direct_b)
        result[name]={'opcodes':dict(ops),'qmma':qmma,'a_quads_direct_from_transpose_ldsm':direct_a,
                      'b_pairs_direct_from_row_ldsm':direct_b}
        if 'ILi16ELi128ELi128ELi2E' in name:
            path=ROOT/'benchmarks/tk_mxfp8_dgrad_c12_rtx5090.sass'
            path.write_text('\n'.join(x.rstrip() for x in ('Function : '+block).splitlines()).rstrip()+'\n')
    assert len(result)==len(CONFIGS)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_mxfp8_dgrad_rtx5090.json')
    parser.add_argument('--rounds',type=int,default=11)
    parser.add_argument('--repeats',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123)
    lib=Library();mixed=GemmLibrary()
    sources=['cuda/persistent_mxfp8_dgrad.cu','cuda/mxfp8_dgrad_configs.inc','cuda/fp8_shared_load.cuh',
             'experiments/tk_mxfp8_dgrad.py','experiments/benchmark_tk_mxfp8_dgrad.py']
    result={'gpu':torch.cuda.get_device_name(),'shape_convention':'M tokens, N output features, K input features; dX[M,K]',
            'method':'Both BF16 outputs; both swapAB with original FP8 W[N,K] and identical 32x32 weight scales. '
                     'Pure consumes prequantized dY[M,N] and packed row/N32 scales; mixed quantizes BF16 dY online. '
                     'GEMM-only excludes operand preparation. Sweep 18 pure tiles with groups 1/13 and 9 mixed tiles '
                     'with group 1 (current mixed AB API). Independent top-3 retest per path, shuffled warm CUDA graphs, '
                     'unlocked clocks. E2E diagnostic includes separate Triton row quantization for pure, weight cached.',
            'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            'binary_sha256':hashlib.sha256((OUTPUT/'libdgrad.so').read_bytes()).hexdigest(),
            'mixed_binary_sha256':hashlib.sha256((DEFAULT_OUTPUT/'libgemm.so').read_bytes()).hexdigest(),
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
        wq,sw=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda',dtype=torch.float32))
        q=RowQuantizePlan(dy);q()
        out=torch.empty((m,k),device='cuda',dtype=torch.bfloat16)
        plans={}
        for group in [1,13]:
            for c in range(len(CONFIGS)):
                plans[f'pure_c{c}_g{group}']=lib.plan(q.out,wq,q.scales,sw,c,out,raster_group=group)
        for c in [0,1,2,3,4,6,7,8,9]:
            plans[f'mixed_c{c}_g1']=mixed.plan_ab(dy,wq,sw,c,out,swap_ab=True)
        ref=plans['mixed_c8_g1']().clone()
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
        def pure_e2e():q();pp()
        e2e=measure({'pure_gemm':pp,'mixed_gemm':bp,'row_quantize':q,'pure_e2e':pure_e2e},
                    'mixed_gemm',args.rounds,args.repeats,654)
        row={'shape_mnk':[m,n,k],'bitwise_equal':True,'plans':{name:p.info for name,p in plans.items()},
             'sweep':sweep,'retest':retest,'best_pure':best['pure'],'best_mixed':best['mixed'],
             'e2e_diagnostic':e2e}
        result['shapes'].append(row);save()
        print((m,n,k),best,{key:round(retest[name]['median_us'],3) for key,name in best.items()},
              'pure TFLOPS',round(retest[best['pure']]['tflops'],2),flush=True)
        del plans,pp,bp,p,dy,q,wq,sw,out,ref
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
