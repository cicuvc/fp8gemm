"""Validate and benchmark contiguous BF16/BF16 AB^T with online MXFP8."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tk_dual_online_gemm import CONFIGS, NATIVE_DW_CONFIGS, DEFAULT_OUTPUT, DualOnlineLibrary, build
from experiments.tk_persistent_gemm import FLAGS
from experiments.benchmark_tk_persistent_gemm import quantized_a, graph_benchmark
import mxtest


@triton.autotune(configs=mxtest.GEMM_CONFIGS, key=['M', 'N', 'K', 'FP32'], cache_results=True)
@triton.jit
def triton_dual(A, B, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                FP32: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    mi = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    ni = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ki = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + ki
        a = tl.load(A + mi[:, None]*K + kk[None, :], (mi[:, None]<M)&(kk[None, :]<K), 0)
        b = tl.load(B + ni[:, None]*K + kk[None, :], (ni[:, None]<N)&(kk[None, :]<K), 0)
        aq, sa = mxtest.online_scale(a)
        bq, sb = mxtest.online_scale(b)
        acc = tl.dot_scaled(aq, sa, 'e4m3', tl.trans(bq), sb, 'e4m3', acc)
    tl.store(O + mi[:, None]*N + ni[None, :], acc, (mi[:, None]<M)&(ni[None, :]<N))


def exact_input(rows, k, seed):
    i = torch.arange(rows*k).reshape(rows, k)
    return torch.ldexp(((i*seed+17)%255-127).float()/64,
                       ((i//k+i//32)%5-2).int()).bfloat16()


def validate(lib, smoke=False):
    results = []
    shapes = [(137, 192, 128)] if smoke else [(1, 64, 128), (137, 192, 128), (777, 320, 384), (1031, 192, 256)]
    for m, n, k in shapes:
        ac, bc = exact_input(m, k, 73), exact_input(n, k, 29)
        ref32 = (quantized_a(ac).double() @ quantized_a(bc).double().T).float()
        a, b = ac.cuda(), bc.cuda()
        bq, bs = mxtest.launch_row_quantize(b)
        for dtype in [torch.bfloat16, torch.float32]:
            ref = ref32.to(dtype)
            # NaN sentinel checks TMA coverage; initialization also lets
            # initcheck track the destination across CPU copies of TMA output.
            guard = torch.full((m*n+32,), float('nan'), dtype=dtype, device='cuda')
            out = guard[16:-16].view(m, n)
            for config in range(len(CONFIGS)):
                for online in [True, False]:
                    p = lib.plan(a, b if online else bq, config, out=out, output_dtype=dtype,
                                 b_scales=None if online else bs, sms=3)
                    out.fill_(float('nan'))
                    got = p().cpu()
                    assert torch.equal(got, ref), (m,n,k,config,online,str(dtype),
                        (got.float()-ref.float()).abs().max().item(), (got!=ref).sum().item())
                    assert torch.isnan(guard[:16]).all() and torch.isnan(guard[-16:]).all()
                    # Changed data and zero K32 blocks through existing descriptors.
                    a[:,32:64] = 0
                    b[:,64:96] = 0
                    refresh_b(b, bq, bs)
                    changed = (quantized_a(a.cpu()).double() @ quantized_a(b.cpu()).double().T).to(dtype)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream): p()
                    torch.cuda.current_stream().wait_stream(stream)
                    assert torch.equal(out.cpu(), changed), (config, online, 'reuse/stream')
                    a.copy_(ac); b.copy_(bc); refresh_b(b,bq,bs)
                    results.append({'shape_mnk':[m,n,k], 'config':config, **p.info,
                                    'dtype':str(dtype), 'exact':True, 'reuse_and_stream':True, 'guards':True})
        print(f'PASS exact/reuse/stream/guards {m,n,k}: all 32 variants', flush=True)
    return results


def refresh_b(b, q, s):
    n,k = b.shape
    mxtest.fp8_row_quantize[(triton.cdiv(n,64)*triton.cdiv(k,64),)](b,q,s,n,k,k,k,k//32,64,64)


def assembly(output, cuda):
    sass = subprocess.check_output([str(cuda/'bin/cuobjdump'), '--dump-sass', str(output/'libdual.so')], text=True)
    (output/'sass.log').write_text(sass)
    with (output/'ptx_build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'), *FLAGS, '-ptx', 'cuda/persistent_dual_online_gemm.cu',
                        '-o', str(output/'kernel.ptx')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {}
    for section in sass.split('Function : ')[1:]:
        name = section.splitlines()[0]
        ops = Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)', section))
        for prefix in ['UTMALDG', 'UTMASTG', 'LDSM', 'QMMA.SF', 'USETMAXREG']:
            assert any(op.startswith(prefix) for op in ops), (name, prefix)
        assert not any(op.startswith(('STG','LDL','STL','CALL')) for op in ops), (name,ops)
        flags=re.search(r'ELb([01])ELb([01])ELb([01])ELb([01])E',name)
        assert flags,name
        online,fp32,native_dy,swap_ab=[v=='1' for v in flags.groups()]
        loads = list(re.finditer(r'\sLDG\.[^;]+;', section))
        if online: assert not loads, name
        else:
            producer = section.index('USETMAXREG.DEALLOC')
            consumer = section.index('USETMAXREG.TRY_ALLOC')
            assert loads and all(producer < load.start() < consumer for load in loads), name
        definitions,qmma,direct={},0,0
        for line in section.splitlines():
            match=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',line)
            if not match:continue
            pc,op,body=match.groups();operands=[o.strip() for o in body.split(',')]
            if op.startswith('QMMA'):
                qmma+=1
                base=int(re.match(r'R(\d+)',operands[1 if swap_ab else 2])[1])
                origins=[definitions.get(base+i) for i in range(4 if swap_ab else 2)]
                if (all(o and o[0]=='LDSM.16.M88.4' for o in origins)
                        and len({o[1] for o in origins})==1):direct+=1
            dest=operands[1] if op.startswith('SHFL') else operands[0]
            if re.fullmatch(r'R\d+',dest):
                base=int(dest[1:])
                width=(4 if op.startswith(('QMMA','LDSM')) or '.128' in op else
                       2 if '.64' in op or op.startswith(('IMAD.WIDE','LEA.WIDE')) else 1)
                for i in range(width):definitions[base+i]=(op,pc)
        if not online:assert direct==qmma>0,(name,direct,qmma)
        if native_dy:assert 'LDSM.16.MT88.4' in ops,name
        result[name] = {'opcodes':dict(ops), 'online_b':online,'fp32_output':fp32,
                        'native_dy':native_dy,'swap_ab':swap_ab,
                        'qmma_count':qmma,'cached_x_operands_direct_from_ldsm':direct,
                        'consumer_direct_global_access':False}
    assert len(result)==6*len(CONFIGS)+len(NATIVE_DW_CONFIGS)-len(CONFIGS)
    return result


def sanitize(args):
    results={}
    for tool in ['memcheck','synccheck','initcheck','racecheck']:
        print('Running',tool,flush=True)
        with (args.output/f'{tool}.log').open('w') as log:
            subprocess.run([str(args.cuda/'bin/compute-sanitizer'),'--tool',tool,'--error-exitcode','1',
                            sys.executable,__file__,'--skip-build','--validate-only','--smoke','--output',str(args.output)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        text=(args.output/f'{tool}.log').read_text()
        summary='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
        assert summary in text and 'all 32 variants' in text, text[-4000:]
        results[tool]=summary
    return results


def benchmark(lib, args):
    results=[]
    shapes=[(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096)]
    for m,n,k in shapes:
        torch.manual_seed(123)
        a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        b=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
        bq,bs=mxtest.launch_row_quantize(b)
        ref=quantized_a(a) @ quantized_a(b).T
        true_ref=a.float() @ b.float().T
        for dtype in [torch.bfloat16,torch.float32]:
            out=torch.empty((m,n),device='cuda',dtype=dtype)
            plans={f'{mode}_{i}':lib.plan(a,b if mode=='online' else bq,i,out=out,output_dtype=dtype,
                        b_scales=None if mode=='online' else bs)
                   for mode in ['online','cached'] for i in range(len(CONFIGS))}
            errors={}
            for name,p in plans.items():
                got=p().float()
                rel=((got-ref.to(dtype).float()).norm()/ref.norm()).item()
                assert torch.isfinite(got).all() and rel<(5e-4 if dtype==torch.bfloat16 else 1e-5), (name,rel)
                errors[name]={'relative_l2_vs_quantized':rel,
                             'relative_l2_vs_bf16_inputs':((got-true_ref).norm()/true_ref.norm()).item()}
            def tri():
                return triton_dual[lambda meta:(triton.cdiv(m,meta['BLOCK_M']),triton.cdiv(n,meta['BLOCK_N']))](
                    a,b,out,m,n,k,dtype==torch.float32)
            tri()
            tri_error=((out.float()-ref.to(dtype).float()).norm()/ref.norm()).item()
            assert tri_error<(5e-4 if dtype==torch.bfloat16 else 1e-5), tri_error
            times=graph_benchmark({**plans,'triton':tri,'quantize_b':lambda:refresh_b(b,bq,bs)},args.rounds,args.repeats)
            best_online=min((v for v in plans if v.startswith('online')),key=lambda v:times[v]['median_us'])
            best_cached=min((v for v in plans if v.startswith('cached')),key=lambda v:times[v]['median_us'])
            # A second paired run compares complete alternatives, with allocations excluded.
            def fresh(): refresh_b(b,bq,bs); plans[best_cached]()
            paired=graph_benchmark({'online':plans[best_online],'cached':plans[best_cached],
                                    'refresh_b_and_cached':fresh,'triton':tri},args.rounds,args.repeats)
            for name,t in times.items():
                if name!='quantize_b':t['tflops']=2*m*n*k/(t['median_us']*1e6)
                if name in plans:t.update(plans[name].info); t.update(errors[name])
            for t in paired.values():t['tflops']=2*m*n*k/(t['median_us']*1e6)
            row={'shape_mnk':[m,n,k],'dtype':str(dtype),'best_online':best_online,'best_cached':best_cached,
                 'timings':times,'paired':paired,'triton_config':str(triton_dual.best_config),'triton_relative_l2':tri_error}
            results.append(row)
            (args.output/'timings.json').write_text(json.dumps(results,indent=2)+'\n')
            print(f'{m,n,k} {dtype}: {best_online} {paired["online"]["median_us"]:.3f} us, '
                  f'{best_cached} {paired["cached"]["median_us"]:.3f}, refresh+cached '
                  f'{paired["refresh_b_and_cached"]["median_us"]:.3f}, Triton {paired["triton"]["median_us"]:.3f}',flush=True)
            del plans
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    for flag in ['skip-build','validate-only','smoke','sanitize']:p.add_argument('--'+flag,action='store_true')
    p.add_argument('--rounds',type=int,default=7);p.add_argument('--repeats',type=int,default=32)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if not args.skip_build:build(args.output,args.cuda)
    lib=DualOnlineLibrary(args.output)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'configs':CONFIGS,
            'validation':validate(lib,args.smoke)}
    if not args.validate_only:
        result['sass']=assembly(args.output,args.cuda)
        result['benchmarks']=benchmark(lib,args)
        result['timing']='CUDA graphs, preallocated buffers, warm resident inputs; rotated order; no locked clocks; transpose excluded'
        result['rounds']=args.rounds;result['repeats']=args.repeats
    if args.sanitize:result['sanitizers']=sanitize(args)
    files=[Path(__file__),ROOT/'cuda/persistent_dual_online_gemm.cu',ROOT/'cuda/tma_ring_pipe.cuh',
           ROOT/'experiments/tk_dual_online_gemm.py',ROOT/'mxtest.py',
           *sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))]
    result['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    name='smoke.json' if args.smoke else 'validation.json' if args.validate_only else 'result.json'
    (args.output/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
