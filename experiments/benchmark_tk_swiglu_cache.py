"""Measure forward overhead of caching transpose-quantized X for weight gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F
import triton

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_persistent_swiglu import CONFIGS,SwiGLULibrary,build
from experiments.benchmark_tk_persistent_swiglu import inputs,assembly
from experiments.benchmark_tk_persistent_gemm import graph_benchmark
from experiments.cached_fp8_wgrad import _pack_columns,packed_wgrad


def oracle(x):
    m,k=x.shape;mp=(m+127)//128*128
    z=F.pad(x.cpu().T.contiguous(),(0,mp-m)).reshape(k,mp//32,32)
    exp=((z.view(torch.int16).int()&0x7f80)>>7).amax(-1)
    reciprocal=(((254-exp)<<7)&0x7f80).short().view(torch.bfloat16)
    q=(z.float()*reciprocal.float()[...,None]).bfloat16().to(torch.float8_e4m3fn).reshape(k,mp)
    return q,exp.byte()


def pack(x,q,s):
    m,k=x.shape;mp=q.shape[1]
    _pack_columns[(mp//64,triton.cdiv(k,64))](x,x,q,s,q,s,m,k,k,mp,False,BM=64,BF=64,num_warps=4)


def guarded(shape,dtype,device):
    # Guards surround a contiguous, aligned sub-allocation. Tail tokens are
    # inside the descriptor and must be explicitly zeroed by the kernel.
    size=1
    for d in shape:size*=d
    storage=torch.full((size+256,),0x7f,device=device,dtype=torch.uint8)
    return storage[128:-128].view(dtype).view(shape),storage


def validate(lib,smoke=False):
    rows=[]
    shapes=[(777,192,256,False)] if smoke else [(1,64,128,False),(31,192,128,False),
        (33,64,256,False),(137,192,256,False),(777,320,384,False),(1031,192,256,False),(65,64,256,True)]
    for m,n,k,edge in shapes:
        args=inputs(m,n,k,edge);x,g,gs,u,us=args
        qref,sref=oracle(x);mp=qref.shape[1]
        tq=torch.empty_like(qref,device='cuda');ts=torch.empty_like(sref,device='cuda')
        pack(x,tq,ts)
        assert torch.equal(tq.cpu().view(torch.uint8),qref.view(torch.uint8))
        assert torch.equal(ts.cpu(),sref)
        for aux in [False,True]:
            for c in range(len(CONFIGS)):
                bufs=[torch.full((m,n),float('nan'),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                # Initcheck does not mark TMA output writes initialized before
                # device copies. NaN sentinels independently catch missing writes.
                plain=lib.plan(*args,config=c,out=bufs[0],gate_out=bufs[1] if aux else None,up_out=bufs[2] if aux else None,sms=3)
                plain();torch.cuda.synchronize()
                expected=[plain.out.clone()]+([bufs[1].clone(),bufs[2].clone()] if aux else [])
                for buf in bufs:buf.fill_(float('nan'))
                cq,qstorage=guarded((k,mp),torch.float8_e4m3fn,'cuda')
                cs,sstorage=guarded((k,mp//32),torch.uint8,'cuda')
                cached=lib.plan(*args,config=c,out=bufs[0],gate_out=bufs[1] if aux else None,
                    up_out=bufs[2] if aux else None,save_x=True,cached_x=cq,cached_scale=cs,sms=3)
                cached();torch.cuda.synchronize()
                assert torch.equal(cq.cpu().view(torch.uint8),qref.view(torch.uint8)),(m,n,k,c,aux,'fp8')
                assert torch.equal(cs.cpu(),sref),(m,n,k,c,aux,'scale')
                assert all(torch.equal(v,e) for v,e in zip(bufs,expected)),(m,n,k,c,aux,'forward')
                for storage in [qstorage,sstorage]:
                    assert (storage[:128]==0x7f).all() and (storage[-128:]==0x7f).all(),(c,'guard')
                assert (cq[:,m:].view(torch.uint8)==0).all(),(c,'padding')
                # Clear input, re-use the same persistent plan on another stream.
                saved=x.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):x.zero_();cached()
                stream.synchronize()
                assert (cq.view(torch.uint8)==0).all() and (cs==0).all(),(c,'reuse')
                assert all((v==0).all() for v in bufs[:3 if aux else 1])
                x.copy_(saved)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):cached()
                graph.replay();torch.cuda.synchronize()
                assert torch.equal(cq.cpu().view(torch.uint8),qref.view(torch.uint8))
                assert torch.equal(cs.cpu(),sref)
                rows.append({'shape_mnk':[m,n,k],'edge':edge,'config':c,**cached.info,
                             'cache_bytes':k*mp+k*mp//32,'fp8_mismatches':0,'scale_mismatches':0,'forward_mismatches':0})
        print(f'PASS cache {m,n,k}, edge={edge}, all 16 configurations, reference/reuse/graph/guards',flush=True)
    # Consume the cache with the existing packed FP8 weight-gradient GEMM.
    # This checks the actual future-B orientation, not just a byte-level pack.
    m,n,k=137,64,128
    args=inputs(m,n,k);x=args[0]
    plan=lib.plan(*args,save_x=True,sms=3)
    plan.cached_x.view(torch.uint8).fill_(0x7f)  # TMA/initcheck limitation above.
    plan()
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    dq=torch.empty((n,plan.cached_x.shape[1]),device='cuda',dtype=torch.float8_e4m3fn)
    ds=torch.empty((n,dq.shape[1]//32),device='cuda',dtype=torch.uint8)
    pack(dy,dq,ds)
    dw=packed_wgrad(dq,ds,plan.cached_x,plan.cached_scale)
    def dq_cpu(q,s):
        return torch.ldexp(q.cpu().double().reshape(q.shape[0],-1,32),
                           (s.cpu().int()-127)[...,None]).reshape(q.shape)
    expected=(dq_cpu(dq,ds)@dq_cpu(plan.cached_x,plan.cached_scale).T).float()
    assert torch.equal(dw.cpu(),expected),'cached B wgrad orientation'
    print('PASS cached X -> existing FP8 wgrad',flush=True)
    return rows


def sanitize(args):
    results={}
    for tool in ['memcheck','synccheck','initcheck','racecheck']:
        path=args.output/f'cache_{tool}.log'
        with path.open('w') as f:
            subprocess.run([str(args.cuda/'bin/compute-sanitizer'),'--tool',tool,'--error-exitcode','1',sys.executable,
                __file__,'--skip-build','--validate-only','--smoke','--output',str(args.output)],
                stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=240,check=True)
        text=path.read_text();expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
        assert expected in text and 'all 16 configurations' in text,text[-4000:]
        results[tool]=expected;print(tool,expected,flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/tk_persistent_swiglu')
    parser.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    for flag in ['skip-build','validate-only','smoke','sanitize']:parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if not args.skip_build:build(args.output,args.cuda)
    lib=SwiGLULibrary(args.output)
    result={'gpu':torch.cuda.get_device_name(),'configs':CONFIGS,'validation':validate(lib,args.smoke),
            'timing':'Preallocated CUDA graphs; 32 launches per graph; 9 rotated rounds; unlocked clocks; weight quantization excluded',
            'cache_layout':'X_T[K,ceil(M/128)*128] E4M3, scales[K,ceil(M/128)*4] E8M0; scales reduce M32',
            'benchmarks':[]}
    if not args.validate_only:
        result['sass']=assembly(args.output,args.cuda)
        for m,n,k in [(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096),
                      (4096,256,4096)]:
            torch.manual_seed(123)
            x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            g,gs=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda')/k**.5)
            u,us=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda')/k**.5)
            for aux in [False,True]:
                plans={}
                for config in [0,1,3,4]:
                    for save in [False,True]:
                        go=torch.empty((m,n),device='cuda',dtype=torch.bfloat16) if aux else None
                        uo=torch.empty_like(go) if aux else None
                        plans[f'c{config}_save{int(save)}']=lib.plan(x,g,gs,u,us,config=config,gate_out=go,up_out=uo,save_x=save)
                # Compare fusion to the existing standalone transpose/quantize
                # kernel with identical padded outputs and no extra allocations.
                mp=(m+127)//128*128
                cq=torch.empty((k,mp),device='cuda',dtype=torch.float8_e4m3fn)
                cs=torch.empty((k,mp//32),device='cuda',dtype=torch.uint8)
                packing=lambda:pack(x,cq,cs)
                functions=dict(plans)
                functions['triton_pack_only']=packing
                for config in [0,1,3,4]:
                    def separate(p=plans[f'c{config}_save0']):
                        p();packing()
                    functions[f'c{config}_separate']=separate
                times=graph_benchmark(functions,rounds=9,repeats=32)
                for name,p in plans.items():times[name].update(p.info)
                comparisons={}
                for config in [0,1,3,4]:
                    base=times[f'c{config}_save0']['median_us'];saved=times[f'c{config}_save1']['median_us']
                    comparisons[str(config)]={'added_us':saved-base,'overhead_pct':100*(saved/base-1),
                        'fused_vs_separate_speedup':times[f'c{config}_separate']['median_us']/saved}
                best_base=min((f'c{c}_save0' for c in [0,1,3,4]),key=lambda key:times[key]['median_us'])
                best_saved=min((f'c{c}_save1' for c in [0,1,3,4]),key=lambda key:times[key]['median_us'])
                result['benchmarks'].append({'shape_mnk':[m,n,k],'store_aux':aux,'cache_bytes':k*mp+k*mp//32,
                    'timings':times,'comparisons':comparisons,'best_base':best_base,'best_cached':best_saved})
                print(f'{m,n,k}, aux={aux}: best {best_base} {times[best_base]["median_us"]:.2f} -> '
                    f'{best_saved} {times[best_saved]["median_us"]:.2f} us; config3 overhead {comparisons["3"]["overhead_pct"]:.2f}%',flush=True)
                (args.output/'cache_result.json').write_text(json.dumps(result,indent=2)+'\n')
                del plans,functions
    if args.sanitize:result['sanitizers']=sanitize(args)
    files=['cuda/persistent_fp8_swiglu.cu','cuda/tma_ring_pipe.cuh','experiments/tk_persistent_swiglu.py',
           'experiments/benchmark_tk_swiglu_cache.py','experiments/benchmark_tk_persistent_swiglu.py']
    result['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    name='cache_smoke.json' if args.smoke else 'cache_validation.json' if args.validate_only else 'cache_result.json'
    (args.output/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
