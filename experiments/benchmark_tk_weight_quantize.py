"""Validate unswizzled WASP weight quantization and compare redux/shuffle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tk_weight_quantize import CONFIGS,QuantLibrary,build
from experiments.tk_persistent_gemm import FLAGS
from experiments.benchmark_tk_persistent_gemm import graph_benchmark
import mxtest


def oracle(weight,round_bf16=False):
    # Independent CPU reference: integer exponent extraction and FP64 ldexp.
    x=weight.cpu().float()
    if round_bf16:
        x=x.bfloat16().float()
    assert torch.isfinite(x).all()
    n,k=x.shape
    nr,kc=(n+31)//32,(k+31)//32
    blocks=F.pad(x,(0,kc*32-k,0,nr*32-n)).reshape(nr,32,kc,32).permute(0,2,1,3).contiguous()
    exponents=((blocks.view(torch.int32)>>23)&255).amax((-1,-2))
    q=torch.ldexp(blocks.double(),(127-exponents)[...,None,None]).float().to(torch.float8_e4m3fn)
    return q.permute(0,2,1,3).reshape(nr*32,kc*32)[:n,:k].contiguous(),exponents.to(torch.uint8)


def data_cases(smoke):
    torch.manual_seed(73)
    shapes=[(777,97)] if smoke else [(1,1),(31,31),(32,32),(33,35),(137,192),(777,97),(1031,320)]
    for n,k in shapes:
        x=torch.randn(n,k)*3
        x[::7,:]=0
        x[::5,::3]=-0.
        yield f'random_{n}_{k}',x,[False,True]
    if smoke:
        return
    # Every finite FP32 exponent; sign/mantissa vary, max positions vary by block.
    block=torch.arange(255)[:,None,None]
    i=torch.arange(1024).reshape(1,32,32)
    bits=(block<<23)|((i*123457+block*991)%0x7e0000)|((i%2)<<31)
    x=bits.to(torch.int32).view(torch.float32).reshape(-1,32)
    yield 'all_finite_exponents',x,[False,True]
    edge=torch.tensor([0x00000000,0x80000000,0x00000001,0x80000001,0x007fffff,0x807fffff,
                       0x00800000,0x80800000,0x7f7fffff,0xff7fffff],dtype=torch.int64).to(torch.int32).view(torch.float32)
    yield 'subnormal_and_max_finite',edge[:,None].repeat(1,1024).reshape(-1,32),[False]
    # BF16/FP8 double-rounding boundary: direct FP32 and compatibility modes
    # must each match their own oracle, not silently use the same conversion.
    ties=torch.tensor([1.0625,1.1875,1.3125,1.4375,1.5625,1.6875,1.8125,1.9375])
    points=torch.stack([torch.nextafter(ties,torch.zeros_like(ties)),ties,
                        torch.nextafter(ties,torch.full_like(ties,2.))]).flatten()
    yield 'rounding_boundaries',points.repeat(128).reshape(96,32),[False,True]
    # Put the unique maximum in every possible position of a 32x32 block.
    x=torch.full((1024,1024),.125)
    x[torch.arange(1024),torch.arange(1024)]=-8.
    yield 'maximum_at_each_position',x.reshape(32768,32),[False,True]


def validate(lib,smoke=False):
    results=[]
    for name,x,modes in data_cases(smoke):
        n,k=x.shape
        input_storage=torch.full((n,(k+3)//4*4+4),2.**80,device='cuda')
        w=input_storage[:,:k]; w.copy_(x)
        expected={bf:oracle(x,bf) for bf in modes}
        for bf in modes:
            refq,refs=expected[bf]
            for config in range(len(CONFIGS)):
                out_bytes=torch.full((n,(k+15)//16*16+16),0xa5,dtype=torch.uint8,device='cuda')
                out=out_bytes.view(torch.float8_e4m3fn)[:,:k]
                scale_storage=torch.full((refs.numel()+32,),255,dtype=torch.uint8,device='cuda')
                scales=scale_storage[16:-16].view_as(refs)
                p=lib.plan(w,config,bf,out,scales,sms=3)
                q,s=p()
                assert torch.equal(q.cpu().view(torch.uint8),refq.view(torch.uint8)),(name,bf,config,'fp8')
                assert torch.equal(s.cpu(),refs),(name,bf,config,'scale')
                padded_k=(k+15)//16*16
                assert (out_bytes[:,k:padded_k]==0).all(),(name,'TMA atom padding')
                assert (out_bytes[:,padded_k:]==0xa5).all(),(name,'output guard')
                assert (scale_storage[:16]==255).all() and (scale_storage[-16:]==255).all(),(name,'scale guard')
                results.append({'case':name,'shape':[n,k],'config':config,**p.info,
                                'fp8_values':n*k,'scale_values':refs.numel(),'mismatches':0})
        # Same plan, same descriptors, different data on a non-default stream.
        w.zero_(); stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            q,s=p()
        stream.synchronize()
        assert (q.view(torch.uint8)==0).all() and (s==0).all(),(name,'reuse')
        print(f'PASS {name}, {len(modes)*len(CONFIGS)} variants + plan reuse',flush=True)
    # Compatibility with the repository's existing preprocessing, including tails.
    for n,k in ([(137,96)] if smoke else [(137,96),(512,512),(1024,2048)]):
        w=torch.randn((n,k),device='cuda')
        gotq,gots=lib.plan(w,round_bf16=True)()
        tq,ts=mxtest.launch_weight_quantize(w)
        assert torch.equal(gotq.view(torch.uint8),tq.view(torch.uint8))
        assert torch.equal(gots,ts)
        print(f'PASS Triton-compatible {n,k}',flush=True)
    return results


def assembly(output,cuda):
    sass=subprocess.check_output([str(cuda/'bin/cuobjdump'),'--dump-sass',str(output/'libquant.so')],text=True)
    (output/'sass.log').write_text(sass)
    with (output/'ptx_build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'),*FLAGS,'-ptx','cuda/persistent_weight_quantize.cu',
                        '-o',str(output/'kernel.ptx')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result={}
    for section in sass.split('Function : ')[1:]:
        name=section.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',section))
        assert not any(op.startswith(('LDG','LDL','STL','CALL','LDSM','QMMA')) for op in ops),name
        assert all(op.startswith('STG.E.U8') for op in ops if op.startswith('STG')),name
        wide='ELi128E' in name
        assert ops['LDS.128']==(32 if wide else 8),(name,ops)
        assert any(op.startswith('UTMALDG') for op in ops) and any(op.startswith('UTMASTG') for op in ops)
        redux=bool(re.search(r'ELb1ELb[01]E',name))
        if redux:
            assert any(op.startswith('REDUX') for op in ops),(name,ops)
            assert not any(op.startswith('SHFL') for op in ops),(name,ops)
        else:
            assert sum(v for op,v in ops.items() if op.startswith('SHFL'))==(3 if wide else 5),(name,ops)
        result[name]=dict(ops)
    assert len(result)==16
    return result


def sanitize(args):
    result={}
    for tool in ['memcheck','synccheck','initcheck','racecheck']:
        print(f'Running {tool}...',flush=True)
        with (args.output/f'{tool}.log').open('w') as log:
            subprocess.run([str(args.cuda/'bin/compute-sanitizer'),'--tool',tool,'--error-exitcode','1',
                            sys.executable,__file__,'--skip-build','--validate-only','--smoke','--output',str(args.output)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
        text=(args.output/f'{tool}.log').read_text()
        expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
        assert expected in text and 'PASS random_777_97' in text,text[-4000:]
        result[tool]=expected
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/tk_weight_quantize')
    parser.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    parser.add_argument('--skip-build',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if not args.skip_build:build(args.output,args.cuda)
    lib=QuantLibrary(args.output)
    validation=validate(lib,args.smoke)
    if args.validate_only:
        (args.output/('smoke.json' if args.smoke else 'validation.json')).write_text(json.dumps(validation,indent=2)+'\n')
        if args.sanitize:
            (args.output/'sanitizers.json').write_text(json.dumps(sanitize(args),indent=2)+'\n')
        return
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'configs':CONFIGS,
            'validation':validation,'sass':assembly(args.output,args.cuda),'benchmarks':[],
            'timing':'CUDA graph, 32 launches/graph, 5 rotated rounds, warm buffers, unlocked clocks',
            'bandwidth_note':'Logical read/write traffic divided by time; cache-resident cases are not DRAM bandwidth measurements',
            'scale_rule':'maximum biased FP32 exponent of each 32x32 block; e=0 means 2^-127; no FP8 range offset',
            'output_layout':'E4M3[N,K], uint8 E8M0[ceil(N/32),ceil(K/32)], optional padded row stride'}
    for n,k in [(1024,1024),(4096,4096),(8192,8192),(16384,8192)]:
        torch.manual_seed(123)
        w=torch.randn((n,k),device='cuda')
        plans={f'direct_{c}':lib.plan(w,c) for c in range(len(CONFIGS))}
        plans['bf16_compat']=lib.plan(w,2,True)
        tq=torch.empty((n,k),device='cuda',dtype=torch.float8_e4m3fn)
        ts=torch.empty((n//32,k//32),device='cuda',dtype=torch.uint8)
        def triton():
            mxtest.fp8_weight_block_quantize[(n//32*k//32,)](w,tq,ts,n,k,k,k,k//32,32,32)
        times=graph_benchmark({**plans,'triton_bf16_compat':triton})
        for name,row in times.items():
            row['logical_GBs']=(5*n*k+(n//32)*(k//32))/(row['median_us']*1000)
            if name in plans:row.update(plans[name].info)
        direct,ds=plans['direct_2']();compat,cs=plans['bf16_compat']()
        # Normal inputs have comparable scales; dequantize before evaluating error.
        def dq(q,s):
            scale=torch.ldexp(torch.ones_like(s,dtype=torch.float32),s.int()-127)
            return (q.float().reshape(n//32,32,k//32,32)*scale[:,None,:,None]).reshape(n,k)
        derror=(dq(direct,ds)-w).norm()/w.norm()
        cerror=(dq(compat,cs)-w).norm()/w.norm()
        result['benchmarks'].append({'shape':[n,k],'timings':times,
                                    'relative_l2_direct':derror.item(),'relative_l2_bf16_compat':cerror.item(),
                                    'different_fp8_fraction':(direct.view(torch.uint8)!=compat.view(torch.uint8)).float().mean().item(),
                                    'different_scales':(ds!=cs).sum().item()})
        best=min((f'direct_{c}' for c in [0,1,2,6]),key=lambda x:times[x]['median_us'])
        print(f'{n,k}: {best} {times[best]["median_us"]:.2f} us {times[best]["logical_GBs"]:.1f} GB/s; '
              f'Triton {times["triton_bf16_compat"]["median_us"]:.2f} us',flush=True)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.sanitize:result['sanitizers']=sanitize(args)
    files=[ROOT/'cuda/persistent_weight_quantize.cu',ROOT/'cuda/linear_tma_buffer.cuh',ROOT/'cuda/tma_ring_pipe.cuh',
           ROOT/'experiments/tk_weight_quantize.py',Path(__file__).resolve(),
           ROOT/'experiments/tk_persistent_gemm.py',ROOT/'experiments/benchmark_tk_persistent_gemm.py']
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
