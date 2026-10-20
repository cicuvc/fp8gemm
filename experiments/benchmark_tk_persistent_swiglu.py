"""Validate and benchmark the experimental fused CUDA gate/up/SwiGLU forward."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_persistent_swiglu import CONFIGS,FLAGS,SwiGLULibrary,build
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


@triton.jit
def reference_epilogue(G,U,O,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    g=tl.load(G+i,i<N,other=0);u=tl.load(U+i,i<N,other=0)
    v=g*(mxtest.fast_tanh(g)+1.)*u
    tl.store(O+i,v,i<N)


def cpu_reference(x,g,gs,u,us):
    # Independent BF16 exponent/reciprocal bit rule and FP64 dot products.
    m,k=x.shape;n=g.shape[0]
    blocks=x.cpu().reshape(m,k//32,32)
    exp=((blocks.view(torch.int16).int()&0x7f80)>>7).amax(-1)
    recip=(((254-exp)<<7)&0x7f80).short().view(torch.bfloat16)
    q=(blocks.float()*recip.float()[...,None]).bfloat16().to(torch.float8_e4m3fn).double()
    up_x=torch.ldexp(q,(exp-127)[...,None]).reshape(m,k)
    gate_x=torch.ldexp(q,((exp-1).clamp_min(0)-127)[...,None]).reshape(m,k)
    def dq(w,s):
        return torch.ldexp(w.cpu().double().reshape(n//32,32,k//32,32),
                           (s.cpu().int()-127)[:,None,:,None]).reshape(n,k)
    half=(gate_x@dq(g,gs).T).float().cuda()
    up=(up_x@dq(u,us).T).float().cuda()
    out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    reference_epilogue[(triton.cdiv(m*n,256),)](half,up,out,m*n,256)
    return out,(half*2).bfloat16(),up.bfloat16()


def triton_fixed(x,g,gs,u,us,o,go,uo):
    m,k=x.shape;n=g.shape[0];aux=go is not None
    return mxtest.fp8_swiglu_fwd.fn[(triton.cdiv(m,32)*triton.cdiv(n,64),)](
        x,g,gs,u,us,o,go if aux else o,uo if aux else o,m,n,k,k,k,gs.stride(0),k,us.stride(0),n,n,n,
        aux,32,64,64,num_warps=4,num_stages=2)


def inputs(m,n,k,edge=False):
    i=torch.arange(m*k).reshape(m,k)
    x=torch.ldexp(((i*73+17)%127-63).float()/64,((i//k+i//32)%5-2).int()).bfloat16().cuda()
    j=torch.arange(n*k).reshape(n,k)
    gf=((j*29+11)%31-15).float()/32
    uf=((j*13+7)%29-14).float()/32
    powers=((j//k//32*3+j%k//32)%7-3).int()
    g,gs=mxtest.launch_weight_quantize(torch.ldexp(gf,powers).cuda())
    u,us=mxtest.launch_weight_quantize(torch.ldexp(uf,-powers).cuda())
    if edge:
        # Exercise e=0 clipping, e=1, high exponent reciprocal corner, zeros,
        # and sign/SiLU saturation without overflow in the dequantized products.
        gs.fill_(127);us.fill_(127)
        # Rescale weights per K32, then each X row uses one matching K block.
        x.zero_()
        for row,e in enumerate([0,1,2,126,127,128,253,254]):
            block=row%(k//32);value=torch.tensor((e<<7)|16,dtype=torch.int16).view(torch.bfloat16)
            x[row::8,block*32:(block+1)*32]=value
            g[:,block*32:(block+1)*32]=.5;u[:,block*32:(block+1)*32]=.25
            gs[:,block]=max(1,min(254,254-e));us[:,block]=max(1,min(254,254-e))
        for start in range(8,m,16):x[start:start+8].neg_()
    return x,g,gs,u,us


def metrics(got,ref):
    d=got.float()-ref.float()
    return {'mismatches':(got!=ref).sum().item(),'relative_l2':(d.norm()/ref.float().norm().clamp_min(1e-30)).item(),
            'max_abs':d.abs().max().item()}


def validate(lib,smoke=False):
    rows=[]
    cases=[(137,192,256,False)] if smoke else [(1,64,128,False),(137,192,128,False),
        (777,320,384,False),(1031,192,256,False),(65,64,256,True)]
    for m,n,k,edge in cases:
        tensors=inputs(m,n,k,edge);x,g,gs,u,us=tensors
        refs=cpu_reference(*tensors)
        tr=[torch.empty_like(refs[0]) for _ in range(3)]
        triton_fixed(*tensors,*tr)
        for name,a,b in zip(['hidden','gate','up'],tr,refs):
            err=metrics(a,b);assert err['relative_l2']<1e-5,(name,'CPU/Triton',err)
        for aux in [False,True]:
            for c in range(len(CONFIGS)):
                # Sentinels detect incomplete TMA output writes, including M tails.
                buffers=[torch.full_like(refs[0],float('nan')) for _ in range(3)]
                p=lib.plan(*tensors,config=c,out=buffers[0],
                           gate_out=buffers[1] if aux else None,up_out=buffers[2] if aux else None,sms=3)
                p();torch.cuda.synchronize()
                errors=[metrics(got,ref) for got,ref in zip(buffers[:3 if aux else 1],refs)]
                assert all(torch.isfinite(v).all() for v in buffers[:3 if aux else 1])
                assert all(e['relative_l2']<1e-5 for e in errors),(m,n,k,edge,aux,c,errors)
                # Changed activation and scale allocations, same descriptors,
                # non-default stream, and zeros produced by persistent reuse.
                saved=x.clone();oldgs=gs.clone();oldus=us.clone()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    x.zero_();gs.fill_(0);us.fill_(0);p()
                stream.synchronize()
                assert all((v==0).all() for v in buffers[:3 if aux else 1]),(c,'reuse')
                x.copy_(saved);gs.copy_(oldgs);us.copy_(oldus)
                # Capture/replay both optional-aux variants.
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):p()
                graph.replay();torch.cuda.synchronize()
                assert metrics(buffers[0],refs[0])['relative_l2']<1e-5
                rows.append({'shape_mnk':[m,n,k],'edge_scales':edge,'config':c,**p.info,'errors':errors})
        print(f'PASS M,N,K={m,n,k}, edge={edge}, all 16 configurations; CPU/Triton, reuse, graph',flush=True)
    return rows


def assembly(output,cuda,fused_silu=True):
    sass=subprocess.check_output([str(cuda/'bin/cuobjdump'),'--dump-sass',str(output/'libswiglu.so')],text=True)
    (output/'sass.log').write_text(sass)
    with (output/'ptx_build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'),*FLAGS,f'-DFP8_SWIGLU_FMA={int(fused_silu)}','-ptx','cuda/persistent_fp8_swiglu.cu',
                        '-o',str(output/'kernel.ptx')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result={}
    for section in sass.split('Function : ')[1:]:
        name=section.splitlines()[0]
        ops=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',section))
        cached=bool(re.search(r'ELb[01]ELb1E',name))
        assert not any(op.startswith(('LDL','STL','CALL')) for op in ops),name
        stores=[op for op in ops if op.startswith('STG')]
        assert (stores and all(op=='STG.E.U8' for op in stores)) if cached else not stores,name
        start=section.index('USETMAXREG.DEALLOC');end=section.index('USETMAXREG.TRY_ALLOC')
        loads=list(re.finditer(r'\sLDG\.[^;]+;',section))
        assert loads and all(start<load.start()<end for load in loads),name
        for prefix in ['QMMA.SF','LDSM','STSM','UTMALDG','UTMASTG','MUFU.TANH']:
            assert any(op.startswith(prefix) for op in ops),(name,prefix)
        # Lexical physical-register audit for the two FP8 B operands, as in
        # the standalone GEMM. Both words must still originate in one LDSM.
        defs={};count=direct=0
        for line in section.splitlines():
            match=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',line)
            if not match:continue
            pc,op,body=match.groups();operands=body.split(',')
            if op.startswith('QMMA'):
                b=int(operands[2].strip()[1:]);origins=[defs.get(b),defs.get(b+1)];count+=1
                if all(o and o[0]=='LDSM' for o in origins) and origins[0][1]==origins[1][1]:direct+=1
            if re.fullmatch(r'R\d+',operands[0].strip()):
                base=int(operands[0].strip()[1:]);width=4 if op.startswith(('LDSM','QMMA')) else 2 if '.64' in op else 1
                for i in range(width):defs[base+i]=('LDSM' if op.startswith('LDSM') else op,pc)
        assert direct==count>0,(name,direct,count)
        result[name]={'opcodes':dict(ops),'qmma':count,'direct_b_pairs':direct}
    assert len(result)==4*len(CONFIGS)
    return result


def sanitize(args):
    result={}
    for tool in ['memcheck','synccheck','initcheck','racecheck']:
        print('Running',tool,flush=True)
        path=args.output/f'{tool}.log'
        with path.open('w') as log:
            subprocess.run([str(args.cuda/'bin/compute-sanitizer'),'--tool',tool,'--error-exitcode','1',sys.executable,
                            __file__,'--skip-build','--validate-only','--smoke','--output',str(args.output),
                            *(['--no-fma'] if args.no_fma else [])],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        text=path.read_text();expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
        assert expected in text and 'all 16 configurations' in text,text[-4000:]
        result[tool]=expected
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/tk_persistent_swiglu')
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    for flag in ['skip-build','validate-only','smoke','sanitize','no-fma']:p.add_argument('--'+flag,action='store_true')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if not args.skip_build:build(args.output,args.cuda,fused_silu=not args.no_fma)
    lib=SwiGLULibrary(args.output);validation=validate(lib,args.smoke)
    if args.validate_only:
        result={'validation':validation}
        if args.sanitize:result['sanitizers']=sanitize(args)
        (args.output/('smoke.json' if args.smoke else 'validation.json')).write_text(json.dumps(result,indent=2)+'\n')
        return
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':triton.__version__,
            'configs':CONFIGS,'flags':FLAGS+[f'-DFP8_SWIGLU_FMA={int(not args.no_fma)}'],
            'fused_silu':not args.no_fma,'validation':validation,'sass':assembly(args.output,args.cuda,not args.no_fma),
            'timing':'CUDA graph; 32 launches per graph; 5 rotated rounds; warm buffers; unlocked clocks; weight quantization excluded',
            'peak_tflops':419.,'mfu_note':'MFU is relative to the supplied nominal 419 TFLOPS; unlocked boost clocks may produce values above 100%, not physical utilization',
            'benchmarks':[]}
    for m,n,k in [(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096)]:
        torch.manual_seed(123)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        g,gs=mxtest.launch_weight_quantize(torch.randn(n,k,device='cuda')/k**.5)
        u,us=mxtest.launch_weight_quantize(torch.randn(n,k,device='cuda')/k**.5)
        for aux in [False,True]:
            go=torch.empty(m,n,device='cuda',dtype=torch.bfloat16) if aux else None
            uo=torch.empty_like(go) if aux else None
            # Autotune once, then benchmark the chosen Triton kernel with
            # preallocated output exactly as the native reusable plans do.
            ref=mxtest.launch_fp8_swiglu(x,g,gs,u,us,go,uo)
            kwargs=mxtest.fp8_swiglu_fwd.best_config.all_kwargs()
            to=torch.empty_like(ref)
            def tf():
                return mxtest.fp8_swiglu_fwd.fn[(triton.cdiv(m,kwargs['BLOCK_M'])*triton.cdiv(n,kwargs['BLOCK_N']),)](
                    x,g,gs,u,us,to,go if aux else to,uo if aux else to,m,n,k,k,k,k//32,k,k//32,n,n,n,aux,**kwargs)
            plans={f'cuda_{c}':lib.plan(x,g,gs,u,us,config=c,gate_out=torch.empty_like(go) if aux else None,
                                       up_out=torch.empty_like(uo) if aux else None) for c in range(len(CONFIGS))}
            errors={}
            for name,plan in plans.items():
                out=plan();errors[name]=metrics(out,ref)
                assert torch.isfinite(out).all() and errors[name]['relative_l2']<5e-4,(m,n,k,aux,name,errors[name])
                if aux:
                    for got,expected in [(plan.gate_out,go),(plan.up_out,uo)]:
                        assert metrics(got,expected)['relative_l2']<5e-4
            times=graph_benchmark({**plans,'triton':tf})
            for name,row in times.items():
                row['tflops']=4*m*n*k/(row['median_us']*1e6);row['mfu_pct']=row['tflops']/419*100
                if name in plans:row.update(plans[name].info);row.update(errors[name])
            best=min(plans,key=lambda name:times[name]['median_us'])
            result['benchmarks'].append({'shape_mnk':[m,n,k],'store_aux':aux,'triton_config':kwargs,'best_cuda':best,'timings':times})
            print(f'{m,n,k} aux={aux}: {best} {times[best]["median_us"]:.2f} us, {times[best]["tflops"]:.1f} TF/s; Triton {times["triton"]["median_us"]:.2f} us',flush=True)
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            del plans
    if args.sanitize:result['sanitizers']=sanitize(args)
    files=['cuda/persistent_fp8_swiglu.cu','cuda/tma_ring_pipe.cuh','experiments/tk_persistent_swiglu.py',
           'experiments/benchmark_tk_persistent_swiglu.py','mxtest.py']
    result['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
