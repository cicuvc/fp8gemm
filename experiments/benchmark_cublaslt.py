"""cuBLASLt BF16/FP8/MXFP8 versus TK online-A GEMM, with heuristic search."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.cublaslt_gemm import Library
from experiments.tk_persistent_gemm import GemmLibrary
from experiments.benchmark_tk_forward_pipeline import measure

@triton.jit
def quantize_mx(x,q,s,M:tl.constexpr,K:tl.constexpr):
    r=tl.program_id(0)*32+tl.arange(0,32)
    k=tl.program_id(1)*128+tl.arange(0,128)
    a=tl.load(x+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),other=0)
    v,sc=mxtest.online_scale(a)
    tl.store(q+r[:,None]*K+k[None,:],v,(r[:,None]<M)&(k[None,:]<K))
    kb=tl.program_id(1)*4+tl.arange(0,4)
    offset=(r[:,None]//128)*(tl.cdiv(K,128)*512)+(kb[None,:]//4)*512+(r[:,None]%32)*16+((r[:,None]%128)//32)*4+kb[None,:]%4
    tl.store(s+offset,sc,r[:,None]<M)

def pack_scales(s):
    r,k=s.shape
    out=torch.zeros((triton.cdiv(r,128)*128,triton.cdiv(k,4)*4),device=s.device,dtype=torch.uint8)
    out[:r,:k]=s
    # [outer128,outer4,outer32,inner4groups,inner4] -> tiled cuBLAS layout.
    return out.reshape(out.shape[0]//128,4,32,out.shape[1]//4,4).permute(0,3,2,1,4).contiguous().flatten()

def error(out,ref):
    diff=out.float()-ref
    return {'relative_l2':(diff.norm()/ref.norm()).item(),'max_abs':diff.abs().max().item()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=11)
    ap.add_argument('--repeats',type=int,default=64)
    ap.add_argument('--shape',type=int,nargs=3,action='append')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lib=Library();tk=GemmLibrary()
    workspace=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    data={'gpu':torch.cuda.get_device_name(),'cublas_version':lib.lib.lt_version(),
          'library_maps':sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in line)),
          'method':'Warm CUDA graphs, shared output/workspace, shuffled and rotated order, unlocked clocks; 64 heuristic candidates requested per mode, FP32 compute and split-K reduction, FAST_ACCUM=0. MX A quantization uses original online_scale and writes cuBLAS tiled scales in the same kernel. B FP8 and scale packing are offline in all FP8 comparisons.',
          'workspace_limit_bytes':workspace.numel(),'peak_fp8_tflops':[877.9,1007.6],
          'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['cuda/cublaslt_benchmark.cu','experiments/cublaslt_gemm.py','experiments/benchmark_cublaslt.py']},'shapes':[]}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2)+'\n')
    for m,n,k in args.shape or [(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096)]:
        torch.manual_seed(123)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        w=torch.randn((n,k),device='cuda').bfloat16()
        wq,ws=mxtest.launch_weight_quantize(w)
        xq,xs=mxtest.launch_row_quantize(x)
        sx,sw=pack_scales(xs),pack_scales(ws.repeat_interleave(32,0))
        # Verify direct packed quantization against the existing quantizer.
        aq=torch.empty_like(xq);sa=torch.empty_like(sx)
        quant=lambda:quantize_mx[(triton.cdiv(m,32),triton.cdiv(k,128))](x,aq,sa,m,k,num_warps=4)
        quant()
        assert torch.equal(aq.view(torch.uint8),xq.view(torch.uint8)) and torch.equal(sa,sx)
        tensor_sx=(x.float().abs().max()/448).reshape(1)
        tensor_sw=(w.float().abs().max()/448).reshape(1)
        tx=(x.float()/tensor_sx).to(torch.float8_e4m3fn)
        tw=(w.float()/tensor_sw).to(torch.float8_e4m3fn)
        refs={
          'bf16':x.float()@w.float().T,
          'fp8_tensor':(tx.float()*tensor_sx)@(tw.float()*tensor_sw).T,
          'mxfp8':mxtest.dequantize_row(xq,xs)@mxtest.dequantize_weight(wq,ws).T,
        }
        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        config=7 if triton.cdiv(m,128)*triton.cdiv(n,128)<torch.cuda.get_device_properties(0).multi_processor_count else 8
        plan=tk.plan(x,wq,config,out=out,scales=ws)
        row={'shape_mnk':[m,n,k],'tk_config':config,'quantizer_bitwise_equal':True,'modes':{}}
        data['shapes'].append(row)
        functions={'tk_online_a':plan,'torch_bf16':lambda:torch.mm(x,w.T,out=out)}
        plan();row['tk_error']=error(out,refs['mxfp8']);assert row['tk_error']['relative_l2']<.004
        owned=[]
        for name,mode,a,b,as_,bs_ in [('bf16',0,x,w,None,None),('fp8_tensor',1,tx,tw,tensor_sx,tensor_sw),('mxfp8',2,aq,wq,sa,sw)]:
            info={};row['modes'][name]=info
            try:
                p=lib.plan(a,b,as_,bs_,out,workspace,mode)
            except RuntimeError as e:
                info['unsupported']=str(e);print(name,str(e),flush=True);continue
            owned.append(p)
            candidates={};info['algorithms']={}
            for i in range(p.count):
                try:
                    p.launch(i);torch.cuda.synchronize()
                    e=error(out,refs[name])
                    assert torch.isfinite(out).all() and e['relative_l2']<.004,e
                    key=f'{name}_{i}'
                    info['algorithms'][key]=p.info(i)|{'error':e}
                    # Some MX algorithms return zero for optional capability metadata.
                    flags=info['algorithms'][key]['numerical_flags']
                    assert not flags or flags & (2<<8), 'Expected FP32 accumulator'
                    assert info['algorithms'][key]['reduction'] in [0,2], 'Expected FP32 reduction'
                    candidates[key]=lambda p=p,i=i:p.launch(i)
                except (RuntimeError,AssertionError) as e:
                    info['algorithms'][f'{name}_{i}']={'rejected':str(e)}
            if not candidates:raise RuntimeError(f'No correct candidates for {name}')
            first=next(iter(candidates))
            sweep=measure(candidates,first,5,16,123)
            info['sweep']=sweep
            ranked=sorted(candidates,key=lambda z:sweep[z]['median_us'])
            # Independently compare top three and the heuristic first choice.
            for key in dict.fromkeys([first,*ranked[:3]]):
                functions['lt_'+key]=candidates[key]
            info['shortlist']=[key for key in candidates if 'lt_'+key in functions]
            info['heuristic_first']=first
            info['sweep_best']=ranked[0]
            if name=='mxfp8':
                info['e2e_algorithm']=ranked[0]
                mxfn=candidates[ranked[0]]
                def mx_e2e(fn=mxfn):
                    quant();fn()
                functions['mx_quantize_only']=quant
                functions['mx_quantize_plus_gemm']=mx_e2e
            print((m,n,k),name,'candidates',len(candidates),'best',ranked[0],sweep[ranked[0]]['median_us'],flush=True)
            save()
        row['retest']=measure(functions,'tk_online_a',args.rounds,args.repeats,456)
        for key,v in row['retest'].items():
            if key!='mx_quantize_only':
                v['tflops']=2*m*n*k/(v['median_us']*1e6)
                if not ('bf16' in key):v['fp8_mfu_percent_range']=[100*v['tflops']/1007.6,100*v['tflops']/877.9]
        for name,info in row['modes'].items():
            if 'unsupported' not in info:
                info['retest_best']=min(('lt_'+z for z in info['shortlist']),key=lambda z:row['retest'][z]['median_us'])
        # Final graph replay output check for MX quantize+GEMM.
        if 'mx_quantize_plus_gemm' in functions:
            functions['mx_quantize_plus_gemm']();row['mx_e2e_error']=error(out,refs['mxfp8']);assert row['mx_e2e_error']['relative_l2']<.004
        save()
        print((m,n,k),{key:round(v['median_us'],3) for key,v in row['retest'].items()},flush=True)
        # Destroy every plan and captured closure before reclaiming shape buffers.
        del functions,candidates,owned,p,plan,refs,x,w,xq,xs,wq,ws,sx,sw,aq,sa,tx,tw,out
        gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
