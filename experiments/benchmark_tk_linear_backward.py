"""End-to-end linear backward, including transposes, X quantization and bias."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import triton

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_linear_backward import LinearBackwardPlan, OUTPUT
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=7)
    parser.add_argument('--repeats',type=int,default=32)
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'rounds':args.rounds,'repeats':args.repeats,
            'timing':'Preallocated CUDA graphs; rotated order; warm inputs; no locked clocks. All transposes/padding/quantization/bias included. Weight transpose refreshed every backward. GEMM config defaults, no per-shape tuning.',
            'benchmarks':[]}
    shapes=[(256,2048,512),(1024,8192,1024),(2048,16384,2048),(4096,4096,4096),(8192,8192,4096)]
    for m,n,k in shapes:
        torch.manual_seed(123)
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        dy=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
        w,ws=mxtest.launch_weight_quantize(torch.randn((n,k),device='cuda'))
        p=LinearBackwardPlan(dy,x,w,ws)
        tdx=torch.empty_like(p.dx);tdw=torch.empty_like(p.dw);tdb=torch.empty_like(p.db)
        dy_float=torch.empty_like(dy,dtype=torch.float32)
        def triton_dx():
            mxtest.fp8gemm_bwd[lambda c:(triton.cdiv(m,c['BLOCK_M'])*triton.cdiv(k,c['BLOCK_K']),)](
                dy,w,ws,tdx,m,n,k,dy.stride(0),w.stride(0),ws.stride(0),tdx.stride(0))
        def triton_dw():
            mxtest.fp8gemm_wgrad[lambda c:(triton.cdiv(n,c['BLOCK_N'])*triton.cdiv(k,c['BLOCK_K']),)](
                dy,x,tdw,m,n,k,dy.stride(0),x.stride(0),tdw.stride(0))
        def triton_db():
            dy_float.copy_(dy)
            torch.sum(dy_float,dim=0,out=tdb)
        def triton_all():triton_dx();triton_dw();triton_db()
        triton_all();p()
        errors={}
        for name,a,b,tol in [('dx',p.dx,tdx,5e-4),('dw',p.dw,tdw,1e-6),('db',p.db,tdb,2e-6)]:
            rel=((a.float()-b.float()).norm()/b.float().norm()).item()
            assert torch.isfinite(a).all() and rel<tol,(m,n,k,name,rel)
            errors[name]={'relative_l2_vs_triton':rel,'max_abs_vs_triton':(a.float()-b.float()).abs().max().item()}
        components=graph_benchmark({'prepare_dy':p.prepare_dy,'transpose_weight':p.prepare_weight,
            'cuda_dx':p.dx_plan,'quantize_transpose_x':p.quantize_x,'cuda_dw':p.dw_plan,'finish_bias_crop':p.finish,
            'triton_dx':triton_dx,'triton_dw':triton_dw,'triton_db':triton_db},args.rounds,args.repeats)
        paired=graph_benchmark({'cuda_backward':p,'triton_backward':triton_all},args.rounds,args.repeats)
        row={**p.info,'errors':errors,'components':components,'paired':paired,
             'speedup':paired['triton_backward']['median_us']/paired['cuda_backward']['median_us'],
             'triton_dx_config':str(mxtest.fp8gemm_bwd.best_config),
             'triton_dw_config':str(mxtest.fp8gemm_wgrad.best_config)}
        result['benchmarks'].append(row)
        print(f'{m,n,k}: CUDA total {paired["cuda_backward"]["median_us"]:.3f} us; '
              f'Triton {paired["triton_backward"]["median_us"]:.3f} us; {row["speedup"]:.3f}x; '
              f'prep DY/W/X {components["prepare_dy"]["median_us"]:.2f}/'
              f'{components["transpose_weight"]["median_us"]:.2f}/{components["quantize_transpose_x"]["median_us"]:.2f} us',flush=True)
        (OUTPUT/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.sanitize:
        result['sanitizers']={}
        for tool in ['memcheck','synccheck','initcheck','racecheck']:
            print('Running',tool,flush=True)
            path=OUTPUT/f'{tool}.log'
            with path.open('w') as log:
                subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','1',
                    sys.executable,'-m','pytest','tests/test_tk_linear_backward.py::test_exact_backward','-q'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
            summary='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
            text=path.read_text()
            assert summary in text and '8 passed' in text,text[-4000:]
            result['sanitizers'][tool]=summary
    files=[Path(__file__),ROOT/'experiments/tk_linear_backward.py',ROOT/'cuda/linear_backward_prepare.cu',
           ROOT/'fp8_linear.py',ROOT/'tests/test_tk_linear_backward.py',ROOT/'cuda/persistent_fp8_gemm.cu',
           ROOT/'cuda/persistent_dual_online_gemm.cu',ROOT/'mxtest.py',
           ROOT/'experiments/tk_persistent_gemm.py',ROOT/'experiments/tk_dual_online_gemm.py']
    result['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
