"""Check native weight quantization -> scaled GEMM, including persistent reuse.

Build both libraries first with experiments.tk_{weight_quantize,persistent_gemm}.build().
This script can also run under each compute-sanitizer tool.
"""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tk_weight_quantize import QuantLibrary
from experiments.tk_persistent_gemm import GemmLibrary, CONFIGS
from experiments.benchmark_tk_weight_quantize import oracle
from experiments.benchmark_tk_persistent_gemm import exact_inputs, quantized_a


def validate():
    torch.set_num_threads(4)
    quant,gemm=QuantLibrary(),GemmLibrary()
    m,n,k=777,192,192
    ac,_=exact_inputs(m,n,k)
    i=torch.arange(n*k).reshape(n,k)
    wc=torch.ldexp(((i*23+13)%129-64).float()/32,
                   ((i//k//32*3+i%k//32)%11-5).int())
    a,w=ac.cuda(),wc.cuda()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rows=[]
    for mode in [False,True]:
        for config in range(len(CONFIGS)):
            # Exercise both narrow and wide quantizer tiles in the chain.
            qp=quant.plan(w,config=6 if config%2 else 2,round_bf16=mode,sms=3)
            # As in the standalone GEMM validation, use a NaN sentinel to
            # detect missing TMA writes. Initcheck does not mark these TMA
            # output writes initialized before a subsequent host copy.
            out=torch.full((m,n),float('nan'),device='cuda',dtype=torch.bfloat16)
            gp=gemm.plan(a,qp.out,config=config,scales=qp.scales,sms=3,out=out)
            stream.wait_stream(torch.cuda.current_stream())
            for reuse in [False,True]:
                current=wc.clone()
                if reuse:
                    current[:64,:32]=0
                    current[64:,64:]*=4
                rq,rs=oracle(current,mode)
                b=torch.ldexp(rq.double().reshape(n//32,32,k//32,32),
                              (rs.int()-127)[:,None,:,None]).reshape(n,k)
                ref=(quantized_a(ac).double()@b.T).bfloat16()
                with torch.cuda.stream(stream):
                    w.copy_(current)
                    qp()
                    got=gp()
                stream.synchronize()
                assert torch.equal(got.cpu(),ref),(mode,config,reuse)
            # Capture both stages together; descriptors and buffers are retained.
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                qp();gp()
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(got.cpu(),ref),(mode,config,'graph')
            rows.append({'round_bf16':mode,'gemm_config':config,'shape_mnk':[m,n,k],
                         'calls':3,'mismatches':0})
        print(f'PASS quantize -> scaled GEMM: mode={mode}, all 10 configurations, reuse and CUDA graph',flush=True)
    return rows


if __name__=='__main__':
    rows=validate()
    (ROOT/'runs/tk_weight_quantize/gemm_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
