"""Paired forward-only benchmark of a saved baseline versus current MXFP8."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mxtest
from experiments.tk_mxfp8_gemm import Library,OUTPUT,pack_a_scales
from experiments.benchmark_tk_forward_pipeline import measure
from experiments.benchmark_tk_mxfp8 import audit_sass

def hot_loops(binary,output):
    sass=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",str(binary)],text=True)
    output.write_text(sass)
    loops={}
    for part in sass.split("Function : ")[1:]:
        name=part.splitlines()[0]
        inst=[(int(pc,16),op,args) for pc,op,args in
              re.findall(r"/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s*([^;]*);",part)]
        candidates=[]
        for pc,op,args in inst:
            target=re.search(r"0x([0-9a-f]+)",args)
            if op.startswith("BRA") and target and int(target[1],16)<pc:
                block=[v for v in inst if int(target[1],16)<=v[0]<=pc]
                if any(v[1].startswith("QMMA") for v in block):candidates.append(block)
        block=min(candidates,key=len)
        ops=Counter(v[1] for v in block)
        loads=[v[2] for v in block if v[1].startswith("LDSM")]
        loops[name]={"start":hex(block[0][0]),"end":hex(block[-1][0]),
                     "opcodes":dict(ops),"instructions":len(block),"ldsm":len(loads),
                     "ldsm_with_ur":sum("UR" in x for x in loads),
                     "gpr_integer_instructions":sum(n for op,n in ops.items() if op.startswith(
                         ("IADD","IMAD","LOP3","SHF","LEA","MOV","ISETP","SEL","BMSK","PRMT"))),
                     "loop_sass":[f"{pc:04x}: {op} {args};" for pc,op,args in block]}
    return loops

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"runs/tk_mxfp8_ldmatrix/baseline")
    parser.add_argument("--output",type=Path,default=ROOT/"benchmarks/tk_mxfp8_ldmatrix_rtx5090.json")
    parser.add_argument("--rounds",type=int,default=11)
    parser.add_argument("--repeats",type=int,default=64)
    parser.add_argument("--shape",type=int,nargs=3,action="append")
    parser.add_argument("--artifacts",type=Path,default=ROOT/"runs/tk_mxfp8_ldmatrix")
    args=parser.parse_args();torch.set_num_threads(4)
    old,new=Library(args.baseline),Library()
    audit=audit_sass()
    evidence=args.artifacts;evidence.mkdir(parents=True,exist_ok=True)
    d={"gpu":torch.cuda.get_device_name(),"rounds":args.rounds,"repeats":args.repeats,
       "method":"Paired shuffled/rotated warm CUDA graphs, same inputs/output/config/grid, G=1, unlocked clocks; pure forward GEMM only. Quantization excluded.",
       "sass_audit":audit,"sass_loops":{name:hot_loops(path/"libgemm.so",evidence/f"{name}.sass")
                                      for name,path in [("old",args.baseline),("new",OUTPUT)]},
       "binary_sha256":{name:hashlib.sha256((path/"libgemm.so").read_bytes()).hexdigest()
                        for name,path in [("old",args.baseline),("new",OUTPUT)]},
       "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                        ["cuda/persistent_mxfp8_gemm.cu","cuda/mxfp8_configs.inc","cuda/fp8_shared_load.cuh",
                         "cuda/tma_ring_pipe.cuh","cuda/linear_tma_buffer.cuh",
                         "experiments/benchmark_tk_mxfp8_ldmatrix.py"]},"shapes":[]}
    shapes=args.shape or [(256,2048,512),(1024,8192,1024),(2048,16384,2048),
                          (4096,4096,4096),(8192,8192,4096)]+[
                              (16384,n,k) for n in [1024,2048,4096] for k in [1024,2048,4096]]
    for m,n,k in shapes:
        torch.manual_seed(123)
        a,sa=mxtest.launch_row_quantize(torch.randn((m,k),device="cuda",dtype=torch.bfloat16))
        sa=pack_a_scales(sa)
        b,sb=mxtest.launch_weight_quantize(torch.randn((n,k),device="cuda"))
        config=7 if ((m+127)//128)*((n+127)//128)<torch.cuda.get_device_properties(0).multi_processor_count else 8
        out=torch.empty((m,n),device="cuda",dtype=torch.bfloat16)
        po=old.plan(a,b,sa,sb,config,out=out);pn=new.plan(a,b,sa,sb,config,out=out)
        ref=po().clone();assert torch.equal(pn(),ref);del ref
        timing=measure({"old":po,"new":pn},"old",args.rounds,args.repeats,123)
        for t in timing.values():t["tflops"]=2*m*n*k/(t["median_us"]*1e6)
        row={"shape_mnk":[m,n,k],"config":config,"bitwise_equal":True,"old_plan":po.info,
             "new_plan":pn.info,"timing":timing}
        d["shapes"].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(d,indent=2)+"\n")
        print((m,n,k),f'old={timing["old"]["median_us"]:.3f}us new={timing["new"]["median_us"]:.3f}us '
              f'speedup={timing["new"]["paired_speedup"]:.3f}x TFLOPS={timing["new"]["tflops"]:.1f}',flush=True)
        del po,pn,a,b,sa,sb,out
        gc.collect();torch.cuda.empty_cache()

if __name__=="__main__":main()
