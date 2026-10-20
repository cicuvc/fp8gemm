"""Compare existing FP8, factored FP8 and byte-equivalent BF16 shared loads."""
import argparse
from collections import Counter
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tk_persistent_gemm import FLAGS
from experiments.benchmark_tk_persistent_gemm import graph_benchmark

OUT=ROOT/"runs/tk_ldmatrix_probe"
CASES=[(16,32,4,False),(16,64,4,False),(16,128,4,False),
       (32,64,4,False),(32,128,4,False),(32,256,4,False),
       (32,32,1,True),(64,32,1,True),(64,64,1,True),
       (128,64,1,True),(128,128,1,True),(64,256,1,True)]
NAMES=["fp8_original","fp8_factored","bf16_control"]

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"build.log").open("w") as log:
        subprocess.run(["/usr/local/cuda/bin/nvcc",*FLAGS,"-shared","-Xcompiler=-fPIC","-Xptxas=-v",
                        "cuda/probe_fp8_ldmatrix.cu","-o",str(OUT/"libprobe.so")],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def assembly():
    raw=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",str(OUT/"libprobe.so")],text=True)
    (OUT/"kernels.sass").write_text(raw)
    result={}
    for section in raw.split("Function : ")[1:]:
        name=section.splitlines()[0]
        instructions=[]
        for pc,op,args in re.findall(r"/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s*([^;]*);",section):
            instructions.append((int(pc,16),op,args))
        loops=[]
        for pc,op,args in instructions:
            target=re.search(r"0x([0-9a-f]+)",args)
            if op.startswith("BRA") and target and int(target[1],16)<pc:
                block=[v for v in instructions if int(target[1],16)<=v[0]<=pc]
                loads=[v for v in block if v[1].startswith("LDSM")]
                if loads:
                    counts=dict(Counter(v[1] for v in block))
                    loops.append({"start":hex(block[0][0]),"end":hex(pc),"opcodes":counts,
                                  "ldsm_addresses":[v[2].split(", ",1)[-1].strip() for v in loads],
                                  "instructions":[f"{a:04x}: {b} {c};" for a,b,c in block]})
        assert len(loops)==1,(name,loops)
        assert not any(op.startswith(("LDL","STL")) for _,op,_ in instructions),name
        result[name]=loops[0]
    assert len(result)==len(CASES)*3
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--output",type=Path,default=OUT/"result.json")
    args=parser.parse_args()
    if not args.skip_build:build()
    torch.set_num_threads(4)
    lib=ct.CDLL(str(OUT/"libprobe.so"))
    lib.ldsm_probe.argtypes=[ct.c_int]*7+[ct.c_void_p]*3
    lib.ldsm_probe_error.restype=ct.c_char_p
    result={"gpu":torch.cuda.get_device_name(),"cases":[],"scope":"Load-only microprobe, one CTA; not GEMM performance.",
            "alignment":"Shared tile base aligned to 8*swizzle_bytes; complete tiles only."}
    for rows,k,gw,swap in CASES:
        threads=32*gw;words=(rows//16)*(k//32)*4
        outputs=[torch.empty((threads,words),device="cuda",dtype=torch.int32) for _ in NAMES]
        clocks=[torch.empty(threads,device="cuda",dtype=torch.int64) for _ in NAMES]
        def launch(kind,iters,offset):
            if lib.ldsm_probe(rows,k,gw,int(swap),kind,iters,offset,outputs[kind].data_ptr(),
                              clocks[kind].data_ptr(),torch.cuda.current_stream().cuda_stream):
                raise RuntimeError(lib.ldsm_probe_error().decode())
        alignment=8*min(k,128)
        assert lib.ldsm_probe(rows,k,gw,int(swap),1,1,16,outputs[1].data_ptr(),
                              clocks[1].data_ptr(),torch.cuda.current_stream().cuda_stream) != 0
        comparisons=0
        for offset in [0,alignment,2*alignment,4*alignment]:
            for iters in [1,2,3,17]:
                for kind in range(3):launch(kind,iters,offset)
                for kind in [1,2]:
                    assert torch.equal(outputs[0],outputs[kind]),(rows,k,gw,swap,kind,iters,offset)
                    comparisons+=1
        row={"rows":rows,"k_bytes":k,"group_warps":gw,"swap":swap,
             "ldsm_per_warp_per_iteration":words//4,"bitwise_comparisons":comparisons,
             "misaligned_base_rejected":True}
        if not args.validate_only:
            funcs={name:(lambda kind=kind:launch(kind,4096,alignment)) for kind,name in enumerate(NAMES)}
            row["timing"]=graph_benchmark(funcs,rounds=9,repeats=16)
            row["cycles_per_ldsm"]={name:clocks[i].double().median().item()/(4096*(words//4))
                                     for i,name in enumerate(NAMES)}
        result["cases"].append(row)
        print((rows,k,gw,swap),"PASS",row.get("cycles_per_ldsm",""),flush=True)
    result["sass"]=assembly()
    result["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        "cuda/fp8_ldmatrix_probe.cuh","cuda/probe_fp8_ldmatrix.cu","experiments/probe_tk_ldmatrix.py",
        "third_party/thunderkittens/include/ops/group/memory/tile/s2r_impl.cuh"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":main()
