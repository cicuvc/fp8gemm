"""Sustained 4096^3 online/pure comparison with concurrent SM-clock samples."""
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mxtest
from experiments.tk_mxfp8_gemm import Library,pack_a_scales
from experiments.tk_persistent_gemm import GemmLibrary

def main():
    torch.set_num_threads(4)
    a=torch.randn((4096,4096),device="cuda",dtype=torch.bfloat16)
    b,bs=mxtest.launch_weight_quantize(torch.randn((4096,4096),device="cuda"))
    aq,s=mxtest.launch_row_quantize(a);s=pack_a_scales(s)
    plans={"online":GemmLibrary().plan(a,b,8,scales=bs),"pure":Library().plan(aq,b,s,bs)}
    result={"shape_mnk":[4096]*3,"method":"Sustained CUDA graph replay, unlocked clocks. Each round: 20 warmup graphs; time 80 graphs, 256 GEMMs per graph, sample clocks while running. Alternate online/pure for 3 rounds. Timing excludes CPU gaps between graph batches.","rounds":[]}
    for rnd in range(3):
        for name in (["online","pure"] if rnd%2==0 else ["pure","online"]):
            p=plans[name];p();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(256):p()
            for _ in range(20):g.replay()
            torch.cuda.synchronize()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(80):g.replay()
            end.record();samples=[]
            for _ in range(8):
                samples.append(subprocess.check_output(["nvidia-smi","--query-gpu=timestamp,clocks.current.sm,power.draw,temperature.gpu","--format=csv,noheader"],text=True).strip())
                time.sleep(.2)
            end.synchronize();us=start.elapsed_time(end)*1000/(80*256)
            row={"round":rnd,"mode":name,"us":us,"tflops":2*4096**3/(us*1e6),"samples":samples}
            result["rounds"].append(row);print(row,flush=True)
            Path("benchmarks/tk_mxfp8_rtx5090_sustained.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":main()
