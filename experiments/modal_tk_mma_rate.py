"""Verify plain versus block-scaled FP8 instruction throughput on PRO6000."""
import json
import subprocess
from pathlib import Path
import modal

app=modal.App("fp8-mma-rate")
ROOT=Path(__file__).resolve().parents[1]
image=modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04",add_python="3.12")
image=image.add_local_file(ROOT/"cuda/probe_mxfp8_rate.cu",remote_path="/workspace/cuda/probe_mxfp8_rate.cu")

@app.function(image=image,gpu="RTX-PRO-6000",timeout=180)
def probe():
    cmd=["/usr/local/cuda/bin/nvcc","-std=c++20","-gencode=arch=compute_120a,code=sm_120a",
         "-O3","-lineinfo","/workspace/cuda/probe_mxfp8_rate.cu","-o","/tmp/mma_rate"]
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads(subprocess.check_output(["/tmp/mma_rate"],text=True))
    result["nvidia_smi"]=subprocess.check_output(["nvidia-smi"],text=True)
    result["command"]=cmd
    return result

@app.local_entrypoint()
def main():
    result=probe.remote()
    Path("benchmarks/tk_mxfp8_instruction_rate_rtx_pro_6000.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
