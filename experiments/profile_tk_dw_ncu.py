"""Profile exactly one prepared dW launch with NCU --profile-from-start off.

Example (run with performance-counter privileges):
  /usr/local/cuda/bin/ncu --profile-from-start off --set full \
      --clock-control none --cache-control none -o runs/tk_dw_ncu/square \
      /home/cicuvc/miniconda3/envs/blkw/bin/python experiments/profile_tk_dw_ncu.py \
      --shape 4096 4096 4096 --output runs/tk_dw_ncu/square.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.benchmark_tk_persistent_gemm import graph_benchmark
from experiments.tk_linear_backward import LinearBackwardPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=[4096, 4096, 4096],
                        metavar=("M", "N", "K"))
    parser.add_argument("--mode", choices=["native", "baseline", "swap"], default="native")
    parser.add_argument("--config", type=int, default=3)
    parser.add_argument("--raster-group", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    m, n, k = args.shape
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    w, ws = mxtest.launch_weight_quantize(torch.randn((n, k), device="cuda"))
    plan = LinearBackwardPlan(dy, x, w, ws, dw_config=args.config,
                              dw_native_dy=args.mode != "baseline",
                              dw_swap_ab=args.mode == "swap",
                              dw_raster_group=args.raster_group)
    plan.prepare_dy()
    plan.quantize_x()
    for _ in range(32):
        plan.dw_plan()
    timing = graph_benchmark({"dw": plan.dw_plan}, rounds=5, repeats=64)
    torch.cuda.synchronize()
    reference = plan.dwp.clone()
    if not args.no_profile:
        torch.cuda.cudart().cudaProfilerStart()
        plan.dw_plan()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    torch.testing.assert_close(plan.dwp, reference, rtol=0, atol=0)
    assert torch.isfinite(plan.dwp).all().item()
    files = [Path(__file__), ROOT / "cuda/persistent_dual_online_gemm.cu",
             ROOT / "experiments/tk_linear_backward.py",
             ROOT / "runs/tk_dual_online_gemm/libdual.so"]
    result = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "shape_mnk": args.shape, "mode": args.mode, "config": args.config,
        "plan": plan.info["dw"], "graph_timing": timing,
        "profile_region": "One dW GEMM only; preparation and graph timing precede cudaProfilerStart.",
        "output_bitwise_equal_after_profile": True,
        "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in files},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
