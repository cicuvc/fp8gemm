"""Sweep dW grouped rasterization using shared buffers and rotated CUDA graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.tk_linear_backward import LinearBackwardPlan
from experiments.tk_dual_online_gemm import DualOnlineLibrary
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=int,
                        default=[1, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--shape", nargs=3, type=int, action="append")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--baseline-library", type=Path,
                        default=ROOT / "runs/tk_group_raster/old",
                        help="Directory containing the saved pre-rasterization libdual.so")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tk_group_raster/sweep.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    result = {"gpu": torch.cuda.get_device_name(), "sm_count": sms,
              "sqrt_sm_count": sms ** .5, "groups": args.groups,
              "rounds": args.rounds, "repeats": args.repeats,
              "full_backward": args.backward,
              "method": "Warm CUDA graphs; all variants share input/output buffers; "
                        "rotated timing order; unlocked clocks; paired ratios by round.",
              "shapes": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for m, n, k in args.shape or [(256, 2048, 512), (1024, 8192, 1024),
                                  (2048, 16384, 2048), (4096, 4096, 4096),
                                  (8192, 8192, 4096)]:
        torch.manual_seed(123)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        w, ws = mxtest.launch_weight_quantize(torch.randn((n, k), device="cuda"))
        p = LinearBackwardPlan(dy, x, w, ws, dw_raster_group=1)
        p()
        owner = p.dw_plan.owner
        old_owner = DualOnlineLibrary(args.baseline_library)
        # All main shapes use config3; preserve the plan's explicit selection.
        config = 0 if p.np <= 256 and p.kp <= 2048 else 3
        plans = {"old": old_owner.plan_native_dy(p.dyp, p.xq, p.xs, config, out=p.dwp)}
        plans.update({f"g{g}": owner.plan_native_dy(
            p.dyp, p.xq, p.xs, config, out=p.dwp, raster_group=g) for g in args.groups})
        reference = plans["old"]().clone()
        for name, plan in plans.items():
            torch.testing.assert_close(plan(), reference, rtol=0, atol=0)
        functions = {}
        for name, plan in plans.items():
            if args.backward:
                def run(plan=plan):
                    p.prepare_dy()
                    p.prepare_weight()
                    p.dx_plan()
                    p.quantize_x()
                    plan()
                    p.finish()
                functions[name] = run
            else:
                functions[name] = plan
        # Bring the GPU to sustained load before the rotated measurements.
        warm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warm):
            for _ in range(64):
                functions["old"]()
        until = time.monotonic() + .5
        while time.monotonic() < until:
            warm.replay()
            torch.cuda.synchronize()
        timing = graph_benchmark(functions, args.rounds, args.repeats)
        paired = {name: statistics.median(a / b for a, b in zip(
            timing["old"]["round_us"], values["round_us"])) for name, values in timing.items()}
        best = max((name for name in plans if name != "old"), key=lambda name: paired[name])
        flops = (4 if args.backward else 2) * m * n * k
        row = {"shape_mnk": [m, n, k], "timings": timing, "paired_speedup_vs_old": paired,
               "best": best, "bitwise_equal": True,
               "plans": {name: plan.info for name, plan in plans.items()},
               "mfu_percent": {name: flops / (v["median_us"] * 1e-6) / 419e12 * 100
                               for name, v in timing.items()}}
        result["shapes"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{m,n,k}: " + ", ".join(
            f"{name}={v['median_us']:.3f}us ({paired[name]:.3f}x)"
            for name, v in timing.items()) + f"; best={best}", flush=True)
    paths = ["cuda/persistent_dual_online_gemm.cu", "experiments/tk_dual_online_gemm.py",
             "experiments/tk_linear_backward.py", "experiments/benchmark_tk_group_raster.py",
             "runs/tk_dual_online_gemm/libdual.so"]
    result["sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    result["baseline_binary_sha256"] = hashlib.sha256(
        (args.baseline_library / "libdual.so").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
