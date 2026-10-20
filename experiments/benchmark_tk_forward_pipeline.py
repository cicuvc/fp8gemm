"""Compare the saved forward binary, L2 grouping, and K32 register prefetch."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.tk_persistent_gemm import CONFIGS, GemmLibrary
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def measure(functions, baseline, rounds, repeats, seed):
    warm = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm):
        for _ in range(32):
            functions[baseline]()
    until = time.monotonic()+.5
    while time.monotonic() < until:
        warm.replay()
        torch.cuda.synchronize()
    names = list(functions)
    random.Random(seed).shuffle(names)
    results = graph_benchmark({name: functions[name] for name in names}, rounds, repeats)
    for value in results.values():
        value["paired_speedup"] = statistics.median(
            a/b for a, b in zip(results[baseline]["round_us"], value["round_us"]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", type=Path, default=ROOT/"runs/tk_forward_pipeline/old")
    parser.add_argument("--groups", nargs="+", type=int,
                        default=[1, 2, 4, 8, 10, 12, 13, 14, 16, 18, 20, 24, 32, 64])
    parser.add_argument("--configs", nargs="+", type=int, default=list(range(len(CONFIGS))))
    parser.add_argument("--shape", nargs=3, type=int, action="append")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--retest-rounds", type=int, default=11)
    parser.add_argument("--retest-repeats", type=int, default=64)
    parser.add_argument("--peak-tflops", type=float, nargs="+", default=[419],
                        help="One peak or lower/upper peak estimates for MFU")
    parser.add_argument("--output", type=Path, default=ROOT/"benchmarks/tk_forward_pipeline_rtx5090.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    lib, old = GemmLibrary(), GemmLibrary(args.baseline_library)
    result = {
        "gpu": torch.cuda.get_device_name(), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "configs": CONFIGS, "args": vars(args) | {"baseline_library": str(args.baseline_library), "output": str(args.output)},
        "method": "Shared buffers, prequantized weights with 32x32 scales; shuffled then rotated warm CUDA graphs; "
                  "unlocked clocks. Compare to saved pre-change binary; independent shortlist retest.",
        "peak_tflops": args.peak_tflops,
        "baseline_binary_sha256": hashlib.sha256((args.baseline_library/"libgemm.so").read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            "cuda/persistent_fp8_gemm.cu", "cuda/tma_ring_pipe.cuh", "experiments/tk_persistent_gemm.py",
            "experiments/benchmark_tk_forward_pipeline.py", "runs/tk_persistent_gemm/libgemm.so"]},
        "shapes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    for m, n, k in args.shape or [(256, 2048, 512), (1024, 8192, 1024), (2048, 16384, 2048),
                                  (4096, 4096, 4096), (8192, 8192, 4096)]:
        torch.manual_seed(123)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b, scales = mxtest.launch_weight_quantize(torch.randn((n, k), device="cuda"))
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        default = 7 if ((m+127)//128)*((n+127)//128) < result["sm_count"] else 8
        configs = sorted(set(args.configs) | {default})
        plans = {f"old_c{c}": old.plan(a, b, c, out=out, scales=scales) for c in configs}
        baseline = f"old_c{default}"
        ref = plans[baseline]().clone()
        for c in configs:
            for prefetch in [False, True]:
                for g in args.groups:
                    g = min(g, (m+8*CONFIGS[c][0]-1)//(8*CONFIGS[c][0]))
                    name = f"{'pref' if prefetch else 'base'}_c{c}_g{g}"
                    if name not in plans:
                        plans[name] = lib.plan(a, b, c, out=out, scales=scales,
                                               raster_group=g, prefetch=prefetch)
        for name, plan in plans.items():
            assert torch.equal(plan(), ref), (m, n, k, name)
        row = {"shape_mnk": [m, n, k], "baseline": baseline, "default_config": default,
               "plans": {name: p.info for name, p in plans.items()}, "bitwise_equal": True}
        result["shapes"].append(row)
        print(f"{m,n,k}: {len(plans)} variants validated", flush=True)
        row["sweep"] = measure(plans, baseline, args.rounds, args.repeats, 123)
        ranked = sorted(plans, key=lambda name: row["sweep"][name]["median_us"])
        shortlist = set(ranked[:6]) | {baseline, f"base_c{default}_g1", f"pref_c{default}_g1"}
        for branch in ["base", "pref"]:
            best = next(name for name in ranked if name.startswith(branch+"_"))
            shortlist.add(best)
            cfg = best.split("_")[1]
            shortlist.update({f"{branch}_{cfg}_g1", f"old_{cfg}",
                              next(name for name in ranked if name.startswith(f"{branch}_c{default}_"))})
        # Same config/group on both paths isolates register prefetch from
        # rasterization; group=1 controls isolate each change from the old binary.
        for name in list(shortlist):
            if name.startswith(("base_", "pref_")):
                branch, cfg, group = name.split("_")
                shortlist.update({f"base_{cfg}_{group}", f"pref_{cfg}_{group}",
                                  f"base_{cfg}_g1", f"pref_{cfg}_g1", f"old_{cfg}"})
        candidates = {name: plans[name] for name in sorted(shortlist) if name in plans}
        save()
        print(f"{m,n,k}: leading {ranked[:6]}, retesting {len(candidates)}", flush=True)
        row["retest"] = measure(candidates, baseline, args.retest_rounds, args.retest_repeats, 456)
        for v in row["retest"].values():
            v["tflops"] = 2*m*n*k/(v["median_us"]*1e6)
            v["mfu_percent"] = 100*v["tflops"]/max(args.peak_tflops)
            v["mfu_percent_range"] = [100*v["tflops"]/max(args.peak_tflops),
                                      100*v["tflops"]/min(args.peak_tflops)]
        row["best"] = min(candidates, key=lambda name: row["retest"][name]["median_us"])
        for branch in ["base", "pref"]:
            row["best_"+branch] = min((name for name in candidates if name.startswith(branch+"_")),
                                      key=lambda name: row["retest"][name]["median_us"])
        save()
        winner = row["retest"][row["best"]]
        print(f"{m,n,k}: old={row['retest'][baseline]['median_us']:.3f}us, "
              f"best={row['best']} {winner['median_us']:.3f}us ({winner['paired_speedup']:.3f}x)", flush=True)
        del plans, candidates, plan, a, b, scales, out, ref
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
