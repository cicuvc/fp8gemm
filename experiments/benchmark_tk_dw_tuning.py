"""Joint native-dY dW tile/group sweep and independent full-backward retest."""
import argparse
import gc
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
from experiments.tk_linear_backward import LinearBackwardPlan
from experiments.tk_dual_online_gemm import NATIVE_DW_CONFIGS
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def warm(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            fn()
    until = time.monotonic() + .5
    while time.monotonic() < until:
        graph.replay()
        torch.cuda.synchronize()


def measure(functions, rounds, repeats, seed):
    names = list(functions)
    random.Random(seed).shuffle(names)
    warm(functions["c3_g1"])
    timings = graph_benchmark({name: functions[name] for name in names}, rounds, repeats)
    baseline = timings["c3_g1"]["round_us"]
    for values in timings.values():
        values["paired_speedup"] = statistics.median(
            a / b for a, b in zip(baseline, values["round_us"]))
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+",
                        default=[1, 4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 24, 26, 32])
    parser.add_argument("--configs", type=int, nargs="+", default=list(range(len(NATIVE_DW_CONFIGS))))
    parser.add_argument("--shape", type=int, nargs=3, action="append")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--retest-rounds", type=int, default=11)
    parser.add_argument("--retest-repeats", type=int, default=64)
    parser.add_argument("--backward-sweep", action="store_true",
                        help="Also screen every tile/group in the full backward context")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "configs": NATIVE_DW_CONFIGS, "groups": args.groups, "args": vars(args) | {"output": str(args.output)},
        "baseline": "c3_g1", "bias": False, "peak_tflops": [877.9, 1007.6],
        "method": "Shared preallocated buffers, warm CUDA graphs, shuffled variant order "
                  "then rotated by round, unlocked clocks. Full Cartesian sweep (deduplicate "
                  "clamped groups), shortlist independently retested as dW and full backward. "
                  "No weight refresh or forward in backward timings.",
        "shapes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for m, n, k in args.shape or [(256, 2048, 512), (1024, 8192, 1024),
                                  (2048, 16384, 2048), (4096, 4096, 4096),
                                  (8192, 8192, 4096)]:
        torch.manual_seed(123)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        w, ws = mxtest.launch_weight_quantize(torch.randn((n, k), device="cuda"))
        p = LinearBackwardPlan(dy, x, w, ws, bias=False, dw_config=3)
        p()
        reference = p.dwp.clone()
        dx_ref = p.dx.clone()
        plans = {"c3_g1": p.dw_plan}
        skipped = {}
        aliases = {}
        for config in args.configs:
            seen = set()
            for group in args.groups:
                actual = min(group, (p.np + 8*NATIVE_DW_CONFIGS[config][0]-1) //
                             (8*NATIVE_DW_CONFIGS[config][0]))
                name = f"c{config}_g{actual}"
                aliases[f"c{config}_g{group}"] = name
                if actual in seen or name in plans:
                    continue
                seen.add(actual)
                try:
                    plan = p.dw_plan.owner.plan_native_dy(
                        p.dyp, p.xq, p.xs, config, out=p.dwp, raster_group=actual)
                except RuntimeError as error:
                    # Record infeasible register/shared-resource candidates; do not
                    # silently discard numerical errors or launch failures.
                    skipped[name] = str(error)
                    continue
                torch.testing.assert_close(plan(), reference, rtol=0, atol=0)
                plans[name] = plan
        row = {"shape_mnk": [m, n, k], "plans": {name: q.info for name, q in plans.items()},
               "aliases": aliases, "skipped": skipped, "bitwise_equal": True}
        result["shapes"].append(row)
        save()
        print(f"{m,n,k}: validated {len(plans)} variants, skipped {len(skipped)}", flush=True)
        row["sweep"] = measure(plans, args.rounds, args.repeats, 123)
        save()
        ranked = sorted(plans, key=lambda name: row["sweep"][name]["median_us"])
        print(f"{m,n,k}: sweep complete, leading candidates {ranked[:6]}", flush=True)
        shortlist = set(ranked[:6]) | {"c3_g1"}
        functions = {}
        for name, plan in plans.items():
            def backward(plan=plan):
                p.prepare_dy()
                p.prepare_weight()
                p.dx_plan()
                p.quantize_x()
                plan()
                p.finish()
            functions[name] = backward
        if args.backward_sweep:
            row["backward_sweep"] = measure(functions, args.rounds, args.repeats, 321)
            backward_ranked = sorted(plans, key=lambda name: row["backward_sweep"][name]["median_us"])
            shortlist.update(backward_ranked[:6])
            for config in args.configs:
                choices = [name for name in backward_ranked if name.startswith(f"c{config}_")]
                if choices:
                    shortlist.add(choices[0])
            save()
            print(f"{m,n,k}: full-backward sweep complete, leaders {backward_ranked[:6]}", flush=True)
        # Include each tile's best group so timing noise cannot hide a tile.
        for config in args.configs:
            candidates = [name for name in ranked if name.startswith(f"c{config}_")]
            if candidates:
                shortlist.add(candidates[0])
                if f"c{config}_g1" in plans:
                    shortlist.add(f"c{config}_g1")
        # Row-major at the leading tile + best group at the baseline tile
        # separate the effects of tiling and rasterization.
        shortlist.add(ranked[0].split("_")[0] + "_g1")
        shortlist.add(next(name for name in ranked if name.startswith("c3_")))
        candidates = {name: plans[name] for name in sorted(shortlist) if name in plans}
        row["dw_retest"] = measure(candidates, args.retest_rounds, args.retest_repeats, 456)
        row["best_dw"] = min(candidates, key=lambda name: row["dw_retest"][name]["median_us"])
        print(f"{m,n,k}: dW retest complete, best={row['best_dw']}; "
              f"retesting {len(candidates)} full-backward variants", flush=True)
        save()
        functions = {name: functions[name] for name in candidates}
        for fn in functions.values():
            fn()
            torch.testing.assert_close(p.dwp, reference, rtol=0, atol=0)
            torch.testing.assert_close(p.dx, dx_ref, rtol=0, atol=0)
        row["backward_retest"] = measure(
            functions, args.retest_rounds, args.retest_repeats, 789)
        row["best_backward"] = min(candidates, key=lambda name: row["backward_retest"][name]["median_us"])
        for phase, factor in [("dw_retest", 2), ("backward_retest", 4)]:
            for values in row[phase].values():
                tflops = factor*m*n*k / (values["median_us"]*1e6)
                values["effective_tflops"] = tflops
                values["mfu_percent"] = [100*tflops/1007.6, 100*tflops/877.9]
        save()
        for phase, best in [("dw_retest", row["best_dw"]), ("backward_retest", row["best_backward"])]:
            base, winner = row[phase]["c3_g1"], row[phase][best]
            print(f"{m,n,k} {phase}: c3_g1={base['median_us']:.3f}us, "
                  f"{best}={winner['median_us']:.3f}us ({winner['paired_speedup']:.3f}x)", flush=True)
        del functions, candidates, plans, plan, p, reference, dx_ref, x, dy, w, ws
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
