"""Benchmark fused dG@Wg + dU@Wu against two standalone MXFP8 dgrad kernels."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_gemm import pack_a_scales
from experiments.tk_mxfp8_swiglu_dgrad import (
    CONFIGS,
    OUTPUT,
    SwiGLUDgradLibrary,
    build,
)


def graph_benchmark(functions, rounds=7, repeats=24):
    results = {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                function()
        values = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000 / repeats)
        results[name] = {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
    return results


def operands(m, n, k, seed=123):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate = torch.randn((m, n), device="cuda", generator=generator).bfloat16()
    up = torch.randn((m, n), device="cuda", generator=generator).bfloat16()
    gate = gate.to(torch.float8_e4m3fn)
    up = up.to(torch.float8_e4m3fn)
    weight_gate = torch.randn((n, k), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    weight_up = torch.randn((n, k), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    gate_scale = pack_a_scales(
        torch.full((m, n // 32), 127, device="cuda", dtype=torch.uint8)
    )
    up_scale = gate_scale.clone()
    weight_gate_scale = torch.full(
        (n // 32, k // 32), 127, device="cuda", dtype=torch.uint8
    )
    weight_up_scale = weight_gate_scale.clone()
    return {
        "gate": gate,
        "up": up,
        "weight_gate": weight_gate,
        "weight_up": weight_up,
        "gate_scale": gate_scale,
        "up_scale": up_scale,
        "weight_gate_scale": weight_gate_scale,
        "weight_up_scale": weight_up_scale,
    }


def benchmark_shape(m, n, k):
    data = operands(m, n, k)
    fused_lib = SwiGLUDgradLibrary()
    single_lib = DgradLibrary()
    functions = {}
    metadata = {}

    for config, (_, _, _, block_n, _) in enumerate(CONFIGS):
        for group in [1, 13]:
            plan = fused_lib.plan(
                *data.values(),
                config=config,
                raster_group=group,
            )
            name = f"fused_c{config}_g{group}"
            functions[name] = plan
            metadata[name] = plan.info

    gate_out = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    up_out = torch.empty_like(gate_out)
    added = torch.empty_like(gate_out)
    metadata["two_gemm_add"] = {"gate_scale": 127, "weight_scale": 127}
    for config, group in [(6, 1), (7, 1), (12, 1), (6, 13), (7, 13), (12, 13)]:
        gate_plan = single_lib.plan(
            data["gate"],
            data["weight_gate"],
            data["gate_scale"],
            data["weight_gate_scale"],
            config=config,
            out=gate_out,
            raster_group=group,
        )
        up_plan = single_lib.plan(
            data["up"],
            data["weight_up"],
            data["up_scale"],
            data["weight_up_scale"],
            config=config,
            out=up_out,
            raster_group=group,
        )

        def two_gemm_add(gate_plan=gate_plan, up_plan=up_plan):
            gate_plan()
            up_plan()
            torch.add(gate_out, up_out, out=added)

        name = f"two_gemm_c{config}_g{group}"
        functions[name] = two_gemm_add
        metadata[name] = {"gate": gate_plan.info, "up": up_plan.info}

    timing = graph_benchmark(functions, rounds=7, repeats=24)
    for name, row in timing.items():
        row["tflops"] = 4 * m * n * k / (row["median_us"] * 1e6)
        row.update(metadata[name])
    fused_names = [name for name in timing if name.startswith("fused_")]
    baseline_names = [name for name in timing if name.startswith("two_gemm_")]
    best_fused = min(fused_names, key=lambda name: timing[name]["median_us"])
    best_baseline = min(baseline_names, key=lambda name: timing[name]["median_us"])
    return {
        "shape_mnk": [m, n, k],
        "best_fused": best_fused,
        "best_baseline": best_baseline,
        "speedup": (
            timing[best_baseline]["median_us"] / timing[best_fused]["median_us"]
        ),
        "timings": timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build()
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "configs": CONFIGS,
        "method": (
            "CUDA graphs, 24 launches per graph, 7 rotated rounds, unlocked "
            "clocks. Fused FLOPs include both N-reductions; wrapper overhead "
            "and weight quantization are excluded. Baseline runs two existing "
            "single-gradient MXFP8 dgrad kernels then adds their BF16 outputs."
        ),
        "benchmarks": [],
    }
    for m, n, k in [
        (1024, 8192, 1024),
        (2048, 16384, 2048),
        (4096, 16384, 4096),
    ]:
        row = benchmark_shape(m, n, k)
        result["benchmarks"].append(row)
        fused = row["timings"][row["best_fused"]]
        baseline = row["timings"][row["best_baseline"]]
        print(
            f"{m} {n} {k} | {row['best_fused']} "
            f"{fused['median_us']:.3f} us {fused['tflops']:.1f} TF/s | "
            f"{row['best_baseline']} {baseline['median_us']:.3f} us | "
            f"{row['speedup']:.3f}x",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    result["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [
            "cuda/persistent_mxfp8_swiglu_dgrad.cu",
            "cuda/mxfp8_swiglu_dgrad_configs.inc",
            "experiments/tk_mxfp8_swiglu_dgrad.py",
            "experiments/benchmark_tk_mxfp8_swiglu_dgrad.py",
        ]
    }
    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    output = ROOT / "benchmarks/tk_mxfp8_swiglu_dgrad_rtx5090.json"
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
