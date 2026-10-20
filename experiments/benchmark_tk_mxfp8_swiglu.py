"""Validate and benchmark the prequantized-MXFP8 fused SwiGLU kernel."""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.tk_mxfp8_gemm import Library as GemmLibrary, pack_a_scales
from experiments.tk_mxfp8_swiglu import (
    CONFIGS,
    MXFP8SwiGLULibrary,
    OUTPUT,
    build,
)
from experiments.tk_persistent_swiglu import (
    CONFIGS as BF16_CONFIGS,
    SwiGLULibrary,
    build as build_bf16,
)


def operands(m, n, k, seed=123):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), device="cuda", generator=generator).div(8)
    gate = torch.randn((n, k), device="cuda", generator=generator).div(k**0.5 * 8)
    up = torch.randn((n, k), device="cuda", generator=generator).div(k**0.5 * 8)
    x_scale = torch.full(
        (m, k // 32), 127, device="cuda", dtype=torch.uint8
    )
    gate_scale = torch.full(
        (n // 32, k // 32), 127, device="cuda", dtype=torch.uint8
    )
    up_scale = gate_scale.clone()
    return {
        "x_bf16": x.bfloat16(),
        "x": x.to(torch.float8_e4m3fn),
        "x_scale": pack_a_scales(x_scale),
        "gate": gate.to(torch.float8_e4m3fn),
        "gate_scale": gate_scale,
        "up": up.to(torch.float8_e4m3fn),
        "up_scale": up_scale,
    }


def graph_benchmark(functions, rounds=7, repeats=24):
    graphs = {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                function()
        graphs[name] = graph
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    timings = {name: [] for name in graphs}
    names = list(graphs)
    for round_index in range(rounds):
        rotated = names[round_index % len(names) :] + names[: round_index % len(names)]
        for name in rotated:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000 / repeats)
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "round_us": values,
        }
        for name, values in timings.items()
    }


def audit_sass():
    binary = OUTPUT / "libswiglu.so"
    raw = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)], text=True
    )
    (OUTPUT / "kernels.sass").write_text(raw)
    result = {}
    for section in raw.split("Function : ")[1:]:
        name = section.splitlines()[0]
        ops = Counter(
            re.findall(
                r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)",
                section,
            )
        )
        assert not any(op.startswith(("LDL", "STL", "CALL")) for op in ops), name
        for prefix in ["QMMA.SF", "LDSM", "STSM", "UTMALDG", "UTMASTG", "MUFU.TANH"]:
            assert any(op.startswith(prefix) for op in ops), (name, prefix)
        result[name] = {
            "opcodes": dict(ops),
            "qmma": sum(value for op, value in ops.items() if op.startswith("QMMA")),
            "tanh": sum(value for op, value in ops.items() if op.startswith("MUFU.TANH")),
        }
    assert len(result) == 2 * len(CONFIGS)
    assert sum("ELb1EEE" in name for name in result) == len(CONFIGS)
    assert sum("ELb0EEE" in name for name in result) == len(CONFIGS)
    return result


def benchmark_shape(m, n, k, rounds, repeats, include_bf16, store_aux):
    data = operands(m, n, k)
    fused_lib = MXFP8SwiGLULibrary()
    gemm_lib = GemmLibrary()
    functions = {}
    metadata = {}
    fused_outputs = {}
    fused_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    gate_linear = (
        torch.empty((m, n), device="cuda", dtype=torch.bfloat16) if store_aux else None
    )
    up_linear = torch.empty_like(gate_linear) if store_aux else None

    for config, (_, _, _, block_k, _) in enumerate(CONFIGS):
        if k % block_k or n % 64:
            continue
        for raster_group in [1, 8, 13]:
            plan = fused_lib.plan(
                data["x"],
                data["x_scale"],
                data["gate"],
                data["gate_scale"],
                data["up"],
                data["up_scale"],
                config=config,
                out=fused_out,
                gate_out=gate_linear,
                up_out=up_linear,
                raster_group=raster_group,
            )
            name = (
                f"fused_aux_c{config}_g{raster_group}"
                if store_aux
                else f"fused_c{config}_g{raster_group}"
            )
            functions[name] = plan
            metadata[name] = plan.info
            fused_outputs[name] = plan

    gate_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    up_out = torch.empty_like(gate_out)
    materialized = torch.empty_like(gate_out)
    pure_pairs = {}
    for config, (_, _, _, block_k, _) in enumerate(CONFIGS):
        if k % block_k or n % 64:
            continue
        gate_plan = gemm_lib.plan(
            data["x"],
            data["gate"],
            data["x_scale"],
            data["gate_scale"],
            config=config,
            out=gate_out,
        )
        up_plan = gemm_lib.plan(
            data["x"],
            data["up"],
            data["x_scale"],
            data["up_scale"],
            config=config,
            out=up_out,
        )

        def materialized_path(gate_plan=gate_plan, up_plan=up_plan):
            gate_plan()
            up_plan()
            torch.sigmoid(gate_out, out=materialized)
            materialized.mul_(gate_out).mul_(up_out)

        name = f"two_gemm_plus_epilogue_c{config}"
        functions[name] = materialized_path
        pure_pairs[name] = (gate_plan, up_plan)

    if include_bf16:
        bf16_lib = SwiGLULibrary()
        for config in range(len(BF16_CONFIGS)):
            try:
                plan = bf16_lib.plan(
                    data["x_bf16"],
                    data["gate"],
                    data["gate_scale"],
                    data["up"],
                    data["up_scale"],
                    config=config,
                    out=torch.empty_like(gate_out),
                )
            except RuntimeError:
                continue
            name = f"bf16_input_fused_c{config}"
            functions[name] = plan
            metadata[name] = plan.info

    # Verify the native fused FP32 epilogue against the materialized pure-GEMM
    # path. BF16 intermediate rounding makes this a tolerance check.
    pure_plan = pure_pairs["two_gemm_plus_epilogue_c3"]
    pure_plan[0]()
    pure_plan[1]()
    torch.sigmoid(gate_out, out=materialized)
    materialized.mul_(gate_out).mul_(up_out)
    reference = materialized.clone()
    errors = {}
    for name, plan in fused_outputs.items():
        delta = plan().float() - reference.float()
        errors[name] = {
            "relative_l2": (delta.norm() / reference.float().norm()).item(),
            "max_abs": delta.abs().max().item(),
        }
        del delta
    del pure_pairs
    gc.collect()
    torch.cuda.empty_cache()

    timing = graph_benchmark(functions, rounds=rounds, repeats=repeats)
    for name, row in timing.items():
        row["tflops"] = 4 * m * n * k / (row["median_us"] * 1e6)
        if name in metadata:
            row.update(metadata[name])
        if name in errors:
            row.update(errors[name])

    fused_names = [
        name
        for name in timing
        if name.startswith("fused_aux_c" if store_aux else "fused_c")
    ]
    pure_names = [
        name for name in timing if name.startswith("two_gemm_plus_epilogue_c")
    ]
    best_fused = min(fused_names, key=lambda name: timing[name]["median_us"])
    best_pure = min(pure_names, key=lambda name: timing[name]["median_us"])
    for name in fused_names:
        timing[name]["speedup_vs_materialized"] = (
            timing[best_pure]["median_us"] / timing[name]["median_us"]
        )
    return {
        "shape_mnk": [m, n, k],
        "best_fused": best_fused,
        "best_materialized": best_pure,
        "timings": timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-bf16-baseline", action="store_true")
    parser.add_argument("--aux", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    result_path = ROOT / (
        "benchmarks/tk_mxfp8_swiglu_aux_rtx5090.json"
        if args.aux
        else "benchmarks/tk_mxfp8_swiglu_rtx5090.json"
    )
    torch.set_num_threads(4)
    if not args.skip_build:
        build()
        build_bf16()

    shapes = (
        [(args.m, args.n, args.k)]
        if args.m is not None
        else [
            (256, 2048, 512),
            (1024, 8192, 1024),
            (2048, 16384, 2048),
            (4096, 16384, 4096),
        ]
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "configs": CONFIGS,
        "method": (
            "CUDA graphs, 24 launches per graph, 7 rotated warm rounds, "
            "unlocked clocks. Weight and X quantization are excluded from the "
            "MXFP8 paths. Fused candidates scan raster groups 1/8/13. FLOPs "
            "count both gate and up GEMMs; the SiLU epilogue is not counted."
        ),
        "store_aux": args.aux,
        "sass": audit_sass(),
        "benchmarks": [],
    }
    for m, n, k in shapes:
        row = benchmark_shape(
            m,
            n,
            k,
            args.rounds,
            args.repeats,
            not args.no_bf16_baseline,
            args.aux,
        )
        result["benchmarks"].append(row)
        best = row["timings"][row["best_fused"]]
        materialized = row["timings"][row["best_materialized"]]
        print(
            f"{m} {n} {k} | {row['best_fused']} "
            f"{best['median_us']:.3f} us {best['tflops']:.1f} TF/s | "
            f"materialized {materialized['median_us']:.3f} us | "
            f"{materialized['median_us'] / best['median_us']:.3f}x",
            flush=True,
        )
        serialized = json.dumps(result, indent=2) + "\n"
        (OUTPUT / "result.json").write_text(serialized)
        result_path.write_text(serialized)
        gc.collect()
        torch.cuda.empty_cache()

    files = [
        "cuda/persistent_mxfp8_swiglu.cu",
        "cuda/mxfp8_swiglu_configs.inc",
        "experiments/tk_mxfp8_swiglu.py",
        "experiments/benchmark_tk_mxfp8_swiglu.py",
    ]
    result["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files
    }
    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    serialized = json.dumps(result, indent=2) + "\n"
    (OUTPUT / "result.json").write_text(serialized)
    result_path.write_text(serialized)


if __name__ == "__main__":
    main()
