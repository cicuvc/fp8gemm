"""Warm CUDA Graph timings, excluding quantization and plan construction.

Run from the repository root with `python benchmarks/bench_fp8_gemm.py`.
The BF16 torch.mm baseline uses dequantized, BF16-rounded operands; it is a
latency comparison, not a bitwise numerical reference for the FP8 operation.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mxfp8 import FP8GemmPlan
from mxfp8._kernels.fp8_gemm import CONFIGS


def time_graph(fn, repeats=50, samples=9):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(10):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / repeats)
    return sorted(times)[len(times) // 2]


def benchmark(m, n, k):
    a = (torch.randn((m, k), device="cuda") * 32).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 32).to(torch.float8_e4m3fn)
    sa = torch.rand((m, k // 128), device="cuda") * 0.02 + 0.01
    sb = torch.rand((n // 128, k // 128), device="cuda") * 0.02 + 0.01
    ah = (a.float() * sa.repeat_interleave(128, 1)).bfloat16()
    bh = (b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    results = []
    for config in range(len(CONFIGS)):
        plan = FP8GemmPlan(a, b, sa, sb, out=out, config=config)
        us = time_graph(plan)
        results.append(
            dict(
                config=config,
                microseconds=us,
                tflops=2 * m * n * k / us / 1e6,
                **plan.info,
            )
        )
    baseline = time_graph(lambda: torch.mm(ah, bh.T, out=out))
    return dict(
        shape=[m, n, k],
        configs=results,
        bf16_torch_mm_us=baseline,
        bf16_torch_mm_tflops=2 * m * n * k / baseline / 1e6,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1024,1024,1024", "4096,4096,4096", "1024,4096,4096"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/tk_fp8_gemm/benchmark.json")
    )
    args = parser.parse_args()
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        timing="median of 9 warm graph replays; 50 GEMMs per graph",
        results=[],
    )
    for shape in args.shapes:
        result = benchmark(*map(int, shape.split(",")))
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
