"""Evaluate all-BF16 elementwise SwiGLU backward accuracy and bandwidth."""

import argparse
from collections import Counter
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

from experiments.tk_silu_bwd import OUTPUT, SiluBwdLibrary, build


def reference(dy, half_gate, up):
    z = half_gate.double()
    tanh = torch.tanh(z)
    a = z + z * tanh
    b = 0.5 - 0.5 * tanh
    c = 0.5 + 0.5 * tanh
    return (
        (dy.double() * up.double() * (a * b + c)).float().bfloat16(),
        (dy.double() * a).float().bfloat16(),
    )


def metrics(got, expected):
    delta = got.float() - expected.float()
    expected_f = expected.float()
    return {
        "relative_l2": (delta.norm() / expected_f.norm().clamp_min(1e-30)).item(),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            got.float().flatten(), expected_f.flatten(), dim=0
        ).item(),
    }


def subset_relative_l2(got, expected, mask):
    delta = got.float()[mask] - expected.float()[mask]
    return (delta.norm() / expected.float()[mask].norm()).item()


def accuracy(lib):
    rows = []
    for scale in [0.1, 1.0, 2.0, 4.0, 8.0]:
        torch.manual_seed(123)
        elements = 1 << 20
        dy = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
        half_gate = (torch.randn(elements, device="cuda") * scale).bfloat16()
        up = (torch.randn(elements, device="cuda") * scale).bfloat16()
        expected_gate, expected_up = reference(dy, half_gate, up)
        dgate, dup = lib.plan(dy, half_gate, up)()
        torch.cuda.synchronize()
        row = {
            "half_gate_std": scale,
            "dgate": metrics(dgate, expected_gate),
            "dgate_negative_relative_l2": subset_relative_l2(
                dgate, expected_gate, half_gate < 0
            ),
            "dgate_positive_relative_l2": subset_relative_l2(
                dgate, expected_gate, half_gate >= 0
            ),
            "dup": metrics(dup, expected_up),
        }
        rows.append(row)
        print(
            f"scale={scale:g}: dgate rel={row['dgate']['relative_l2']:.5f}, "
            f"negative={row['dgate_negative_relative_l2']:.5f}, "
            f"positive={row['dgate_positive_relative_l2']:.5f}, "
            f"dup rel={row['dup']['relative_l2']:.5f}",
            flush=True,
        )
    return rows


def graph_benchmark(function, rounds=7, repeats=64):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
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
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "round_us": values,
    }


def performance(lib):
    rows = []
    for elements in [1 << 20, 1024 * 8192, 2048 * 16384, 4096 * 16384]:
        torch.manual_seed(321)
        dy = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
        half_gate = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
        plan = lib.plan(dy, half_gate, up)
        timing = graph_benchmark(plan)
        timing["bytes"] = elements * 10
        timing["effective_gbps"] = timing["bytes"] / (timing["median_us"] * 1e3)
        timing.update(plan.info)
        rows.append({"elements": elements, **timing})
        print(
            f"{elements}: {timing['median_us']:.3f} us, "
            f"{timing['effective_gbps']:.1f} GB/s",
            flush=True,
        )
    return rows


def sass_audit():
    raw = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(OUTPUT / "libsilu_bwd.so")],
        text=True,
    )
    (OUTPUT / "kernels.sass").write_text(raw)
    section = raw.split("Function : ")[1]
    instructions = Counter(
        re.findall(
            r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)",
            section,
        )
    )
    assert not any(op.startswith(("LDL", "STL")) for op in instructions)
    for opcode in ["HFMA2.BF16_V2", "HMUL2.BF16_V2", "MUFU.TANH.BF16"]:
        assert instructions[opcode] > 0
    return {opcode: count for opcode, count in instructions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build()
    lib = SiluBwdLibrary()
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "accuracy": accuracy(lib),
        "performance": performance(lib),
        "sass": sass_audit(),
        "method": (
            "Accuracy compares BF16 outputs with an FP64 formula cast to BF16. "
            "Performance uses CUDA graphs, 64 launches per graph, and 7 rounds. "
            "Effective bandwidth counts three BF16 reads and two BF16 writes."
        ),
    }
    result["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [
            "cuda/silu_bwd_elementwise.cu",
            "experiments/tk_silu_bwd.py",
            "experiments/benchmark_silu_bwd.py",
        ]
    }
    output = ROOT / "benchmarks/tk_silu_bwd_bf16_rtx5090.json"
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
