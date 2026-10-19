"""Benchmark prenorm RMSNorm fused into FP8 SwiGLU forward."""

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.testing

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mxtest
from experiments.flash_norm import launch_inverse_rms
from experiments.flash_norm_swiglu import (
    FlashNormSwiGLU,
    launch_prenorm_swiglu_gateup,
    launch_rmsnorm,
)
from fp8_swiglu import FP8SwiGLU


DEFAULT_CASES = (
    (1024, 512, 1408, 512),
    (8192, 512, 1408, 512),
    (8192, 1024, 2816, 1024),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--out", type=int, default=None)
    return parser.parse_args()


def benchmark_case(M, K, H, O):
    fused = FlashNormSwiGLU(K, H, O, device="cuda")
    baseline = FP8SwiGLU(K, H, O, bias=False, device="cuda")
    with torch.no_grad():
        baseline.gate_weight.copy_(fused.gate_weight)
        baseline.up_weight.copy_(fused.up_weight)
        baseline.down_proj.master_weight.copy_(fused.down_proj.master_weight)
        fused.elementwise_factor.copy_(
            torch.randn(K, device="cuda").mul_(0.1).add_(1.0)
        )
    baseline.refresh_fp8_weights()

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    down_weight, down_scale = baseline.down_proj.quantized_weight

    def baseline_gate_up():
        normalized = launch_rmsnorm(
            x,
            fused.elementwise_factor,
            fused.eps,
        )
        return mxtest.launch_fp8_swiglu(
            normalized,
            baseline._gate_weight_fp8,
            baseline._gate_weight_scale,
            baseline._up_weight_fp8,
            baseline._up_weight_scale,
        )

    def fused_gate_up():
        inverse_rms = launch_inverse_rms(x, fused.eps)
        return launch_prenorm_swiglu_gateup(
            x,
            fused._gate_weight_fp8,
            fused._gate_weight_scale,
            fused._up_weight_fp8,
            fused._up_weight_scale,
            inverse_rms,
        )

    def baseline_forward():
        normalized = launch_rmsnorm(
            x,
            fused.elementwise_factor,
            fused.eps,
        )
        hidden = mxtest.launch_fp8_swiglu(
            normalized,
            baseline._gate_weight_fp8,
            baseline._gate_weight_scale,
            baseline._up_weight_fp8,
            baseline._up_weight_scale,
        )
        return mxtest.launch_fp8gemm(hidden, down_weight, down_scale)

    def fused_forward():
        return fused(x)

    baseline_output = baseline_forward()
    fused_output = fused_forward()
    relative_l2 = (
        torch.linalg.vector_norm(
            fused_output.float() - baseline_output.float()
        )
        / torch.linalg.vector_norm(baseline_output.float())
    ).item()

    baseline_ms = triton.testing.do_bench(
        baseline_forward,
        warmup=25,
        rep=100,
    )
    fused_ms = triton.testing.do_bench(
        fused_forward,
        warmup=25,
        rep=100,
    )

    def norm_only():
        return launch_rmsnorm(
            x,
            fused.elementwise_factor,
            fused.eps,
        )

    norm_ms = triton.testing.do_bench(
        norm_only,
        warmup=25,
        rep=100,
    )
    baseline_gate_up_ms = triton.testing.do_bench(
        baseline_gate_up,
        warmup=25,
        rep=100,
    )
    fused_gate_up_ms = triton.testing.do_bench(
        fused_gate_up,
        warmup=25,
        rep=100,
    )
    return {
        "shape": (M, K, H, O),
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "norm_ms": norm_ms,
        "speedup": baseline_ms / fused_ms,
        "baseline_gate_up_ms": baseline_gate_up_ms,
        "fused_gate_up_ms": fused_gate_up_ms,
        "gate_up_speedup": baseline_gate_up_ms / fused_gate_up_ms,
        "relative_l2": relative_l2,
    }


def main():
    args = parse_args()
    if args.m is not None:
        cases = ((args.m, args.k, args.hidden, args.out),)
    else:
        cases = DEFAULT_CASES

    print(
        f"{torch.cuda.get_device_name(0)}, "
        f"Triton {triton.__version__}"
    )
    print(
        "M K H O | norm ms | gate/up base | gate/up fused | "
        "full base | full fused | full speedup | rel L2"
    )
    for case in cases:
        result = benchmark_case(*case)
        M, K, H, O = result["shape"]
        print(
            f"{M} {K} {H} {O} | "
            f"{result['norm_ms']:.4f} | "
            f"{result['baseline_gate_up_ms']:.4f} | "
            f"{result['fused_gate_up_ms']:.4f} | "
            f"{result['baseline_ms']:.4f} | "
            f"{result['fused_ms']:.4f} | "
            f"{result['speedup']:.3f}x | "
            f"{result['relative_l2']:.4%}"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
