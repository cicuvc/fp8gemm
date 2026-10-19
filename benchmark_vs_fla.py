"""Reproducible full-MLP baseline: cached FP8 vs FLA BF16 vs eager torch BF16."""
import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing
import fla
from fla.modules.mlp import GatedMLP

from fp8_swiglu import FP8SwiGLU

DEFAULT_CASES = [(256, 512, 1024), (1024, 1024, 4096), (2048, 2048, 8192)]


class TorchSwiGLU(torch.nn.Module):
    def __init__(self, k, hidden):
        super().__init__()
        self.gate_proj = torch.nn.Linear(k, hidden, bias=False, device="cuda", dtype=torch.bfloat16)
        self.up_proj = torch.nn.Linear(k, hidden, bias=False, device="cuda", dtype=torch.bfloat16)
        self.down_proj = torch.nn.Linear(hidden, k, bias=False, device="cuda", dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def params(layer):
    if isinstance(layer, FP8SwiGLU):
        return layer.gate_weight, layer.up_weight, layer.down_proj.master_weight
    return layer.gate_proj.weight, layer.up_proj.weight, layer.down_proj.weight


def error(actual, reference):
    diff = actual.float() - reference.float()
    if not torch.isfinite(actual).all():
        raise RuntimeError("Non-finite output/gradient")
    return {"relative_l2": (diff.norm() / reference.float().norm().clamp_min(1e-12)).item(),
            "max_abs": diff.abs().max().item()}


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def benchmark_case(m, k, hidden, args):
    torch.manual_seed(args.seed)
    layers = {
        "fp8": FP8SwiGLU(k, hidden, k, bias=False, device="cuda"),
        "fla_bf16": GatedMLP(hidden_size=k, intermediate_size=hidden,
                             hidden_act="swish", fuse_swiglu=True).cuda().bfloat16(),
        "torch_bf16": TorchSwiGLU(k, hidden),
    }
    # Use identical, exactly BF16-representable source weights for all providers.
    with torch.no_grad():
        for weights in zip(*(params(layer) for layer in layers.values())):
            source = weights[-1]
            for weight in weights[:-1]:
                weight.copy_(source)
    layers["fp8"].refresh_fp8_weights()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    reference = layers["torch_bf16"](x)
    reference_grads = torch.autograd.grad(reference, (x,) + params(layers["torch_bf16"]), dy)
    results, functions = {}, {}
    for name, layer in layers.items():
        y = layer(x)
        grads = torch.autograd.grad(y, (x,) + params(layer), dy)
        assert y.shape == (m, k) and y.dtype == torch.bfloat16
        results[name] = {"errors_vs_torch": {"output": error(y, reference), **{
            label: error(g, r) for label, g, r in zip(
                ("dx", "dgate", "dup", "ddown"), grads, reference_grads)}}}

        def forward(layer=layer):
            with torch.no_grad():
                return layer(x)

        def forward_backward(layer=layer):
            return torch.autograd.grad(layer(x), (x,) + params(layer), dy)

        functions[name] = {"forward": forward, "forward_backward": forward_backward}
        forward()
        forward_backward()
    del reference, reference_grads, y, grads
    torch.cuda.synchronize()
    modes = ["eager", "cuda_graph"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for name in layers:
            results[name][mode] = {phase: {"round_ms": []} for phase in functions[name]}
        for repeat in range(args.repeats):
            # Rotate provider order to reduce systematic thermal/order bias.
            names = list(layers)
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in names:
                for phase, fn in functions[name].items():
                    if mode == "eager":
                        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep,
                                                   return_mode="median")
                    else:
                        ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="median")
                    results[name][mode][phase]["round_ms"].append(ms)
        for name in layers:
            for phase, values in results[name][mode].items():
                values["median_ms"] = statistics.median(values["round_ms"])
                values["min_ms"] = min(values["round_ms"])
                values["max_ms"] = max(values["round_ms"])
                flops = (6 if phase == "forward" else 18) * m * k * hidden
                values["effective_tflops"] = flops / values["median_ms"] / 1e9
        for phase in ("forward", "forward_backward"):
            times = [results[name][mode][phase]["median_ms"] for name in layers]
            print(f"{m:5} {k:5} {hidden:5} | {mode:10} | {phase:16} | "
                  f"{times[0]:.4f} | {times[1]:.4f} | {times[2]:.4f} | "
                  f"{times[1]/times[0]:.2f}x | {times[2]/times[0]:.2f}x", flush=True)
    return {"shape": {"M": m, "K": k, "H": hidden, "O": k}, "providers": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--mode", choices=("eager", "cuda_graph", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=25, help="Eager warmup milliseconds")
    parser.add_argument("--rep", type=int, default=200, help="Timing budget in milliseconds")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=Path("runs/swiglu_baseline.json"))
    args = parser.parse_args()
    dimensions = (args.m, args.k, args.hidden)
    if any(v is not None for v in dimensions) and not all(v is not None and v > 0 for v in dimensions):
        parser.error("--m, --k and --hidden must be supplied together and positive")
    if min(args.repeats, args.rep, args.warmup) <= 0:
        parser.error("--repeats, --rep and --warmup must be positive")
    cases = [dimensions] if args.m is not None else DEFAULT_CASES
    result = {
        "environment": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "fla": fla.__version__, "cuda": torch.version.cuda,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "nvidia_smi_before": command_output(["nvidia-smi"]),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
        "settings": {**vars(args), "output": str(args.output)},
        "method": {
            "scope": "Full bias-free MLP; BF16 input/output; forward no_grad; backward computes dx and all three dW; no optimizer or weight refresh",
            "fp8": "Cached FP8 weights, FP32 master weights and dW; default fp8_wgrad=False",
            "bf16": "BF16 parameters and gradients; FLA fuse_swiglu=True; torch uses Linear/silu/multiply/Linear without compile",
            "timing": "Median of round medians; eager CUDA events with L2 flush per iteration; CUDA graph replay without L2 flush, excludes Python launch overhead",
            "flops": "Effective dense GEMM FLOPs only: forward 6*M*K*H, forward+backward 18*M*K*H",
        }, "cases": [],
    }
    print(f"{result['environment']['gpu']}; torch {torch.__version__}; Triton {triton.__version__}; FLA {fla.__version__}")
    print("M K H | mode | phase | FP8 ms | FLA ms | torch ms | FLA/FP8 | torch/FP8", flush=True)
    for case in cases:
        result["cases"].append(benchmark_case(*case, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        torch.cuda.empty_cache()
    result["environment"]["nvidia_smi_after"] = command_output(["nvidia-smi"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
