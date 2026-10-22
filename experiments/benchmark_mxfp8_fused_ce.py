"""Compare experimental MXFP8 and installed FLA LM-head CE on one CUDA GPU.

Run: python -m experiments.benchmark_mxfp8_fused_ce --output benchmarks/mxfp8_fused_ce.json
FLA uses BF16 GEMMs. MXFP8 is approximate, with cached quantized weights in both
warmup and timing; weight refresh/optimizer time is excluded and reported as such.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics

import torch

from experiments.mxfp8_core import quantize_weight
from experiments.tk_mxfp8_fused_ce import MXFP8FusedLinearCrossEntropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--forward-config", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.vocab, args.repeats) <= 0:
        parser.error("dimensions and repeats must be positive")
    from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss as fla_ce

    torch.manual_seed(581)
    x = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn((args.vocab, args.hidden), device="cuda", dtype=torch.bfloat16) * .02).requires_grad_()
    y = torch.randint(args.vocab, (args.tokens,), device="cuda")
    result = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "method": "CUDA events around loss and autograd.grad; eager dispatch; no optimizer. MXFP8 cached weight refresh excluded. Peak includes cached workspaces and inputs, measured per backend.",
    }
    outputs = {}
    for backend in ("fla_bf16", "mxfp8"):
        criterion = None
        cache = None
        if backend == "mxfp8":
            criterion = MXFP8FusedLinearCrossEntropy(chunk_size=args.chunk_size, forward_config=args.forward_config)
            cache = quantize_weight(w.detach().float())
        def step():
            if criterion is None:
                loss = fla_ce(x, y, w, num_chunks=max(1, (args.tokens+args.chunk_size-1)//args.chunk_size), reduction="sum")
            else:
                loss = criterion(x, y, w, quantized_weight=cache)
            loss = loss / args.tokens
            grads = torch.autograd.grad(loss, (x, w))
            return loss, *grads
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        values = step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        outputs[backend] = tuple(v.detach().float().cpu() for v in values)
        del values
        timings = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            values = step()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
            del values
        result[backend] = {"median_ms": statistics.median(timings), "samples_ms": timings, "peak_allocated_mib": peak/2**20}
        if criterion is not None:
            criterion.clear_workspace()
        del criterion, cache, step
        gc.collect()
        torch.cuda.empty_cache()
    a, b = outputs["mxfp8"], outputs["fla_bf16"]
    result["accuracy_vs_fla"] = {
        "mxfp8_mean_loss": a[0].item(), "fla_mean_loss": b[0].item(),
        "abs_loss_error": (a[0]-b[0]).abs().item(),
        "dx_relative_l2": ((a[1]-b[1]).norm()/b[1].norm()).item(),
        "dw_relative_l2": ((a[2]-b[2]).norm()/b[2].norm()).item(),
    }
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
