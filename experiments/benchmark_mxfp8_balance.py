"""Measure full MXFP8 CE forward/backward overhead of balance and softcap.

All modes use the same packed inputs, ignore mask, and cached quantized weight.
The sum baseline is unnormalized. Optimizer and weight refresh are excluded.
"""

import argparse
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
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=1, help="Steps per timing sample; use >1 to reduce timing noise")
    parser.add_argument("--softcap", type=float, help="Also measure softcap against each matching balance mode")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.documents <= 0 or args.documents % 4 or args.tokens <= 0 or
            args.tokens % (4 * args.documents) or args.repeats <= 0 or args.iterations <= 0):
        parser.error("documents must be a positive multiple of 4; tokens a positive multiple of 4*documents; repeats/iterations positive")
    torch.manual_seed(911)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(args.vocab, args.hidden, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_()
    wq, ws = quantize_weight(w.detach().float())
    y = torch.randint(args.vocab, (args.tokens,), device="cuda")
    lengths = [v * (args.tokens // (4 * args.documents)) for v in [1, 3, 5, 7] * (args.documents // 4)]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    y[::7] = -100
    y[:lengths[0]] = -100
    names = ("sum", "token", "document")
    if args.softcap is not None:
        names += tuple(name + "_softcap" for name in names)
    steps, graphs, outputs = {}, {}, {}
    for name in names:
        mode = name.removesuffix("_softcap")
        criterion = MXFP8FusedLinearCrossEntropy(
            chunk_size=args.chunk_size, balance=None if mode == "sum" else mode,
            softcap=args.softcap if name.endswith("_softcap") else None,
        )
        def step(criterion=criterion):
            loss = criterion(x, y, w, cu_seqlens=cu, quantized_weight=(wq, ws))
            return loss, *torch.autograd.grad(loss, (x, w))
        steps[name] = step
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = step()
        graphs[name] = graph
    torch.cuda.synchronize()
    result = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "document_lengths": lengths,
        "valid_tokens": int((y != -100).sum()),
        "method": "CUDA events; full forward plus autograd.grad; same inputs with 1/7 ignored and first document fully ignored. Cached quantized W, excluding optimizer/refresh. Rotate mode order each round. Each sample averages config.iterations steps. Graph numbers include loss metadata and gradient normalization.",
    }
    for kind, functions in (("eager", steps), ("graph", {k: g.replay for k, g in graphs.items()})):
        samples = {name: [] for name in names}
        for iteration in range(args.repeats):
            for i in range(len(names)):
                name = names[(i + iteration) % len(names)]
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    values = functions[name]()
                    del values
                end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.iterations)
        result[kind] = {name: {"median_ms": statistics.median(times), "samples_ms": times} for name, times in samples.items()}
        for name in names:
            baseline_name = name.removesuffix("_softcap") if name.endswith("_softcap") else "sum"
            baseline = result[kind][baseline_name]["median_ms"]
            result[kind][name]["baseline"] = baseline_name
            result[kind][name]["overhead_percent"] = (result[kind][name]["median_ms"] / baseline - 1) * 100
    # Graph replay must reproduce eager results with the same normalization.
    for name in names:
        expected = steps[name]()
        graphs[name].replay()
        for actual, reference in zip(outputs[name], expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
