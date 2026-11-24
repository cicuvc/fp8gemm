"""Warm graph FFN latency including activation/gradient quantization.

Weight-cache refresh and optimizer updates are excluded. The PyTorch BF16
baseline casts FP32 master weights each forward and returns FP32 weight grads.
Run from the repository root: python benchmarks/bench_fp8_training.py.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
import torch.nn.functional as F

from mxfp8 import FP8SwiGLU


class BF16SwiGLU(nn.Module):
    def __init__(self, k, h, n):
        super().__init__()
        self.gate = nn.Parameter(torch.randn(h, k, device="cuda") / k**.5)
        self.up = nn.Parameter(torch.randn(h, k, device="cuda") / k**.5)
        self.down = nn.Parameter(torch.randn(n, h, device="cuda") / h**.5)

    def forward(self, x):
        g = F.linear(x, self.gate.bfloat16())
        u = F.linear(x, self.up.bfloat16())
        return F.linear((F.silu(g.float()) * u.float()).bfloat16(), self.down.bfloat16())


def measure(layer, x, backward, samples=30):
    dy = torch.randn(x.shape[0], layer.out_features if isinstance(layer, FP8SwiGLU)
                     else layer.down.shape[0], device=x.device, dtype=x.dtype)

    def run():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with torch.set_grad_enabled(backward):
            y = layer(x)
        if backward:
            y.backward(dy)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(10):
        graph.replay()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    times = []
    for _ in range(samples):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 100)
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["128,1024,2816,1024", "1024,1024,2816,1024",
                                                       "1024,2048,5632,2048"])
    parser.add_argument("--reductions", nargs="+", choices=["persistent", "atomic", "deterministic"],
                        default=["persistent", "atomic", "deterministic"])
    parser.add_argument("--output", type=Path, default=Path("runs/tk_fp8_gemm/ffn_benchmark.json"))
    args = parser.parse_args()
    torch.manual_seed(123)
    report = dict(gpu=torch.cuda.get_device_name(), results=[])
    for shape in args.shapes:
        m, k, h, n = map(int, shape.split(","))
        layers = {"fp8_" + mode: FP8SwiGLU(k,h,n,streamk=None if mode == "persistent" else mode)
                  for mode in args.reductions}
        first = next(iter(layers.values()))
        for layer in layers.values():
            if layer is not first: layer.load_state_dict(first.state_dict())
        bf16 = BF16SwiGLU(k,h,n)
        with torch.no_grad():
            bf16.gate.copy_(first.gate_weight)
            bf16.up.copy_(first.up_weight)
            bf16.down.copy_(first.down_proj.weight)
        layers["bf16"] = bf16
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        result = dict(shape=[m, k, h, n])
        for name, layer in layers.items():
            result[name + "_forward_us"] = measure(layer, x, False)
            result[name + "_forward_backward_us"] = measure(layer, x, True)
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        del layers, first, layer, bf16, x
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
