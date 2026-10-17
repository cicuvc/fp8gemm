import argparse

import torch
import triton
import triton.testing
from fla.modules.mlp import GatedMLP

from fp8_swiglu import FP8SwiGLU


DEFAULT_CASES = [
    (256, 512, 1024),
    (1024, 1024, 4096),
    (2048, 2048, 8192),
]


def make_fp8_layer(k, hidden):
    return FP8SwiGLU(
        in_features=k,
        hidden_features=hidden,
        out_features=k,
        bias=False,
        device="cuda",
    )


def make_fla_layer(k, hidden):
    return (
        GatedMLP(
            hidden_size=k,
            intermediate_size=hidden,
            hidden_act="swish",
            fuse_swiglu=True,
        )
        .cuda()
        .bfloat16()
    )


def params(layer):
    if isinstance(layer, FP8SwiGLU):
        return (
            layer.gate_weight,
            layer.up_weight,
            layer.down_proj.master_weight,
        )
    return (
        layer.gate_proj.weight,
        layer.up_proj.weight,
        layer.down_proj.weight,
    )


def measure(layer, m, k, hidden, grad_output):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.no_grad():
        out = layer(x)
    assert out.shape == (m, k)

    def forward():
        with torch.no_grad():
            return layer(x)

    def forward_backward():
        out = layer(x)
        return torch.autograd.grad(out, (x,) + params(layer), grad_outputs=grad_output)

    out = forward_backward()
    del out

    forward_ms = triton.testing.do_bench(forward, warmup=25, rep=100)
    train_ms = triton.testing.do_bench(forward_backward, warmup=25, rep=100)

    forward_flops = 4 * m * hidden * k + 2 * m * k * hidden
    train_flops = 12 * m * hidden * k + 6 * m * k * hidden
    return {
        "forward_ms": forward_ms,
        "train_ms": train_ms,
        "forward_tflops": forward_flops / forward_ms / 1e9,
        "train_tflops": train_flops / train_ms / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.m is not None:
        cases = [(args.m, args.k, args.hidden)]

    device = torch.cuda.get_device_properties(0)
    print(f"{device.name}, FLA {__import__('fla').__version__}, Triton {triton.__version__}")
    print("M K H | fp8 fwd | fla fwd | speedup | fp8 train | fla train | speedup")

    for m, k, hidden in cases:
        fp8_layer = make_fp8_layer(k, hidden)
        fla_layer = make_fla_layer(k, hidden)
        grad_output = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

        fp8 = measure(fp8_layer, m, k, hidden, grad_output)
        fla = measure(fla_layer, m, k, hidden, grad_output)

        print(
            f"{m} {k} {hidden} | "
            f"{fp8['forward_ms']:.3f} ms | "
            f"{fla['forward_ms']:.3f} ms | "
            f"{fla['forward_ms'] / fp8['forward_ms']:.2f}x | "
            f"{fp8['train_ms']:.3f} ms | "
            f"{fla['train_ms']:.3f} ms | "
            f"{fla['train_ms'] / fp8['train_ms']:.2f}x"
        )
        print(
            f"  TFLOPS: fp8 fwd {fp8['forward_tflops']:.1f}, "
            f"fla fwd {fla['forward_tflops']:.1f}, "
            f"fp8 train {fp8['train_tflops']:.1f}, "
            f"fla train {fla['train_tflops']:.1f}"
        )

        del fp8_layer, fla_layer, grad_output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
