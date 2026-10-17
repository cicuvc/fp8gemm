import argparse

import torch
import triton
import triton.testing

import mxtest
from fp8_swiglu import FP8SwiGLU


DEFAULT_CASES = [
    (256, 512, 1024, 512),
    (1024, 1024, 4096, 1024),
    (2048, 2048, 8192, 2048),
]


def format_config(kernel):
    config = getattr(kernel, "best_config", None)
    if config is None:
        return "not-run"
    return (
        f"{config.kwargs['BLOCK_M']}x{config.kwargs['BLOCK_N']}x{config.kwargs['BLOCK_K']}"
        f"/w{config.num_warps}/s{config.num_stages}"
    )


def benchmark_case(m, k, hidden, out_features):
    layer = FP8SwiGLU(k, hidden, out_features, bias=True, device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(m, out_features, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        layer(x)

    grads = torch.autograd.grad(
        layer(x),
        (
            x,
            layer.gate_weight,
            layer.up_weight,
            layer.down_proj.master_weight,
        ),
        grad_outputs=grad_output,
    )
    del grads

    def forward():
        with torch.no_grad():
            return layer(x)

    def forward_backward():
        out = layer(x)
        return torch.autograd.grad(
            out,
            (
                x,
                layer.gate_weight,
                layer.up_weight,
                layer.down_proj.master_weight,
            ),
            grad_outputs=grad_output,
        )

    forward_ms = triton.testing.do_bench(forward, warmup=25, rep=100)
    train_ms = triton.testing.do_bench(forward_backward, warmup=25, rep=100)

    forward_flops = 4 * m * hidden * k + 2 * m * out_features * hidden
    train_flops = 12 * m * hidden * k + 6 * m * out_features * hidden

    configs = {
        "swiglu_fwd": format_config(mxtest.fp8_swiglu_fwd),
        "linear_fwd": format_config(mxtest.fp8gemm_fwd),
        "swiglu_dx": format_config(mxtest.fp8_swiglu_bwd_dx),
        "linear_dx": format_config(mxtest.fp8gemm_bwd),
        "wgrad": format_config(mxtest.fp8gemm_wgrad),
    }

    return {
        "shape": (m, k, hidden, out_features),
        "forward_ms": forward_ms,
        "train_ms": train_ms,
        "forward_tflops": forward_flops / forward_ms / 1e9,
        "train_tflops": train_flops / train_ms / 1e9,
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--out", type=int, default=None)
    args = parser.parse_args()

    if args.m is not None:
        cases = [(args.m, args.k, args.hidden, args.out)]
    else:
        cases = DEFAULT_CASES

    device = torch.cuda.get_device_properties(0)
    print(f"{device.name}, Triton {triton.__version__}, torch {torch.__version__}")
    print("M K H O | fwd ms | fwd TFLOPS | fwd+bwd ms | fwd+bwd TFLOPS")

    results = []
    for case in cases:
        result = benchmark_case(*case)
        results.append(result)
        m, k, hidden, out_features = result["shape"]
        print(
            f"{m} {k} {hidden} {out_features} | "
            f"{result['forward_ms']:.3f} | "
            f"{result['forward_tflops']:.1f} | "
            f"{result['train_ms']:.3f} | "
            f"{result['train_tflops']:.1f}"
        )
        for name, config in result["configs"].items():
            print(f"  {name}: {config}")

        del result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
