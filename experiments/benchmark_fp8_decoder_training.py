"""Measure FP8 decoder training MFU on the historical pretraining shape."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fp8_decoder_transformer import FP8DecoderTransformer
from experiments.fp8_decoder_postnorm_transformer import (
    FP8PostNormDecoderTransformer,
)
from fp8_adamw import AdamWStateDtype, FusedAdamWMXFP8, optimizer_state_nbytes


def decoder_flops(
    tokens,
    layers,
    d_model,
    n_heads,
    mlp_hidden,
    vocab_size,
    sequence_lengths,
):
    """Training FLOPs for projections, lm head, and causal attention.

    Each linear GEMM uses forward=2MNK and backward=4MNK. Causal attention
    counts QK, AV, dQ, dK, and dV over causal token pairs.
    """
    del n_heads  # Head count cancels when using d_model as the reduction width.
    linear = 6 * tokens * layers * (
        4 * d_model * d_model + 3 * d_model * mlp_hidden
    )
    lm_head = 6 * tokens * d_model * vocab_size
    causal_pairs = sum(length * (length + 1) // 2 for length in sequence_lengths)
    attention = 10 * layers * d_model * causal_pairs

    out_projection = 6 * tokens * layers * d_model * d_model
    bf16_flops = out_projection + lm_head + attention
    return {
        "total": linear + lm_head + attention,
        "fp8_linear": linear - out_projection,
        "bf16_o_projection": out_projection,
        "bf16_lm_head": lm_head,
        "bf16_attention": attention,
        "bf16_total": bf16_flops,
    }


def timed(function, *, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden", type=int, default=1408)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--tokens-per-microbatch", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fp8-peak-tflops", type=float, default=419.0)
    parser.add_argument("--no-fused-linear-ce", action="store_true")
    parser.add_argument("--fused-ce-chunks", type=int, default=2)
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--bf16-adam-state", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/fp8_decoder_training_rtx5090.json",
    )
    args = parser.parse_args()
    if args.bf16_adam_state and args.fused_adamw:
        parser.error("--bf16-adam-state and --fused-adamw are mutually exclusive")

    if args.tokens_per_microbatch % args.context_length:
        raise ValueError("tokens-per-microbatch must contain whole sequences")
    sequence_count = args.tokens_per_microbatch // args.context_length
    sequence_lengths = [args.context_length] * sequence_count

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model_class = (
        FP8PostNormDecoderTransformer
        if args.post_norm
        else FP8DecoderTransformer
    )
    model = model_class(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden,
        context_length=args.context_length,
        device="cuda",
    ).train()

    tokens = torch.randint(
        0, args.vocab_size, (args.tokens_per_microbatch,), device="cuda"
    )
    targets = torch.roll(tokens, -1)
    cu_seqlens = torch.tensor(
        [0, *torch.arange(1, sequence_count + 1).mul(args.context_length).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    def forward_backward():
        model.zero_grad(set_to_none=True)
        loss = model.forward_loss(
            tokens,
            targets,
            cu_seqlens,
            args.context_length,
            fused_linear_ce=not args.no_fused_linear_ce,
            num_chunks=args.fused_ce_chunks,
        )
        loss.backward()

    def graph_forward_backward():
        model.zero_grad(set_to_none=True)
        logits = model(tokens, cu_seqlens, args.context_length)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()

    optimizer = (
        FusedAdamWMXFP8(
            model.parameters(),
            lr=1e-4,
        )
        if args.fused_adamw
        else AdamWStateDtype(
            model.parameters(),
            lr=1e-4,
            moment1_dtype=torch.bfloat16,
            moment2_dtype=torch.bfloat16,
        )
        if args.bf16_adam_state
        else torch.optim.AdamW(model.parameters(), lr=1e-4)
    )
    model.bind_optimizer(optimizer)

    def full_step():
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward_loss(
            tokens,
            targets,
            cu_seqlens,
            args.context_length,
            fused_linear_ce=not args.no_fused_linear_ce,
            num_chunks=args.fused_ce_chunks,
        )
        loss.backward()
        optimizer.step()

    fwd_bwd = timed(forward_backward, warmup=args.warmup, repeats=args.repeats)
    full = timed(full_step, warmup=args.warmup, repeats=args.repeats)
    if args.no_fused_linear_ce:
        for _ in range(args.warmup):
            graph_forward_backward()
        torch.cuda.synchronize()
        torch.autograd.graph.set_override_stale_capture_stream(True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_forward_backward()
        graph_fwd_bwd = timed(
            graph.replay,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    else:
        graph_fwd_bwd = None
    flops = decoder_flops(
        args.tokens_per_microbatch,
        args.n_layers,
        args.d_model,
        args.n_heads,
        args.mlp_hidden,
        args.vocab_size,
        sequence_lengths,
    )

    def metrics(timing):
        tflops = flops["total"] / (timing["median_us"] * 1e6)
        return {
            **timing,
            "tflops": tflops,
            "fp8_peak_mfu_percent": tflops / args.fp8_peak_tflops * 100,
        }

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config_source": (
            "Defaults from 3d58143:experiments/train_decoder_muon.py; "
            "vocab size from the referenced tokenizer."
        ),
        "attention_out_projection": "torch.nn.Linear(dtype=torch.bfloat16)",
        "loss_backend": (
            "unfused"
            if args.no_fused_linear_ce
            else "fla.FusedLinearCrossEntropyLoss"
        ),
        "optimizer": (
            "FusedAdamWMXFP8(BF16 moments, fused MXFP8 update)"
            if args.fused_adamw
            else
            "AdamWStateDtype(BF16 moments)"
            if args.bf16_adam_state
            else "torch.optim.AdamW(FP32 state defaults)"
        ),
        "config": {
            **vars(args),
            "sequence_count": sequence_count,
            "sequence_lengths": sequence_lengths,
            "parameters": sum(p.numel() for p in model.parameters()),
            "normalization": "post" if args.post_norm else "pre",
        },
        "flops": flops,
        "method": {
            "forward_backward": (
                "CUDA events; fixed packed-varlen batch; zero_grad, loss, and "
                "backward included."
            ),
            "graph_forward_backward": (
                "Same work with Python layer dispatch removed by CUDA Graph; "
                "position IDs are precomputed before capture."
            ),
            "full_step": (
                "CUDA events; optimizer step and any required FP8 cache "
                "refresh included."
            ),
            "mfu": (
                "Useful model FLOPs divided by nominal 419 TFLOPS FP8 peak. "
                "BF16 O projection, lm head, and attention are included in "
                "the numerator, so this is conservative for mixed precision."
            ),
        },
        "forward_backward": metrics(fwd_bwd),
        "graph_forward_backward": (
            metrics(graph_fwd_bwd)
            if graph_fwd_bwd is not None
            else {
                "unavailable": (
                    "FLA fused linear cross-entropy synchronizes on the host "
                    "to count valid targets and is not CUDA Graph capturable."
                )
            }
        ),
        "full_step": metrics(full),
        "optimizer_state_gib": optimizer_state_nbytes(optimizer) / 2**30,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }

    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")

    print(
        f"{args.n_layers}L d={args.d_model} h={args.n_heads} "
        f"mlp={args.mlp_hidden} tokens={args.tokens_per_microbatch} "
        f"ctx={args.context_length}",
        flush=True,
    )
    for name in ["forward_backward", "graph_forward_backward", "full_step"]:
        row = result[name]
        if "median_us" in row:
            print(
                f"{name}: {row['median_us']:.1f} us, "
                f"{row['tflops']:.1f} TF/s, "
                f"FP8-peak MFU {row['fp8_peak_mfu_percent']:.1f}%",
                flush=True,
            )
        else:
            print(f"{name}: unavailable ({row['unavailable']})", flush=True)
    print(f"peak_memory={result['peak_memory_gib']:.2f} GiB")
    print(f"optimizer_state={result['optimizer_state_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
