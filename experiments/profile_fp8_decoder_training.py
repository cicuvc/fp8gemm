"""Run one FP8 decoder training step for external profilers."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fp8_decoder_transformer import FP8DecoderTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden", type=int, default=1408)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--tokens-per-microbatch", type=int, default=16384)
    args = parser.parse_args()

    model = FP8DecoderTransformer(
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
    sequence_count = args.tokens_per_microbatch // args.context_length
    cu_seqlens = torch.tensor(
        [
            0,
            *torch.arange(1, sequence_count + 1)
            .mul(args.context_length)
            .tolist(),
        ],
        device="cuda",
        dtype=torch.int32,
    )

    loss = model.forward_loss(
        tokens,
        targets,
        cu_seqlens,
        args.context_length,
    )
    loss.backward()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
