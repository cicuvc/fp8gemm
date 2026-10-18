"""Pretrain a small decoder-only transformer with the experimental fused Muon path."""

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import pyarrow.parquet as pq

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import torch
import torch.nn.functional as F
import wandb
from flash_attn import flash_attn_varlen_func
from tokenizers import Tokenizer
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.muon_fused import (
    FusedMuonLinear,
    finalize_muon_step,
    set_accum_first,
)


DEFAULT_DATA = Path(
    "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/"
    "cicuvc/fwedu-100B/sample/100BT"
)
DEFAULT_TOKENIZER = Path(
    "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/"
    "cicuvc/tokenizer/tokenizer.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoder-only pretraining smoke test with fused FP8 Muon."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden", type=int, default=1408)
    parser.add_argument("--tokens-per-microbatch", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=800_000_000)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the step count implied by --target-tokens",
    )
    parser.add_argument("--muon-lr", type=float, default=6e-3)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-fraction", type=float, default=0.1)
    parser.add_argument(
        "--stochastic-rounding",
        action="store_true",
        help="use stochastic rounding when writing BF16 Muon momentum",
    )
    parser.add_argument("--muon-weight-decay", type=float, default=0.01)
    parser.add_argument("--aux-weight-decay", type=float, default=0.01)
    parser.add_argument("--token-batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/fp8_muon_decoder/latest.pt"),
    )
    parser.add_argument("--wandb-project", default="fp8-muon-decoder")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(
            x_fp32.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(torch.bfloat16)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.pow(
            theta,
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim,
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer(
            "cos",
            angles.cos().to(torch.bfloat16)[:, None, :],
            persistent=False,
        )
        self.register_buffer(
            "sin",
            angles.sin().to(torch.bfloat16)[:, None, :],
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        offsets = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        position_ids = (
            torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            - offsets
        )
        cos = self.cos.index_select(0, position_ids)
        sin = self.sin.index_select(0, position_ids)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        stochastic_rounding: bool,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = FusedMuonLinear(
            d_model,
            d_model,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.k_proj = FusedMuonLinear(
            d_model,
            d_model,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.v_proj = FusedMuonLinear(
            d_model,
            d_model,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.out_proj = FusedMuonLinear(
            d_model,
            d_model,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.rope = RotaryEmbedding(self.head_dim, context_length)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        total_tokens, _ = x.shape
        q = self.q_proj(x).view(
            total_tokens,
            self.n_heads,
            self.head_dim,
        )
        k = self.k_proj(x).view(
            total_tokens,
            self.n_heads,
            self.head_dim,
        )
        v = self.v_proj(x).view(
            total_tokens,
            self.n_heads,
            self.head_dim,
        )
        q = self.rope(q, cu_seqlens)
        k = self.rope(k, cu_seqlens)
        attended = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        return self.out_proj(attended.reshape(total_tokens, self.d_model))


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        stochastic_rounding: bool,
    ):
        super().__init__()
        self.gate = FusedMuonLinear(
            d_model,
            hidden_size,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.up = FusedMuonLinear(
            d_model,
            hidden_size,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )
        self.down = FusedMuonLinear(
            hidden_size,
            d_model,
            device="cuda",
            stochastic_rounding=stochastic_rounding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(x)) * self.up(x)
        return self.down(hidden)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden: int,
        context_length: int,
        stochastic_rounding: bool,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = Attention(
            d_model,
            n_heads,
            context_length,
            stochastic_rounding,
        )
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden, stochastic_rounding)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            cu_seqlens,
            max_seqlen,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_hidden: int,
        context_length: int,
        stochastic_rounding: bool,
    ):
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Parameter(
            torch.empty(vocab_size, d_model, dtype=torch.bfloat16)
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    mlp_hidden,
                    context_length,
                    stochastic_rounding,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.apply(self._init_weights)
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = F.embedding(tokens, self.embedding)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen)
        hidden = self.final_norm(x)
        return F.linear(hidden, self.embedding)


def token_documents(
    parquet_files: list[Path],
    tokenizer: Tokenizer,
    token_batch_size: int,
    context_length: int,
    seed: int,
):
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    rng = random.Random(seed)

    while True:
        rng.shuffle(parquet_files)
        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(
                batch_size=token_batch_size,
                columns=["text"],
            ):
                texts = [text for text in batch.column(0).to_pylist() if text]
                encodings = tokenizer.encode_batch(
                    texts,
                    add_special_tokens=False,
                )
                for encoding in encodings:
                    payload = encoding.ids
                    payload_size = max(context_length - 2, 1)
                    for start in range(0, len(payload), payload_size):
                        yield (
                            [bos_token_id]
                            + payload[start : start + payload_size]
                            + [eos_token_id]
                        )


def packed_varlen_batches(
    documents,
    tokens_per_microbatch: int,
    context_length: int,
):
    eos_token_id = 2
    pending_documents: deque[list[int]] = deque()

    while True:
        packed_documents = []
        total_tokens = 0
        while total_tokens < tokens_per_microbatch:
            document = (
                pending_documents.popleft()
                if pending_documents
                else next(documents)
            )
            remaining = tokens_per_microbatch - total_tokens
            if len(document) <= remaining:
                packed_documents.append(document)
                total_tokens += len(document)
                continue

            if total_tokens == 0:
                packed_documents.append(document[:remaining])
                total_tokens = remaining
                if remaining < len(document):
                    pending_documents.appendleft(document[remaining:])
                break

            pending_documents.appendleft(document)
            break

        padding_tokens = tokens_per_microbatch - total_tokens

        flat_tokens = []
        targets = []
        cu_seqlens = [0]
        max_seqlen = 0
        valid_tokens = 0
        for document in packed_documents:
            document_length = len(document)
            if document_length == 0:
                continue
            flat_tokens.extend(document)
            targets.extend(document[1:])
            targets.append(-100)
            cu_seqlens.append(cu_seqlens[-1] + document_length)
            max_seqlen = max(max_seqlen, document_length)
            valid_tokens += max(document_length - 1, 0)

        if padding_tokens:
            flat_tokens.extend([eos_token_id] * padding_tokens)
            targets.extend([-100] * padding_tokens)
            cu_seqlens.append(cu_seqlens[-1] + padding_tokens)
            max_seqlen = max(max_seqlen, padding_tokens)

        if max_seqlen > context_length:
            raise RuntimeError(
                f"document length {max_seqlen} exceeds context {context_length}"
            )

        yield (
            torch.tensor(flat_tokens, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(cu_seqlens, dtype=torch.int32),
            max_seqlen,
            valid_tokens,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def learning_rate_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_fraction: float,
) -> float:
    if not 0.0 <= min_lr_fraction <= 1.0:
        raise ValueError("--min-lr-fraction must be in [0, 1]")
    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / warmup_steps
        return min_lr_fraction + (1.0 - min_lr_fraction) * progress
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_fraction + (1.0 - min_lr_fraction) * cosine


def muon_momentum_stats(model: nn.Module) -> tuple[float, float]:
    rms_values = []
    maximum_values = []
    for module in model.modules():
        if isinstance(module, FusedMuonLinear):
            momentum = module.momentum_buffer.float()
            rms_values.append(momentum.square().mean().sqrt().item())
            maximum_values.append(momentum.abs().max().item())
    return max(rms_values), max(maximum_values)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    aux_optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    if args.memory_limit_gib > 0:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_gib * 2**30 / total_memory,
            device=device,
        )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    parquet_files = sorted(args.data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {args.data_dir}")

    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden,
        context_length=args.context_length,
        stochastic_rounding=args.stochastic_rounding,
    ).to(device)
    model.train()

    muon_parameters = {
        id(module.master_weight)
        for module in model.modules()
        if isinstance(module, FusedMuonLinear)
    }
    aux_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in muon_parameters
    ]
    aux_optimizer = torch.optim.AdamW(
        aux_parameters,
        lr=args.aux_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.aux_weight_decay,
    )

    total_parameters = parameter_count(model)
    muon_parameter_count = sum(
        module.master_weight.numel()
        for module in model.modules()
        if isinstance(module, FusedMuonLinear)
    )
    print(
        f"parameters={total_parameters / 1e6:.2f}M "
        f"muon={muon_parameter_count / 1e6:.2f}M "
        f"vocab={vocab_size} "
        f"layers={args.n_layers} d_model={args.d_model} "
        f"heads={args.n_heads} mlp={args.mlp_hidden}"
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "parameters": total_parameters,
            "muon_parameters": muon_parameter_count,
            "vocab_size": vocab_size,
        },
    )

    documents = token_documents(
        parquet_files,
        tokenizer,
        args.token_batch_size,
        args.context_length,
        args.seed,
    )
    batch_stream = packed_varlen_batches(
        documents,
        args.tokens_per_microbatch,
        args.context_length,
    )

    tokens_per_step = args.tokens_per_microbatch * args.grad_accum
    if args.steps is None:
        if args.target_tokens <= 0:
            raise ValueError("--target-tokens must be positive when --steps is omitted")
        args.steps = math.ceil(args.target_tokens / tokens_per_step)
    target_tokens = args.steps * tokens_per_step
    print(
        f"tokens_per_step={tokens_per_step} steps={args.steps} "
        f"target_tokens={target_tokens / 1e6:.1f}M "
        f"memory_limit={args.memory_limit_gib:.1f}GiB"
    )
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()

    try:
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            lr_scale = learning_rate_scale(
                step,
                args.steps,
                args.warmup_steps,
                args.min_lr_fraction,
            )
            muon_lr = args.muon_lr * lr_scale
            aux_lr = args.aux_lr * lr_scale
            for group in aux_optimizer.param_groups:
                group["lr"] = aux_lr
            aux_optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            valid_tokens_step = 0

            for microstep in range(args.grad_accum):
                inputs, targets, cu_seqlens, max_seqlen, valid_tokens = next(
                    batch_stream
                )
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                cu_seqlens = cu_seqlens.to(device, non_blocking=True)
                set_accum_first(model, microstep == 0)

                logits = model(inputs, cu_seqlens, max_seqlen)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step={step} microstep={microstep}"
                    )
                (loss / args.grad_accum).backward()
                loss_sum += loss.item()
                valid_tokens_step += valid_tokens

            finalize_muon_step(
                model,
                lr=muon_lr,
                weight_decay=args.muon_weight_decay,
            )
            aux_optimizer.step()
            aux_optimizer.zero_grad(set_to_none=True)

            step_time = time.perf_counter() - step_start
            elapsed = time.perf_counter() - run_start
            mean_loss = loss_sum / args.grad_accum
            momentum_rms, momentum_max = muon_momentum_stats(model)
            memory_gib = torch.cuda.max_memory_allocated() / 2**30

            if step % args.log_every == 0:
                metrics = {
                    "train/loss": mean_loss,
                    "train/perplexity": math.exp(min(mean_loss, 20.0)),
                    "train/lr_scale": lr_scale,
                    "train/lr_muon": muon_lr,
                    "train/lr_aux": aux_lr,
                    "train/muon_momentum_rms_max": momentum_rms,
                    "train/muon_momentum_abs_max": momentum_max,
                    "train/tokens_per_second": valid_tokens_step / step_time,
                    "train/step_time_seconds": step_time,
                    "system/tokens_seen": step * tokens_per_step,
                    "system/gpu_memory_gib": memory_gib,
                    "system/elapsed_seconds": elapsed,
                }
                wandb.log(metrics, step=step)
                print(
                    f"step={step:4d} loss={mean_loss:.4f} "
                    f"ppl={math.exp(min(mean_loss, 20.0)):.2f} "
                    f"tok/s={valid_tokens_step / step_time:.0f} "
                    f"time={step_time:.3f}s "
                    f"mem={memory_gib:.2f}GiB "
                    f"m_rms={momentum_rms:.4f}"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint,
                    model,
                    aux_optimizer,
                    step,
                    args,
                )

        save_checkpoint(
            args.checkpoint,
            model,
            aux_optimizer,
            args.steps,
            args,
        )
        wandb.finish()
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
