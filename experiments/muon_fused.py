"""Experimental fused wgrad + Muon-momentum path.

This module is intentionally isolated from the stable FP8Linear and Muon
training paths. It supports:

- FP32 master weights and FP32 wgrad accumulation inside the Triton kernel.
- A persistent BF16 Muon momentum buffer.
- Gradient accumulation over N microsteps without writing FP32 dW to memory.
- A separate Muon Newton-Schulz step after each accumulation group.
- Bias-free linear layers only, so the experiment remains Muon-only.
"""

import math

import torch
import triton
import triton.language as tl
from torch import nn

import mxtest
from fp8_muon import (
    DEFAULT_NS_COEFFICIENTS,
    _adjust_lr,
    _zeropower_via_newtonschulz,
)


@triton.jit
def stochastic_round_bf16_fp32(values, seed, offsets):
    bits = values.to(tl.uint32, bitcast=True)
    high16 = (bits >> 16).to(tl.uint16)
    low16 = bits & 0xFFFF
    random16 = tl.randint(seed, offsets).to(tl.uint32) & 0xFFFF
    round_up = (random16 < low16).to(tl.uint16)
    result_bits = (high16 + round_up).to(tl.uint16)
    return result_bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def stochastic_round_bf16_kernel(
    values,
    output,
    n_elements,
    seed,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values_block = tl.load(values + offsets, mask=mask, other=0.0)
    rounded = stochastic_round_bf16_fp32(
        values_block,
        seed,
        offsets.to(tl.uint32),
    )
    tl.store(output + offsets, rounded, mask=mask)


def launch_stochastic_round_bf16(
    values: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    if values.dtype != torch.float32 or not values.is_cuda:
        raise TypeError("stochastic rounding requires CUDA FP32 input")
    output = torch.empty_like(values, dtype=torch.bfloat16)
    block = 256
    stochastic_round_bf16_kernel[(triton.cdiv(values.numel(), block),)](
        values,
        output,
        values.numel(),
        seed,
        BLOCK=block,
    )
    return output


@triton.autotune(
    configs=mxtest.WGRAD_CONFIGS,
    key=["M", "N", "K", "ACCUM_FIRST", "STOCHASTIC_ROUNDING"],
    restore_value=["momentum"],
    cache_results=True,
)
@triton.jit
def fp8gemm_wgrad_momentum(
    dy_input,
    x_input,
    momentum,
    beta,
    M,
    N,
    K,
    dy_stride1,
    x_stride1,
    momentum_stride1,
    rounding_seed,
    rounding_counter,
    ACCUM_FIRST: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    dy_block_ptr = tl.make_block_ptr(
        dy_input,
        shape=(M, N),
        strides=(dy_stride1, 1),
        offsets=(0, row_idx * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    x_block_ptr = tl.make_block_ptr(
        x_input,
        shape=(M, K),
        strides=(x_stride1, 1),
        offsets=(0, col_idx * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for _ in tl.range(0, M, BLOCK_M):
        dy_block = tl.load(
            dy_block_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        dy_block_scaled, dy_scale = mxtest.online_scale(tl.trans(dy_block))

        x_block = tl.load(
            x_block_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        x_block_scaled, x_scale = mxtest.online_scale(tl.trans(x_block))
        x_block_scaled = tl.trans(x_block_scaled)

        acc = tl.dot_scaled(
            dy_block_scaled,
            dy_scale,
            "e4m3",
            x_block_scaled,
            x_scale,
            "e4m3",
            acc,
        )
        dy_block_ptr = tl.advance(dy_block_ptr, (BLOCK_M, 0))
        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_M, 0))

    momentum_block_ptr = tl.make_block_ptr(
        momentum,
        shape=(N, K),
        strides=(momentum_stride1, 1),
        offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    previous = tl.load(
        momentum_block_ptr,
        boundary_check=(1, 0),
        padding_option="zero",
    ).to(tl.float32)
    if ACCUM_FIRST:
        updated = beta * previous + (1.0 - beta) * acc
    else:
        updated = previous + (1.0 - beta) * acc

    if STOCHASTIC_ROUNDING:
        row_offsets = row_idx * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
        col_offsets = col_idx * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        flat_offsets = (
            row_offsets.to(tl.uint32) * K
            + col_offsets.to(tl.uint32)
            + rounding_counter * 0x3C6EF35F
        ).to(tl.uint32)
        rounded = stochastic_round_bf16_fp32(
            updated,
            rounding_seed,
            flat_offsets,
        )
    else:
        rounded = updated.to(tl.bfloat16)

    tl.store(
        momentum_block_ptr,
        rounded,
        boundary_check=(1, 0),
    )


def launch_wgrad_momentum(
    dy: torch.Tensor,
    x: torch.Tensor,
    momentum: torch.Tensor,
    beta: float,
    accum_first: bool,
    stochastic_rounding: bool,
    rounding_seed: int,
    rounding_counter: int,
) -> None:
    M, N = dy.shape
    M_x, K = x.shape
    if M_x != M:
        raise ValueError("dy and x must have the same M dimension")
    if dy.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise TypeError("dy and x must be BF16")
    if momentum.dtype != torch.bfloat16:
        raise TypeError("experimental Muon momentum must be BF16")
    if momentum.shape != (N, K):
        raise ValueError("momentum shape must match [N, K]")

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    fp8gemm_wgrad_momentum[grid](
        dy,
        x,
        momentum,
        beta,
        M,
        N,
        K,
        dy.stride(0),
        x.stride(0),
        momentum.stride(0),
        rounding_seed,
        rounding_counter,
        accum_first,
        stochastic_rounding,
    )


class _FusedMuonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        master_weight,
        weight_fp8,
        weight_scale,
        momentum,
        beta,
        accum_first,
        stochastic_rounding,
        rounding_seed,
        rounding_counter,
    ):
        out = mxtest.launch_fp8gemm(x, weight_fp8, weight_scale)
        ctx.save_for_backward(x, weight_fp8, weight_scale, momentum)
        ctx.beta = beta
        ctx.accum_first = accum_first
        ctx.stochastic_rounding = stochastic_rounding
        ctx.rounding_seed = rounding_seed
        ctx.rounding_counter = rounding_counter
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_fp8, weight_scale, momentum = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        grad_x = mxtest.launch_fp8gemm_backward(
            grad_output,
            weight_fp8,
            weight_scale,
        )
        launch_wgrad_momentum(
            grad_output,
            x,
            momentum,
            ctx.beta,
            ctx.accum_first,
            ctx.stochastic_rounding,
            ctx.rounding_seed,
            ctx.rounding_counter,
        )
        return grad_x, None, None, None, None, None, None, None, None, None


class FusedMuonLinear(nn.Module):
    """Bias-free FP8Linear whose backward directly accumulates BF16 momentum."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        muon_momentum: float = 0.95,
        stochastic_rounding: bool = False,
        stochastic_rounding_seed: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.muon_momentum = muon_momentum
        self.stochastic_rounding = stochastic_rounding
        self.stochastic_rounding_seed = (
            stochastic_rounding_seed
            if stochastic_rounding_seed is not None
            else int(torch.randint(0, 2**31 - 1, ()))
        )
        self._rounding_counter = 0
        self.master_weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.register_buffer(
            "momentum_buffer",
            torch.zeros(
                (out_features, in_features),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_buffer(
            "_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._accum_first = True
        self.reset_parameters()
        self.refresh_fp8_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        self.momentum_buffer.zero_()

    def refresh_fp8_weight(self):
        with torch.no_grad():
            self._weight_fp8, self._weight_scale = mxtest.launch_weight_quantize(
                self.master_weight.detach()
            )
        return self

    def set_accum_first(self, accum_first: bool):
        self._accum_first = accum_first
        return self

    def get_extra_state(self):
        return {
            "stochastic_rounding_seed": self.stochastic_rounding_seed,
            "rounding_counter": self._rounding_counter,
        }

    def set_extra_state(self, state):
        self.stochastic_rounding_seed = state["stochastic_rounding_seed"]
        self._rounding_counter = state["rounding_counter"]

    def finalize_muon_step(
        self,
        lr: float,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ):
        with torch.no_grad():
            update = _zeropower_via_newtonschulz(
                self.momentum_buffer.float(),
                ns_coefficients,
                ns_steps,
                1e-7,
            )
            adjusted_lr = _adjust_lr(lr, "original", self.master_weight.shape)
            self.master_weight.mul_(1.0 - lr * weight_decay)
            self.master_weight.add_(update, alpha=-adjusted_lr)
        self.refresh_fp8_weight()
        return self

    def forward(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input shape [M, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError("FusedMuonLinear requires CUDA BF16 input")
        rounding_counter = self._rounding_counter
        if self.stochastic_rounding:
            self._rounding_counter += 1
        return _FusedMuonLinearFunction.apply(
            x.contiguous(),
            self.master_weight,
            self._weight_fp8,
            self._weight_scale,
            self.momentum_buffer,
            self.muon_momentum,
            self._accum_first,
            self.stochastic_rounding,
            self.stochastic_rounding_seed,
            rounding_counter,
        )


def set_accum_first(module: nn.Module, accum_first: bool) -> None:
    for child in module.modules():
        if isinstance(child, FusedMuonLinear):
            child.set_accum_first(accum_first)


def finalize_muon_step(
    module: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
) -> None:
    for child in module.modules():
        if isinstance(child, FusedMuonLinear):
            child.finalize_muon_step(lr, weight_decay)
