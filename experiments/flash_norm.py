"""FlashNorm linear experiment.

Computes ``Linear(elementwise_factor * RMSNorm(x))`` without materializing the
normalized activation:

1. Reduce each row to ``1 / rms(x)``.
2. Absorb ``elementwise_factor`` into the effective linear weight.
3. Apply ``1 / rms`` to the FP32 GEMM accumulator before writing the output.

The backward path derives the row RMS gradient from ``dot(x, B)``, where ``B``
is the pre-correction input gradient.
"""

import math

import torch
import triton
import triton.language as tl
from torch import nn

import mxtest


@triton.jit
def inverse_rms_kernel(
    x,
    inverse_rms,
    M,
    K,
    x_stride,
    eps,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    x_row = tl.load(
        x + row * x_stride + offsets,
        mask=offsets < K,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(x_row * x_row, axis=0) / K
    tl.store(inverse_rms + row, tl.rsqrt(mean_square + eps))


@triton.jit
def quantize_factored_weight_kernel(
    weight,
    factor,
    weight_fp8,
    weight_scale,
    N,
    K,
    weight_stride,
    factor_stride,
    out_stride,
    scale_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    pid = tl.program_id(0)
    row_block = pid // NUM_KBLOCKS
    col_block = pid % NUM_KBLOCKS

    weight_ptr = tl.make_block_ptr(
        weight,
        shape=(N, K),
        strides=(weight_stride, 1),
        offsets=(row_block * BLOCK_N, col_block * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    factor_ptr = tl.make_block_ptr(
        factor,
        shape=(K,),
        strides=(factor_stride,),
        offsets=(col_block * BLOCK_K,),
        block_shape=(BLOCK_K,),
        order=(0,),
    )
    output_ptr = tl.make_block_ptr(
        weight_fp8,
        shape=(N, K),
        strides=(out_stride, 1),
        offsets=(row_block * BLOCK_N, col_block * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    scale_ptr = tl.make_block_ptr(
        weight_scale,
        shape=(tl.cdiv(N, 32), tl.cdiv(K, 32)),
        strides=(scale_stride, 1),
        offsets=(row_block, col_block),
        block_shape=(1, 1),
        order=(1, 0),
    )

    weight_block = tl.load(
        weight_ptr,
        boundary_check=(1, 0),
        padding_option="zero",
    ).to(tl.float32)
    factor_block = tl.load(
        factor_ptr,
        boundary_check=(0,),
        padding_option="zero",
    ).to(tl.float32)
    effective_weight = (weight_block * factor_block[None, :]).to(tl.bfloat16)
    quantized, scale = mxtest.block_online_scale(effective_weight)
    tl.store(output_ptr, quantized, boundary_check=(1, 0))
    tl.store(scale_ptr, scale, boundary_check=(1, 0))


@triton.autotune(
    configs=mxtest.GEMM_CONFIGS,
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def flash_norm_linear_fwd(
    x,
    weight,
    weight_scale,
    inverse_rms,
    bias,
    out,
    M,
    N,
    K,
    x_stride,
    weight_stride,
    scale_stride,
    out_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    PRESET_GROUP_SIZE: tl.constexpr = 4
    MX_BLOCK: tl.constexpr = 32
    BLOCK_SCALES: tl.constexpr = BLOCK_K // MX_BLOCK
    SCALE_ROWS: tl.constexpr = BLOCK_N // MX_BLOCK

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_MBLOCKS = tl.cdiv(M, BLOCK_M)
    NUM_SCALE_ROWS = tl.cdiv(N, MX_BLOCK)
    NUM_SCALE_BLOCKS = tl.cdiv(K, MX_BLOCK)
    GROUP_SIZE = tl.minimum(NUM_MBLOCKS, PRESET_GROUP_SIZE)

    pid = tl.program_id(0)
    NUM_PID_IN_GROUP = NUM_NBLOCKS * GROUP_SIZE
    group_id = pid // NUM_PID_IN_GROUP
    first_row_idx = group_id * GROUP_SIZE
    group_rows = tl.minimum(NUM_MBLOCKS - first_row_idx, GROUP_SIZE)
    pid_in_group = pid % NUM_PID_IN_GROUP
    row_idx = first_row_idx + (pid_in_group % group_rows)
    col_idx = pid_in_group // group_rows

    x_ptr = tl.make_block_ptr(
        x,
        shape=(M, K),
        strides=(x_stride, 1),
        offsets=(row_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    weight_ptr = tl.make_block_ptr(
        weight,
        shape=(N, K),
        strides=(weight_stride, 1),
        offsets=(col_idx * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    scale_ptr = tl.make_block_ptr(
        weight_scale,
        shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS),
        strides=(scale_stride, 1),
        offsets=(col_idx * SCALE_ROWS, 0),
        block_shape=(SCALE_ROWS, BLOCK_SCALES),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K, BLOCK_K):
        x_block = tl.load(
            x_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        x_quantized, x_scale = mxtest.online_scale(x_block)

        weight_block = tl.load(
            weight_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        weight_scale_block = tl.load(
            scale_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        weight_scale_block = tl.reshape(
            tl.broadcast_to(
                weight_scale_block[:, None, :],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )
        accumulator = tl.dot_scaled(
            x_quantized,
            x_scale,
            "e4m3",
            tl.trans(weight_block),
            weight_scale_block,
            "e4m3",
            accumulator,
        )
        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        weight_ptr = tl.advance(weight_ptr, (0, BLOCK_K))
        scale_ptr = tl.advance(scale_ptr, (0, BLOCK_SCALES))

    row_offsets = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_inverse_rms = tl.load(
        inverse_rms + row_offsets,
        mask=row_offsets < M,
        other=0.0,
    )
    accumulator *= row_inverse_rms[:, None]

    if HAS_BIAS:
        col_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_block = tl.load(
            bias + col_offsets,
            mask=col_offsets < N,
            other=0.0,
        )
        accumulator += bias_block[None, :]

    output_ptr = tl.make_block_ptr(
        out,
        shape=(M, N),
        strides=(out_stride, 1),
        offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(
        output_ptr,
        accumulator.to(out.dtype.element_ty),
        boundary_check=(1, 0),
    )


@triton.autotune(
    configs=mxtest.GEMM_CONFIGS,
    key=["M", "N", "K", "B_OUTPUT_BF16"],
    cache_results=True,
)
@triton.jit
def flash_norm_dx_gemm(
    grad_output,
    weight,
    weight_scale,
    inverse_rms,
    output,
    M,
    N,
    K,
    grad_stride,
    weight_stride,
    scale_stride,
    output_stride,
    B_OUTPUT_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    MX_BLOCK: tl.constexpr = 32
    SCALE_ROWS: tl.constexpr = BLOCK_N // MX_BLOCK
    SCALE_COLS: tl.constexpr = BLOCK_K // MX_BLOCK
    DY_SCALES: tl.constexpr = BLOCK_N // MX_BLOCK

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    NUM_SCALE_ROWS = tl.cdiv(N, MX_BLOCK)
    NUM_SCALE_COLS = tl.cdiv(K, MX_BLOCK)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    grad_ptr = tl.make_block_ptr(
        grad_output,
        shape=(M, N),
        strides=(grad_stride, 1),
        offsets=(row_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    weight_ptr = tl.make_block_ptr(
        weight,
        shape=(N, K),
        strides=(weight_stride, 1),
        offsets=(0, col_idx * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    scale_ptr = tl.make_block_ptr(
        weight_scale,
        shape=(NUM_SCALE_ROWS, NUM_SCALE_COLS),
        strides=(scale_stride, 1),
        offsets=(0, col_idx * SCALE_COLS),
        block_shape=(SCALE_ROWS, SCALE_COLS),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for _ in tl.range(0, N, BLOCK_N):
        grad_block = tl.load(
            grad_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        grad_quantized, grad_scale = mxtest.online_scale(grad_block)

        weight_block = tl.load(
            weight_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        weight_scale_block = tl.load(
            scale_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        weight_scale_block = tl.reshape(
            tl.broadcast_to(
                tl.trans(weight_scale_block)[:, None, :],
                (SCALE_COLS, MX_BLOCK, DY_SCALES),
            ),
            (BLOCK_K, DY_SCALES),
        )
        accumulator = tl.dot_scaled(
            grad_quantized,
            grad_scale,
            "e4m3",
            weight_block,
            weight_scale_block,
            "e4m3",
            accumulator,
        )
        grad_ptr = tl.advance(grad_ptr, (0, BLOCK_N))
        weight_ptr = tl.advance(weight_ptr, (BLOCK_N, 0))
        scale_ptr = tl.advance(scale_ptr, (SCALE_ROWS, 0))

    row_offsets = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_inverse_rms = tl.load(
        inverse_rms + row_offsets,
        mask=row_offsets < M,
        other=0.0,
    )
    accumulator *= row_inverse_rms[:, None]

    output_ptr = tl.make_block_ptr(
        output,
        shape=(M, K),
        strides=(output_stride, 1),
        offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    if B_OUTPUT_BF16:
        output_value = accumulator.to(tl.bfloat16)
    else:
        output_value = accumulator
    tl.store(
        output_ptr,
        output_value,
        boundary_check=(1, 0),
    )


@torch.no_grad()
def launch_flash_norm_dx_gemm(
    grad_output: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    inverse_rms: torch.Tensor,
    b_output_bf16: bool,
) -> torch.Tensor:
    M, N = grad_output.shape
    _, K = weight_fp8.shape
    output = torch.empty(
        (M, K),
        dtype=torch.bfloat16 if b_output_bf16 else torch.float32,
        device=grad_output.device,
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    flash_norm_dx_gemm[grid](
        grad_output,
        weight_fp8,
        weight_scale,
        inverse_rms,
        output,
        M,
        N,
        K,
        grad_output.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        output.stride(0),
        b_output_bf16,
    )
    return output


@triton.jit
def rms_correction_kernel(
    x,
    pre_correction_grad,
    inverse_rms,
    output,
    M,
    K,
    x_stride,
    grad_stride,
    output_stride,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    x_row = tl.load(
        x + row * x_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    pre_grad_row = tl.load(
        pre_correction_grad + row * grad_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inverse_rms_row = tl.load(inverse_rms + row)
    rms_gradient = tl.sum(x_row * pre_grad_row, axis=0)
    coefficient = -rms_gradient * inverse_rms_row * inverse_rms_row * inverse_rms_row / K
    corrected = pre_grad_row + coefficient * x_row
    tl.store(
        output + row * output_stride + offsets,
        corrected.to(tl.bfloat16),
        mask=mask,
    )


@torch.no_grad()
def launch_rms_correction(
    x: torch.Tensor,
    pre_correction_grad: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    output = torch.empty_like(x)
    block_k = triton.next_power_of_2(K)
    rms_correction_kernel[(M,)](
        x,
        pre_correction_grad,
        inverse_rms,
        output,
        M,
        K,
        x.stride(0),
        pre_correction_grad.stride(0),
        output.stride(0),
        BLOCK_K=block_k,
    )
    return output


@triton.autotune(
    configs=mxtest.WGRAD_CONFIGS,
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def flash_norm_wgrad(
    grad_output,
    x,
    inverse_rms,
    weight_grad,
    M,
    N,
    K,
    grad_stride,
    x_stride,
    output_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    grad_ptr = tl.make_block_ptr(
        grad_output,
        shape=(M, N),
        strides=(grad_stride, 1),
        offsets=(0, row_idx * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    x_ptr = tl.make_block_ptr(
        x,
        shape=(M, K),
        strides=(x_stride, 1),
        offsets=(0, col_idx * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for block_start in tl.range(0, M, BLOCK_M):
        grad_block = tl.load(
            grad_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        # Scale by 1/rms before the FP8 online quantization.
        row_offsets = block_start + tl.arange(0, BLOCK_M)
        row_scale = tl.load(
            inverse_rms + row_offsets,
            mask=row_offsets < M,
            other=0.0,
        )
        grad_block = (
            grad_block.to(tl.float32) * row_scale[:, None]
        ).to(tl.bfloat16)
        grad_quantized, grad_scale = mxtest.online_scale(tl.trans(grad_block))

        x_block = tl.load(
            x_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        x_quantized, x_scale = mxtest.online_scale(tl.trans(x_block))
        x_quantized = tl.trans(x_quantized)

        accumulator = tl.dot_scaled(
            grad_quantized,
            grad_scale,
            "e4m3",
            x_quantized,
            x_scale,
            "e4m3",
            accumulator,
        )
        grad_ptr = tl.advance(grad_ptr, (BLOCK_M, 0))
        x_ptr = tl.advance(x_ptr, (BLOCK_M, 0))

    output_ptr = tl.make_block_ptr(
        weight_grad,
        shape=(N, K),
        strides=(output_stride, 1),
        offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    tl.store(
        output_ptr,
        accumulator,
        boundary_check=(1, 0),
    )


@torch.no_grad()
def launch_flash_norm_wgrad(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> torch.Tensor:
    M, N = grad_output.shape
    _, K = x.shape
    weight_grad = torch.empty(
        (N, K),
        dtype=torch.float32,
        device=grad_output.device,
    )
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    flash_norm_wgrad[grid](
        grad_output,
        x,
        inverse_rms,
        weight_grad,
        M,
        N,
        K,
        grad_output.stride(0),
        x.stride(0),
        weight_grad.stride(0),
    )
    return weight_grad


@triton.jit
def factor_chain_rule_kernel(
    effective_weight_grad,
    weight,
    factor,
    weight_grad,
    factor_grad,
    N,
    K,
    effective_grad_stride,
    weight_stride,
    weight_grad_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS
    row_offsets = row_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    col_offsets = col_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < K)

    effective_grad = tl.load(
        effective_weight_grad
        + row_offsets[:, None] * effective_grad_stride
        + col_offsets[None, :],
        mask=mask,
        other=0.0,
    )
    weight_block = tl.load(
        weight + row_offsets[:, None] * weight_stride + col_offsets[None, :],
        mask=mask,
        other=0.0,
    )
    factor_block = tl.load(
        factor + col_offsets,
        mask=col_offsets < K,
        other=0.0,
    )
    tl.store(
        weight_grad
        + row_offsets[:, None] * weight_grad_stride
        + col_offsets[None, :],
        effective_grad * factor_block[None, :],
        mask=mask,
    )
    partial_factor_grad = tl.sum(effective_grad * weight_block, axis=0)
    tl.atomic_add(
        factor_grad + col_offsets,
        partial_factor_grad,
        mask=col_offsets < K,
    )


@torch.no_grad()
def launch_factor_chain_rule(
    effective_weight_grad: torch.Tensor,
    weight: torch.Tensor,
    factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, K = weight.shape
    weight_grad = torch.empty_like(weight)
    factor_grad = torch.zeros_like(factor)
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(N, block_n) * triton.cdiv(K, block_k),)
    factor_chain_rule_kernel[grid](
        effective_weight_grad,
        weight,
        factor,
        weight_grad,
        factor_grad,
        N,
        K,
        effective_weight_grad.stride(0),
        weight.stride(0),
        weight_grad.stride(0),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return weight_grad, factor_grad


class _FlashNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        master_weight,
        elementwise_factor,
        weight_fp8,
        weight_scale,
        bias,
        eps,
        b_output_bf16,
    ):
        M, K = x.shape
        N = master_weight.shape[0]
        inverse_rms = launch_inverse_rms(x, eps)
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        bias_ptr = bias if bias is not None else weight_scale
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        flash_norm_linear_fwd[grid](
            x,
            weight_fp8,
            weight_scale,
            inverse_rms,
            bias_ptr,
            out,
            M,
            N,
            K,
            x.stride(0),
            weight_fp8.stride(0),
            weight_scale.stride(0),
            out.stride(0),
            bias is not None,
        )
        ctx.save_for_backward(
            x,
            master_weight,
            elementwise_factor,
            weight_fp8,
            weight_scale,
            inverse_rms,
        )
        ctx.has_bias = bias is not None
        ctx.b_output_bf16 = b_output_bf16
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            master_weight,
            elementwise_factor,
            weight_fp8,
            weight_scale,
            inverse_rms,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        pre_correction_grad = launch_flash_norm_dx_gemm(
            grad_output,
            weight_fp8,
            weight_scale,
            inverse_rms,
            ctx.b_output_bf16,
        )
        grad_x = launch_rms_correction(
            x,
            pre_correction_grad,
            inverse_rms,
        )
        effective_weight_grad = launch_flash_norm_wgrad(
            grad_output,
            x,
            inverse_rms,
        )
        grad_weight, grad_factor = launch_factor_chain_rule(
            effective_weight_grad,
            master_weight,
            elementwise_factor,
        )
        grad_bias = (
            grad_output.float().sum(dim=0)
            if ctx.has_bias
            else None
        )
        return (
            grad_x,
            grad_weight,
            grad_factor,
            None,
            None,
            grad_bias,
            None,
            None,
        )
def launch_inverse_rms(
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    M, K = x.shape
    inverse_rms = torch.empty(M, dtype=torch.float32, device=x.device)
    block_k = triton.next_power_of_2(K)
    inverse_rms_kernel[(M,)](
        x,
        inverse_rms,
        M,
        K,
        x.stride(0),
        eps,
        BLOCK_K=block_k,
    )
    return inverse_rms


def launch_quantize_factored_weight(
    weight: torch.Tensor,
    factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, K = weight.shape
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(
        (triton.cdiv(N, 32), triton.cdiv(K, 32)),
        dtype=torch.uint8,
        device=weight.device,
    )
    grid = (triton.cdiv(N, 32) * triton.cdiv(K, 32),)
    quantize_factored_weight_kernel[grid](
        weight,
        factor,
        weight_fp8,
        weight_scale,
        N,
        K,
        weight.stride(0),
        factor.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return weight_fp8, weight_scale


class FlashNormLinear(nn.Module):
    """Fused FP8 RMSNorm + Linear layer with a custom backward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        b_output_bf16: bool = True,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.b_output_bf16 = b_output_bf16
        self.master_weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.elementwise_factor = nn.Parameter(
            torch.ones(in_features, dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)
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
        self._synced_versions = None
        self.reset_parameters()
        self.refresh_effective_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        nn.init.ones_(self.elementwise_factor)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def refresh_effective_weight(self):
        with torch.no_grad():
            self._weight_fp8, self._weight_scale = (
                launch_quantize_factored_weight(
                    self.master_weight.detach(),
                    self.elementwise_factor.detach(),
                )
            )
        self._synced_versions = (
            self.master_weight._version,
            self.elementwise_factor._version,
        )
        return self

    def _sync_if_needed(self):
        versions = (
            self.master_weight._version,
            self.elementwise_factor._version,
        )
        if self._synced_versions != versions:
            self.refresh_effective_weight()

    @property
    def quantized_effective_weight(self):
        self._sync_if_needed()
        return self._weight_fp8, self._weight_scale

    def mark_dirty(self):
        self._synced_versions = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input shape [M, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError("FlashNormLinear requires CUDA BF16 input")
        x = x.contiguous()
        weight_fp8, weight_scale = self.quantized_effective_weight
        return _FlashNormLinearFunction.apply(
            x,
            self.master_weight,
            self.elementwise_factor,
            weight_fp8,
            weight_scale,
            self.bias,
            self.eps,
            self.b_output_bf16,
        )

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, eps={self.eps}, "
            f"b_output_bf16={self.b_output_bf16}"
        )
