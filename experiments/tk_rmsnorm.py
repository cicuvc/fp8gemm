"""RMSNorm prenorm that emits both MXFP8 activation layouts.

For ``y = x * rsqrt(mean(x^2) + eps) * factor``, this plan produces:

* row-wise ``y[M,K]`` E4M3 and packed 1x32 scales for forward GEMM;
* column-wise ``y[MP,K]`` E4M3 and packed 32x1 scales for wgrad;
* FP32 ``inverse_rms[M]`` for the eventual RMSNorm backward.

Normalized BF16 activations are not materialized in global memory.
"""

import torch
import triton
import triton.language as tl

import mxtest


@triton.jit
def _inverse_rms(
    x,
    inverse_rms,
    M,
    K,
    x_stride,
    eps,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    values = tl.load(
        x + row * x_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / K
    tl.store(inverse_rms + row, tl.rsqrt(mean_square + eps))


@triton.jit
def _rmsnorm_dual_quant(
    x,
    factor,
    inverse_rms,
    row_fp8,
    row_scale,
    column_fp8,
    column_scale,
    M,
    K,
    MP,
    MP16,
    SP,
    x_stride,
    factor_stride,
    row_stride,
    column_stride,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    values = tl.load(
        x + rows[:, None] * x_stride + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    inv = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    normalized = (
        values.to(tl.float32)
        * inv[:, None]
        * scale_factor[None, :]
    ).to(tl.bfloat16)

    row_q, row_s = mxtest.online_scale(normalized)
    tl.store(
        row_fp8 + rows[:, None] * row_stride + cols[None, :],
        row_q,
        mask=row_mask[:, None] & col_mask[None, :],
    )
    # BLOCK_K is 128, exactly four K32 scale blocks per packed word.
    row_word = tl.sum(
        row_s.to(tl.uint32)
        << (8 * tl.arange(0, 4))[None, :],
        axis=1,
    )
    tl.store(
        row_scale + col_block * MP16 + rows,
        row_word,
        mask=rows < MP16,
    )

    column_q_t, column_s = mxtest.online_scale(tl.trans(normalized))
    tl.store(
        column_fp8 + rows[:, None] * column_stride + cols[None, :],
        tl.trans(column_q_t),
        mask=col_mask[None, :],
    )
    scale_shift = 8 * (row_block % 4)
    column_values = tl.reshape(column_s.to(tl.uint32), (BLOCK_K,))
    tl.atomic_or(
        column_scale + (row_block // 4) * SP + cols,
        column_values << scale_shift,
        mask=cols < SP,
    )


class RMSNormDualQuantPlan:
    """Reusable fixed-shape prenorm and two-direction quantization plan."""

    def __init__(self, x, factor, *, eps=1e-6):
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("Expected CUDA BF16 X[M,K]")
        if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
            raise ValueError("Expected elementwise factor[K]")
        if factor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Factor must be FP32 or BF16")
        if not factor.is_cuda or factor.device != x.device:
            raise ValueError("Expected factor on the input device")
        m, k = x.shape
        if k % 32:
            raise ValueError("K must be a multiple of 32")
        self.x = x
        self.factor = factor
        self.eps = float(eps)
        self.m, self.k = m, k
        self.mp = (m + 127) // 128 * 128
        self.mp16 = (m + 15) // 16 * 16
        self.sp = (k + 15) // 16 * 16
        self.inverse_rms = torch.empty(m, device=x.device, dtype=torch.float32)
        self.row_fp8 = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.zeros(
            ((k + 127) // 128, self.mp16),
            device=x.device,
            dtype=torch.int32,
        )
        self.column_fp8 = torch.empty(
            (self.mp, k), device=x.device, dtype=torch.float8_e4m3fn
        )
        self.column_scale = torch.zeros(
            (self.mp // 128, self.sp),
            device=x.device,
            dtype=torch.int32,
        )

    def __call__(self):
        self.column_scale.zero_()
        _inverse_rms[(self.m,)](
            self.x,
            self.inverse_rms,
            self.m,
            self.k,
            self.x.stride(0),
            self.eps,
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        _rmsnorm_dual_quant[(self.mp // 32, triton.cdiv(self.k, 128))](
            self.x,
            self.factor,
            self.inverse_rms,
            self.row_fp8,
            self.row_scale,
            self.column_fp8,
            self.column_scale,
            self.m,
            self.k,
            self.mp,
            self.mp16,
            self.sp,
            self.x.stride(0),
            self.factor.stride(0),
            self.k,
            self.k,
            BLOCK_K=128,
            num_warps=4,
        )
        return (
            self.row_fp8,
            self.row_scale,
            self.column_fp8,
            self.column_scale,
            self.inverse_rms,
        )


@triton.jit
def _rmsnorm_dot(
    dy,
    x,
    factor,
    dot,
    M,
    K,
    dy_stride,
    x_stride,
    factor_stride,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    dy_values = tl.load(
        dy + row * dy_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + row * x_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(dot + row, tl.sum(dy_values * x_values * scale_factor, axis=0))


@triton.jit
def _rmsnorm_backward(
    dy,
    x,
    factor,
    inverse_rms,
    dot,
    dx,
    factor_partial,
    M,
    K,
    dy_stride,
    x_stride,
    dx_stride,
    factor_stride,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    dy_values = tl.load(
        dy + rows[:, None] * dy_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + rows[:, None] * x_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    inverse = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    dy_scaled = dy_values * scale_factor[None, :]
    correction = (inverse * inverse / K) * tl.load(dot + rows, mask=row_mask, other=0.0)
    grad_x = inverse[:, None] * (
        dy_scaled - correction[:, None] * x_values
    )
    tl.store(
        dx + rows[:, None] * dx_stride + cols[None, :],
        grad_x.to(tl.bfloat16),
        mask=mask,
    )
    partial = tl.sum(dy_values * x_values * inverse[:, None], axis=0)
    tl.store(
        factor_partial + row_block * K + cols,
        partial,
        mask=col_mask,
    )


@triton.jit
def _reduce_factor_grad(
    factor_partial,
    factor_grad,
    row_blocks,
    K,
    BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for row_block in range(0, row_blocks):
        accumulator += tl.load(
            factor_partial + row_block * K + cols,
            mask=mask,
            other=0.0,
        )
    tl.store(factor_grad + cols, accumulator, mask=mask)


class RMSNormBackwardPlan:
    """Reusable backward from BF16 dY to BF16 dX and FP32 dFactor."""

    def __init__(self, dy, x, factor, inverse_rms):
        if (
            dy.shape != x.shape
            or dy.ndim != 2
            or dy.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
        ):
            raise ValueError("Expected BF16 dY and X with equal [M,K] shapes")
        if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
            raise ValueError("Expected factor[K]")
        if inverse_rms.shape != (x.shape[0],) or inverse_rms.dtype != torch.float32:
            raise ValueError("Expected FP32 inverse_rms[M]")
        if not x.is_cuda or any(
            tensor.device != x.device
            for tensor in [dy, factor, inverse_rms]
        ):
            raise ValueError("Expected all tensors on one CUDA device")
        self.dy, self.x, self.factor = dy, x, factor
        self.inverse_rms = inverse_rms
        self.m, self.k = x.shape
        self.row_blocks = (self.m + 31) // 32
        self.dx = torch.empty_like(x)
        self.dot = torch.empty(self.m, device=x.device, dtype=torch.float32)
        self.factor_grad = torch.empty(self.k, device=x.device, dtype=torch.float32)
        self.factor_partial = torch.empty(
            (self.row_blocks, self.k), device=x.device, dtype=torch.float32
        )

    def __call__(self):
        _rmsnorm_dot[(self.m,)](
            self.dy,
            self.x,
            self.factor,
            self.dot,
            self.m,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.factor.stride(0),
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        _rmsnorm_backward[(self.row_blocks, triton.cdiv(self.k, 128))](
            self.dy,
            self.x,
            self.factor,
            self.inverse_rms,
            self.dot,
            self.dx,
            self.factor_partial,
            self.m,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.dx.stride(0),
            self.factor.stride(0),
            BLOCK_K=128,
            num_warps=4,
        )
        block = 256
        _reduce_factor_grad[(triton.cdiv(self.k, block),)](
            self.factor_partial,
            self.factor_grad,
            self.row_blocks,
            self.k,
            BLOCK=block,
            num_warps=4,
        )
        return self.dx, self.factor_grad
