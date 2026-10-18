"""FP8 linear layer with fused residual addition."""

import torch
import triton
import triton.language as tl

import mxtest
from fp8_linear import FP8Linear, _fp8_linear_bwd


@triton.autotune(
    configs=mxtest.GEMM_CONFIGS,
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def fp8gemm_residual_fwd(
    x,
    weight,
    weight_scale,
    bias,
    residual,
    out,
    M,
    N,
    K,
    x_stride,
    weight_stride,
    scale_stride,
    residual_stride,
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

    if HAS_BIAS:
        col_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_block = tl.load(
            bias + col_offsets,
            mask=col_offsets < N,
            other=0.0,
        )
        accumulator += bias_block[None, :]

    residual_ptr = tl.make_block_ptr(
        residual,
        shape=(M, N),
        strides=(residual_stride, 1),
        offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    residual_block = tl.load(
        residual_ptr,
        boundary_check=(1, 0),
        padding_option="zero",
    )
    accumulator += residual_block.to(tl.float32)

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


@torch.library.custom_op("fp8gemm::linear_residual_fwd", mutates_args=())
def _fp8_linear_residual_fwd(
    x: torch.Tensor,
    master_weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    N = master_weight.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    bias_ptr = bias if bias is not None else weight_scale
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    fp8gemm_residual_fwd[grid](
        x,
        weight_fp8,
        weight_scale,
        bias_ptr,
        residual,
        out,
        M,
        N,
        K,
        x.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        residual.stride(0),
        out.stride(0),
        bias is not None,
    )
    return out


@_fp8_linear_residual_fwd.register_fake
def _fp8_linear_residual_fwd_fake(
    x: torch.Tensor,
    master_weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], master_weight.shape[0]),
        dtype=torch.bfloat16,
        device=x.device,
    )


def _fp8_linear_residual_setup_context(ctx, inputs, output):
    x, _, weight_fp8, weight_scale, bias, _ = inputs
    ctx.save_for_backward(x, weight_fp8, weight_scale)
    ctx.has_bias = bias is not None


def _fp8_linear_residual_backward(ctx, grad_output):
    x, weight_fp8, weight_scale = ctx.saved_tensors
    grad_x, grad_weight, grad_bias = _fp8_linear_bwd(
        grad_output,
        x,
        weight_fp8,
        weight_scale,
    )
    grad_residual = grad_output.contiguous()
    if grad_residual.dtype != torch.bfloat16:
        grad_residual = grad_residual.to(torch.bfloat16)
    return (
        grad_x,
        grad_weight,
        None,
        None,
        grad_bias if ctx.has_bias else None,
        grad_residual,
    )


torch.library.register_autograd(
    _fp8_linear_residual_fwd,
    _fp8_linear_residual_backward,
    setup_context=_fp8_linear_residual_setup_context,
)


class FP8LinearResidual(FP8Linear):
    """FP8Linear with fused ``linear(x) + bias + residual`` output."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input shape [M, {self.in_features}], got {tuple(x.shape)}"
            )
        expected_residual_shape = (x.shape[0], self.out_features)
        if tuple(residual.shape) != expected_residual_shape:
            raise ValueError(
                f"expected residual shape {expected_residual_shape}, "
                f"got {tuple(residual.shape)}"
            )
        if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise TypeError("input and residual must be BF16")
        if not x.is_cuda or not residual.is_cuda:
            raise RuntimeError("FP8LinearResidual requires CUDA")

        x = x.contiguous()
        residual = residual.contiguous()
        weight_fp8, weight_scale = self.quantized_weight
        return _fp8_linear_residual_fwd(
            x,
            self.master_weight,
            weight_fp8,
            weight_scale,
            self.bias,
            residual,
        )

    def extra_repr(self):
        return super().extra_repr() + ", residual=True"
