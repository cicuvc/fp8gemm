"""Forward-only postnorm residual linear experiment.

Computes::

    linear(z) + last_unnormalized_out * elementwise_factor / rms

where ``1 / rms`` is supplied by the corresponding FlashNorm operation.
"""

import torch
import triton
import triton.language as tl

import mxtest
from fp8_linear import FP8Linear


@triton.autotune(
    configs=mxtest.GEMM_CONFIGS,
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def fp8gemm_post_norm_residual_fwd(
    z,
    weight,
    weight_scale,
    bias,
    residual_source,
    inverse_rms,
    elementwise_factor,
    out,
    M,
    N,
    K,
    z_stride,
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

    z_ptr = tl.make_block_ptr(
        z,
        shape=(M, K),
        strides=(z_stride, 1),
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
        z_block = tl.load(
            z_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        z_quantized, z_scale = mxtest.online_scale(z_block)
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
            z_quantized,
            z_scale,
            "e4m3",
            tl.trans(weight_block),
            weight_scale_block,
            "e4m3",
            accumulator,
        )
        z_ptr = tl.advance(z_ptr, (0, BLOCK_K))
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

    row_offsets = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    residual_ptr = tl.make_block_ptr(
        residual_source,
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
    ).to(tl.float32)
    row_inverse_rms = tl.load(
        inverse_rms + row_offsets,
        mask=row_offsets < M,
        other=0.0,
    )
    factor_block = tl.load(
        elementwise_factor + col_offsets,
        mask=col_offsets < N,
        other=0.0,
    ).to(tl.float32)
    accumulator += (
        residual_block
        * row_inverse_rms[:, None]
        * factor_block[None, :]
    )

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


@torch.library.custom_op("fp8gemm::post_norm_residual_fwd", mutates_args=())
def _post_norm_residual_fwd(
    z: torch.Tensor,
    master_weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual_source: torch.Tensor,
    inverse_rms: torch.Tensor,
    elementwise_factor: torch.Tensor,
) -> torch.Tensor:
    M, K = z.shape
    N = master_weight.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=z.device)
    bias_ptr = bias if bias is not None else weight_scale
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    fp8gemm_post_norm_residual_fwd[grid](
        z,
        weight_fp8,
        weight_scale,
        bias_ptr,
        residual_source,
        inverse_rms,
        elementwise_factor,
        out,
        M,
        N,
        K,
        z.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        residual_source.stride(0),
        out.stride(0),
        bias is not None,
    )
    return out


@_post_norm_residual_fwd.register_fake
def _post_norm_residual_fwd_fake(
    z: torch.Tensor,
    master_weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual_source: torch.Tensor,
    inverse_rms: torch.Tensor,
    elementwise_factor: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (z.shape[0], master_weight.shape[0]),
        dtype=torch.bfloat16,
        device=z.device,
    )


class PostNormResidualLinear(FP8Linear):
    """Forward-only ``linear(z) + x * factor / rms`` layer."""

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        residual_source: torch.Tensor,
        inverse_rms: torch.Tensor,
        elementwise_factor: torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.in_features:
            raise ValueError(
                f"expected z shape [M, {self.in_features}], got {tuple(z.shape)}"
            )
        M = z.shape[0]
        N = self.out_features
        if tuple(residual_source.shape) != (M, N):
            raise ValueError(
                f"expected residual_source shape {(M, N)}, "
                f"got {tuple(residual_source.shape)}"
            )
        if tuple(inverse_rms.shape) != (M,):
            raise ValueError(
                f"expected inverse_rms shape {(M,)}, "
                f"got {tuple(inverse_rms.shape)}"
            )
        if tuple(elementwise_factor.shape) != (N,):
            raise ValueError(
                f"expected elementwise_factor shape {(N,)}, "
                f"got {tuple(elementwise_factor.shape)}"
            )
        if z.dtype != torch.bfloat16 or residual_source.dtype != torch.bfloat16:
            raise TypeError("z and residual_source must be BF16")
        if inverse_rms.dtype != torch.float32:
            raise TypeError("inverse_rms must be FP32")

        z = z.contiguous()
        residual_source = residual_source.contiguous()
        inverse_rms = inverse_rms.contiguous()
        elementwise_factor = elementwise_factor.contiguous()
        weight_fp8, weight_scale = self.quantized_weight
        return _post_norm_residual_fwd(
            z,
            self.master_weight,
            weight_fp8,
            weight_scale,
            self.bias,
            residual_source,
            inverse_rms,
            elementwise_factor,
        )

    def extra_repr(self):
        return super().extra_repr() + ", post_norm_residual=True"
