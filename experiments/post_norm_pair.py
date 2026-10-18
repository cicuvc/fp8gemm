"""Pair-level postnorm forward/backward with shared RMS normalization."""

import math

import torch
import triton
import triton.language as tl
from torch import nn

import mxtest
from experiments.flash_norm import (
    flash_norm_linear_fwd,
    launch_factor_chain_rule,
    launch_flash_norm_dx_gemm,
    launch_flash_norm_wgrad,
    launch_inverse_rms,
    launch_quantize_factored_weight,
)
from experiments.post_norm import fp8gemm_post_norm_residual_fwd


def _launch_first_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    inverse_rms: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    M, K = x.shape
    N = weight_fp8.shape[0]
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
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
        output,
        M,
        N,
        K,
        x.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        output.stride(0),
        bias is not None,
    )
    return output


def _launch_second_linear(
    z: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual_source: torch.Tensor,
    inverse_rms: torch.Tensor,
    elementwise_factor: torch.Tensor,
) -> torch.Tensor:
    M, K = z.shape
    N = weight_fp8.shape[0]
    output = torch.empty((M, N), dtype=torch.bfloat16, device=z.device)
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
        output,
        M,
        N,
        K,
        z.stride(0),
        weight_fp8.stride(0),
        weight_scale.stride(0),
        residual_source.stride(0),
        output.stride(0),
        bias is not None,
    )
    return output


@triton.jit
def pair_rms_correction_kernel(
    x,
    first_pre_correction_grad,
    second_linear_grad,
    elementwise_factor,
    inverse_rms,
    output,
    M,
    K,
    x_stride,
    first_grad_stride,
    second_grad_stride,
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
    first_grad_row = tl.load(
        first_pre_correction_grad + row * first_grad_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    second_grad_row = tl.load(
        second_linear_grad + row * second_grad_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    factor_row = tl.load(
        elementwise_factor + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inverse_rms_row = tl.load(inverse_rms + row)

    first_rms_grad = tl.sum(x_row * first_grad_row, axis=0)
    second_rms_grad = tl.sum(
        x_row * factor_row * second_grad_row,
        axis=0,
    )
    direct_second_grad = inverse_rms_row * factor_row * second_grad_row
    coefficient = (
        -(first_rms_grad + second_rms_grad)
        * inverse_rms_row
        * inverse_rms_row
        * inverse_rms_row
        / K
    )
    corrected = first_grad_row + direct_second_grad + coefficient * x_row
    tl.store(
        output + row * output_stride + offsets,
        corrected.to(tl.bfloat16),
        mask=mask,
    )


@torch.no_grad()
def _launch_pair_rms_correction(
    x: torch.Tensor,
    first_pre_correction_grad: torch.Tensor,
    second_linear_grad: torch.Tensor,
    elementwise_factor: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    output = torch.empty_like(x)
    block_k = triton.next_power_of_2(K)
    pair_rms_correction_kernel[(M,)](
        x,
        first_pre_correction_grad,
        second_linear_grad,
        elementwise_factor,
        inverse_rms,
        output,
        M,
        K,
        x.stride(0),
        first_pre_correction_grad.stride(0),
        second_linear_grad.stride(0),
        output.stride(0),
        BLOCK_K=block_k,
    )
    return output


@triton.jit
def second_factor_grad_kernel(
    x,
    second_linear_grad,
    inverse_rms,
    factor_grad,
    M,
    K,
    x_stride,
    second_grad_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for row_start in tl.range(0, M, BLOCK_M):
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        mask = (
            (row_offsets[:, None] < M)
            & (col_offsets[None, :] < K)
        )
        x_block = tl.load(
            x + row_offsets[:, None] * x_stride + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_block = tl.load(
            second_linear_grad
            + row_offsets[:, None] * second_grad_stride
            + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inverse_rms_block = tl.load(
            inverse_rms + row_offsets,
            mask=row_offsets < M,
            other=0.0,
        )
        accumulator += tl.sum(
            x_block * grad_block * inverse_rms_block[:, None],
            axis=0,
        )
    tl.store(
        factor_grad + col_offsets,
        accumulator,
        mask=col_offsets < K,
    )


@torch.no_grad()
def _launch_second_factor_grad(
    x: torch.Tensor,
    second_linear_grad: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    output = torch.zeros(K, dtype=torch.float32, device=x.device)
    block_m = 128
    block_k = 128
    grid = (triton.cdiv(K, block_k),)
    second_factor_grad_kernel[grid](
        x,
        second_linear_grad,
        inverse_rms,
        output,
        M,
        K,
        x.stride(0),
        second_linear_grad.stride(0),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )
    return output


class _PostNormPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        z,
        first_weight,
        second_weight,
        elementwise_factor,
        first_weight_fp8,
        first_weight_scale,
        second_weight_fp8,
        second_weight_scale,
        first_bias,
        second_bias,
        eps,
        b_output_bf16,
    ):
        inverse_rms = launch_inverse_rms(x, eps)
        first_output = _launch_first_linear(
            x,
            first_weight_fp8,
            first_weight_scale,
            inverse_rms,
            first_bias,
        )
        second_output = _launch_second_linear(
            z,
            second_weight_fp8,
            second_weight_scale,
            second_bias,
            x,
            inverse_rms,
            elementwise_factor,
        )
        ctx.save_for_backward(
            x,
            z,
            first_weight,
            second_weight,
            elementwise_factor,
            first_weight_fp8,
            first_weight_scale,
            second_weight_fp8,
            second_weight_scale,
            inverse_rms,
        )
        ctx.has_first_bias = first_bias is not None
        ctx.has_second_bias = second_bias is not None
        ctx.b_output_bf16 = b_output_bf16
        return first_output, second_output

    @staticmethod
    def backward(ctx, grad_first, grad_second):
        (
            x,
            z,
            first_weight,
            second_weight,
            elementwise_factor,
            first_weight_fp8,
            first_weight_scale,
            second_weight_fp8,
            second_weight_scale,
            inverse_rms,
        ) = ctx.saved_tensors
        if grad_first is None:
            grad_first = torch.zeros(
                (x.shape[0], first_weight.shape[0]),
                dtype=torch.bfloat16,
                device=x.device,
            )
        else:
            grad_first = grad_first.contiguous().to(torch.bfloat16)
        if grad_second is None:
            grad_second = torch.zeros(
                (x.shape[0], second_weight.shape[0]),
                dtype=torch.bfloat16,
                device=x.device,
            )
        else:
            grad_second = grad_second.contiguous().to(torch.bfloat16)

        first_pre_correction_grad = launch_flash_norm_dx_gemm(
            grad_first,
            first_weight_fp8,
            first_weight_scale,
            inverse_rms,
            ctx.b_output_bf16,
        )
        grad_x = _launch_pair_rms_correction(
            x,
            first_pre_correction_grad,
            grad_second,
            elementwise_factor,
            inverse_rms,
        )
        first_effective_weight_grad = launch_flash_norm_wgrad(
            grad_first,
            x,
            inverse_rms,
        )
        first_weight_grad, first_factor_grad = launch_factor_chain_rule(
            first_effective_weight_grad,
            first_weight,
            elementwise_factor,
        )
        second_factor_grad = _launch_second_factor_grad(
            x,
            grad_second,
            inverse_rms,
        )
        factor_grad = first_factor_grad + second_factor_grad

        grad_z = mxtest.launch_fp8gemm_backward(
            grad_second,
            second_weight_fp8,
            second_weight_scale,
        )
        second_weight_grad = mxtest.launch_fp8gemm_weight_grad(
            grad_second,
            z,
            out_dtype=torch.float32,
        )
        first_bias_grad = (
            grad_first.float().sum(dim=0)
            if ctx.has_first_bias
            else None
        )
        second_bias_grad = (
            grad_second.float().sum(dim=0)
            if ctx.has_second_bias
            else None
        )
        return (
            grad_x,
            grad_z,
            first_weight_grad,
            second_weight_grad,
            factor_grad,
            None,
            None,
            None,
            None,
            first_bias_grad,
            second_bias_grad,
            None,
            None,
        )


class PostNormPair(nn.Module):
    """Postnorm branch pair sharing one RMSNorm factor and inverse RMS."""

    def __init__(
        self,
        residual_features: int,
        first_features: int,
        second_input_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        b_output_bf16: bool = True,
        device=None,
    ):
        super().__init__()
        self.residual_features = residual_features
        self.first_features = first_features
        self.second_input_features = second_input_features
        self.eps = eps
        self.b_output_bf16 = b_output_bf16
        self.first_weight = nn.Parameter(
            torch.empty(
                (first_features, residual_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.second_weight = nn.Parameter(
            torch.empty(
                (residual_features, second_input_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.elementwise_factor = nn.Parameter(
            torch.ones(residual_features, dtype=torch.float32, device=device)
        )
        if bias:
            self.first_bias = nn.Parameter(
                torch.empty(first_features, dtype=torch.float32, device=device)
            )
            self.second_bias = nn.Parameter(
                torch.empty(
                    residual_features,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            self.register_parameter("first_bias", None)
            self.register_parameter("second_bias", None)
        self.register_buffer(
            "_first_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_first_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_second_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_second_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._synced_versions = None
        self.reset_parameters()
        self.refresh_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.first_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.second_weight, a=math.sqrt(5))
        nn.init.ones_(self.elementwise_factor)
        if self.first_bias is not None:
            bound = 1.0 / math.sqrt(self.residual_features)
            nn.init.uniform_(self.first_bias, -bound, bound)
        if self.second_bias is not None:
            bound = 1.0 / math.sqrt(self.second_input_features)
            nn.init.uniform_(self.second_bias, -bound, bound)

    def refresh_weights(self):
        with torch.no_grad():
            self._first_weight_fp8, self._first_weight_scale = (
                launch_quantize_factored_weight(
                    self.first_weight.detach(),
                    self.elementwise_factor.detach(),
                )
            )
            self._second_weight_fp8, self._second_weight_scale = (
                mxtest.launch_weight_quantize(self.second_weight.detach())
            )
        self._synced_versions = (
            self.first_weight._version,
            self.second_weight._version,
            self.elementwise_factor._version,
        )
        return self

    def _sync_if_needed(self):
        versions = (
            self.first_weight._version,
            self.second_weight._version,
            self.elementwise_factor._version,
        )
        if self._synced_versions != versions:
            self.refresh_weights()

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.residual_features:
            raise ValueError(
                f"expected x shape [M, {self.residual_features}], "
                f"got {tuple(x.shape)}"
            )
        if z.ndim != 2 or z.shape != (
            x.shape[0],
            self.second_input_features,
        ):
            raise ValueError(
                f"expected z shape {(x.shape[0], self.second_input_features)}, "
                f"got {tuple(z.shape)}"
            )
        if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
            raise TypeError("postnorm pair inputs must be BF16")
        self._sync_if_needed()
        return _PostNormPairFunction.apply(
            x.contiguous(),
            z.contiguous(),
            self.first_weight,
            self.second_weight,
            self.elementwise_factor,
            self._first_weight_fp8,
            self._first_weight_scale,
            self._second_weight_fp8,
            self._second_weight_scale,
            self.first_bias,
            self.second_bias,
            self.eps,
            self.b_output_bf16,
        )
