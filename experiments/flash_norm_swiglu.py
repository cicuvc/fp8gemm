"""Prenorm RMSNorm + fused FP8 SwiGLU forward experiment."""

import math

import torch
import triton
import triton.language as tl
from torch import nn

import mxtest
from experiments.flash_norm import (
    launch_inverse_rms,
    launch_quantize_factored_weight,
)
from fp8_linear import FP8Linear


@triton.jit
def rmsnorm_bf16_kernel(
    x,
    factor,
    output,
    M,
    K,
    x_stride,
    output_stride,
    eps,
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
    factor_row = tl.load(
        factor + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x_row * x_row, axis=0) / K + eps)
    normalized = x_row * inverse_rms * factor_row
    tl.store(
        output + row * output_stride + offsets,
        normalized.to(tl.bfloat16),
        mask=mask,
    )


def launch_rmsnorm(
    x: torch.Tensor,
    factor: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    M, K = x.shape
    output = torch.empty_like(x)
    rmsnorm_bf16_kernel[(M,)](
        x,
        factor,
        output,
        M,
        K,
        x.stride(0),
        output.stride(0),
        eps,
        BLOCK_K=triton.next_power_of_2(K),
    )
    return output


@triton.autotune(
    configs=mxtest.DUAL_GEMM_CONFIGS,
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def fp8_swiglu_prenorm_fwd(
    x,
    gate_weight,
    gate_scale,
    up_weight,
    up_scale,
    inverse_rms,
    out,
    M,
    N,
    K,
    x_stride,
    gate_weight_stride,
    gate_scale_stride,
    up_weight_stride,
    up_scale_stride,
    out_stride,
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
    gate_ptr = tl.make_block_ptr(
        gate_weight,
        shape=(N, K),
        strides=(gate_weight_stride, 1),
        offsets=(col_idx * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    gate_scale_ptr = tl.make_block_ptr(
        gate_scale,
        shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS),
        strides=(gate_scale_stride, 1),
        offsets=(col_idx * SCALE_ROWS, 0),
        block_shape=(SCALE_ROWS, BLOCK_SCALES),
        order=(1, 0),
    )
    up_ptr = tl.make_block_ptr(
        up_weight,
        shape=(N, K),
        strides=(up_weight_stride, 1),
        offsets=(col_idx * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    up_scale_ptr = tl.make_block_ptr(
        up_scale,
        shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS),
        strides=(up_scale_stride, 1),
        offsets=(col_idx * SCALE_ROWS, 0),
        block_shape=(SCALE_ROWS, BLOCK_SCALES),
        order=(1, 0),
    )

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K, BLOCK_K):
        x_block = tl.load(
            x_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        x_quantized, x_scale = mxtest.online_scale(x_block)

        gate_block = tl.load(
            gate_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        gate_scale_block = tl.load(
            gate_scale_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        gate_scale_block = tl.reshape(
            tl.broadcast_to(
                gate_scale_block[:, None, :],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )
        gate_acc = tl.dot_scaled(
            x_quantized,
            x_scale,
            "e4m3",
            tl.trans(gate_block),
            gate_scale_block,
            "e4m3",
            gate_acc,
        )

        up_block = tl.load(
            up_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        up_scale_block = tl.load(
            up_scale_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        up_scale_block = tl.reshape(
            tl.broadcast_to(
                up_scale_block[:, None, :],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )
        up_acc = tl.dot_scaled(
            x_quantized,
            x_scale,
            "e4m3",
            tl.trans(up_block),
            up_scale_block,
            "e4m3",
            up_acc,
        )
        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        gate_ptr = tl.advance(gate_ptr, (0, BLOCK_K))
        gate_scale_ptr = tl.advance(gate_scale_ptr, (0, BLOCK_SCALES))
        up_ptr = tl.advance(up_ptr, (0, BLOCK_K))
        up_scale_ptr = tl.advance(up_scale_ptr, (0, BLOCK_SCALES))

    row_offsets = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_inverse_rms = tl.load(
        inverse_rms + row_offsets,
        mask=row_offsets < M,
        other=0.0,
    )
    gate = gate_acc * row_inverse_rms[:, None]
    up = up_acc * row_inverse_rms[:, None]
    half_gate = gate * 0.5
    swiglu = half_gate * (mxtest.fast_tanh(half_gate) + 1.0) * up

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
        swiglu.to(out.dtype.element_ty),
        boundary_check=(1, 0),
    )


def launch_prenorm_swiglu_gateup(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    inverse_rms: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    N = gate_weight.shape[0]
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    fp8_swiglu_prenorm_fwd[grid](
        x,
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        inverse_rms,
        output,
        M,
        N,
        K,
        x.stride(0),
        gate_weight.stride(0),
        gate_scale.stride(0),
        up_weight.stride(0),
        up_scale.stride(0),
        output.stride(0),
    )
    return output


class FlashNormSwiGLU(nn.Module):
    """Forward-only prenorm RMSNorm + FP8 SwiGLU."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        eps: float = 1e-6,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.eps = eps
        self.gate_weight = nn.Parameter(
            torch.empty(
                (hidden_features, in_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.up_weight = nn.Parameter(
            torch.empty(
                (hidden_features, in_features),
                dtype=torch.float32,
                device=device,
            )
        )
        self.elementwise_factor = nn.Parameter(
            torch.ones(in_features, dtype=torch.float32, device=device)
        )
        self.down_proj = FP8Linear(
            hidden_features,
            out_features,
            bias=False,
            device=device,
        )
        self.register_buffer(
            "_gate_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_gate_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._synced_versions = None
        self.reset_parameters()
        self.refresh_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        nn.init.ones_(self.elementwise_factor)

    def refresh_weights(self):
        with torch.no_grad():
            self._gate_weight_fp8, self._gate_weight_scale = (
                launch_quantize_factored_weight(
                    self.gate_weight.detach(),
                    self.elementwise_factor.detach(),
                )
            )
            self._up_weight_fp8, self._up_weight_scale = (
                launch_quantize_factored_weight(
                    self.up_weight.detach(),
                    self.elementwise_factor.detach(),
                )
            )
        self.down_proj.refresh_fp8_weight()
        self._synced_versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.elementwise_factor._version,
            self.down_proj.master_weight._version,
        )
        return self

    def _sync_if_needed(self):
        versions = (
            self.gate_weight._version,
            self.up_weight._version,
            self.elementwise_factor._version,
            self.down_proj.master_weight._version,
        )
        if self._synced_versions != versions:
            self.refresh_weights()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input shape [M, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError("FlashNormSwiGLU requires CUDA BF16 input")
        self._sync_if_needed()
        inverse_rms = launch_inverse_rms(x.contiguous(), self.eps)
        hidden = launch_prenorm_swiglu_gateup(
            x,
            self._gate_weight_fp8,
            self._gate_weight_scale,
            self._up_weight_fp8,
            self._up_weight_scale,
            inverse_rms,
        )
        down_weight, down_scale = self.down_proj.quantized_weight
        return mxtest.launch_fp8gemm(
            hidden,
            down_weight,
            down_scale,
        )
