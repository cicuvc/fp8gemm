"""LM-head weights: one E8M0 scale per embedding and 32 features."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(W, Q, S, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    block = tl.program_id(1)
    cols = block * 32 + tl.arange(0, 32)
    mask = (rows[:, None] < N) & (cols[None, :] < K)
    # Preserve the existing weight quantizer's BF16 rounding and exponent rule.
    w = tl.load(W + rows[:, None] * K + cols[None, :], mask, 0)
    w = w.to(tl.bfloat16).to(tl.float32)
    exponent = (w.to(tl.int32, bitcast=True) >> 23) & 255
    scale = tl.maximum(tl.max(exponent, 1), 7) - 7
    reciprocal = tl.where(scale == 254, 0x00400000, (254 - scale) << 23)
    reciprocal = reciprocal.to(tl.int32).to(tl.float32, bitcast=True)
    q = (w * reciprocal[:, None]).to(tl.float8e4nv)
    tl.store(Q + rows[:, None] * K + cols[None, :], q, mask)
    tl.store(S + rows * tl.cdiv(K, 32) + block, scale.to(tl.uint8), rows < N)


@torch.no_grad()
def quantize_weight_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32/BF16 [V,D] -> E4M3 [V,D], uint8 E8M0 [V,ceil(D/32)].

    Input must be nonempty, contiguous CUDA storage and finite after BF16
    rounding. Each embedding is quantized independently. The returned pair is
    the forward weight cache accepted by fused linear cross entropy; callers
    must refresh it after updating the original weight. dX uses that original
    weight in BF16, not this cache. No native extension build is needed.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if (
        weight.ndim != 2
        or min(weight.shape) <= 0
        or weight.dtype not in (torch.float32, torch.bfloat16)
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected nonempty contiguous CUDA FP32/BF16 weight[V,D]")
    n, k = weight.shape
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (n, triton.cdiv(k, 32)), device=weight.device, dtype=torch.uint8
    )
    with torch.cuda.device(weight.device):
        _quantize_rows[(triton.cdiv(n, 32), triton.cdiv(k, 32))](
            weight, q, scales, n, k, num_warps=4
        )
    return q, scales
