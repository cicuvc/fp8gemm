"""Triton LSE merge and CE with independent row/column MXFP8 gradients.

Losses and dZ are unnormalized; labels must be valid vocabulary indices.
No BF16 dZ tensor is materialized. Quantization rounds dZ to BF16 in registers
and follows cuda/mxfp8_quantize.cuh (E8M0 exponent offset 7).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_lse(Z, Partial, Target, LSE, Loss, M: tl.constexpr,
               V: tl.constexpr, P: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, BP)
    local = tl.load(Partial + row * P + p, p < P, float('-inf'))
    maximum = tl.max(local, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(local - maximum), 0))
    y = tl.load(Target + row)
    tl.device_assert((y >= 0) & (y < V), "target outside vocabulary")
    zy = tl.load(Z + row * V + y, (y >= 0) & (y < V), float('nan')).to(tl.float32)
    tl.store(LSE + row, lse)
    tl.store(Loss + row, lse - zy)


@triton.jit
def _ce_dual_quantize(Z, LSE, Target, Row, RowScale, Col, ColScale,
                      M: tl.constexpr, V: tl.constexpr, MP: tl.constexpr,
                      MP16: tl.constexpr, VP16: tl.constexpr):
    # Each program owns one 32x32 tile: complete groups in both directions.
    tm = tl.program_id(0).to(tl.int64)
    tv = tl.program_id(1).to(tl.int64)
    rows = tm * 32 + tl.arange(0, 32)
    cols = tv * 32 + tl.arange(0, 32)
    z = tl.load(Z + rows[:, None] * V + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < V), 0).to(tl.float32)
    lse = tl.load(LSE + rows, rows < M, 0)
    y = tl.load(Target + rows, rows < M, -1)
    g = tl.exp(z - lse[:, None]) - (y[:, None] == cols[None, :]).to(tl.float32)
    g = tl.where((rows[:, None] < M) & (cols[None, :] < V), g, 0)
    # Match the existing BF16 -> MXFP8 quantizer, including its rounding point.
    g = g.to(tl.bfloat16).to(tl.float32)
    exponent = (g.to(tl.int32, bitcast=True) >> 23) & 255
    er = tl.maximum(tl.max(exponent, 1), 7) - 7
    ec = tl.maximum(tl.max(exponent, 0), 7) - 7
    rr = ((254 - er) << 23).to(tl.float32, bitcast=True)
    rc = ((254 - ec) << 23).to(tl.float32, bitcast=True)
    qr = (g * rr[:, None]).to(tl.bfloat16).to(tl.float8e4nv)
    qc = (g * rc[None, :]).to(tl.bfloat16).to(tl.float8e4nv)
    tl.store(Row + rows[:, None] * V + cols[None, :], qr,
             (rows[:, None] < M) & (cols[None, :] < V))
    tl.store(Col + cols[None, :] * MP + rows[:, None], qc, cols[None, :] < V)
    # Disjoint byte stores, never an int32 read-modify-write across programs.
    tl.store(RowScale + (tv // 4 * MP16 + rows) * 4 + tv % 4,
             er.to(tl.uint8), rows < MP16)
    tl.store(ColScale + (tm // 4 * VP16 + cols) * 4 + tm % 4,
             ec.to(tl.uint8), cols < VP16)


def merge_lse(logits, partial_lse, target, *, lse=None, loss=None):
    """Merge GEMM partials and gather the target logit; return FP32 [M] tensors."""
    m, v = logits.shape
    if (logits.dtype != torch.bfloat16 or not logits.is_cuda or
            not logits.is_contiguous() or m <= 0 or v <= 0):
        raise ValueError("Expected contiguous CUDA BF16 logits[M,V]")
    if (partial_lse.ndim != 2 or partial_lse.shape[0] != m or
            partial_lse.shape[1] <= 0 or partial_lse.dtype != torch.float32 or
            not partial_lse.is_contiguous() or partial_lse.device != logits.device):
        raise ValueError("Expected contiguous FP32 partial_lse[M,P]")
    _validate_target(target, m, logits.device)
    if lse is None:
        lse = torch.empty(m, device=logits.device, dtype=torch.float32)
    if loss is None:
        loss = torch.empty_like(lse)
    for t in (lse, loss):
        if t.shape != (m,) or t.dtype != torch.float32 or t.device != logits.device or not t.is_contiguous():
            raise ValueError("Expected contiguous FP32 output[M]")
    p = partial_lse.shape[1]
    with torch.cuda.device(logits.device):
        _merge_lse[(m,)](logits, partial_lse, target, lse, loss, m, v, p,
                         triton.next_power_of_2(p))
    return lse, loss


def _validate_target(target, m, device):
    if (target.shape != (m,) or target.dtype != torch.int64 or
            target.device != device or not target.is_contiguous()):
        raise ValueError("Expected contiguous CUDA int64 target[M]")


class CEPlan:
    """Reusable quantization workspace compatible with CUDA dgrad/wgrad plans."""
    def __init__(self, m, v, device):
        if m <= 0 or v <= 0:
            raise ValueError("Positive dimensions required")
        self.m, self.v = m, v
        self.mp = triton.cdiv(m, 128) * 128
        self.mp16, self.vp16 = triton.cdiv(m, 16) * 16, triton.cdiv(v, 16) * 16
        self.row_fp8 = torch.empty((m, v), device=device, dtype=torch.float8_e4m3fn)
        self.column_fp8 = torch.empty((v, self.mp), device=device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.zeros((triton.cdiv(v, 128), self.mp16), device=device, dtype=torch.int32)
        self.column_scale = torch.zeros((self.mp // 128, self.vp16), device=device, dtype=torch.int32)

    def __call__(self, logits, lse, target):
        if (logits.shape != (self.m, self.v) or logits.dtype != torch.bfloat16 or
                logits.device != self.row_fp8.device or not logits.is_contiguous()):
            raise ValueError("Expected contiguous BF16 logits matching the CE plan")
        if (lse.shape != (self.m,) or lse.dtype != torch.float32 or
                lse.device != logits.device or not lse.is_contiguous()):
            raise ValueError("Expected contiguous FP32 LSE[M]")
        _validate_target(target, self.m, logits.device)
        with torch.cuda.device(logits.device):
            _ce_dual_quantize[(self.mp // 32, triton.cdiv(self.v, 32))](
                logits, lse, target, self.row_fp8, self.row_scale.view(torch.uint8),
                self.column_fp8, self.column_scale.view(torch.uint8),
                self.m, self.v, self.mp, self.mp16, self.vp16, num_warps=4,
                enable_fp_fusion=False,
            )
        return self.row_fp8, self.row_scale, self.column_fp8, self.column_scale
