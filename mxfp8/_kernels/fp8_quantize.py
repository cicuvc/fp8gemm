"""K128 FP32-scale quantizers used by the SM120 training path.

Row and column representations are quantized independently from the source.
Scales are dequantization multipliers: max(abs(block))/448, or 1 for zero.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _scale(amax):
    return tl.where(amax == 0, 1., tl.maximum(amax / 448., 1.1754943508222875e-38))


@tr.jit
def _store_dual(v, qr, sr, qc, sc, rows, cols, M: tl.constexpr, N: tl.constexpr,
                MP: tl.constexpr):
    rs = _scale(tl.max(tl.abs(v), 1))
    cs = _scale(tl.max(tl.abs(v), 0))
    rv = tl.minimum(tl.maximum(v / rs[:, None], -448.), 448.).to(tl.float8e4nv)
    cv = tl.minimum(tl.maximum(v / cs[None, :], -448.), 448.).to(tl.float8e4nv)
    tl.store(qr + rows[:, None] * N + cols[None, :], rv, rows[:, None] < M)
    tl.store(sr + rows * (N // 128) + tl.program_id(1), rs, rows < M)
    # Explicit zero padding in the transposed operand is part of the dW contract.
    tl.store(qc + cols[None, :] * MP + rows[:, None], cv)
    tl.store(sc + cols * (MP // 128) + tl.program_id(0), cs)


@tr.jit
def _dual(x, gate, up, qr, sr, qc, sc, ur, usr, uc, usc,
          M: tl.constexpr, N: tl.constexpr, MP: tl.constexpr, SILU: tl.constexpr):
    rows = tl.program_id(0) * 128 + tl.arange(0, 128)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    offset = rows[:, None] * N + cols[None, :]
    v = tl.load(x + offset, rows[:, None] < M, 0).to(tl.float32)
    if SILU:
        g = tl.load(gate + offset, rows[:, None] < M, 0).to(tl.float32)
        u = tl.load(up + offset, rows[:, None] < M, 0).to(tl.float32)
        sigmoid = 1. / (1. + tl.exp(-g))
        dg = (v * u * sigmoid * (1. + g * (1. - sigmoid))).to(tl.bfloat16).to(tl.float32)
        du = (v * g * sigmoid).to(tl.bfloat16).to(tl.float32)
        _store_dual(dg, qr, sr, qc, sc, rows, cols, M, N, MP)
        _store_dual(du, ur, usr, uc, usc, rows, cols, M, N, MP)
    else:
        _store_dual(v, qr, sr, qc, sc, rows, cols, M, N, MP)


@tr.jit
def _weight(w, q, s, qt, st, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 128 + tl.arange(0, 128)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    v = tl.load(w + rows[:, None] * K + cols[None, :])
    scale = _scale(tl.max(tl.max(tl.abs(v), 1), 0))
    quant = tl.minimum(tl.maximum(v / scale, -448.), 448.).to(tl.float8e4nv)
    tl.store(q + rows[:, None] * K + cols[None, :], quant)
    tl.store(qt + cols[None, :] * N + rows[:, None], quant)
    tl.store(s + tl.program_id(0) * (K // 128) + tl.program_id(1), scale)
    tl.store(st + tl.program_id(1) * (N // 128) + tl.program_id(0), scale)


class DualQuantized:
    """Stable rowwise and transposed-columnwise buffers for an M x N activation."""

    def __init__(self, m, n, device):
        self.m, self.n, self.mp = m, n, tr.cdiv(m, 128) * 128
        self.row = torch.empty((m, n), device=device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.empty((m, n // 128), device=device, dtype=torch.float32)
        self.col = torch.empty((n, self.mp), device=device, dtype=torch.float8_e4m3fn)
        self.col_scale = torch.empty((n, self.mp // 128), device=device, dtype=torch.float32)

    @property
    def tensors(self):
        return self.row, self.row_scale, self.col, self.col_scale

    def quantize(self, x):
        with torch.cuda.device(x.device):
            _dual[(tr.cdiv(self.m, 128), self.n // 128)](
                x, x, x, *self.tensors, *self.tensors,
                self.m, self.n, self.mp, False, num_warps=8,
            )
        return self

    def silu_backward(self, dy, gate, up, up_grad):
        with torch.cuda.device(dy.device):
            _dual[(tr.cdiv(self.m, 128), self.n // 128)](
                dy, gate, up, *self.tensors, *up_grad.tensors,
                self.m, self.n, self.mp, True, num_warps=8,
            )


class BlockQuantized:
    """128x128 weight blocks; the transpose reuses exactly the same quantization."""

    def __init__(self, n, k, device):
        self.row = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.empty((n // 128, k // 128), device=device, dtype=torch.float32)
        self.col = torch.empty((k, n), device=device, dtype=torch.float8_e4m3fn)
        self.col_scale = torch.empty((k // 128, n // 128), device=device, dtype=torch.float32)

    def quantize(self, weight):
        n, k = weight.shape
        with torch.cuda.device(weight.device):
            _weight[(n // 128, k // 128)](
                weight, self.row, self.row_scale, self.col, self.col_scale,
                n, k, num_warps=8,
            )
        return self
