"""Independent numerical references for the LM-head rowwise weight contract."""

import torch


def unpack_row_scales(words, rows, cols):
    scales = words.view(torch.uint8).reshape(words.shape[0], words.shape[1], 4)
    return scales.permute(1, 0, 2).reshape(words.shape[1], -1)[
        :rows, : (cols + 31) // 32
    ]


def dequant_rows(q, scales):
    exponents = scales.int().repeat_interleave(32, dim=1)[:, : q.shape[1]] - 127
    return torch.ldexp(q.float(), exponents)


def rowwise_logits(xq, wq, xs, ws):
    """FP32 GEMM of represented operands, followed by the logits BF16 rounding."""
    x = dequant_rows(xq, unpack_row_scales(xs, *xq.shape))
    w = dequant_rows(wq, ws)
    return (x @ w.T).bfloat16()
