"""Experimental fused gradient prequantization; not used by FP8SwiGLU.

The extra FP8/scales stores regress medium shapes on RTX 5090. Keep this
variant separate so further experiments do not affect the default path.
"""
import torch
import triton
import triton.language as tl

from mxtest import (
    DUAL_GEMM_CONFIGS, DOWN_ACT_CONFIGS, online_scale, fast_tanh,
    launch_bf16_weight_grad, launch_bf16_weight_grad_dual,
    launch_fp8gemm_weight_grad, launch_fp8gemm_weight_grad_dual,
)

SWIGLU_DX_CONFIGS = DUAL_GEMM_CONFIGS + [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]

@triton.autotune(configs=SWIGLU_DX_CONFIGS, key=["M", "N", "K", "PREQUANTIZED"], cache_results=True)
@triton.jit
def fp8_swiglu_bwd_dx(
    dgate,
    dup,
    gate_weight,
    gate_scale,
    up_weight,
    up_scale,
    out,
    M,
    N,
    K,
    dg_stride1,
    du_stride1,
    gw_stride1,
    gs_stride1,
    uw_stride1,
    us_stride1,
    o_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREQUANTIZED: tl.constexpr = False,
    dg_scale=None,
    du_scale=None,
):
    # dX = dgate @ Wg + dup @ Wu. Both dgate and dup use online 1x32 scales
    # along the contraction dimension N. Weight scales stay in 32x32 blocks.
    MX_BLOCK: tl.constexpr = 32
    SCALE_ROWS: tl.constexpr = BLOCK_N // MX_BLOCK
    SCALE_COLS: tl.constexpr = BLOCK_K // MX_BLOCK
    GRAD_SCALES: tl.constexpr = BLOCK_N // MX_BLOCK

    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    NUM_SCALE_ROWS = tl.cdiv(N, MX_BLOCK)
    NUM_SCALE_COLS = tl.cdiv(K, MX_BLOCK)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    dg_block_ptr = tl.make_block_ptr(dgate, shape=(M,N), strides=(dg_stride1, 1), offsets=(row_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    du_block_ptr = tl.make_block_ptr(dup, shape=(M,N), strides=(du_stride1, 1), offsets=(row_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    gw_block_ptr = tl.make_block_ptr(gate_weight, shape=(N,K), strides=(gw_stride1, 1), offsets=(0, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    gs_block_ptr = tl.make_block_ptr(gate_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_COLS), strides=(gs_stride1, 1), offsets=(0, col_idx * SCALE_COLS), block_shape=(SCALE_ROWS, SCALE_COLS), order=(1,0))
    uw_block_ptr = tl.make_block_ptr(up_weight, shape=(N,K), strides=(uw_stride1, 1), offsets=(0, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    us_block_ptr = tl.make_block_ptr(up_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_COLS), strides=(us_stride1, 1), offsets=(0, col_idx * SCALE_COLS), block_shape=(SCALE_ROWS, SCALE_COLS), order=(1,0))

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for i in tl.range(0, N, BLOCK_N):
        dg_block = tl.load(dg_block_ptr, boundary_check=(1,0), padding_option='zero')
        du_block = tl.load(du_block_ptr, boundary_check=(1,0), padding_option='zero')
        if PREQUANTIZED:
            rows = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = i // MX_BLOCK + tl.arange(0, GRAD_SCALES)
            offsets = rows[:, None] * NUM_SCALE_ROWS + cols[None, :]
            mask = (rows[:, None] < M) & (cols[None, :] < NUM_SCALE_ROWS)
            dgs_block = tl.load(dg_scale + offsets, mask=mask, other=0)
            dus_block = tl.load(du_scale + offsets, mask=mask, other=0)
            dg_block_scaled, du_block_scaled = dg_block, du_block
        else:
            dg_block_scaled, dgs_block = online_scale(dg_block)
            du_block_scaled, dus_block = online_scale(du_block)

        gw_block = tl.load(gw_block_ptr, boundary_check=(1,0), padding_option='zero')
        gs_block = tl.load(gs_block_ptr, boundary_check=(1,0), padding_option='zero')
        gs_block = tl.reshape(
            tl.broadcast_to(
                tl.trans(gs_block)[:,None,:],
                (SCALE_COLS, MX_BLOCK, GRAD_SCALES),
            ),
            (BLOCK_K, GRAD_SCALES),
        )

        uw_block = tl.load(uw_block_ptr, boundary_check=(1,0), padding_option='zero')
        us_block = tl.load(us_block_ptr, boundary_check=(1,0), padding_option='zero')
        us_block = tl.reshape(
            tl.broadcast_to(
                tl.trans(us_block)[:,None,:],
                (SCALE_COLS, MX_BLOCK, GRAD_SCALES),
            ),
            (BLOCK_K, GRAD_SCALES),
        )

        acc = tl.dot_scaled(dg_block_scaled, dgs_block, 'e4m3', gw_block, gs_block, 'e4m3', acc)
        acc = tl.dot_scaled(du_block_scaled, dus_block, 'e4m3', uw_block, us_block, 'e4m3', acc)

        dg_block_ptr = tl.advance(dg_block_ptr, (0, BLOCK_N))
        du_block_ptr = tl.advance(du_block_ptr, (0, BLOCK_N))
        gw_block_ptr = tl.advance(gw_block_ptr, (BLOCK_N, 0))
        gs_block_ptr = tl.advance(gs_block_ptr, (SCALE_ROWS, 0))
        uw_block_ptr = tl.advance(uw_block_ptr, (BLOCK_N, 0))
        us_block_ptr = tl.advance(us_block_ptr, (SCALE_ROWS, 0))

    o_block_ptr = tl.make_block_ptr(out, shape=(M,K), strides=(o_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    tl.store(o_block_ptr, acc.to(out.dtype.element_ty), boundary_check=(1,0))


@triton.autotune(configs=DOWN_ACT_CONFIGS, key=["M", "N", "K", "PREQUANTIZE"], cache_results=True)
@triton.jit
def fp8gemm_bwd_swiglu_act(
    dy_input,
    weight,
    weight_scale,
    gate,
    up,
    dgate,
    dup,
    M,
    N,
    K,
    dy_stride1,
    w_stride1,
    ws_stride1,
    gate_stride1,
    up_stride1,
    dg_stride1,
    du_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREQUANTIZE: tl.constexpr = False,
    dg_quant=None,
    du_quant=None,
    dg_scale=None,
    du_scale=None,
):
    # dY @ Wd and the SwiGLU activation gradient share the same [M, H] tile.
    # dH is rounded through BF16 to preserve the two-kernel numerical behavior.
    MX_BLOCK: tl.constexpr = 32
    SCALE_ROWS: tl.constexpr = BLOCK_N // MX_BLOCK
    SCALE_COLS: tl.constexpr = BLOCK_K // MX_BLOCK
    DY_SCALES: tl.constexpr = BLOCK_N // MX_BLOCK

    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    NUM_SCALE_ROWS = tl.cdiv(N, MX_BLOCK)
    NUM_SCALE_COLS = tl.cdiv(K, MX_BLOCK)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    dy_block_ptr = tl.make_block_ptr(dy_input, shape=(M,N), strides=(dy_stride1, 1), offsets=(row_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    w_block_ptr = tl.make_block_ptr(weight, shape=(N,K), strides=(w_stride1, 1), offsets=(0, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    ws_block_ptr = tl.make_block_ptr(weight_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_COLS), strides=(ws_stride1, 1), offsets=(0, col_idx * SCALE_COLS), block_shape=(SCALE_ROWS, SCALE_COLS), order=(1,0))

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for i in tl.range(0, N, BLOCK_N):
        dy_block = tl.load(dy_block_ptr, boundary_check=(1,0), padding_option='zero')
        dy_block_scaled, dys_block = online_scale(dy_block)

        w_block = tl.load(w_block_ptr, boundary_check=(1,0), padding_option='zero')
        ws_block = tl.load(ws_block_ptr, boundary_check=(1,0), padding_option='zero')
        ws_block = tl.reshape(
            tl.broadcast_to(
                tl.trans(ws_block)[:,None,:],
                (SCALE_COLS, MX_BLOCK, DY_SCALES),
            ),
            (BLOCK_K, DY_SCALES),
        )

        acc = tl.dot_scaled(dy_block_scaled, dys_block, 'e4m3', w_block, ws_block, 'e4m3', acc)

        dy_block_ptr = tl.advance(dy_block_ptr, (0, BLOCK_N))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_N, 0))
        ws_block_ptr = tl.advance(ws_block_ptr, (SCALE_ROWS, 0))

    gate_ptr = tl.make_block_ptr(gate, shape=(M,K), strides=(gate_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    up_ptr = tl.make_block_ptr(up, shape=(M,K), strides=(up_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    gate_block = tl.load(gate_ptr, boundary_check=(1,0), padding_option='zero').to(tl.float32)
    up_block = tl.load(up_ptr, boundary_check=(1,0), padding_option='zero').to(tl.float32)

    dH_block = acc.to(tl.bfloat16).to(tl.float32)
    half_gate = gate_block * 0.5
    tanh = fast_tanh(half_gate)
    tanh_plus_one = tanh + 1.0
    dsilu = 0.5 * (tanh_plus_one + half_gate * (1.0 - tanh * tanh))
    dgate_block = dH_block * dsilu * up_block
    dup_block = dH_block * half_gate * tanh_plus_one

    dgate_ptr = tl.make_block_ptr(dgate, shape=(M,K), strides=(dg_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    dup_ptr = tl.make_block_ptr(dup, shape=(M,K), strides=(du_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    tl.store(dgate_ptr, dgate_block.to(tl.bfloat16), boundary_check=(1,0))
    tl.store(dup_ptr, dup_block.to(tl.bfloat16), boundary_check=(1,0))

    if PREQUANTIZE:
        # Quantize the rounded BF16 values used by the original dX kernel.
        dg_q, dg_s = online_scale(dgate_block.to(tl.bfloat16))
        du_q, du_s = online_scale(dup_block.to(tl.bfloat16))
        rows = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        offsets = rows[:, None] * K + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < K)
        tl.store(dg_quant + offsets, dg_q, mask=mask)
        tl.store(du_quant + offsets, du_q, mask=mask)
        scale_cols = col_idx * SCALE_COLS + tl.arange(0, SCALE_COLS)
        scale_offsets = rows[:, None] * NUM_SCALE_COLS + scale_cols[None, :]
        scale_mask = (rows[:, None] < M) & (scale_cols[None, :] < NUM_SCALE_COLS)
        tl.store(dg_scale + scale_offsets, dg_s, mask=scale_mask)
        tl.store(du_scale + scale_offsets, du_s, mask=scale_mask)


def launch_fp8_down_swiglu_backward(
    dy: torch.Tensor,
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    x: torch.Tensor,
    return_intermediate_grads: bool = False,
    use_fp8_wgrad: bool = False,
    prequantize: bool = False,
):
    M, O = dy.shape
    M_hidden, H = hidden.shape
    N_weight, O_weight = down_weight.shape

    assert M_hidden == M
    assert N_weight == O and O_weight == H
    assert hidden.dtype == torch.bfloat16
    assert hidden.stride(1) == 1

    dgate = torch.empty((M, H), dtype=torch.bfloat16, device=dy.device)
    dup = torch.empty_like(dgate)
    dg_q = du_q = dg_s = du_s = None
    if prequantize:
        dg_q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dy.device)
        du_q = torch.empty_like(dg_q)
        dg_s = torch.empty((M, triton.cdiv(H, 32)), dtype=torch.uint8, device=dy.device)
        du_s = torch.empty_like(dg_s)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(H, meta['BLOCK_K']),)
    fp8gemm_bwd_swiglu_act[grid](
        dy,
        down_weight,
        down_scale,
        gate,
        up,
        dgate,
        dup,
        M,
        O,
        H,
        dy.stride(0),
        down_weight.stride(0),
        down_scale.stride(0),
        gate.stride(0),
        up.stride(0),
        dgate.stride(0),
        dup.stride(0),
        PREQUANTIZE=prequantize,
        dg_quant=dg_q,
        du_quant=du_q,
        dg_scale=dg_s,
        du_scale=du_s,
    )

    grad_x = torch.empty((M, x.shape[1]), dtype=torch.bfloat16, device=dy.device)
    fp8_swiglu_bwd_dx[
        lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(x.shape[1], meta['BLOCK_K']),
        )
    ](
        dg_q if prequantize else dgate,
        du_q if prequantize else dup,
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        grad_x,
        M,
        H,
        x.shape[1],
        dgate.stride(0),
        dup.stride(0),
        gate_weight.stride(0),
        gate_scale.stride(0),
        up_weight.stride(0),
        up_scale.stride(0),
        grad_x.stride(0),
        PREQUANTIZED=prequantize,
        dg_scale=dg_s,
        du_scale=du_s,
    )

    if use_fp8_wgrad:
        grad_gate_weight, grad_up_weight = launch_fp8gemm_weight_grad_dual(
            dgate,
            dup,
            x,
            out_dtype=torch.float32,
        )
        grad_down_weight = launch_fp8gemm_weight_grad(
            dy,
            hidden,
            out_dtype=torch.float32,
        )
    else:
        grad_gate_weight, grad_up_weight = launch_bf16_weight_grad_dual(
            dgate,
            dup,
            x,
            out_dtype=torch.float32,
        )
        grad_down_weight = launch_bf16_weight_grad(
            dy,
            hidden,
            out_dtype=torch.float32,
        )

    if return_intermediate_grads:
        return (
            grad_x,
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            dgate,
            dup,
        )
    return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight

