"""Fused producers for the cached FP8 wgrad experiment.

These fork the existing GEMMs to measure the actual cost of storing column
packs from their epilogues. They do not alter the default implementation.
"""
import torch
import triton
import triton.language as tl
import mxtest

@triton.autotune(configs=mxtest.DUAL_GEMM_CONFIGS, key=["M", "N", "K", "SAVE_X"], cache_results=True)
@triton.jit
def _swiglu_fwd_save_hidden(
    x_input,
    gate_weight,
    gate_scale,
    up_weight,
    up_scale,
    out,
    gate_out,
    up_out,
    M,
    N,
    K,
    x_stride1,
    gw_stride1,
    gs_stride1,
    uw_stride1,
    us_stride1,
    o_stride1,
    go_stride1,
    uo_stride1,
    STORE_AUX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    hidden_q,
    hidden_s,
    MP: tl.constexpr,
    SAVE_X: tl.constexpr = False,
    input_q=None,
    input_s=None,
):
    # Both weights use [32, 32] block scales. X is quantized online once and
    # reused for the gate and up projections.
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

    x_block_ptr = tl.make_block_ptr(x_input, shape=(M, K), strides=(x_stride1, 1), offsets=(row_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    gw_block_ptr = tl.make_block_ptr(gate_weight, shape=(N, K), strides=(gw_stride1, 1), offsets=(col_idx * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    gs_block_ptr = tl.make_block_ptr(gate_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS), strides=(gs_stride1, 1), offsets=(col_idx * SCALE_ROWS, 0), block_shape=(SCALE_ROWS, BLOCK_SCALES), order=(1,0))
    uw_block_ptr = tl.make_block_ptr(up_weight, shape=(N, K), strides=(uw_stride1, 1), offsets=(col_idx * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    us_block_ptr = tl.make_block_ptr(up_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS), strides=(us_stride1, 1), offsets=(col_idx * SCALE_ROWS, 0), block_shape=(SCALE_ROWS, BLOCK_SCALES), order=(1,0))

    gate_half_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, K, BLOCK_K):
        x_block = tl.load(x_block_ptr, boundary_check=(1,0), padding_option='zero')
        x_block_scaled, xs_block = mxtest.online_scale(x_block)
        if SAVE_X:
            # Each input tile has exactly one writer despite reuse by all
            # output-feature CTAs. All BLOCK_M values are multiples of 32.
            if col_idx == 0:
                xq, xsc = mxtest.online_scale(tl.trans(x_block))
                xrows = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                xfeatures = i + tl.arange(0, BLOCK_K)
                xoffs = xfeatures[:, None] * MP + xrows[None, :]
                tl.store(input_q + xoffs, xq,
                         mask=(xfeatures[:, None] < K) & (xrows[None, :] < MP))
                xgroups = row_idx * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)
                xsoffs = xfeatures[:, None] * (MP // 32) + xgroups[None, :]
                tl.store(input_s + xsoffs, xsc,
                         mask=(xfeatures[:, None] < K) & (xgroups[None, :] < MP // 32))
        # Fold the 0.5 from 0.5*x*(tanh(0.5*x)+1) into the gate GEMM.
        # Only the activation scale changes; the FP8 weight scales stay intact.
        gate_xs_block = tl.maximum(xs_block.to(tl.int32) - 1, 0).to(tl.uint8)

        gw_block = tl.load(gw_block_ptr, boundary_check=(1,0), padding_option='zero')
        gs_block = tl.load(gs_block_ptr, boundary_check=(1,0), padding_option='zero')
        gs_block = tl.reshape(
            tl.broadcast_to(
                gs_block[:,None,:],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )

        uw_block = tl.load(uw_block_ptr, boundary_check=(1,0), padding_option='zero')
        us_block = tl.load(us_block_ptr, boundary_check=(1,0), padding_option='zero')
        us_block = tl.reshape(
            tl.broadcast_to(
                us_block[:,None,:],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )

        gate_half_acc = tl.dot_scaled(x_block_scaled, gate_xs_block, 'e4m3', tl.trans(gw_block), gs_block, 'e4m3', gate_half_acc)
        up_acc = tl.dot_scaled(x_block_scaled, xs_block, 'e4m3', tl.trans(uw_block), us_block, 'e4m3', up_acc)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        gw_block_ptr = tl.advance(gw_block_ptr, (0, BLOCK_K))
        gs_block_ptr = tl.advance(gs_block_ptr, (0, BLOCK_SCALES))
        uw_block_ptr = tl.advance(uw_block_ptr, (0, BLOCK_K))
        us_block_ptr = tl.advance(us_block_ptr, (0, BLOCK_SCALES))

    swiglu = gate_half_acc * (mxtest.fast_tanh(gate_half_acc) + 1.0) * up_acc
    o_block_ptr = tl.make_block_ptr(out, shape=(M,N), strides=(o_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    tl.store(o_block_ptr, swiglu.to(out.dtype.element_ty), boundary_check=(1,0))

    if STORE_AUX:
        gate_block_ptr = tl.make_block_ptr(gate_out, shape=(M,N), strides=(go_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
        up_block_ptr = tl.make_block_ptr(up_out, shape=(M,N), strides=(uo_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
        # Keep the public aux output as the full XW_g value needed by backward.
        tl.store(gate_block_ptr, (gate_half_acc * 2.0).to(gate_out.dtype.element_ty), boundary_check=(1,0))
        tl.store(up_block_ptr, up_acc.to(up_out.dtype.element_ty), boundary_check=(1,0))

    q, scales = mxtest.online_scale(tl.trans(swiglu.to(tl.bfloat16)))
    rows = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    features = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = features[:, None] * MP + rows[None, :]
    valid = (features[:, None] < N) & (rows[None, :] < MP)
    tl.store(hidden_q + offsets, q, mask=valid)
    groups = row_idx * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)
    scale_offsets = features[:, None] * (MP // 32) + groups[None, :]
    tl.store(hidden_s + scale_offsets, scales,
             mask=(features[:, None] < N) & (groups[None, :] < MP // 32))


@triton.autotune(configs=mxtest.DOWN_ACT_CONFIGS, key=["M", "N", "K"], cache_results=True)
@triton.jit
def _down_bwd_save_grads(
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
    dg_q,
    dg_s,
    du_q,
    du_s,
    MP: tl.constexpr,
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
        dy_block_scaled, dys_block = mxtest.online_scale(dy_block)

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
    tanh = mxtest.fast_tanh(half_gate)
    tanh_plus_one = tanh + 1.0
    dsilu = 0.5 * (tanh_plus_one + half_gate * (1.0 - tanh * tanh))
    dgate_block = dH_block * dsilu * up_block
    dup_block = dH_block * half_gate * tanh_plus_one

    dgate_ptr = tl.make_block_ptr(dgate, shape=(M,K), strides=(dg_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    dup_ptr = tl.make_block_ptr(dup, shape=(M,K), strides=(du_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    tl.store(dgate_ptr, dgate_block.to(tl.bfloat16), boundary_check=(1,0))
    tl.store(dup_ptr, dup_block.to(tl.bfloat16), boundary_check=(1,0))

    q0, s0 = mxtest.online_scale(tl.trans(dgate_block.to(tl.bfloat16)))
    q1, s1 = mxtest.online_scale(tl.trans(dup_block.to(tl.bfloat16)))
    rows = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    features = col_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = features[:, None] * MP + rows[None, :]
    valid = (features[:, None] < K) & (rows[None, :] < MP)
    tl.store(dg_q + offsets, q0, mask=valid)
    tl.store(du_q + offsets, q1, mask=valid)
    groups = row_idx * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)
    scale_offsets = features[:, None] * (MP // 32) + groups[None, :]
    scale_valid = (features[:, None] < K) & (groups[None, :] < MP // 32)
    tl.store(dg_s + scale_offsets, s0, mask=scale_valid)
    tl.store(du_s + scale_offsets, s1, mask=scale_valid)


def saved_hidden_forward(x, gw, gs, uw, us, dw, ds, bias, save_x=False):
    m,k=x.shape
    h=gw.shape[0]
    mp=triton.cdiv(m,32)*32
    gate=torch.empty((m,h),device=x.device,dtype=torch.bfloat16)
    up,hidden=torch.empty_like(gate),torch.empty_like(gate)
    hq=torch.empty((h,mp),device=x.device,dtype=torch.float8_e4m3fn)
    hs=torch.empty((h,mp//32),device=x.device,dtype=torch.uint8)
    xq=xs=None
    if save_x:
        xq=torch.empty((k,mp),device=x.device,dtype=torch.float8_e4m3fn)
        xs=torch.empty((k,mp//32),device=x.device,dtype=torch.uint8)
    _swiglu_fwd_save_hidden[lambda c:(triton.cdiv(m,c['BLOCK_M'])*triton.cdiv(h,c['BLOCK_N']),)](
        x,gw,gs,uw,us,hidden,gate,up,m,h,k,x.stride(0),gw.stride(0),gs.stride(0),
        uw.stride(0),us.stride(0),hidden.stride(0),gate.stride(0),up.stride(0),True,
        hidden_q=hq,hidden_s=hs,MP=mp,SAVE_X=save_x,input_q=xq,input_s=xs)
    y=mxtest.launch_fp8gemm(hidden,dw,ds,bias)
    return y,gate,up,hq,hs,xq,xs


def saved_grads_backward(dy, gate, up, dw, ds):
    m,o=dy.shape
    h=gate.shape[1]
    mp=triton.cdiv(m,32)*32
    dg,du=torch.empty_like(gate),torch.empty_like(up)
    dq=torch.empty((h,mp),device=dy.device,dtype=torch.float8_e4m3fn)
    uq=torch.empty_like(dq)
    dqs=torch.empty((h,mp//32),device=dy.device,dtype=torch.uint8)
    uqs=torch.empty_like(dqs)
    _down_bwd_save_grads[lambda c:(triton.cdiv(m,c['BLOCK_M'])*triton.cdiv(h,c['BLOCK_K']),)](
        dy,dw,ds,gate,up,dg,du,m,o,h,dy.stride(0),dw.stride(0),ds.stride(0),
        gate.stride(0),up.stride(0),dg.stride(0),du.stride(0),
        dg_q=dq,dg_s=dqs,du_q=uq,du_s=uqs,MP=mp)
    return dg,du,dq,dqs,uq,uqs
