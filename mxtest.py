import triton
import triton.language as tl
import torch

@triton.jit
def u16_max(x, y):
    return tl.inline_asm_elementwise('max.u16x2 $0, $1, $2;', '=r,r,r',[x,y], tl.uint16, True, 2)

@triton.jit
def online_scale(x: tl.tensor):
    tl.static_assert(x.dtype == tl.bfloat16)
    # x: [M, K]
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    MX_BLOCK: tl.constexpr = 32

    x = tl.reshape(x, (M, K // MX_BLOCK, MX_BLOCK))
    x_exp = tl.inline_asm_elementwise("and.b32 $0, $1, 0x7F807F80;", "=r,r", [x], tl.uint16, True, 2) # [M, BLK, MX_BLOCK]
    x_scale_u16 = tl.reduce(x_exp, axis=-1, combine_fn=u16_max) # [M, BLK]
    x_scale_rcp = tl.inline_asm_elementwise("sub.u32 $0, 0x7F007F00, $1; and.b32 $0, $0, 0x7F807F80;", "=r,r", [x_scale_u16], tl.bfloat16, True, 2)
    x_scaled = tl.cast(x * x_scale_rcp[:,:,None], tl.float8e4nv).reshape(M, K)
    return x_scaled, tl.cast(x_scale_u16 >> 7, tl.uint8)


@triton.jit
def block_online_scale(x: tl.tensor):
    tl.static_assert(x.dtype == tl.bfloat16)
    # Use one E8M0 scale for the whole [BLOCK_N, BLOCK_K] block.
    BLOCK_N: tl.constexpr = x.shape[0]
    BLOCK_K: tl.constexpr = x.shape[1]

    x_exp = tl.inline_asm_elementwise(
        "and.b32 $0, $1, 0x7F807F80;",
        "=r,r",
        [x],
        tl.uint16,
        True,
        2,
    )
    x_scale_u16 = tl.max(x_exp, axis=None)
    x_scale = tl.cast(x_scale_u16 >> 7, tl.uint8)
    x_scale_rcp = tl.exp2(127.0 - x_scale.to(tl.float32))
    x_scaled = tl.cast(x * x_scale_rcp, tl.float8e4nv)
    return x_scaled, x_scale


@triton.jit
def fp8gemm_fwd(x_input, weight, weight_scale, bias, out, M, N, K, x_stride1, w_stride1, ws_stride1, o_stride1, HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # x is quantized online with one scale per 32 elements.
    # weight uses one scale per [32, 32] block.
    # x: [M, K], w: [N, K], ws: [ceil(N, 32), ceil(K, 32)]
    PRESET_GROUP_SIZE: tl.constexpr = 4
    BLOCK_K: tl.constexpr = 64
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
    w_block_ptr = tl.make_block_ptr(weight, shape=(N, K), strides=(w_stride1, 1), offsets=(col_idx * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    ws_block_ptr = tl.make_block_ptr(weight_scale, shape=(NUM_SCALE_ROWS, NUM_SCALE_BLOCKS), strides=(ws_stride1, 1), offsets=(col_idx * SCALE_ROWS, 0), block_shape=(SCALE_ROWS, BLOCK_SCALES), order=(1,0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, K, BLOCK_K):
        x_block = tl.load(x_block_ptr, boundary_check=(1,0), padding_option='zero')
        x_block_scaled, xs_block = online_scale(x_block)

        w_block = tl.load(w_block_ptr, boundary_check=(1,0), padding_option='zero')
        ws_block = tl.load(ws_block_ptr, boundary_check=(1,0), padding_option='zero')
        ws_block = tl.reshape(
            tl.broadcast_to(
                ws_block[:,None,:],
                (SCALE_ROWS, MX_BLOCK, BLOCK_SCALES),
            ),
            (BLOCK_N, BLOCK_SCALES),
        )

        acc = tl.dot_scaled(x_block_scaled, xs_block, 'e4m3', tl.trans(w_block), ws_block, 'e4m3', acc)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (0, BLOCK_K))
        ws_block_ptr = tl.advance(ws_block_ptr, (0, BLOCK_SCALES))

    if HAS_BIAS:
        bias_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_block = tl.load(bias + bias_offsets, mask=bias_offsets < N, other=0.0)
        acc += bias_block[None, :]

    o_block_ptr = tl.make_block_ptr(out, shape=(M,N), strides=(o_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    tl.store(o_block_ptr, acc.to(out.dtype.element_ty), boundary_check=(1,0))


@triton.jit
def fp8gemm_bwd(dy_input, weight, weight_scale, out, M, N, K, dy_stride1, w_stride1, ws_stride1, o_stride1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # dy is quantized online with one scale per 32 elements.
    # The weight keeps its [32, 32] block scales.
    # dy: [M, N], w: [N, K], out: [M, K]
    # The same [ceil(N, 32), ceil(K, 32)] weight scales are transposed and
    # broadcast for the backward GEMM.
    MX_BLOCK: tl.constexpr = 32
    SCALE_ROWS: tl.constexpr = BLOCK_N // MX_BLOCK
    SCALE_COLS: tl.constexpr = BLOCK_K // MX_BLOCK
    DY_SCALES: tl.constexpr = BLOCK_N // MX_BLOCK

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    NUM_SCALE_ROWS = tl.cdiv(N, MX_BLOCK)
    NUM_SCALE_COLS = tl.cdiv(K, MX_BLOCK)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    dy_block_ptr = tl.make_block_ptr(dy_input, shape=(M, N), strides=(dy_stride1, 1), offsets=(row_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    w_block_ptr = tl.make_block_ptr(weight, shape=(N, K), strides=(w_stride1, 1), offsets=(0, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
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

    o_block_ptr = tl.make_block_ptr(out, shape=(M,K), strides=(o_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))
    tl.store(o_block_ptr, acc.to(out.dtype.element_ty), boundary_check=(1,0))


@triton.jit
def fp8gemm_wgrad(dy_input, x_input, weight_grad, M, N, K, dy_stride1, x_stride1, dw_stride1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # dY: [M, N], X: [M, K], dW: [N, K]
    # Both inputs are quantized online with one scale per 32 contraction
    # elements along M.
    MX_BLOCK: tl.constexpr = 32

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)

    pid = tl.program_id(0)
    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    dy_block_ptr = tl.make_block_ptr(dy_input, shape=(M,N), strides=(dy_stride1, 1), offsets=(0, row_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    x_block_ptr = tl.make_block_ptr(x_input, shape=(M,K), strides=(x_stride1, 1), offsets=(0, col_idx * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1,0))

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for i in tl.range(0, M, BLOCK_M):
        dy_block = tl.load(dy_block_ptr, boundary_check=(1,0), padding_option='zero')
        dy_block_scaled, dys_block = online_scale(tl.trans(dy_block))

        x_block = tl.load(x_block_ptr, boundary_check=(1,0), padding_option='zero')
        x_block_scaled, xs_block = online_scale(tl.trans(x_block))
        x_block_scaled = tl.trans(x_block_scaled)

        acc = tl.dot_scaled(dy_block_scaled, dys_block, 'e4m3', x_block_scaled, xs_block, 'e4m3', acc)

        dy_block_ptr = tl.advance(dy_block_ptr, (BLOCK_M, 0))
        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_M, 0))

    o_block_ptr = tl.make_block_ptr(weight_grad, shape=(N,K), strides=(dw_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    tl.store(o_block_ptr, acc.to(weight_grad.dtype.element_ty), boundary_check=(1,0))


@triton.jit
def fp8_row_quantize(weight, weight_out, weight_scale, N, K, w_stride1, wo_stride1, ws_stride1, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    MX_BLOCK: tl.constexpr = 32
    BLOCK_SCALES: tl.constexpr = BLOCK_K // MX_BLOCK

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)
    NUM_SCALE_BLOCKS = tl.cdiv(K, MX_BLOCK)

    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    w_block_ptr = tl.make_block_ptr(weight, shape=(N, K), strides=(w_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    wo_block_ptr = tl.make_block_ptr(weight_out, shape=(N, K), strides=(wo_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    ws_block_ptr = tl.make_block_ptr(weight_scale, shape=(N, NUM_SCALE_BLOCKS), strides=(ws_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_SCALES), block_shape=(BLOCK_N, BLOCK_SCALES), order=(1,0))

    w_block = tl.load(w_block_ptr, boundary_check=(1,0), padding_option='zero').to(tl.bfloat16)
    w_block_scaled, ws_block = online_scale(w_block)

    tl.store(wo_block_ptr, w_block_scaled, boundary_check=(1,0))
    tl.store(ws_block_ptr, ws_block, boundary_check=(1,0))


@triton.jit
def fp8_weight_block_quantize(weight, weight_out, weight_scale, N, K, w_stride1, wo_stride1, ws_stride1, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_KBLOCKS = tl.cdiv(K, BLOCK_K)

    row_idx = pid // NUM_KBLOCKS
    col_idx = pid % NUM_KBLOCKS

    w_block_ptr = tl.make_block_ptr(weight, shape=(N, K), strides=(w_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    wo_block_ptr = tl.make_block_ptr(weight_out, shape=(N, K), strides=(wo_stride1, 1), offsets=(row_idx * BLOCK_N, col_idx * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1,0))
    ws_block_ptr = tl.make_block_ptr(weight_scale, shape=(NUM_NBLOCKS, NUM_KBLOCKS), strides=(ws_stride1, 1), offsets=(row_idx, col_idx), block_shape=(1, 1), order=(1,0))

    w_block = tl.load(w_block_ptr, boundary_check=(1,0), padding_option='zero').to(tl.bfloat16)
    w_block_scaled, ws_block = block_online_scale(w_block)

    tl.store(wo_block_ptr, w_block_scaled, boundary_check=(1,0))
    tl.store(ws_block_ptr, ws_block, boundary_check=(1,0))


def launch_row_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations with the existing 1x32 E8M0 scaling."""
    assert weight.dtype == torch.bfloat16 or weight.dtype == torch.float32
    assert weight.stride(1) == 1
    N, K = weight.shape
    wo_buffer = torch.empty(weight.shape, dtype = torch.float8_e4m3fn, device = weight.device)
    ws_buffer = torch.empty((N, triton.cdiv(K, 32)), dtype = torch.uint8, device = weight.device)

    fp8_row_quantize[lambda meta: (triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(K, meta['BLOCK_K']), )](weight, wo_buffer, ws_buffer, N, K, weight.stride(0), wo_buffer.stride(0), ws_buffer.stride(0), 64, 64)

    return wo_buffer, ws_buffer


def launch_weight_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix with one E8M0 scale per 32x32 block."""
    assert weight.dtype == torch.bfloat16 or weight.dtype == torch.float32
    assert weight.stride(1) == 1
    N, K = weight.shape
    wo_buffer = torch.empty(weight.shape, dtype = torch.float8_e4m3fn, device = weight.device)
    ws_buffer = torch.empty((triton.cdiv(N, 32), triton.cdiv(K, 32)), dtype = torch.uint8, device = weight.device)

    fp8_weight_block_quantize[lambda meta: (triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(K, meta['BLOCK_K']), )](weight, wo_buffer, ws_buffer, N, K, weight.stride(0), wo_buffer.stride(0), ws_buffer.stride(0), 32, 32)

    return wo_buffer, ws_buffer

def launch_fp8gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    ws: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    M, K = x.shape
    N, _ = weight.shape

    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.float8_e4m3fn and ws.dtype == torch.uint8
    assert ws.shape == (triton.cdiv(N, 32), triton.cdiv(K, 32))
    if bias is not None:
        assert bias.dtype == torch.float32
        assert bias.shape == (N,)
        assert bias.device == x.device

    out_buffer = torch.empty((M, N), dtype = torch.bfloat16, device=x.device)
    bias_ptr = bias if bias is not None else ws
    fp8gemm_fwd[lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )](x, weight, ws, bias_ptr, out_buffer, M, N, K, x.stride(0), weight.stride(0), ws.stride(0), out_buffer.stride(0), bias is not None, 64, 64)

    return out_buffer


def launch_fp8gemm_backward(dy: torch.Tensor, weight: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    M, N = dy.shape
    N_weight, K = weight.shape

    assert N == N_weight
    assert dy.dtype == torch.bfloat16
    assert weight.dtype == torch.float8_e4m3fn and ws.dtype == torch.uint8
    assert ws.shape == (triton.cdiv(N, 32), triton.cdiv(K, 32))

    out_buffer = torch.empty((M, K), dtype = torch.bfloat16, device=dy.device)
    fp8gemm_bwd[lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_K']), )](dy, weight, ws, out_buffer, M, N, K, dy.stride(0), weight.stride(0), ws.stride(0), out_buffer.stride(0), 64, 64, 64)

    return out_buffer


def launch_fp8gemm_weight_grad(dy: torch.Tensor, x: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    M, N = dy.shape
    M_x, K = x.shape

    assert M == M_x
    assert dy.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    assert dy.stride(1) == 1 and x.stride(1) == 1
    assert out_dtype in (torch.bfloat16, torch.float32)

    weight_grad = torch.empty((N, K), dtype = out_dtype, device=dy.device)
    fp8gemm_wgrad[lambda meta: (triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(K, meta['BLOCK_K']), )](dy, x, weight_grad, M, N, K, dy.stride(0), x.stride(0), weight_grad.stride(0), 64, 64, 64)

    return weight_grad


def check_quantize(w: torch.Tensor, wo: torch.Tensor, ws: torch.Tensor):
    N, K = w.shape
    back_w = dequantize_weight(wo, ws)
    torch.testing.assert_close(back_w, w.to(torch.float32), rtol=6e-2, atol=1e-2)


def dequantize_row(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    scale_values = torch.pow(2.0, scales.float() - 127)[...,None].expand(-1, -1, 32).reshape(N, -1)[:,0:K]
    return q.to(torch.float32) * scale_values


def dequantize_weight(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    scale_values = torch.pow(2.0, scales.float() - 127)
    scale_values = scale_values.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return q.to(torch.float32) * scale_values[:N, :K]


if __name__ == "__main__":
    torch.manual_seed(53)
    torch.cuda.manual_seed(654)
    torch.set_printoptions(3, 100000, linewidth=100000, sci_mode=False)
    torch.set_default_device('cuda:0')
    x = torch.randn((128, 128), dtype = torch.float32).to(torch.bfloat16)
    w = torch.randn((128, 128), dtype = torch.float32).to(torch.bfloat16)

    wo, ws = launch_weight_quantize(w)

    check_quantize(w, wo, ws)

    o = launch_fp8gemm(x, wo, ws)
    xo, xs = launch_row_quantize(x)
    w_dequant = dequantize_weight(wo, ws)

    o_ref = dequantize_row(xo, xs) @ w_dequant.T
    torch.testing.assert_close(o.to(torch.float32), o_ref, rtol=1e-2, atol=1e-2)

    o_original_ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))
    relative_l2_error = torch.linalg.vector_norm(o.float() - o_original_ref) / torch.linalg.vector_norm(o_original_ref)
    assert relative_l2_error < 5e-2

    dy = torch.randn(o.shape, dtype=torch.float32).to(torch.bfloat16)
    dyo, dys = launch_row_quantize(dy)
    dx = launch_fp8gemm_backward(dy, wo, ws)
    dx_ref = dequantize_row(dyo, dys) @ w_dequant
    torch.testing.assert_close(dx.to(torch.float32), dx_ref, rtol=1e-2, atol=1e-2)

    dx_original_ref = dy.to(torch.float32) @ w.to(torch.float32)
    backward_relative_l2_error = torch.linalg.vector_norm(dx.float() - dx_original_ref) / torch.linalg.vector_norm(dx_original_ref)
    assert backward_relative_l2_error < 5e-2

    dw = launch_fp8gemm_weight_grad(dy, x)
    dyt_q, dyt_s = launch_row_quantize(dy.T.contiguous())
    xt_q, xt_s = launch_row_quantize(x.T.contiguous())
    dw_ref = dequantize_row(dyt_q, dyt_s) @ dequantize_row(xt_q, xt_s).T
    torch.testing.assert_close(dw.to(torch.float32), dw_ref, rtol=1e-2, atol=1e-2)

    dw_original_ref = dy.to(torch.float32).T @ x.to(torch.float32)
    weight_grad_relative_l2_error = torch.linalg.vector_norm(dw.float() - dw_original_ref) / torch.linalg.vector_norm(dw_original_ref)
    assert weight_grad_relative_l2_error < 5e-2

    print(f"fp8 forward check passed, relative L2 error: {relative_l2_error.item():.4f}")
    print(f"fp8 backward check passed, relative L2 error: {backward_relative_l2_error.item():.4f}")
    print(f"fp8 weight-grad check passed, relative L2 error: {weight_grad_relative_l2_error.item():.4f}")
