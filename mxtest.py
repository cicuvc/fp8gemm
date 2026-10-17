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
def fp8gemm_fwd(x_input, weight, weight_scale, out, M, N, K, x_stride1, w_stride1, ws_stride1, o_stride1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # x_input: bf16, weight: fp8, weight_scale: e8m0 uint8
    # x: [M, K], w: [N, K], ws: [N, BLK]
    PRESET_GROUP_SIZE: tl.constexpr = 4
    BLOCK_K: tl.constexpr = 64
    MX_BLOCK: tl.constexpr = 32
    BLOCK_SCALES: tl.constexpr = BLOCK_K // MX_BLOCK

    NUM_NBLOCKS = tl.cdiv(N, BLOCK_N)
    NUM_MBLOCKS = tl.cdiv(M, BLOCK_M)
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
    ws_block_ptr = tl.make_block_ptr(weight_scale, shape=(N, NUM_SCALE_BLOCKS), strides=(ws_stride1, 1), offsets=(col_idx * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_SCALES), order=(1,0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, K, BLOCK_K):
        x_block = tl.load(x_block_ptr, boundary_check=(1,0), padding_option='zero')
        x_block_scaled, xs_block = online_scale(x_block)

        w_block = tl.load(w_block_ptr, boundary_check=(1,0), padding_option='zero')
        ws_block = tl.load(ws_block_ptr, boundary_check=(1,0), padding_option='zero')

        acc = tl.dot_scaled(x_block_scaled, xs_block, 'e4m3', tl.trans(w_block), ws_block, 'e4m3', acc)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (0, BLOCK_K))
        ws_block_ptr = tl.advance(ws_block_ptr, (0, BLOCK_SCALES))

    o_block_ptr = tl.make_block_ptr(out, shape=(M,N), strides=(o_stride1, 1), offsets=(row_idx * BLOCK_M, col_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1,0))
    tl.store(o_block_ptr, acc.to(out.dtype.element_ty), boundary_check=(1,0))


@triton.jit
def fp8_weight_quantize(weight, weight_out, weight_scale, N, K, w_stride1, wo_stride1, ws_stride1, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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

def launch_weight_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight.dtype == torch.bfloat16 or weight.dtype == torch.float32
    assert weight.stride(1) == 1
    N, K = weight.shape
    wo_buffer = torch.empty(weight.shape, dtype = torch.float8_e4m3fn, device = weight.device)
    ws_buffer = torch.empty((N, triton.cdiv(K, 32)), dtype = torch.uint8, device = weight.device)

    fp8_weight_quantize[lambda meta: (triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(K, meta['BLOCK_K']), )](weight, wo_buffer, ws_buffer, N, K, weight.stride(0), wo_buffer.stride(0), ws_buffer.stride(0), 64, 64)

    return wo_buffer, ws_buffer

def launch_fp8gemm(x: torch.Tensor, weight: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N, _ = weight.shape

    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.float8_e4m3fn and ws.dtype == torch.uint8

    out_buffer = torch.empty((M, N), dtype = torch.bfloat16, device=x.device)
    fp8gemm_fwd[lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )](x, weight, ws, out_buffer, M, N, K, x.stride(0), weight.stride(0), ws.stride(0), out_buffer.stride(0), 64, 64)

    return out_buffer

def check_quantize(w: torch.Tensor, wo: torch.Tensor, ws: torch.Tensor):
    N, K = w.shape

    scales = torch.pow(2.0, ws - 127)[...,None].expand((-1,-1,32)).reshape(N, -1)[:,0:K]
    back_w = wo.to(torch.float32) * scales
    torch.testing.assert_close(back_w, w.to(torch.float32), rtol=6e-2, atol=1e-2)

def dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    scale_values = torch.pow(2.0, scales.float() - 127)[...,None].expand(-1, -1, 32).reshape(N, -1)[:,0:K]
    return q.to(torch.float32) * scale_values
    

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
    xo, xs = launch_weight_quantize(x)

    o_ref = dequantize(xo, xs) @ dequantize(wo, ws).T
    torch.testing.assert_close(o.to(torch.float32), o_ref, rtol=1e-2, atol=1e-2)

    o_original_ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))
    relative_l2_error = torch.linalg.vector_norm(o.float() - o_original_ref) / torch.linalg.vector_norm(o_original_ref)
    assert relative_l2_error < 5e-2

    print(f"fp8 gemm check passed, relative L2 error: {relative_l2_error.item():.4f}")

    
