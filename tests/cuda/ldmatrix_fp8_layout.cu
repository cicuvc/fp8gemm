// Byte-coordinate probe for native SM120 8-bit transposing LDSM.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void probe_ldmatrix_fp8(uint32_t *out, int columns, int swap_rows) {
    __shared__ __align__(16) uint8_t tile[32][16];
    int lane = threadIdx.x;
    for (int c = 0; c < 16; ++c)
        tile[lane][c] = columns ? c : lane;
    __syncthreads();
    int row = swap_rows ? ((lane & 7) | ((lane & 8) << 1) | ((lane & 16) >> 1)) : lane;
    uint32_t addr = __cvta_generic_to_shared(&tile[row][0]);
    uint32_t a, b, c, d;
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
    out[lane*4+0] = a;
    out[lane*4+1] = b;
    out[lane*4+2] = c;
    out[lane*4+3] = d;
}

extern "C" int probe_fp8(void *out, int columns, int swap_rows, void *stream) {
    probe_ldmatrix_fp8<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t*>(out), columns, swap_rows);
    return cudaGetLastError();
}

// Select each K coordinate with an identity A, so the result must reproduce W.
__device__ __forceinline__ uint32_t identity_word(int row, int k, int offset) {
    uint32_t word = 0;
    for (int byte = 0; byte < 4; ++byte)
        if (row + offset == k + byte) word |= uint32_t(0x38) << (8*byte); // E4M3 1.0
    return word;
}

template<bool SwapRows, bool AdjacentPairs>
__global__ void probe_ldmatrix_qmma(const uint8_t *weight, float *out) {
    __shared__ __align__(16) uint8_t tile[32][16];
    int lane = threadIdx.x;
    for (int c = 0; c < 16; ++c) tile[lane][c] = weight[lane*16+c];
    __syncthreads();
    int address_row = SwapRows ? ((lane & 7) | ((lane & 8) << 1) | ((lane & 16) >> 1)) : lane;
    uint32_t addr = __cvta_generic_to_shared(&tile[address_row][0]);
    uint32_t b[4];
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
    int row = lane / 4, k = (lane % 4)*4, offset = blockIdx.x*16;
    uint32_t a0 = identity_word(row, k, offset);
    uint32_t a1 = identity_word(row+8, k, offset);
    uint32_t a2 = identity_word(row, k+16, offset);
    uint32_t a3 = identity_word(row+8, k+16, offset);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float d0, d1, d2, d3;
        uint32_t b0 = b[AdjacentPairs ? 2*half : half];
        uint32_t b1 = b[AdjacentPairs ? 2*half+1 : half+2];
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, "
            "%11, {0,0}, %11, {0,0};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(0.0f), "r"(127));
        int col = half*8 + 2*(lane%4);
        out[(offset+row)*16+col] = d0;
        out[(offset+row)*16+col+1] = d1;
        out[(offset+row+8)*16+col] = d2;
        out[(offset+row+8)*16+col+1] = d3;
    }
}

extern "C" int probe_fp8_qmma(void *weight, void *out, int mode, void *stream) {
    auto s = static_cast<cudaStream_t>(stream);
    auto w = static_cast<const uint8_t*>(weight);
    auto d = static_cast<float*>(out);
    switch (mode) {
        case 0: probe_ldmatrix_qmma<false,false><<<2,32,0,s>>>(w,d); break;
        case 1: probe_ldmatrix_qmma<false,true><<<2,32,0,s>>>(w,d); break;
        case 2: probe_ldmatrix_qmma<true,false><<<2,32,0,s>>>(w,d); break;
        case 3: probe_ldmatrix_qmma<true,true><<<2,32,0,s>>>(w,d); break;
        default: return int(cudaErrorInvalidValue);
    }
    return cudaGetLastError();
}
