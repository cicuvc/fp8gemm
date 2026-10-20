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
