#pragma once

#include "kittens.cuh"

namespace fp8gemm {

// Round the accumulator to the exact BF16 logits consumed by CE, then reduce
// each warp's complete rows. No inter-warp communication or logits reload.
template<int R, int C>
__device__ __forceinline__ void store_partial_lse(
    kt::rt_fl<R,C> &z, float *lse, int mr, int nc, int m, int n) {
    const int lane = kt::laneid(), parts = (n+C-1)/C;
    __syncwarp();
#pragma unroll
    for (int i=0; i<R/16; ++i) {
        float maxima[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j=0; j<C/16; ++j) {
#pragma unroll
            for (int p=0; p<4; ++p) {
                auto &v = z.tiles[i][j].data[p];
                v.x = __bfloat162float(__float2bfloat16_rn(v.x));
                v.y = __bfloat162float(__float2bfloat16_rn(v.y));
                int col = nc+j*16+(lane%4)*2+(p/2)*8;
                maxima[p%2] = fmaxf(maxima[p%2], col<n ? v.x : -INFINITY);
                maxima[p%2] = fmaxf(maxima[p%2], col+1<n ? v.y : -INFINITY);
            }
        }
#pragma unroll
        for (int r=0; r<2; ++r) {
            maxima[r] = fmaxf(maxima[r], __shfl_xor_sync(0xffffffff, maxima[r], 1));
            maxima[r] = fmaxf(maxima[r], __shfl_xor_sync(0xffffffff, maxima[r], 2));
        }
        float sums[2] = {0.f, 0.f};
#pragma unroll
        for (int j=0; j<C/16; ++j) {
#pragma unroll
            for (int p=0; p<4; ++p) {
                const auto &v = z.tiles[i][j].data[p];
                int col = nc+j*16+(lane%4)*2+(p/2)*8;
                sums[p%2] += col<n ? __expf(v.x-maxima[p%2]) : 0.f;
                sums[p%2] += col+1<n ? __expf(v.y-maxima[p%2]) : 0.f;
            }
        }
#pragma unroll
        for (int r=0; r<2; ++r) {
            sums[r] += __shfl_xor_sync(0xffffffff, sums[r], 1);
            sums[r] += __shfl_xor_sync(0xffffffff, sums[r], 2);
            int row = mr+i*16+lane/4+r*8;
            if (lane%4==0 && row<m)
                lse[size_t(row)*parts+nc/C] = maxima[r]+__logf(sums[r]);
        }
    }
}
} // namespace fp8gemm
