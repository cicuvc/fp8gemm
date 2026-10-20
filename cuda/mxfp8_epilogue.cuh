#pragma once
#include "kittens.cuh"

namespace fp8gemm {
// Next GEMM's A-scale layout: uint32 [ceil(N/128), roundup(M,16)].
// Complete groups use word stores; C32/C64 tiles own disjoint scale bytes.
template<int R,int C>
__device__ __forceinline__ void store_output_scales(
    uint32_t *dst,const kt::mxfp8_scale<R,C> &scales,int mr,int nc,int m,int n) {
    static_assert(C%32==0);
    int lane=kt::laneid(),mp=(m+15)/16*16;
    if((lane&3)<2) {
#pragma unroll
        for(int i=0;i<R/16;i++) {
            int row=mr+i*16+lane/4+8*(lane&1);
            if(row<mp) {
                if constexpr(C%128==0) {
#pragma unroll
                    for(int j=0;j<C/128;j++) {
                        uint32_t word=0;
#pragma unroll
                        for(int k=0;k<4;k++)word|=(scales.data[i][4*j+k]&255u)<<(8*k);
                        if(nc/128+j<(n+127)/128)
                            dst[size_t(nc/128+j)*mp+row]=word;
                    }
                } else {
                    auto *bytes=reinterpret_cast<uint8_t*>(dst);
#pragma unroll
                    for(int j=0;j<C/32;j++) {
                        int block=nc/32+j;
                        if(block<(n+31)/32)
                            bytes[(size_t(block/4)*mp+row)*4+block%4]=scales.data[i][j];
                    }
                    // Deterministic zero padding, including reused graph outputs.
                    if(nc+C>=n) {
                        for(int block=(n+31)/32;block<(n+127)/128*4;block++)
                            bytes[(size_t(block/4)*mp+row)*4+block%4]=0;
                    }
                }
            }
        }
    }
}
} // namespace fp8gemm
