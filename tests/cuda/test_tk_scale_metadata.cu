// Isolated comparison of shift/extract versus BF16x2 -> E8M0x2 conversion.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(c) do { auto e=(c); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#c,cudaGetErrorString(e)); std::exit(2); } } while(0)

template<int Mode>
__global__ void metadata(const uint32_t *src, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    uint32_t e=src[i];
    if constexpr(Mode==0) {
        out[i]=(e >> (7+16*(threadIdx.x&1))) & 255;
    } else {
        uint16_t packed;
        if constexpr(Mode!=2)
            asm("cvt.rz.ue8m0x2.bf16x2 %0, %1;" : "=h"(packed) : "r"(e));
        else
            asm("cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1;" : "=h"(packed) : "r"(e));
        if constexpr(Mode==3)
            out[i]=__byte_perm(uint32_t(packed),0,0x4440u | (threadIdx.x&1));
        else
            out[i]=(uint32_t(packed) >> (8*(threadIdx.x&1))) & 255;
    }
}

__global__ void reverse(uint32_t *out) {
    int i=threadIdx.x;
    uint16_t packed=uint16_t(i | (i<<8));
    uint32_t v;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(v) : "h"(packed));
    out[i]=v;
}

int main() {
    // Every pair of exponent bytes; repeat twice so both lane parities select
    // both rows, including zero and the Inf (255) exponent field.
    constexpr int n=256*256*2;
    std::vector<uint32_t> src(n), got(n), shift(n);
    for(int i=0;i<n;++i) src[i]=uint32_t((i/2)%256)<<7 | uint32_t((i/2)/256)<<23;
    uint32_t *dx,*dy; CHECK(cudaMalloc(&dx,n*4)); CHECK(cudaMalloc(&dy,n*4));
    CHECK(cudaMemcpy(dx,src.data(),n*4,cudaMemcpyHostToDevice));
    metadata<0><<<(n+127)/128,128>>>(dx,dy,n);
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(shift.data(),dy,n*4,cudaMemcpyDeviceToHost));
    for(int mode=1;mode<=3;++mode) {
        if(mode==1) metadata<1><<<(n+127)/128,128>>>(dx,dy,n);
        else if(mode==2) metadata<2><<<(n+127)/128,128>>>(dx,dy,n);
        else metadata<3><<<(n+127)/128,128>>>(dx,dy,n);
        CHECK(cudaGetLastError()); CHECK(cudaMemcpy(got.data(),dy,n*4,cudaMemcpyDeviceToHost));
        int finite_diffs=0,inf_diffs=0;
        for(int i=0;i<n;++i) {
            uint32_t expected=uint32_t((i/2) >> (8*(i&1))) & 255;
            if(shift[i]!=expected || got[i]!=(mode==2 && expected==255 ? 254 : expected)) {
                std::fprintf(stderr,"FAIL mode %d i %d shift %u cvt %u expected %u\n",mode,i,shift[i],got[i],expected);
                return 1;
            }
            if(got[i]!=shift[i]) {if(expected<255) ++finite_diffs; else ++inf_diffs;}
        }
        std::fprintf(stdout,"PASS mode=%d values=%d finite_mismatches=%d exponent255_mismatches=%d\n",mode,n,finite_diffs,inf_diffs);
    }
    reverse<<<1,256>>>(dy); CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(got.data(),dy,256*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<255;++i) {
        uint32_t h=i ? uint32_t(i)<<7 : 0x40u;
        if(got[i]!=(h | h<<16)) return 1;
    }
    std::fprintf(stdout,"PASS reverse finite=255; ue8m0=0 -> bf16x2=0x%08x (2^-127, not zero)\n",got[0]);
    CHECK(cudaFree(dx)); CHECK(cudaFree(dy));
}
