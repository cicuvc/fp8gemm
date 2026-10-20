// One CTA, four warps: compare plain and block-scaled FP8 instruction rates
// using SM clock cycles, independently of GPU boost and global-memory traffic.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(expr) do { auto e=(expr); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1); } } while(0)
template<bool Scaled>
__global__ void rate(unsigned long long *cycles,float *result,int iterations) {
    float d[8][4]={};
    const unsigned a=0x38383838u,s=0x7f7f7f7fu;
    __syncthreads();
    auto start=clock64();
    for(int k=0;k<iterations;k++) {
#pragma unroll
        for(int i=0;i<8;i++) {
            if constexpr(Scaled) {
                asm volatile(
                    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                    "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3},{%5},{%6,%6},{%5},{%6,%6};"
                    : "+f"(d[i][0]),"+f"(d[i][1]),"+f"(d[i][2]),"+f"(d[i][3])
                    : "r"(a),"r"(s),"h"((unsigned short)0));
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3};"
                    : "+f"(d[i][0]),"+f"(d[i][1]),"+f"(d[i][2]),"+f"(d[i][3])
                    : "r"(a));
            }
        }
    }
    auto elapsed=clock64()-start;
    float sum=0;
#pragma unroll
    for(int i=0;i<8;i++)for(int j=0;j<4;j++)sum+=d[i][j];
    cycles[threadIdx.x]=elapsed;result[threadIdx.x]=sum;
}
int main() {
    constexpr int iterations=4096;
    unsigned long long *cycles,hcycles[128];float *result,hresult[128];
    CHECK(cudaMalloc(&cycles,sizeof(hcycles)));CHECK(cudaMalloc(&result,sizeof(hresult)));
    printf("{\"iterations\":%d,\"warps\":4,\"independent_accumulators_per_warp\":8,\"rounds\":[",iterations);
    for(int round=0;round<5;round++)for(int scaled=0;scaled<2;scaled++) {
        if(scaled)rate<true><<<1,128>>>(cycles,result,iterations);
        else rate<false><<<1,128>>>(cycles,result,iterations);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(hcycles,cycles,sizeof(hcycles),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hresult,result,sizeof(hresult),cudaMemcpyDeviceToHost));
        for(float x:hresult)if(x!=float(iterations*32*32))return 2;
        printf("%s{\"round\":%d,\"mode\":\"%s\",\"cycles_per_mma_per_warp\":[",round||scaled?",":"",round,scaled?"block_scale":"plain");
        for(int w=0;w<4;w++)printf("%s%.6f",w?",":"",double(hcycles[w*32])/(iterations*8));
        printf("]}");
    }
    printf("]}\n");CHECK(cudaFree(cycles));CHECK(cudaFree(result));
}
