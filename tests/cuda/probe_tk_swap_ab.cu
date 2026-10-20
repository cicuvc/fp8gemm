// One warp / one small GEMM per CTA. No TMA or persistent pipeline.
// Original tensors: dy[M,N], w[N,K]. Swapped computation: w^T[K,N] @ dy^T[N,M].
#include "kittens.cuh"
namespace kt = kittens;

template<int Rows, int Cols>
__device__ void dump_float(float *dst, const kt::rt_fl<Rows,Cols> &src) {
    int lane = threadIdx.x;
#pragma unroll
    for (int i=0; i<Rows/16; ++i)
#pragma unroll
        for (int j=0; j<Cols/16; ++j)
#pragma unroll
            for (int p=0; p<4; ++p) {
                int r=i*16+lane/4+8*(p%2), c=j*16+2*(lane%4)+8*(p/2);
                *reinterpret_cast<float2*>(dst+r*Cols+c)=src.tiles[i][j].data[p];
            }
}

template<int M, int N, int K, bool SwapAB>
__global__ void swap_ab_probe(const kt::bf16 *dy, const kt::fp8e4m3 *w,
                             const uint8_t *ws, float *raw, kt::bf16 *dx) {
    __shared__ kt::st_bf<M,N> sy;
    __shared__ kt::st_fp8e4m3<N,K> sw;
    __shared__ kt::st_bf<M,K> so;
    int lane=threadIdx.x, batch=blockIdx.x;
    dy+=batch*M*N; w+=batch*N*K; ws+=batch*(N/32)*(K/32);
    raw+=batch*M*K; dx+=batch*M*K;
    // Ordinary cooperative global loads intentionally isolate fragment layout.
    for (int p=lane; p<M*N; p+=32) sy[{p/N,p%N}]=dy[p];
    for (int p=lane; p<N*K; p+=32) sw[{p/K,p%K}]=w[p];
    __syncthreads();

    kt::rt_bf<M,N> yr;
    kt::rt_fp8e4m3<M,N> yq;
    kt::mxfp8_scale<M,N> ys;
    kt::warp::load(yr,sy);
    kt::online_quantize_mxfp8(yq,ys,yr);
    kt::rt<kt::fp8e4m3,N,K,kt::ducks::rt_layout::col> wr;
    kt::warp::load(wr,sw); // LDSM.8.MT1616.2: four consecutive registers.
    kt::mxfp8_scale<K,N> weights;
#pragma unroll
    for (int i=0; i<K/16; ++i)
#pragma unroll
        for (int j=0; j<N/32; ++j)
            weights.data[i][j]=ws[j*(K/32)+i/2];

    if constexpr (SwapAB) {
        kt::rt_fp8e4m3<K,N> wa;
        // Logical transpose: only exchange base-tile indices. The four words
        // already have the native A order; no byte/lane permutation.
#pragma unroll
        for (int i=0; i<K/16; ++i)
#pragma unroll
            for (int j=0; j<N/32; ++j)
#pragma unroll
                for (int p=0; p<4; ++p)
                    wa.tiles[i][j].data[p]=wr.tiles[j][i].data[p];
        kt::rt_fl<K,M> acc;
        kt::warp::rt_maps::zero(acc);
        kt::warp::wmma::mma_ABt_block_scaled(acc,wa,yq,acc,weights,ys);
        dump_float(raw,acc); // FP32 result in [K,M] for the independent oracle.

        // Reinterpret the logical transpose as col-layout registers; store
        // converts FP32 pairs to BF16 and uses STSM trans into original [M,K].
        kt::rt<float,M,K,kt::ducks::rt_layout::col> transposed;
#pragma unroll
        for (int i=0; i<K/16; ++i)
#pragma unroll
            for (int j=0; j<M/16; ++j)
#pragma unroll
                for (int p=0; p<4; ++p)
                    transposed.tiles[j][i].data[p]=acc.tiles[i][j].data[p];
        kt::warp::store(so,transposed);
    } else {
        kt::rt_fl<M,K> acc;
        kt::warp::rt_maps::zero(acc);
        kt::warp::wmma::mma_AB(acc,yq,wr,acc,ys,weights);
        dump_float(raw,acc);
        kt::warp::store(so,acc);
    }
    __syncthreads();
    for (int p=lane; p<M*K; p+=32) dx[p]=so[{p/K,p%K}];
}

extern "C" int probe_swap_ab(void *dy, void *w, void *ws, void *raw, void *dx,
                           int config, int swap, int batches, void *stream) {
    auto s=static_cast<cudaStream_t>(stream);
#define LAUNCH(M,N,K,S) swap_ab_probe<M,N,K,S><<<batches,32,0,s>>>( \
    static_cast<kt::bf16*>(dy),static_cast<kt::fp8e4m3*>(w),static_cast<uint8_t*>(ws), \
    static_cast<float*>(raw),static_cast<kt::bf16*>(dx))
#define CASE(ID,M,N,K) case ID: if(swap) { LAUNCH(M,N,K,true); } \
    else { LAUNCH(M,N,K,false); } break
    switch(config) {
        CASE(0,16,32,32);
        CASE(1,32,64,32);
        CASE(2,16,96,64);
        CASE(3,64,128,32);
        CASE(4,32,64,64);
        default: return int(cudaErrorInvalidValue);
    }
#undef CASE
#undef LAUNCH
    return int(cudaGetLastError());
}
