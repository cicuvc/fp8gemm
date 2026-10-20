#include "fp8_ldmatrix_probe.cuh"
#include <stdexcept>
namespace kt = kittens;

// Kind 0: existing FP8; 1: factored FP8; 2: existing BF16 with identical bytes.
template<int Rows, int KBytes, int GW, bool Swap, int Kind>
__global__ void probe_load(uint32_t *output, unsigned long long *cycles,
                          int iterations, int base_offset) {
    using T = std::conditional_t<Kind == 2, kt::bf16, kt::fp8e4m3>;
    using ST = kt::st<T, GW * Rows, KBytes / sizeof(T)>;
    using RT = kt::rt<T, Rows, KBytes / sizeof(T)>;
    extern __shared__ __align__(1024) unsigned char storage[];
    auto *tiles = reinterpret_cast<ST *>(storage + base_offset);
    // Fill physical shared bytes with an arbitrary, position-dependent pattern.
    auto *words = reinterpret_cast<uint32_t *>(tiles);
    for (int i = threadIdx.x; i < 2 * sizeof(ST) / 4; i += blockDim.x)
        words[i] = (uint32_t(i) * 1664525u + 1013904223u) ^ (uint32_t(i) >> 5);
    __syncthreads();
    RT reg;
    auto start = clock64();
#pragma unroll 1
    for (int it = 0; it < iterations; ++it) {
        const ST &tile = tiles[(it + blockIdx.x) & 1];
        if constexpr (Kind == 1) fp8gemm::probe::load_factored<GW, Swap>(reg, tile);
        else kt::group<GW>::template load<Swap>(reg, tile);
    }
    auto end = clock64();
    constexpr int words_per_thread = RT::height * RT::width * 4;
#pragma unroll
    for (int i = 0; i < RT::height; ++i) {
#pragma unroll
        for (int j = 0; j < RT::width; ++j) {
#pragma unroll
            for (int d = 0; d < 4; ++d)
                output[threadIdx.x * words_per_thread + (i * RT::width + j) * 4 + d]
                    = *reinterpret_cast<const uint32_t *>(&reg.tiles[i][j].data[d]);
        }
    }
    cycles[threadIdx.x] = end - start;
}

template<int R, int K, int GW, bool Swap, int Kind>
void launch(uint32_t *out, unsigned long long *cycles, int iters, int offset, cudaStream_t stream) {
    using T = std::conditional_t<Kind == 2, kt::bf16, kt::fp8e4m3>;
    using ST = kt::st<T, GW * R, K / sizeof(T)>;
    if (offset % ST::required_alignment) throw std::runtime_error("Unaligned shared base");
    static const cudaError_t attribute = cudaFuncSetAttribute(
        probe_load<R,K,GW,Swap,Kind>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096+2*sizeof(ST));
    if (attribute != cudaSuccess) throw std::runtime_error(cudaGetErrorString(attribute));
    constexpr int threads = GW == 1 ? 32 : GW * 32;
    probe_load<R,K,GW,Swap,Kind><<<1,threads,offset+2*sizeof(ST),stream>>>(out,cycles,iters,offset);
}
static thread_local const char *error = "";
extern "C" const char *ldsm_probe_error() { return error; }
extern "C" int ldsm_probe(int rows,int k,int gw,int swap,int kind,int iters,int offset,
                          uint32_t *out,unsigned long long *cycles,cudaStream_t stream) {
    try {
        if(iters<1 || offset<0 || offset>4096) throw std::runtime_error("Invalid iteration count or offset");
#define KIND(R,K,G,S) do { \
    if(kind==0) launch<R,K,G,S,0>(out,cycles,iters,offset,stream); \
    else if(kind==1) launch<R,K,G,S,1>(out,cycles,iters,offset,stream); \
    else if(kind==2) launch<R,K,G,S,2>(out,cycles,iters,offset,stream); \
    else throw std::runtime_error("Invalid kind"); \
} while(0)
#define CASE(R,K,G,S) if(rows==R && k==K && gw==G && swap==S) { KIND(R,K,G,S); } else
        CASE(16,32,4,false)
        CASE(16,64,4,false)
        CASE(16,128,4,false)
        CASE(32,64,4,false)
        CASE(32,128,4,false)
        CASE(32,256,4,false)
        CASE(32,32,1,true)
        CASE(64,32,1,true)
        CASE(64,64,1,true)
        CASE(128,64,1,true)
        CASE(128,128,1,true)
        CASE(64,256,1,true)
        { throw std::runtime_error("Invalid configuration"); }
#undef CASE
#undef KIND
        auto e=cudaGetLastError();
        if(e!=cudaSuccess){error=cudaGetErrorString(e);return 1;}
        return 0;
    } catch(const std::exception &) { error="Invalid probe arguments";return 1; }
}
