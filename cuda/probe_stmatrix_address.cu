#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "mxfp8_epilogue.cuh"
#include <stdexcept>
namespace kt=kittens;

// Kinds: BF16, native FP8, MOVM-layout FP8, FP32 -> BF16.
template<int R,int C,int GW,int Kind,bool Factored>
__global__ void probe_store(uint32_t *out,unsigned long long *cycles,int iterations,int offset) {
    using T=std::conditional_t<Kind==0 || Kind==3,kt::bf16,kt::fp8e4m3>;
    using RT=kt::rt<std::conditional_t<Kind==3,float,T>,R,C>;
    using ST=kt::st<T,GW*R,C>;
    extern __shared__ __align__(1024) unsigned char storage[];
    auto *tiles=reinterpret_cast<ST*>(storage+offset);
    auto *words=reinterpret_cast<uint32_t*>(tiles);
    for(int i=threadIdx.x;i<2*sizeof(ST)/4;i+=blockDim.x)words[i]=0xcccccccc;
    RT src;
#pragma unroll
    for(int i=0;i<RT::height;i++) {
#pragma unroll
        for(int j=0;j<RT::width;j++) {
#pragma unroll
            for(int p=0;p<4;p++) {
                uint32_t id=threadIdx.x*RT::height*RT::width*4+(i*RT::width+j)*4+p;
                if constexpr(Kind==3)src.tiles[i][j].data[p]=make_float2(float(id%2048)/16,float(id%1024)/32);
                else {
                    uint32_t bits=(id*1664525u+1013904223u)^(id>>5);
                    __builtin_memcpy(&src.tiles[i][j].data[p],&bits,4);
                }
            }
        }
    }
    fp8gemm::MXOutputTransposed<R,(C+31)/32*32> transposed;
    if constexpr(Kind==2) {
#pragma unroll
        for(int i=0;i<R/16;i++) {
#pragma unroll
            for(int j=0;j<C/32;j++) {
#pragma unroll
                for(int p=0;p<4;p++)
                    transposed.data[i][j][p]=std::bit_cast<uint32_t>(src.tiles[i][j].data[p]);
            }
        }
    }
    __syncthreads();
    auto begin=clock64();
#pragma unroll 1
    for(int it=0;it<iterations;it++) {
        ST &tile=tiles[it&1];
        if constexpr(Kind==2)fp8gemm::store_output_movmatrix<GW,Factored>(tile,transposed);
        else if constexpr(Factored)fp8gemm::store_factored<GW>(tile,src);
        else kt::group<GW>::store(tile,src);
    }
    auto end=clock64();
    __syncthreads();
    for(int i=threadIdx.x;i<2*sizeof(ST)/4;i+=blockDim.x)out[i]=words[i];
    cycles[threadIdx.x]=end-begin;
}
template<int R,int C,int GW,int Kind,bool F>
void launch(uint32_t *o,unsigned long long *cy,int it,int off,cudaStream_t stream) {
    using T=std::conditional_t<Kind==0 || Kind==3,kt::bf16,kt::fp8e4m3>;
    using ST=kt::st<T,GW*R,C>;
    if(off%ST::required_alignment)throw std::runtime_error("Unaligned shared base");
    auto e=cudaFuncSetAttribute(probe_store<R,C,GW,Kind,F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,4096+2*sizeof(ST));
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    probe_store<R,C,GW,Kind,F><<<1,32*GW,off+2*sizeof(ST),stream>>>(o,cy,it,off);
}
static thread_local std::string error;
extern "C" const char *stsm_probe_error(){return error.c_str();}
extern "C" int stsm_probe(int r,int c,int gw,int kind,int factored,int it,int off,
                         uint32_t *o,unsigned long long *cy,cudaStream_t stream) {
    try {
        if(it<1||off<0||off>4096)throw std::runtime_error("Invalid iterations/offset");
#define RUN(R,C,G,K) if(factored)launch<R,C,G,K,true>(o,cy,it,off,stream);else launch<R,C,G,K,false>(o,cy,it,off,stream)
#define KIND(R,C,G) switch(kind){case 0:RUN(R,C,G,0);break;case 1:RUN(R,C,G,1);break;case 2:RUN(R,C,G,2);break;case 3:RUN(R,C,G,3);break;default:throw std::runtime_error("Invalid kind");}
#define CASE(R,C,G) if(r==R && c==C && gw==G){KIND(R,C,G)}else
        CASE(16,32,1)
        CASE(16,64,1)
        CASE(16,128,1)
        CASE(16,32,4)
        CASE(16,64,4)
        CASE(16,128,4)
        CASE(16,256,4)
        CASE(32,64,4)
        CASE(32,128,4)
        CASE(64,32,4)
        CASE(64,64,4)
        {throw std::runtime_error("Invalid shape");}
#undef CASE
#undef KIND
#undef RUN
        auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
        return 0;
    }catch(const std::exception &e){error=e.what();return 1;}
}
