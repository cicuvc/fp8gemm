// SM120 forward FP8 GEMM with FP32 1x128 / 128x128 dequantization scales.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm {
template<class Tile> using FP8Global = kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

template<int C, int S, class Output> struct FP8Config {
    static_assert(C == 64 || C == 128);
    using ATile = kt::st_fp8e4m3<64,128>;
    using BTile = kt::st_fp8e4m3<C,128>;
    struct Packet { ATile a[2]; BTile b; };
    struct Shared { Packet input[S]; };
    struct Args {
        FP8Global<ATile> a;
        FP8Global<BTile> b;
        const float *sa, *sb;
        Output *out;
        int m, n, ktiles, ntiles, mtiles, total, raster_group;
    };
    static constexpr int smem = 1024 + sizeof(Shared);
    static_assert(smem <= 101376);
};

__device__ __forceinline__ int2 fp8_raster(int task, int nt, int mt, int g) {
    // Keep one formula even for g=1 so ptxas retains uniform task arithmetic.
    int span = g * nt, first = task / span * g;
    int rows = min(g,mt-first), local = task % span;
    return {first + local % rows, local / rows};
}

template<class Output>
__device__ __forceinline__ void fp8_store_pair(Output *p, float2 v) {
    if constexpr(std::is_same_v<Output,float>) *reinterpret_cast<float2*>(p) = v;
    else *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v.x,v.y);
}

template<int C, int S, class Output>
__global__ __launch_bounds__(384,1) void persistent_fp8(
    const __grid_constant__ typename FP8Config<C,S,Output>::Args args) {
    using CF = FP8Config<C,S,Output>;
    using Shared = typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group = kt::warpgroup::groupid(), warp = kt::warpgroup::warpid();
    int lane = kt::laneid();
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto &shared = allocator.allocate<Shared>();
    if (group == 2) {
        kt::warpgroup::decrease_registers<40>();
        if (warp == 0) {
            bool leader = kt::warp::elect_leader();
            int last = 0;
            for (int task=blockIdx.x; task<args.total; task+=gridDim.x) {
                int2 coord = fp8_raster(task,args.ntiles,args.mtiles,args.raster_group);
                for (int g=0; g<args.ktiles; ++g) {
                    auto packet = pipe.waitBuffer(0,shared.input);
                    auto &tile = packet.template get<0>();
                    if (leader) {
                        auto &bar = packet.getBarrier();
                        kt::tma::expect_bytes(bar,(128+C)*128);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,coord.x*128,g*128},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,coord.x*128+64,g*128},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,coord.y*C,g*128},bar);
                    }
                    last = packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        for (int task=blockIdx.x; task<args.total; task+=gridDim.x) {
            int2 coord = fp8_raster(task,args.ntiles,args.mtiles,args.raster_group);
            int mr = coord.x*128 + group*64 + warp*16;
            int nc = coord.y*C;
            int row0 = mr + lane/4, row1 = row0+8;
            kt::rt_fl<16,C> accum;
            kt::warp::rt_maps::zero(accum);
            for (int g=0; g<args.ktiles; ++g) {
                // Values are dequantization multipliers, not inverse scales.
                float sb = args.sb[size_t(nc/128)*args.ktiles+g];
                float s0 = row0<args.m ? args.sa[size_t(row0)*args.ktiles+g]*sb : 0.f;
                float s1 = row1<args.m ? args.sa[size_t(row1)*args.ktiles+g]*sb : 0.f;
                auto packet = pipe.waitBuffer(2,shared.input);
                auto &tile = packet.template get<0>();
                kt::rt_fl<16,C> partial;
                kt::warp::rt_maps::zero(partial);
#pragma unroll
                for (int k32=0; k32<4; ++k32) {
                    kt::rt_fp8e4m3<16,32> ar;
                    kt::rt_fp8e4m3<C,32> br;
                    kt::mxfp8_scale<16,32> as;
                    kt::mxfp8_scale<C,32> bs;
                    load_fp8_factored<4,false>(ar,tile.a[group],k32);
                    load_fp8_factored<1,true>(br,tile.b,k32);
                    // E8M0 byte 127 encodes 1.0 for every selected lane.
                    // External FP32 scales are applied only after K128.
#pragma unroll
                    for (int j=0; j<C/16; ++j)
                        bs.data[j][0]=127u;
                    as.data[0][0]=127u;
                    kt::warp::wmma::mma_ABt_block_scaled(partial,ar,br,partial,as,bs);
                }
                packet.submitToNextAndTrigger();
                pipe.moveNext();
#pragma unroll
                for (int j=0; j<C/16; ++j) {
#pragma unroll
                    for (int p=0; p<4; ++p) {
                        auto &d = accum.tiles[0][j].data[p];
                        auto v = partial.tiles[0][j].data[p];
                        float s = (p&1) ? s1 : s0;
                        d.x = fmaf(v.x,s,d.x);
                        d.y = fmaf(v.y,s,d.y);
                    }
                }
            }
            // Direct stores leave room for three input stages even with FP32
            // output. N is a multiple of 128, so only M needs a tail predicate.
#pragma unroll
            for (int j=0; j<C/16; ++j) {
#pragma unroll
                for (int p=0; p<4; ++p) {
                    int row = (p&1) ? row1 : row0;
                    int col = nc+j*16+(p/2)*8+(lane%4)*2;
                    if (row<args.m)
                        fp8_store_pair(args.out+size_t(row)*args.n+col,accum.tiles[0][j].data[p]);
                }
            }
        }
    }
    kt::warpgroup::sync(group+1);
}

static void fp8_check(cudaError_t e) {
    if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct FP8PlanBase {
    virtual ~FP8PlanBase() = default;
    virtual void launch(cudaStream_t) = 0;
    virtual void info(int*) = 0;
};
template<int C, int S, class Output> struct FP8Plan : FP8PlanBase {
    using CF = FP8Config<C,S,Output>;
    typename CF::Args args;
    int grid, regs, local_bytes;
    FP8Plan(void *a,void *b,void *sa,void *sb,void *out,int m,int n,int k,int sms,int raster)
        : args{FP8Global<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,k),
               FP8Global<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,k),
               (float*)sa,(float*)sb,(Output*)out,m,n,k/128,n/C,(m+127)/128,
               ((m+127)/128)*(n/C),std::min(raster,(m+127)/128)} {
        int device;
        fp8_check(cudaGetDevice(&device));
        cudaDeviceProp prop;
        fp8_check(cudaGetDeviceProperties(&prop,device));
        if (prop.major!=12 || prop.minor!=0) throw std::runtime_error("FP8 GEMM requires SM120");
        grid = std::min(args.total,sms ? std::min(sms,prop.multiProcessorCount) : prop.multiProcessorCount);
        fp8_check(cudaFuncSetAttribute(persistent_fp8<C,S,Output>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;
        fp8_check(cudaFuncGetAttributes(&attr,persistent_fp8<C,S,Output>));
        regs=attr.numRegs; local_bytes=attr.localSizeBytes;
    }
    void launch(cudaStream_t stream) override {
        persistent_fp8<C,S,Output><<<grid,384,CF::smem,stream>>>(args);
        fp8_check(cudaGetLastError());
    }
    void info(int *v) override {
        v[0]=128; v[1]=C; v[2]=128; v[3]=S; v[4]=CF::smem;
        v[5]=regs; v[6]=local_bytes; v[7]=grid;
    }
};
static thread_local std::string fp8_error;
template<class Output>
FP8PlanBase *fp8_create(void *a,void *b,void *sa,void *sb,void *out,
                      int m,int n,int k,int config,int sms,int raster) {
#define CFG(I,C,S) case I: return new FP8Plan<C,S,Output>(a,b,sa,sb,out,m,n,k,sms,raster)
    switch(config) {
#include "fp8_configs.inc"
        default: throw std::runtime_error("Invalid configuration");
    }
#undef CFG
}
} // namespace fp8gemm

extern "C" const char *tk_fp8_error() { return fp8gemm::fp8_error.c_str(); }
extern "C" void *tk_fp8_create(void *a,void *b,void *sa,void *sb,void *out,
                             int m,int n,int k,int config,int sms,int raster,int fp32) {
    using namespace fp8gemm;
    try {
        if (!a||!b||!sa||!sb||!out||m<=0||n<=0||k<=0||n%128||k%128||sms<0||raster<1)
            throw std::runtime_error("Invalid arguments; N and K must be positive multiples of 128");
        if (fp32) return fp8_create<float>(a,b,sa,sb,out,m,n,k,config,sms,raster);
        return fp8_create<kt::bf16>(a,b,sa,sb,out,m,n,k,config,sms,raster);
    } catch(const std::exception &e) { fp8_error=e.what(); return nullptr; }
}
extern "C" int tk_fp8_launch(fp8gemm::FP8PlanBase *p,void *stream) {
    try { p->launch((cudaStream_t)stream); return 0; }
    catch(const std::exception &e) { fp8gemm::fp8_error=e.what(); return -1; }
}
extern "C" void tk_fp8_info(fp8gemm::FP8PlanBase *p,int *v) { p->info(v); }
extern "C" void tk_fp8_destroy(fp8gemm::FP8PlanBase *p) { delete p; }
