// SM120 FP8 training GEMMs with FP32 K128 dequantization scales.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm {
template<class Tile> using FP8Global = kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

template<int C, int S, class Output, int Mode=0, bool RowB=false> struct FP8Config {
    static_assert(C == 64 || C == 128);
    using ATile = kt::st_fp8e4m3<64,128>;
    using BTile = kt::st_fp8e4m3<C,128>;
    // Mode 0: GEMM, 1: fused gate/up + SiLU, 2: fused gate/up dX.
    struct Packet { ATile a[Mode==2 ? 4 : 2]; BTile b[Mode ? 2 : 1]; };
    struct Shared { Packet input[S]; };
    struct Args {
        FP8Global<ATile> a;
        FP8Global<BTile> b;
        FP8Global<ATile> a2;
        FP8Global<BTile> b2;
        const float *sa, *sb, *sa2, *sb2;
        Output *out;
        kt::bf16 *gate, *up;
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

template<int C, int S, class Output, int Mode=0, bool RowB=false>
__global__ __launch_bounds__(384,1) void persistent_fp8(
    const __grid_constant__ typename FP8Config<C,S,Output,Mode,RowB>::Args args) {
    using CF = FP8Config<C,S,Output,Mode,RowB>;
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
                        kt::tma::expect_bytes(bar,((Mode==2 ? 256 : 128)+(Mode ? 2*C : C))*128);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,coord.x*128,g*128},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,coord.x*128+64,g*128},bar);
                        kt::tma::load_async(tile.b[0],args.b,{0,0,coord.y*C,g*128},bar);
                        if constexpr(Mode!=0)
                            kt::tma::load_async(tile.b[1],args.b2,{0,0,coord.y*C,g*128},bar);
                        if constexpr(Mode==2) {
                            kt::tma::load_async(tile.a[2],args.a2,{0,0,coord.x*128,g*128},bar);
                            kt::tma::load_async(tile.a[3],args.a2,{0,0,coord.x*128+64,g*128},bar);
                        }
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
            kt::rt_fl<16,C> accum, accum_up;
            kt::warp::rt_maps::zero(accum);
            if constexpr(Mode==1) kt::warp::rt_maps::zero(accum_up);
            for (int g=0; g<args.ktiles; ++g) {
                auto packet = pipe.waitBuffer(2,shared.input);
                auto &tile = packet.template get<0>();
#pragma unroll
                for (int projection=0; projection<(Mode ? 2 : 1); ++projection) {
                    const float *sa = Mode==2 && projection ? args.sa2 : args.sa;
                    const float *sb = projection ? args.sb2 : args.sb;
                    float s0 = row0<args.m ? sa[size_t(row0)*args.ktiles+g] : 0.f;
                    float s1 = row1<args.m ? sa[size_t(row1)*args.ktiles+g] : 0.f;
                    if constexpr(!RowB) {
                        float bscale = sb[size_t(nc/128)*args.ktiles+g];
                        s0 *= bscale; s1 *= bscale;
                    }
                    kt::rt_fl<16,C> partial;
                    kt::warp::rt_maps::zero(partial);
#pragma unroll
                    for (int k32=0; k32<4; ++k32) {
                        kt::rt_fp8e4m3<16,32> ar;
                        kt::rt_fp8e4m3<C,32> br;
                        kt::mxfp8_scale<16,32> as;
                        kt::mxfp8_scale<C,32> bs;
                        load_fp8_factored<4,false>(ar,tile.a[group+(Mode==2 ? projection*2 : 0)],k32);
                        load_fp8_factored<1,true>(br,tile.b[projection],k32);
                        // Unit E8M0 scales select QMMA.SF; apply FP32 scales after K128.
#pragma unroll
                        for (int j=0; j<C/16; ++j) bs.data[j][0]=127u;
                        as.data[0][0]=127u;
                        kt::warp::wmma::mma_ABt_block_scaled(partial,ar,br,partial,as,bs);
                    }
                    // All shared loads are complete after the last projection.
                    // Release before scaling so the producer can fill the next tile.
                    if (projection == (Mode ? 1 : 0)) {
                        packet.submitToNextAndTrigger();
                        pipe.moveNext();
                    }
#pragma unroll
                    for (int j=0; j<C/16; ++j) {
#pragma unroll
                        for (int p=0; p<4; ++p) {
                            auto &d = Mode==1 && projection ? accum_up.tiles[0][j].data[p]
                                                          : accum.tiles[0][j].data[p];
                            auto v = partial.tiles[0][j].data[p];
                            float sx = (p&1) ? s1 : s0, sy = sx;
                            if constexpr(RowB) {
                                int col = nc+j*16+(p/2)*8+(lane%4)*2;
                                sx *= sb[size_t(col)*args.ktiles+g];
                                sy *= sb[size_t(col+1)*args.ktiles+g];
                            }
                            d.x = fmaf(v.x,sx,d.x);
                            d.y = fmaf(v.y,sy,d.y);
                        }
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
                    if (row<args.m) {
                        auto v = accum.tiles[0][j].data[p];
                        size_t offset = size_t(row)*args.n+col;
                        if constexpr(Mode==1) {
                            // Save BF16 projections; forward and backward use the same values.
                            auto u = accum_up.tiles[0][j].data[p];
                            auto gv = __floats2bfloat162_rn(v.x,v.y);
                            auto uv = __floats2bfloat162_rn(u.x,u.y);
                            *reinterpret_cast<__nv_bfloat162*>(args.gate+offset) = gv;
                            *reinterpret_cast<__nv_bfloat162*>(args.up+offset) = uv;
                            v = __bfloat1622float2(gv); u = __bfloat1622float2(uv);
                            v.x = v.x/(1.f+expf(-v.x))*u.x;
                            v.y = v.y/(1.f+expf(-v.y))*u.y;
                        }
                        fp8_store_pair(args.out+offset,v);
                    }
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
    virtual void set_output(void*) = 0;
};
template<int C, int S, class Output, int Mode=0, bool RowB=false> struct FP8Plan : FP8PlanBase {
    using CF = FP8Config<C,S,Output,Mode,RowB>;
    typename CF::Args args;
    int grid, regs, local_bytes;
    FP8Plan(void *a,void *b,void *sa,void *sb,void *out,int m,int n,int k,int sms,int raster,
            void *a2=nullptr,void *b2=nullptr,void *sa2=nullptr,void *sb2=nullptr,
            void *gate=nullptr,void *up=nullptr)
        : args{FP8Global<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,k),
               FP8Global<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,k),
               FP8Global<typename CF::ATile>((kt::fp8e4m3*)(a2 ? a2 : a),1,1,m,k),
               FP8Global<typename CF::BTile>((kt::fp8e4m3*)(b2 ? b2 : b),1,1,n,k),
               (float*)sa,(float*)sb,(float*)sa2,(float*)sb2,(Output*)out,
               (kt::bf16*)gate,(kt::bf16*)up,m,n,k/128,n/C,(m+127)/128,
               ((m+127)/128)*(n/C),std::min(raster,(m+127)/128)} {
        int device;
        fp8_check(cudaGetDevice(&device));
        cudaDeviceProp prop;
        fp8_check(cudaGetDeviceProperties(&prop,device));
        if (prop.major!=12 || prop.minor!=0) throw std::runtime_error("FP8 GEMM requires SM120");
        grid = std::min(args.total,sms ? std::min(sms,prop.multiProcessorCount) : prop.multiProcessorCount);
        fp8_check(cudaFuncSetAttribute(persistent_fp8<C,S,Output,Mode,RowB>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;
        fp8_check(cudaFuncGetAttributes(&attr,persistent_fp8<C,S,Output,Mode,RowB>));
        regs=attr.numRegs; local_bytes=attr.localSizeBytes;
    }
    void launch(cudaStream_t stream) override {
        persistent_fp8<C,S,Output,Mode,RowB><<<grid,384,CF::smem,stream>>>(args);
        fp8_check(cudaGetLastError());
    }
    void set_output(void *out) override { args.out = (Output*)out; }
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

// Training variants use the same plan lifecycle and launch entry point.
extern "C" void *tk_fp8_training_create(void *a,void *b,void *sa,void *sb,void *out,
    void *a2,void *b2,void *sa2,void *sb2,void *gate,void *up,
    int m,int n,int k,int mode) {
    using namespace fp8gemm;
    try {
        if (!a||!b||!sa||!sb||!out||m<=0||n<=0||k<=0||n%128||k%128)
            throw std::runtime_error("Invalid training GEMM arguments");
        if (mode==0) return new FP8Plan<64,2,float,0,true>(a,b,sa,sb,out,m,n,k,0,8);
        if (!b2||!sb2) throw std::runtime_error("Missing second projection");
        if (mode==1 && gate && up)
            return new FP8Plan<64,2,kt::bf16,1>(a,b,sa,sb,out,m,n,k,0,8,a,b2,sa,sb2,gate,up);
        if (mode==2 && a2 && sa2)
            return new FP8Plan<64,2,kt::bf16,2>(a,b,sa,sb,out,m,n,k,0,8,a2,b2,sa2,sb2);
        throw std::runtime_error("Invalid training GEMM mode");
    } catch(const std::exception &e) { fp8_error=e.what(); return nullptr; }
}
extern "C" void tk_fp8_set_output(fp8gemm::FP8PlanBase *p,void *out) { p->set_output(out); }
