// SM120 FP8 training GEMMs with FP32 K128 dequantization scales.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include "mxfp8_streamk.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm {
template<class Tile> using FP8Global = kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

template<int C, int S, class Output, int Mode=0, bool RowB=false, int SK=0> struct FP8Config {
    static_assert(C == 64 || C == 128);
    using ATile = kt::st_fp8e4m3<64,128>;
    using BTile = kt::st_fp8e4m3<C,128>;
    // Mode 0: GEMM, 1: fused gate/up + SiLU, 2: fused gate/up dX.
    struct Packet { ATile a[Mode==2 ? 4 : 2]; BTile b[Mode ? 2 : 1]; };
    // Precombine row and block-weight scales in the producer. Even the dual
    // dX packet then fits in exactly 1 KiB of metadata per slot.
    struct BlockScales { float a[Mode ? 2 : 1][128]; };
    struct RowScales { float a[1][128]; float b[C]; };
    using Scales = std::conditional_t<RowB,RowScales,BlockScales>;
    using Partial = kt::st_fl<64,C/2>;
    struct SharedBase { Packet input[S]; Scales scales[S]; };
    struct SharedSK { Packet input[S]; Scales scales[S]; Partial output[2]; };
    using Shared = std::conditional_t<SK==0,SharedBase,SharedSK>;
    struct Empty {};
    using SKArgs = std::conditional_t<SK==0,Empty,sk::Params<16,C,(Mode==1 ? 2 : 1)>>;
    static SKArgs make_sk(void *partial,void *work,void *offsets,int slots) {
        if constexpr(SK) return {
            FP8Global<Partial>((float*)partial,1,1,std::max(1,slots)*(Mode==1 ? 2 : 1)*128,C),
            (sk::Work*)work,(int*)offsets};
        else return {};
    }
    struct Args {
        FP8Global<ATile> a;
        FP8Global<BTile> b;
        FP8Global<ATile> a2;
        FP8Global<BTile> b2;
        const float *sa, *sb, *sa2, *sb2;
        Output *out;
        kt::bf16 *gate, *up;
        int m, n, ktiles, ntiles, mtiles, total, raster_group;
        SKArgs sk;
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

__device__ __forceinline__ float2 fp8_activate_pair(float2 g,float2 u,
    kt::bf16 *gate,kt::bf16 *up,size_t offset) {
    auto gv=__floats2bfloat162_rn(g.x,g.y),uv=__floats2bfloat162_rn(u.x,u.y);
    *reinterpret_cast<__nv_bfloat162*>(gate+offset)=gv;
    *reinterpret_cast<__nv_bfloat162*>(up+offset)=uv;
    g=__bfloat1622float2(gv);u=__bfloat1622float2(uv);
    return {g.x/(1.f+expf(-g.x))*u.x,g.y/(1.f+expf(-g.y))*u.y};
}

template<int C,class Output,int Mode,bool Atomic>
__global__ __launch_bounds__(256) void fp8_fixup(const float *partial,Output *out,
    kt::bf16 *gate,kt::bf16 *up,const sk::Fixup *fix,const int *ids,int m,int n) {
    const auto f=fix[blockIdx.x];
    constexpr int components=Mode==1 ? 2 : 1;
    for (int index=threadIdx.x*2; index<128*C; index+=256*2) {
        float2 sums[components];
#pragma unroll
        for (int comp=0; comp<components; ++comp) {
            float2 v={0,0};
            if constexpr(Atomic)
                v=*reinterpret_cast<const float2*>(partial+(size_t(blockIdx.x)*components+comp)*128*C+index);
            else for (int i=0; i<f.count; ++i) {
                auto p=*reinterpret_cast<const float2*>(partial+(size_t(ids[f.first+i])*components+comp)*128*C+index);
                v.x+=p.x;v.y+=p.y;
            }
            sums[comp]=v;
        }
        int row=f.nr+index/C,col=f.kc+index%C;
        if (row<m) {
            size_t offset=size_t(row)*n+col;
            auto v=sums[0];
            if constexpr(Mode==1) v=fp8_activate_pair(v,sums[1],gate,up,offset);
            fp8_store_pair(out+offset,v);
        }
    }
}

template<int C, int S, class Output, int Mode=0, bool RowB=false, int SK=0>
__global__ __launch_bounds__(384,1) void persistent_fp8(
    const __grid_constant__ typename FP8Config<C,S,Output,Mode,RowB,SK>::Args args) {
    using CF = FP8Config<C,S,Output,Mode,RowB,SK>;
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
            int first=blockIdx.x,stop=args.total,stride=gridDim.x;
            if constexpr(SK) {first=args.sk.offsets[blockIdx.x];stop=args.sk.offsets[blockIdx.x+1];stride=1;}
            for (int task=first; task<stop; task+=stride) {
                sk::Work work;int2 coord;int begin=0,end=args.ktiles;
                if constexpr(SK) {
                    work=args.sk.work[task];coord={work.nr/128,work.kc/C};begin=work.begin;end=work.end;
                } else coord=fp8_raster(task,args.ntiles,args.mtiles,args.raster_group);
                for (int g=begin; g<end; ++g) {
                    // Coalesced global prefetch before waiting for the free slot.
                    float pa[Mode ? 2 : 1][4];
                    float pb[RowB ? C/32 : 1];
#pragma unroll
                    for (int projection=0; projection<(Mode ? 2 : 1); ++projection) {
                        const float *sa = Mode==2 && projection ? args.sa2 : args.sa;
                        const float *sb = projection ? args.sb2 : args.sb;
                        float bscale = 1.f;
                        if constexpr(!RowB) bscale=sb[size_t(coord.y*C/128)*args.ktiles+g];
#pragma unroll
                        for (int i=0; i<4; ++i) {
                            int row=coord.x*128+lane+i*32;
                            pa[projection][i]=row<args.m ? sa[size_t(row)*args.ktiles+g]*bscale : 0.f;
                        }
                    }
                    if constexpr(RowB) {
#pragma unroll
                        for (int i=0; i<C/32; ++i)
                            pb[i]=args.sb[size_t(coord.y*C+lane+i*32)*args.ktiles+g];
                    }
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
                    auto &scale=shared.scales[packet.SlotIdx];
#pragma unroll
                    for (int projection=0; projection<(Mode ? 2 : 1); ++projection) {
#pragma unroll
                        for (int i=0; i<4; ++i) scale.a[projection][lane+i*32]=pa[projection][i];
                    }
                    if constexpr(RowB) {
#pragma unroll
                        for (int i=0; i<C/32; ++i) scale.b[lane+i*32]=pb[i];
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
        bool writer=false;
        if constexpr(SK) writer=warp==0 && kt::warp::elect_leader();
        int first=blockIdx.x,stop=args.total,stride=gridDim.x;
            if constexpr(SK) {first=args.sk.offsets[blockIdx.x];stop=args.sk.offsets[blockIdx.x+1];stride=1;}
            for (int task=first; task<stop; task+=stride) {
            sk::Work work;int2 coord;int begin=0,end=args.ktiles;
                if constexpr(SK) {
                    work=args.sk.work[task];coord={work.nr/128,work.kc/C};begin=work.begin;end=work.end;
                } else coord=fp8_raster(task,args.ntiles,args.mtiles,args.raster_group);
            int mr = coord.x*128 + group*64 + warp*16;
            int nc = coord.y*C;
            int row0 = mr + lane/4, row1 = row0+8;
            kt::rt_fl<16,C> accum, accum_up;
            kt::warp::rt_maps::zero(accum);
            if constexpr(Mode==1) kt::warp::rt_maps::zero(accum_up);
            for (int g=begin; g<end; ++g) {
                auto packet = pipe.waitBuffer(2,shared.input);
                auto &tile = packet.template get<0>();
#pragma unroll
                for (int projection=0; projection<(Mode ? 2 : 1); ++projection) {
                    const auto &scale=shared.scales[packet.SlotIdx];
                    int local_row=group*64+warp*16+lane/4;
                    float s0=scale.a[projection][local_row];
                    float s1=scale.a[projection][local_row+8];
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
                    float bvalues[C/16][4];
                    if constexpr(RowB) {
#pragma unroll
                        for (int j=0; j<C/16; ++j) {
#pragma unroll
                            for (int i=0; i<4; ++i)
                                bvalues[j][i]=scale.b[j*16+(i/2)*8+(lane%4)*2+i%2];
                        }
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
                                sx *= bvalues[j][(p/2)*2];
                                sy *= bvalues[j][(p/2)*2+1];
                            }
                            d.x = fmaf(v.x,sx,d.x);
                            d.y = fmaf(v.y,sy,d.y);
                        }
                    }
                }
            }
            if constexpr(SK) {
                if (work.slot>=0) {
                    sk::store_partial<16,C,(Mode==1 ? 2 : 1),kt::rt_fl<16,C>,SK==1>(
                        shared.output[group],accum,args.sk,work.slot,0,group,writer);
                    if constexpr(Mode==1)
                        sk::store_partial<16,C,2,kt::rt_fl<16,C>,SK==1>(
                            shared.output[group],accum_up,args.sk,work.slot,1,group,writer);
                    continue;
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
                            v = fp8_activate_pair(v,accum_up.tiles[0][j].data[p],args.gate,args.up,offset);
                        }
                        fp8_store_pair(args.out+offset,v);
                    }
                }
            }
        }
        if constexpr(SK) if(writer) kt::tma::store_async_wait<0>();
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
template<int C, int S, class Output, int Mode=0, bool RowB=false, int SK=0> struct FP8Plan : FP8PlanBase {
    using CF = FP8Config<C,S,Output,Mode,RowB,SK>;
    typename CF::Args args;
    int grid, regs, local_bytes;
    void *workspace;
    const sk::Fixup *fix;
    const int *ids;
    int slots,nfix;
    FP8Plan(void *a,void *b,void *sa,void *sb,void *out,int m,int n,int k,int sms,int raster,
            void *a2=nullptr,void *b2=nullptr,void *sa2=nullptr,void *sb2=nullptr,
            void *gate=nullptr,void *up=nullptr,
            void *partial=nullptr,void *work=nullptr,void *offsets=nullptr,
            void *fixups=nullptr,void *contributors=nullptr,int sk_grid=0,int sk_slots=0,int sk_nfix=0)
        : args{FP8Global<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,k),
               FP8Global<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,k),
               FP8Global<typename CF::ATile>((kt::fp8e4m3*)(a2 ? a2 : a),1,1,m,k),
               FP8Global<typename CF::BTile>((kt::fp8e4m3*)(b2 ? b2 : b),1,1,n,k),
               (float*)sa,(float*)sb,(float*)sa2,(float*)sb2,(Output*)out,
               (kt::bf16*)gate,(kt::bf16*)up,m,n,k/128,n/C,(m+127)/128,
               ((m+127)/128)*(n/C),std::min(raster,(m+127)/128),CF::make_sk(partial,work,offsets,sk_slots)},
          workspace(partial),fix((sk::Fixup*)fixups),ids((int*)contributors),slots(sk_slots),nfix(sk_nfix) {
        int device;
        fp8_check(cudaGetDevice(&device));
        cudaDeviceProp prop;
        fp8_check(cudaGetDeviceProperties(&prop,device));
        if (prop.major!=12 || prop.minor!=0) throw std::runtime_error("FP8 GEMM requires SM120");
        grid = std::min(args.total,sms ? std::min(sms,prop.multiProcessorCount) : prop.multiProcessorCount);
        if constexpr(SK) grid=sk_grid;
        fp8_check(cudaFuncSetAttribute(persistent_fp8<C,S,Output,Mode,RowB,SK>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;
        fp8_check(cudaFuncGetAttributes(&attr,persistent_fp8<C,S,Output,Mode,RowB,SK>));
        regs=attr.numRegs; local_bytes=attr.localSizeBytes;
    }
    void launch(cudaStream_t stream) override {
        if constexpr(SK==1) if(nfix)
            fp8_check(cudaMemsetAsync(workspace,0,size_t(slots)*(Mode==1 ? 2 : 1)*128*C*sizeof(float),stream));
        persistent_fp8<C,S,Output,Mode,RowB,SK><<<grid,384,CF::smem,stream>>>(args);
        fp8_check(cudaGetLastError());
        if constexpr(SK) if(nfix) {
            fp8_fixup<C,Output,Mode,SK==1><<<nfix,256,0,stream>>>((float*)workspace,args.out,
                args.gate,args.up,fix,ids,args.m,args.n);
            fp8_check(cudaGetLastError());
        }
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

namespace fp8gemm {
template<int SK>
FP8PlanBase *fp8_streamk_create(void **p,int m,int n,int k,int mode,int row_b,int fp32,
    int grid,int slots,int nfix) {
#define SK_PLAN(O,M,R,S) new FP8Plan<64,S,O,M,R,SK>(p[0],p[1],p[2],p[3],p[4],m,n,k,0,8, \
    p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],grid,slots,nfix)
    if(mode==0 && row_b && fp32) return SK_PLAN(float,0,true,2);
    if(mode==0 && !row_b) {
        if(fp32) return SK_PLAN(float,0,false,2);
        return SK_PLAN(kt::bf16,0,false,2);
    }
    if(!row_b && !fp32) {
        if(mode==1 && p[6] && p[8] && p[9] && p[10]) return SK_PLAN(kt::bf16,1,false,2);
        // A single input stage leaves room for FP32 TMA partial stores.
        if(mode==2 && p[5] && p[6] && p[7] && p[8]) return SK_PLAN(kt::bf16,2,false,1);
    }
    throw std::runtime_error("Invalid FP8 Stream-K mode");
#undef SK_PLAN
}
}
extern "C" void *tk_fp8_streamk_create(void **p,int m,int n,int k,int mode,int row_b,int fp32,
    int atomic,int grid,int slots,int nfix) {
    using namespace fp8gemm;
    try {
        if(!p||!p[0]||!p[1]||!p[2]||!p[3]||!p[4]||!p[11]||!p[12]||!p[13]||!p[14]||!p[15]||
           m<=0||n<=0||k<=0||n%128||k%128||grid<=0||slots<0||nfix<0)
            throw std::runtime_error("Invalid FP8 Stream-K arguments");
        if(atomic) return fp8_streamk_create<1>(p,m,n,k,mode,row_b,fp32,grid,slots,nfix);
        return fp8_streamk_create<2>(p,m,n,k,mode,row_b,fp32,grid,slots,nfix);
    } catch(const std::exception &e) {fp8_error=e.what();return nullptr;}
}
