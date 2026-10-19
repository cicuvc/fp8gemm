// Torch-free SM120a WASP GEMM: BF16 A (online MX scale), E4M3 B, BF16 A*B^T.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm {
namespace kt = kittens;
template<class Tile> using Global = kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

template<int R,int C,int BK,int Stages> struct Config {
    static_assert(R%16==0 && C%16==0 && BK%32==0 && 4*R<=256);
    using ATile = kt::st_bf<4*R,BK>;
    using BTile = kt::st_fp8e4m3<C,BK>;
    using OTile = kt::st_bf<4*R,C>;
    struct Packet { ATile a[2]; BTile b; };
    struct Shared { Packet input[Stages]; OTile output[2]; };
    struct Args {
        Global<ATile> a;
        Global<BTile> b;
        Global<OTile> out;
        int ntiles, total, ktiles;
        const uint8_t *weight_scales;
        int scale_rows,scale_cols;
    };
    // First 1024-byte region holds both endpoints' mbarriers.
    static constexpr int smem = 1024+sizeof(Shared);
    static_assert(smem<=101376, "Configuration exceeds SM120 shared memory");
};

template<int R,int C,int BK,int Stages,bool HasWeightScale=false>
__global__ __launch_bounds__(384,1) void persistent_gemm(
    const __grid_constant__ typename Config<R,C,BK,Stages>::Args args) {
    using CF = Config<R,C,BK,Stages>;
    using Shared = typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group = kt::warpgroup::groupid();
    int warp = kt::warpgroup::warpid();
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    // Metadata fits in the existing first 1024-byte allocation region, next
    // to mbarriers. Thus even the largest tile retains its previous SMEM size.
    constexpr int ScaleRows=(C+31)/32,ScaleCols=BK/32;
    static_assert(2*Stages*sizeof(kt::semaphore)+16+Stages*ScaleRows*ScaleCols*4<=1024);
    auto &scale_buffer=allocator.template allocate<16,uint32_t,Stages,ScaleRows,ScaleCols>();
    auto &shared = allocator.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader = kt::warp::elect_leader();
            int last = 0;
            // Static persistent assignment: no global counter or task loads.
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int mr = (task/args.ntiles)*(8*R), nc = (task%args.ntiles)*C;
                for(int k=0;k<args.ktiles;++k) {
                    auto packet = pipe.waitBuffer(0,shared.input);
                    auto &tile = packet.template get<0>();
                    if constexpr(HasWeightScale) {
                        int lane=threadIdx.x%32;
                        if(lane<ScaleRows*ScaleCols) {
                            int sr=lane/ScaleCols,sc=lane%ScaleCols;
                            int gr=nc/32+sr,gc=k*ScaleCols+sc;
                            uint32_t e=0;
                            if(gr<args.scale_rows && gc<args.scale_cols)
                                e=args.weight_scales[size_t(gr)*args.scale_cols+gc];
                            scale_buffer[packet.SlotIdx][sr][sc]=e;
                        }
                    }
                    if(leader) {
                        auto &bar = packet.getBarrier();
                        kt::tma::expect_bytes(bar,8*R*BK*2+C*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                    }
                    last = packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    pipe.moveNext();
                }
            }
            // All CTAs have at least one task and K>0. No phase reset between tasks.
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer = warp==0 && kt::warp::elect_leader();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> accum;
            kt::warp::rt_maps::zero(accum);
            for(int k=0;k<args.ktiles;++k) {
                auto packet = pipe.waitBuffer(2,shared.input);
                auto &tile = packet.template get<0>();
                kt::rt_bf<R,BK> ar;
                kt::rt_fp8e4m3<R,BK> aq;
                kt::rt_fp8e4m3<C,BK> br;
                kt::mxfp8_scale<R,BK> as;
                kt::mxfp8_scale<C,BK> bs;
                kt::warpgroup::load(ar,tile.a[group]);
                kt::warp::load<true>(br,tile.b);
#pragma unroll
                for(int n=0;n<C/16;++n)
#pragma unroll
                    for(int kk=0;kk<BK/32;++kk) {
                        if constexpr(HasWeightScale)
                            bs.data[n][kk]=scale_buffer[packet.SlotIdx][n/2][kk];
                        else bs.data[n][kk]=127;
                    }
                // Publish only after every consumer has completed both LDSM loads.
                packet.submitToNextAndTrigger();
                pipe.moveNext();
                kt::online_quantize_mxfp8(aq,as,ar);
                kt::warp::wmma::mma_ABt_block_scaled(accum,aq,br,accum,as,bs);
            }
            // Each WG owns one output tile. Let its previous TMA store overlap
            // this GEMM, then wait for shared reads before reusing that tile.
            if(writer) kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            // TK converts each FP32 pair to BF16 before stmatrix; never stores FP32.
            kt::warpgroup::store(shared.output[group],accum);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                int mr = (task/args.ntiles)*(8*R)+group*(4*R), nc = (task%args.ntiles)*C;
                kt::tma::store_async(args.out,shared.output[group],{0,0,mr,nc});
                kt::tma::store_commit_group();
            }
        }
        if(writer) kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct PlanBase {
    virtual ~PlanBase() = default;
    virtual void launch(cudaStream_t stream) = 0;
    virtual void info(int *data) = 0;
};
template<int R,int C,int BK,int S,bool HasWeightScale=false> struct Plan final : PlanBase {
    using CF=Config<R,C,BK,S>;
    typename CF::Args args;
    int grid, regs;
    Plan(void *a,void *b,void *o,int m,int n,int k,int sms,const uint8_t *scales=nullptr):
        args{Global<typename CF::ATile>(static_cast<kt::bf16*>(a),1,1,m,k),
             Global<typename CF::BTile>(static_cast<kt::fp8e4m3*>(b),1,1,n,k),
             Global<typename CF::OTile>(static_cast<kt::bf16*>(o),1,1,m,n),
             (n+C-1)/C,((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,
             scales,(n+31)/32,(k+31)/32} {
        if(k%(CF::BTile::swizzle_bytes) || n%(CF::OTile::swizzle_bytes/2))
            throw std::runtime_error("K/N must be aligned to the standard TMA swizzle panel");
        int dev; check(cudaGetDevice(&dev));
        cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0 ? std::min(sms,prop.multiProcessorCount) : prop.multiProcessorCount);
        check(cudaFuncSetAttribute(persistent_gemm<R,C,BK,S,HasWeightScale>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr; check(cudaFuncGetAttributes(&attr,persistent_gemm<R,C,BK,S,HasWeightScale>));
        regs=attr.numRegs;
        // 256 consumers*232 + 128 producers*40 need 64512 registers.
        // ptxas accounts for setmaxnreg and supplies 168 initial registers/thread.
        if(regs<168) throw std::runtime_error("Insufficient initial CTA register pool for WASP redistribution");
        if(attr.localSizeBytes) throw std::runtime_error("Kernel spills to local memory");
    }
    void launch(cudaStream_t stream) override {
        persistent_gemm<R,C,BK,S,HasWeightScale><<<grid,384,CF::smem,stream>>>(args);
        check(cudaGetLastError());
    }
    void info(int *d) override {d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;}
};
static thread_local std::string error;
} // namespace fp8gemm

// Pointer-only C ABI. Plans cache descriptors and own no input/output allocations.
extern "C" const char *tk_gemm_error() {return fp8gemm::error.c_str();}
extern "C" void *tk_gemm_create(void *a,void *b,void *out,int m,int n,int k,int config,int sms) {
    using namespace fp8gemm;
    try {
        if(m<=0 || n<=0 || k<=0 || !a || !b || !out) throw std::runtime_error("Invalid GEMM dimensions/pointers");
#define CFG(ID,R,C,BK,S) case ID: return new Plan<R,C,BK,S>(a,b,out,m,n,k,sms)
        switch(config) {
            CFG(0,16,32,32,2); CFG(1,16,64,32,2); CFG(2,16,128,32,2);
            CFG(3,32,32,32,2); CFG(4,32,64,32,2); CFG(5,64,16,32,2);
            CFG(6,64,32,32,2);
            CFG(7,16,64,64,2); CFG(8,16,128,64,2); CFG(9,32,32,64,2);
            default: throw std::runtime_error("Unknown configuration");
        }
#undef CFG
    } catch(const std::exception &e) {error=e.what();return nullptr;}
}
extern "C" void *tk_gemm_create_scaled(void *a,void *b,void *scales,void *out,int m,int n,int k,int config,int sms) {
    using namespace fp8gemm;
    try {
        if(m<=0 || n<=0 || k<=0 || !a || !b || !scales || !out) throw std::runtime_error("Invalid GEMM dimensions/pointers");
#define CFG(ID,R,C,BK,S) case ID:return new Plan<R,C,BK,S,true>(a,b,out,m,n,k,sms,static_cast<uint8_t*>(scales))
        switch(config) {
            CFG(0,16,32,32,2); CFG(1,16,64,32,2); CFG(2,16,128,32,2);
            CFG(3,32,32,32,2); CFG(4,32,64,32,2); CFG(5,64,16,32,2);
            CFG(6,64,32,32,2);
            CFG(7,16,64,64,2); CFG(8,16,128,64,2); CFG(9,32,32,64,2);
            default:throw std::runtime_error("Unknown configuration");
        }
#undef CFG
    } catch(const std::exception &e) {error=e.what();return nullptr;}
}
extern "C" int tk_gemm_launch(void *plan,void *stream) {
    try {static_cast<fp8gemm::PlanBase*>(plan)->launch(static_cast<cudaStream_t>(stream));return 0;}
    catch(const std::exception &e) {fp8gemm::error=e.what();return -1;}
}
extern "C" void tk_gemm_info(void *plan,int *data) {static_cast<fp8gemm::PlanBase*>(plan)->info(data);}
extern "C" void tk_gemm_destroy(void *plan) {delete static_cast<fp8gemm::PlanBase*>(plan);}
