// SM120a WASP AB^T with BF16 A and B quantized online along each row/K32.
// A cached FP8 B + per-row/K32 scales path provides an equivalent comparison.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::dual_online {
namespace kt=kittens;
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

// Group output tile rows, traversing rows first within each column of a group.
// The final group can be shorter. Both TMA producers and output writers must
// use the same bijection; persistent task ownership itself is unchanged.
__device__ __forceinline__ int2 raster_tile(int task,int ntiles,int mtiles,int group_m) {
    if(group_m==1)return make_int2(task/ntiles,task%ntiles);
    int group_tiles=group_m*ntiles;
    int first_row=(task/group_tiles)*group_m;
    int rows=min(group_m,mtiles-first_row);
    int local=task%group_tiles;
    return make_int2(first_row+local%rows,local/rows);
}

template<int R,int C,int BK,int S,bool OnlineB,bool FloatOut,bool NativeDY=false> struct Config {
    using BType=std::conditional_t<OnlineB,kt::bf16,kt::fp8e4m3>;
    using OType=std::conditional_t<FloatOut,float,kt::bf16>;
    static constexpr int OC=FloatOut?C/2:C;
    using A=kt::st_bf<NativeDY?BK:4*R,NativeDY?4*R:BK>;
    using B=kt::st<BType,C,BK>;
    // FP32 output is stored in two C/2 slices, preserving the BF16 kernel's
    // shared-memory budget and leaving the full FP32 accumulator in registers.
    using O=kt::st<OType,4*R,OC>;
    struct Packet {A a[2];B b;};
    struct Shared {Packet input[S];O output[2];};
    struct Args {
        Global<A> a;Global<B> b;Global<O> out;
        const uint8_t *bs;
        int ntiles,total,ktiles,n,scale_cols,mtiles,raster_group;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
};

template<int R,int C,int BK,int S,bool OnlineB,bool FloatOut,bool NativeDY=false,bool SwapAB=false>
__global__ __launch_bounds__(384,1) void dual_online_gemm(
    const __grid_constant__ typename Config<R,C,BK,S,OnlineB,FloatOut,NativeDY>::Args args) {
    static_assert(!SwapAB || (NativeDY && !OnlineB && FloatOut));
    using CF=Config<R,C,BK,S,OnlineB,FloatOut,NativeDY>;using Shared=typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> alloc(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid();
    MbarrierRingPipe pipe(alloc,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    static_assert(2*S*sizeof(kt::semaphore)+16+S*C*(BK/32)<=1024);
    auto &scales=alloc.template allocate<16,uint8_t,S,C,BK/32>();
    auto &shared=alloc.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int2 coord=raster_tile(task,args.ntiles,args.mtiles,args.raster_group);
                int mr=coord.x*(8*R),nc=coord.y*C;
                for(int k=0;k<args.ktiles;++k) {
                    auto p=pipe.waitBuffer(0,shared.input);auto &tile=p.template get<0>();
                    if constexpr(!OnlineB) {
#pragma unroll
                        for(int i=threadIdx.x%32;i<C*(BK/32);i+=32) {
                            int row=i/(BK/32),col=i%(BK/32),gc=k*(BK/32)+col;
                            uint8_t s=0;
                            if(nc+row<args.n && gc<args.scale_cols)
                                s=args.bs[size_t(nc+row)*args.scale_cols+gc];
                            scales[p.SlotIdx][row][col]=s;
                        }
                    }
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,8*R*BK*2+C*BK*sizeof(typename CF::BType));
                        if constexpr(NativeDY) {
                            kt::tma::load_async(tile.a[0],args.a,{0,0,k*BK,mr},bar);
                            kt::tma::load_async(tile.a[1],args.a,{0,0,k*BK,mr+4*R},bar);
                        } else {
                            kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                            kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        }
                        kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                    }
                    last=p.SlotIdx;p.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<SwapAB?C:R,SwapAB?R:C> acc;kt::warp::rt_maps::zero(acc);
            for(int k=0;k<args.ktiles;++k) {
                auto p=pipe.waitBuffer(2,shared.input);auto &tile=p.template get<0>();
                kt::rt_bf<R,BK> ar;
                kt::rt_fp8e4m3<R,BK> aq;
                kt::mxfp8_scale<R,BK> as;
                if constexpr(NativeDY) {
                    auto panel=tile.a[group].template subtile<BK,R>({0,warp});
                    kt::rt<kt::bf16,BK,R,kt::ducks::rt_layout::col> transposed;
                    kt::warp::load(transposed,panel);
#pragma unroll
                    for(int i=0;i<R/16;++i)
#pragma unroll
                        for(int j=0;j<BK/16;++j)
#pragma unroll
                            for(int p=0;p<4;++p)
                                ar.tiles[i][j].data[p]=transposed.tiles[j][i].data[p];
                } else kt::warpgroup::load(ar,tile.a[group]);
                if constexpr(OnlineB) {
                    // Retire BF16 A before loading the larger BF16 B tile.
                    // Keeping both live spills at R16/C128/BK64.
                    kt::online_quantize_mxfp8(aq,as,ar);
                    constexpr int BC=C>64?64:C;
#pragma unroll
                    for(int part=0;part<C/BC;++part) {
                        kt::rt_bf<BC,BK> br;
                        kt::rt_fp8e4m3<BC,BK> q;
                        kt::mxfp8_scale<BC,BK> s;
                        auto panel=tile.b.template subtile<BC,BK>({part,0});
                        kt::warp::load(br,panel);
                        if(part==C/BC-1) {
                            p.submitToNextAndTrigger();pipe.moveNext();
                        }
                        kt::online_quantize_mxfp8(q,s,br);
                        kt::rt_fl<R,BC> partial;
#pragma unroll
                        for(int i=0;i<R/16;++i) {
#pragma unroll
                            for(int j=0;j<BC/16;++j)
                                partial.tiles[i][j]=acc.tiles[i][part*(BC/16)+j];
                        }
                        kt::warp::wmma::mma_ABt_block_scaled(partial,aq,q,partial,as,s);
#pragma unroll
                        for(int i=0;i<R/16;++i) {
#pragma unroll
                            for(int j=0;j<BC/16;++j)
                                acc.tiles[i][part*(BC/16)+j]=partial.tiles[i][j];
                        }
                    }
                } else {
                    kt::rt_fp8e4m3<C,BK> bq;
                    kt::mxfp8_scale<C,BK> bs;
                    if constexpr(SwapAB) kt::warp::load(bq,tile.b);
                    else kt::warp::load<true>(bq,tile.b);
                    int row=threadIdx.x%32/4+8*(threadIdx.x%2);
#pragma unroll
                    for(int i=0;i<C/16;++i) {
#pragma unroll
                        for(int j=0;j<BK/32;++j)
                            bs.data[i][j]=scales[p.SlotIdx][i*16+row][j];
                    }
                    p.submitToNextAndTrigger();pipe.moveNext();
                    kt::online_quantize_mxfp8(aq,as,ar);
                    if constexpr(SwapAB) kt::warp::wmma::mma_ABt_block_scaled(acc,bq,aq,acc,bs,as);
                    else kt::warp::wmma::mma_ABt_block_scaled(acc,aq,bq,acc,as,bs);
                }
            }
            int2 coord=raster_tile(task,args.ntiles,args.mtiles,args.raster_group);
            int mr=coord.x*(8*R)+group*4*R,nc=coord.y*C;
#pragma unroll
            for(int slice=0;slice<(FloatOut?2:1);++slice) {
                if(writer) kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                if constexpr(FloatOut) {
                    kt::rt<float,R,CF::OC,std::conditional_t<SwapAB,
                        kt::ducks::rt_layout::col,kt::ducks::rt_layout::row>> part;
#pragma unroll
                    for(int i=0;i<R/16;++i) {
#pragma unroll
                        for(int j=0;j<CF::OC/16;++j) {
                            if constexpr(SwapAB) {
#pragma unroll
                                for(int p=0;p<4;++p)
                                    part.tiles[i][j].data[p]=acc.tiles[j+slice*(CF::OC/16)][i].data[p];
                            } else part.tiles[i][j]=acc.tiles[i][j+slice*(CF::OC/16)];
                        }
                    }
                    kt::warpgroup::store(shared.output[group],part);
                } else kt::warpgroup::store(shared.output[group],acc);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.out,shared.output[group],{0,0,mr,nc+slice*CF::OC});
                    kt::tma::store_commit_group();
                }
            }
        }
        if(writer) kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct PlanBase{
    virtual ~PlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void set_raster_group(int)=0;
};
template<int R,int C,int BK,int S,bool OnlineB,bool FloatOut,bool NativeDY=false,bool SwapAB=false> struct Plan final:PlanBase {
    using CF=Config<R,C,BK,S,OnlineB,FloatOut,NativeDY>;
    typename CF::Args args;int grid,regs;
    Plan(void*a,void*b,void*bs,void*out,int m,int n,int k,int sms,int arows=0):args{
        Global<typename CF::A>(static_cast<kt::bf16*>(a),1,1,NativeDY?arows:m,NativeDY?m:k),
        Global<typename CF::B>(static_cast<typename CF::BType*>(b),1,1,n,k),
        Global<typename CF::O>(static_cast<typename CF::OType*>(out),1,1,m,n),
        static_cast<uint8_t*>(bs),(n+C-1)/C,((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,n,(k+31)/32,
        (m+8*R-1)/(8*R),1} {
        if(k%(CF::B::swizzle_bytes/sizeof(typename CF::BType)) || n%(CF::O::swizzle_bytes/sizeof(typename CF::OType)))
            throw std::runtime_error("Input/output strides must align to the TMA panels");
        if constexpr(NativeDY) {
            if(arows<=0 || arows>k || m%(CF::A::swizzle_bytes/2))
                throw std::runtime_error("Invalid native dY rows/stride");
        }
        int dev;check(cudaGetDevice(&dev));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        check(cudaFuncSetAttribute(dual_online_gemm<R,C,BK,S,OnlineB,FloatOut,NativeDY,SwapAB>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,dual_online_gemm<R,C,BK,S,OnlineB,FloatOut,NativeDY,SwapAB>));
        regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes)throw std::runtime_error("Insufficient WASP register pool or local memory spills");
    }
    void launch(cudaStream_t stream)override{
        dual_online_gemm<R,C,BK,S,OnlineB,FloatOut,NativeDY,SwapAB><<<grid,384,CF::smem,stream>>>(args);check(cudaGetLastError());
    }
    void info(int*d)override{d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;d[7]=OnlineB;d[8]=FloatOut;}
    void set_raster_group(int size)override{
        if(size<1)throw std::runtime_error("Raster group must be positive");
        args.raster_group=std::min(size,args.mtiles);
    }
};
static thread_local std::string error;
template<bool OnlineB,bool FloatOut,bool NativeDY=false,bool SwapAB=false> PlanBase*create(void*a,void*b,void*bs,void*out,int m,int n,int k,int cfg,int sms,int arows=0){
#define CFG(ID,R,C,BK,S) case ID:return new Plan<R,C,BK,S,OnlineB,FloatOut,NativeDY,SwapAB>(a,b,bs,out,m,n,k,sms,arows)
    switch(cfg){
        CFG(0,16,32,64,2);CFG(1,16,64,64,2);CFG(2,16,128,32,2);CFG(3,16,128,64,2);
        CFG(4,32,64,32,2);CFG(5,32,32,64,2);CFG(6,16,64,64,3);CFG(7,16,32,128,2);
        // Extra native-dY dW candidates under the 99 KiB shared-memory budget;
        // keep generic ABt and swapped-operand instantiations unchanged.
#define DW_CFG(ID,R,C,BK,S) case ID: \
        if constexpr(NativeDY && !SwapAB) \
            return new Plan<R,C,BK,S,OnlineB,FloatOut,NativeDY,SwapAB>(a,b,bs,out,m,n,k,sms,arows); \
        else throw std::runtime_error("Configuration requires native dY without swapAB")
        DW_CFG(8,32,128,32,1);DW_CFG(9,16,256,32,1);
        DW_CFG(10,32,64,32,3);DW_CFG(11,16,128,32,3);
        DW_CFG(12,64,32,32,1);DW_CFG(13,32,64,32,1);
        DW_CFG(14,16,256,32,2);DW_CFG(15,16,128,32,4);DW_CFG(16,16,64,64,4);
#undef DW_CFG
        default:throw std::runtime_error("Unknown configuration");
    }
#undef CFG
}
}
extern "C" const char*tk_dual_error(){return fp8gemm::dual_online::error.c_str();}
extern "C" void*tk_dual_create(void*a,void*b,void*bs,void*out,int m,int n,int k,int cfg,int sms,int online,int fp32){
    using namespace fp8gemm::dual_online;
    try{
        if(m<=0 || n<=0 || k<=0 || !a || !b || !out || (!online && !bs))throw std::runtime_error("Invalid shape or pointers");
        if(online)return fp32?create<true,true>(a,b,bs,out,m,n,k,cfg,sms):create<true,false>(a,b,bs,out,m,n,k,cfg,sms);
        return fp32?create<false,true>(a,b,bs,out,m,n,k,cfg,sms):create<false,false>(a,b,bs,out,m,n,k,cfg,sms);
    }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void*tk_dual_create_native_dy(void*a,void*b,void*bs,void*out,
    int m,int n,int k,int arows,int cfg,int sms,int swap){
    using namespace fp8gemm::dual_online;
    try{
        if(m<=0 || n<=0 || k<=0 || !a || !b || !bs || !out)throw std::runtime_error("Invalid shape or pointers");
        return swap?create<false,true,true,true>(a,b,bs,out,m,n,k,cfg,sms,arows):
                    create<false,true,true,false>(a,b,bs,out,m,n,k,cfg,sms,arows);
    }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" int tk_dual_launch(void*p,void*s){
    try{static_cast<fp8gemm::dual_online::PlanBase*>(p)->launch(static_cast<cudaStream_t>(s));return 0;}
    catch(const std::exception&e){fp8gemm::dual_online::error=e.what();return -1;}
}
extern "C" void tk_dual_info(void*p,int*d){static_cast<fp8gemm::dual_online::PlanBase*>(p)->info(d);}
// Configure before launching/capturing the plan; preserves the create/info ABI.
extern "C" int tk_dual_set_raster_group(void*p,int size){
    try{
        if(!p)throw std::runtime_error("Invalid plan");
        static_cast<fp8gemm::dual_online::PlanBase*>(p)->set_raster_group(size);return 0;
    }catch(const std::exception&e){fp8gemm::dual_online::error=e.what();return -1;}
}
extern "C" void tk_dual_destroy(void*p){delete static_cast<fp8gemm::dual_online::PlanBase*>(p);}
