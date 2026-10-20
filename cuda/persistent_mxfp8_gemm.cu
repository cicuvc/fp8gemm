// Matched ablation of persistent_fp8_gemm: prequantized A/B, no online quantization.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "linear_tma_buffer.cuh"
#include <stdexcept>
#include <string>
namespace fp8gemm {
template<class T> using MXGlobal=kt::gl<typename T::dtype,-1,-1,-1,-1,T>;
template<class T> using MXLinearGlobal=kt::gl<typename T::dtype,-1,-1,-1,-1,kt::tma::descriptor<T,false>>;
__device__ __forceinline__ int2 mx_raster(int task,int ntiles,int mtiles,int g) {
    if(g==1)return make_int2(task/ntiles,task%ntiles);
    int span=g*ntiles,first=task/span*g,rows=min(g,mtiles-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}
template<int R,int C,int BK,int S> struct MXConfig {
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using OTile=kt::st_bf<4*R,C>;
    using STile=LinearTile<float,1,4*R>; // Bitwise uint32: four K32 E8M0 bytes per row.
    static constexpr int SC=BK/32,SR=(C+31)/32;
    struct Packet {
        ATile a[2];BTile b;
        STile as[2];
        uint32_t bs[SR][SC];
    };
    struct Shared {Packet input[S];OTile output[2];};
    struct Args {
        MXGlobal<ATile> a;MXGlobal<BTile> b;MXGlobal<OTile> out;
        MXLinearGlobal<STile> as; // packed [ceil(K/128),roundup(M,16)] uint32 words
        const uint8_t *bs; // bs[ceil(N/32),K/32]
        int m,n,sc,ntiles,mtiles,total,ktiles,raster_group;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
};
template<int R,int C,int BK,int S>
__global__ __launch_bounds__(384,1) void persistent_mxfp8(
    const __grid_constant__ typename MXConfig<R,C,BK,S>::Args args) {
    using CF=MXConfig<R,C,BK,S>;using Shared=typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=threadIdx.x%32;
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto &shared=allocator.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int2 coord=mx_raster(task,args.ntiles,args.mtiles,args.raster_group);
                int mr=coord.x*(8*R),nc=coord.y*C;
                for(int k=0;k<args.ktiles;k++) {
                    auto packet=pipe.waitBuffer(0,shared.input);auto &tile=packet.template get<0>();
                    if(lane<CF::SR*CF::SC) {
                        int sr=lane/CF::SC,sc=lane%CF::SC,gr=nc/32+sr,gc=k*CF::SC+sc;
                        tile.bs[sr][sc]=(gr<(args.n+31)/32 && gc<args.sc)?args.bs[size_t(gr)*args.sc+gc]:0;
                    }
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BK+8*R*4);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                        kt::tma::load_async(tile.as[0],args.as,{0,0,(k*CF::SC)/4,mr},bar);
                        kt::tma::load_async(tile.as[1],args.as,{0,0,(k*CF::SC)/4,mr+4*R},bar);
                    }
                    last=packet.SlotIdx;packet.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> accum;kt::warp::rt_maps::zero(accum);
            for(int k=0;k<args.ktiles;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);auto &tile=packet.template get<0>();
                kt::rt_fp8e4m3<R,BK> ar;kt::rt_fp8e4m3<C,BK> br;
                kt::mxfp8_scale<R,BK> as;kt::mxfp8_scale<C,BK> bs;
                kt::warpgroup::load(ar,tile.a[group]);kt::warp::load<true>(br,tile.b);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    int row=warp*R+i*16+lane/4+8*(lane&1);
                    uint32_t word=std::bit_cast<uint32_t>(tile.as[group].data[row]);
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)as.data[i][kk]=word >> (8*((k*CF::SC+kk)%4));
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)bs.data[i][kk]=tile.bs[i/2][kk];
                }
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(accum,ar,br,accum,as,bs);
            }
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::warpgroup::store(shared.output[group],accum);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                int2 coord=mx_raster(task,args.ntiles,args.mtiles,args.raster_group);
                kt::tma::store_async(args.out,shared.output[group],{0,0,coord.x*8*R+group*4*R,coord.y*C});
                kt::tma::store_commit_group();
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}
static void mx_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct MXPlanBase {virtual ~MXPlanBase()=default;virtual void launch(cudaStream_t)=0;virtual void info(int*)=0;};
template<int R,int C,int BK,int S> struct MXPlan:MXPlanBase {
    using CF=MXConfig<R,C,BK,S>;typename CF::Args args;int grid,regs;
    MXPlan(void *a,void *b,void *as,void *bs,void *o,int m,int n,int k,int sms,int g):
        args{MXGlobal<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,k),
             MXGlobal<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,k),
             MXGlobal<typename CF::OTile>((kt::bf16*)o,1,1,m,n),
             MXLinearGlobal<typename CF::STile>((float*)as,1,1,(k+127)/128,((m+15)/16)*16),
             (uint8_t*)bs,m,n,(k+31)/32,(n+C-1)/C,(m+8*R-1)/(8*R),
             ((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,std::min(g,(m+8*R-1)/(8*R))} {
        if(k%CF::ATile::swizzle_bytes || k%CF::BTile::swizzle_bytes || n%(CF::OTile::swizzle_bytes/2))
            throw std::runtime_error("K/N must align to TMA swizzle panels");
        int dev;mx_check(cudaGetDevice(&dev));cudaDeviceProp prop;mx_check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        mx_check(cudaFuncSetAttribute(persistent_mxfp8<R,C,BK,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;mx_check(cudaFuncGetAttributes(&attr,persistent_mxfp8<R,C,BK,S>));regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes)throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t stream) override {
        persistent_mxfp8<R,C,BK,S><<<grid,384,CF::smem,stream>>>(args);mx_check(cudaGetLastError());
    }
    void info(int *d) override {d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;}
};
static thread_local std::string mx_error;
}
extern "C" const char *tk_mx_error(){return fp8gemm::mx_error.c_str();}
extern "C" void *tk_mx_create(void *a,void *b,void *as,void *bs,void *o,int m,int n,int k,int config,int sms,int g) {
    using namespace fp8gemm;
    try {
        if(!a||!b||!as||!bs||!o||m<=0||n<=0||k<=0||g<1)throw std::runtime_error("Invalid arguments");
#define CFG(I,R,C,K,S) case I:return new MXPlan<R,C,K,S>(a,b,as,bs,o,m,n,k,sms,g)
        switch(config) {
            CFG(0,16,32,32,2);CFG(1,16,64,32,2);CFG(2,16,128,32,2);CFG(3,32,32,32,2);
            CFG(4,32,64,32,2);CFG(5,64,16,32,2);CFG(6,64,32,32,2);
            CFG(7,16,64,64,2);CFG(8,16,128,64,2);CFG(9,32,32,64,2);
            default:throw std::runtime_error("Invalid configuration");
        }
#undef CFG
    }catch(const std::exception &e){mx_error=e.what();return nullptr;}
}
extern "C" int tk_mx_launch(fp8gemm::MXPlanBase *p,void *s){
    try{p->launch((cudaStream_t)s);return 0;}catch(const std::exception &e){fp8gemm::mx_error=e.what();return -1;}
}
extern "C" void tk_mx_info(fp8gemm::MXPlanBase *p,int *d){p->info(d);}
extern "C" void tk_mx_destroy(fp8gemm::MXPlanBase *p){delete p;}
