// Inference-only MXFP8 GEMM + register topK. The persistent MMA pipeline follows
// persistent_mxfp8_gemm.cu; no full logits or output shared tile is allocated.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include "mxfp8_topk.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::topk {
template<class T> using Global=kt::gl<typename T::dtype,-1,-1,-1,-1,T>;
__device__ __forceinline__ int2 raster(int task,int nt,int mt,int g) {
    int span=g*nt,first=task/span*g,rows=min(g,mt-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}
template<int R,int C,int BK,int S> struct Config {
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using BBuffer=TopKPermutationBuffer<C,BK>;
    static constexpr int SC=BK/32,SR=C/32;
    static_assert(SR*SC<=32);
    struct Packet {ATile a[2];BBuffer b;};
    struct Shared {Packet input[S];};
    struct Args {
        Global<ATile> a;
        Global<BBuffer> b;
        Global<BTile> btail;
        const uint32_t *as;
        const uint8_t *bs;
        uint64_t *out;
        int m,n,h,sc,ntiles,mtiles,total,ktiles,raster_group;
        uint32_t as_stride;
        bool permute_n;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    struct ScalePrefetch {
        uint32_t a[R/16],b;
        __device__ __forceinline__ void load(const Args& args,int mr,int nc,int k,int lane) {
#pragma unroll
            for(int i=0;i<R/16;i++) {
                int row=mr+i*16+lane/4+8*(lane&1);
                a[i]=(row<args.m && k<args.ktiles)
                    ?args.as[size_t(uint32_t(k)*SC/4)*args.as_stride+uint32_t(row)]:0;
            }
            int row=nc/32+(lane/SC)%SR,col=k*SC+lane%SC;
            b=(row<(args.n+31)/32 && col<args.sc)
                ?args.bs[size_t(row)*args.sc+col]:0;
        }
    };
};

template<int R,int C,int BK,int S,int K>
__global__ __launch_bounds__(384,1) void kernel(
    const __grid_constant__ typename Config<R,C,BK,S>::Args args) {
    using CF=Config<R,C,BK,S>;using Shared=typename CF::Shared;
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
                int2 coord=raster(task,args.ntiles,args.mtiles,args.raster_group);
                int mr=coord.x*8*R,nc=coord.y*C;
                bool permuted=args.permute_n && nc+C<=args.n;
                for(int k=0;k<args.ktiles;k++) {
                    auto packet=pipe.waitBuffer(0,shared.input);auto &tile=packet.template get<0>();
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        if(permuted)kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                        else kt::tma::load_async(tile.b.payload,args.btail,{0,0,nc,k*BK},bar);
                    }
                    last=packet.SlotIdx;packet.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> accum;kt::warp::rt_maps::zero(accum);
            int2 coord=raster(task,args.ntiles,args.mtiles,args.raster_group);
            int mr=coord.x*8*R+group*4*R+warp*R,nc=coord.y*C;
            bool permuted=args.permute_n && nc+C<=args.n;
            typename CF::ScalePrefetch next;
            next.load(args,mr,nc,0,lane);
            for(int k=0;k<args.ktiles;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);auto &tile=packet.template get<0>();
                kt::rt_fp8e4m3<R,BK> ar;kt::rt_fp8e4m3<C,BK> br;
                kt::mxfp8_scale<R,BK> as;kt::mxfp8_scale<C,BK> bs;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_factored<1,true>(br,tile.b.payload);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    uint32_t word=next.a[i];
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)as.data[i][kk]=word >> (8*((k*CF::SC+kk)%4));
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
                    // Metadata is per physical MMA row, not per output tile.
                    // Follow exactly the same N permutation as the FP8 values.
                    int physical=i*16+lane/4+8*(lane%2);
                    int logical=permuted ? topk_permute_n<C>(physical):physical;
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)
                        bs.data[i][kk]=__shfl_sync(0xffffffff,next.b,(logical/32)*CF::SC+kk);
                }
                next.load(args,mr,nc,k+1,lane);
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(accum,ar,br,accum,as,bs);
            }
            store_topk<R,C,K>(accum,args.out,mr,nc,args.m,args.n,permuted);
        }
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Base {
    virtual ~Base()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void rebind(void*,void*,void*,void*,void*)=0;
    virtual void info(int*)=0;
};
template<int R,int C,int BK,int S,int K> struct Plan:Base {
    using CF=Config<R,C,BK,S>;
    typename CF::Args args;int grid,regs;
    Plan(void*a,void*b,void*as,void*bs,void*out,int m,int n,int h,int sms,int g,bool permute):
        args{Global<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,h),
             Global<typename CF::BBuffer>((kt::fp8e4m3*)b,1,1,n,h),
             Global<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,h),
             (uint32_t*)as,(uint8_t*)bs,(uint64_t*)out,m,n,h,h/32,
             (n+C-1)/C,(m+8*R-1)/(8*R),((m+8*R-1)/(8*R))*((n+C-1)/C),
             (h+BK-1)/BK,std::min(g,(m+8*R-1)/(8*R)),uint32_t((m+15)/16*16),permute} {
        if(h%CF::ATile::swizzle_bytes || h%CF::BTile::swizzle_bytes)
            throw std::runtime_error("H must align to TMA panels");
        int dev;check(cudaGetDevice(&dev));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        check(cudaFuncSetAttribute(kernel<R,C,BK,S,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,kernel<R,C,BK,S,K>));regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes)throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t stream) override {
        kernel<R,C,BK,S,K><<<grid,384,CF::smem,stream>>>(args);check(cudaGetLastError());
    }
    void rebind(void*a,void*b,void*as,void*bs,void*out) override {
        args.a=Global<typename CF::ATile>((kt::fp8e4m3*)a,1,1,args.m,args.h);
        args.b=Global<typename CF::BBuffer>((kt::fp8e4m3*)b,1,1,args.n,args.h);
        args.btail=Global<typename CF::BTile>((kt::fp8e4m3*)b,1,1,args.n,args.h);
        args.as=(uint32_t*)as;args.bs=(uint8_t*)bs;args.out=(uint64_t*)out;
    }
    void info(int*d) override {d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;}
};
static thread_local std::string error;
template<int K> static Base* create(void*a,void*b,void*as,void*bs,void*out,
    int m,int n,int h,int config,int sms,int g,bool permute) {
#define CFG(I,R,C,BK,S) case I:return new Plan<R,C,BK,S,K>(a,b,as,bs,out,m,n,h,sms,g,permute)
    switch(config) {
#include "mxfp8_topk_configs.inc"
        default:throw std::runtime_error("Invalid topK config");
    }
#undef CFG
}
}
extern "C" const char *tk_mxt_error(){return fp8gemm::topk::error.c_str();}
extern "C" void *tk_mxt_create(void*a,void*b,void*as,void*bs,void*out,
    int m,int n,int h,int config,int k,int sms,int g,int permute) {
    using namespace fp8gemm::topk;
    try {
        if(!a||!b||!as||!bs||!out||m<=0||n<=0||h<=0||h%32||g<=0||sms<0)
            throw std::runtime_error("Invalid topK arguments");
#define CASE(K) case K:return create<K>(a,b,as,bs,out,m,n,h,config,sms,g,permute)
        switch(k){CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);default:throw std::runtime_error("Invalid topK bucket");}
#undef CASE
    }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" int tk_mxt_launch(fp8gemm::topk::Base*p,void*s){
    try{p->launch((cudaStream_t)s);return 0;}catch(const std::exception&e){fp8gemm::topk::error=e.what();return -1;}
}
extern "C" int tk_mxt_rebind(fp8gemm::topk::Base*p,void*a,void*b,void*as,void*bs,void*out){
    try{p->rebind(a,b,as,bs,out);return 0;}catch(const std::exception&e){fp8gemm::topk::error=e.what();return -1;}
}
extern "C" void tk_mxt_info(fp8gemm::topk::Base*p,int*d){p->info(d);}
extern "C" void tk_mxt_destroy(fp8gemm::topk::Base*p){delete p;}
