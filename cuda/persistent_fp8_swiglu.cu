// SM120a fused up(x)*SiLU(gate(x)); weight quantization remains in Triton.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include <stdexcept>
#include <string>

#ifndef FP8_SWIGLU_FMA
#define FP8_SWIGLU_FMA 1
#endif

namespace fp8gemm::swiglu {
namespace kt=kittens;
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

template<int R,int C,int BK,int S,bool SaveX=false> struct Config {
    using A=kt::st_bf<4*R,BK>;
    using B=kt::st_fp8e4m3<C,BK>;
    using O=kt::st_bf<4*R,C>;
    struct Packet {A x[2]; B gate,up;};
    using Cache=kt::st_fp8e4m3<BK,4*R>;
    union CachedWorkspace {O output;Cache cache;};
    using Workspace=std::conditional_t<SaveX,CachedWorkspace,O>;
    struct Shared {Packet input[S]; Workspace workspace[2];};
    __device__ static O& output(Shared&s,int group) {
        if constexpr(SaveX) return s.workspace[group].output;
        else return s.workspace[group];
    }
    struct Args {
        Global<A> x;
        Global<B> gate,up;
        Global<O> out,gate_out,up_out;
        const uint8_t *gs,*us;
        int ntiles,total,ktiles,scale_rows,scale_cols;
        Global<Cache> cached_x;
        uint8_t *cached_scale;
        int cached_scale_cols;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
};

__device__ __forceinline__ float activate(float gate_half,float up) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(gate_half));
    // h*(t+1)*up -> fma(h,t,h)*up. This saves one FP32 instruction;
    // one fewer rounding can change BF16 ties relative to Triton's ordering.
#if FP8_SWIGLU_FMA
    return fmaf(gate_half,t,gate_half)*up;
#else
    return (gate_half*(t+1.f))*up;
#endif
}

// Each warp owns CR consecutive feature columns and all 4R tokens of its
// consumer WG. LDSM.T produces row-layout registers of X^T directly, without
// an intermediate shared transpose or global input load.
template<int R,int BK>
__device__ __forceinline__ void save_transposed_x(
    const kt::st_bf<4*R,BK>&src,kt::st_fp8e4m3<BK,4*R>&dst,
    const Global<kt::st_fp8e4m3<BK,4*R>>&global,uint8_t*scale,int scale_cols,
    int token_base,int feature_base,bool writer,int group) {
    constexpr int CR=BK>=64?BK/4:16,CW=BK/CR;
    int warp=kt::warpgroup::warpid(),lane=threadIdx.x%32;
    // The shared workspace aliases the prior output/cache store. No overwrite
    // until its asynchronous shared reads finish; complete the wait in writer.
    if(writer) kt::tma::store_async_read_wait<0>();
    kt::warpgroup::sync(group+1);
    if(warp<CW) {
        kt::rt_bf<CR,4*R> xt;
        kt::rt_fp8e4m3<CR,4*R> q;
        kt::mxfp8_scale<CR,4*R> scales;
        uint32_t base=uint32_t(__cvta_generic_to_shared(src.data));
#pragma unroll
        for(int i=0;i<CR/16;++i) {
#pragma unroll
            for(int j=0;j<4*R/16;++j) {
                auto &d=xt.tiles[i][j];
                uint32_t address=src.idx(base,{j*16+lane%16,warp*CR+i*16+(lane/16)*8});
                kt::move<kt::bf16_2>::ldsm4t(d.data[0],d.data[2],d.data[1],d.data[3],address);
            }
        }
        kt::online_quantize_mxfp8(q,scales,xt);
#pragma unroll
        for(int i=0;i<CR/16;++i) {
#pragma unroll
            for(int j=0;j<4*R/32;++j) {
                int col=token_base/32+j;
                int row=feature_base+warp*CR+i*16+lane/4+8*(lane&1);
                if((lane&2)==0 && col<scale_cols)
                    scale[size_t(row)*scale_cols+col]=uint8_t(scales.data[i][j]);
            }
        }
        kt::group<CW>::store(dst,q);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    kt::warpgroup::sync(group+1);
    if(writer) {
        kt::tma::store_async(global,dst,{0,0,feature_base,token_base});
        kt::tma::store_commit_group();
    }
}

template<int R,int C,int BK,int S,bool Aux,bool SaveX=false>
__global__ __launch_bounds__(384,1) void persistent_swiglu(
    const __grid_constant__ typename Config<R,C,BK,S,SaveX>::Args args) {
    using CF=Config<R,C,BK,S,SaveX>;
    using Shared=typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> alloc(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid();
    MbarrierRingPipe pipe(alloc,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    constexpr int SR=(C+31)/32,SC=BK/32;
    static_assert(2*S*sizeof(kt::semaphore)+16+S*2*SR*SC*4<=1024);
    auto &scales=alloc.template allocate<16,uint32_t,S,2,SR,SC>();
    auto &shared=alloc.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int mr=task/args.ntiles*(8*R),nc=task%args.ntiles*C;
                for(int k=0;k<args.ktiles;++k) {
                    auto p=pipe.waitBuffer(0,shared.input);
                    auto &tile=p.template get<0>();
                    int lane=threadIdx.x%32;
                    if(lane<SR*SC) {
                        int sr=lane/SC,sc=lane%SC,gr=nc/32+sr,gc=k*SC+sc;
                        uint32_t gs=0,us=0;
                        if(gr<args.scale_rows && gc<args.scale_cols) {
                            gs=args.gs[size_t(gr)*args.scale_cols+gc];
                            us=args.us[size_t(gr)*args.scale_cols+gc];
                        }
                        scales[p.SlotIdx][0][sr][sc]=gs;
                        scales[p.SlotIdx][1][sr][sc]=us;
                    }
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,8*R*BK*2+2*C*BK);
                        kt::tma::load_async(tile.x[0],args.x,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.x[1],args.x,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.gate,args.gate,{0,0,nc,k*BK},bar);
                        kt::tma::load_async(tile.up,args.up,{0,0,nc,k*BK},bar);
                    }
                    last=p.SlotIdx;p.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        auto &output_tile=CF::output(shared,group);
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> gate,up;
            kt::warp::rt_maps::zero(gate);kt::warp::rt_maps::zero(up);
            int next_cache_k=0;
            if constexpr(SaveX) {
                // For each M tile, rotate cache ownership across N tiles as
                // K advances. This avoids making N=0 process the entire cache
                // and becoming a persistent-CTA straggler. Every (M,K) tile
                // has exactly one owner, regardless of grid size or ordering.
                next_cache_k=task%args.ntiles-(task/args.ntiles)%args.ntiles;
                if(next_cache_k<0) next_cache_k+=args.ntiles;
            }
            for(int k=0;k<args.ktiles;++k) {
                auto p=pipe.waitBuffer(2,shared.input);
                auto &tile=p.template get<0>();
                if constexpr(SaveX) {
                    // Transpose quantization reduces M32, not forward K32.
                    if(k==next_cache_k) {
                        save_transposed_x<R,BK>(tile.x[group],shared.workspace[group].cache,
                            args.cached_x,args.cached_scale,args.cached_scale_cols,
                            task/args.ntiles*(8*R)+group*4*R,k*BK,writer,group);
                        next_cache_k+=args.ntiles;
                    }
                }
                kt::rt_bf<R,BK> xr;
                kt::rt_fp8e4m3<R,BK> xq;
                kt::rt_fp8e4m3<C,BK> gr,ur;
                kt::mxfp8_scale<R,BK> xs,half_xs;
                kt::mxfp8_scale<C,BK> gs,us;
                kt::warpgroup::load(xr,tile.x[group]);
                kt::warp::load<true>(gr,tile.gate);
                kt::warp::load<true>(ur,tile.up);
#pragma unroll
                for(int n=0;n<C/16;++n) {
#pragma unroll
                    for(int kk=0;kk<BK/32;++kk) {
                        gs.data[n][kk]=scales[p.SlotIdx][0][n/2][kk];
                        us.data[n][kk]=scales[p.SlotIdx][1][n/2][kk];
                    }
                }
                p.submitToNextAndTrigger();pipe.moveNext();
                kt::online_quantize_mxfp8(xq,xs,xr);
#pragma unroll
                for(int r=0;r<R/16;++r) {
#pragma unroll
                    for(int kk=0;kk<BK/32;++kk)
                        // Online quantization leaves unrelated bits in upper
                        // bytes. Only byte 0 is selected by the MMA operand.
                        half_xs.data[r][kk]=max(int(xs.data[r][kk]&255)-1,0);
                }
                kt::warp::wmma::mma_ABt_block_scaled(gate,xq,gr,gate,half_xs,gs);
                kt::warp::wmma::mma_ABt_block_scaled(up,xq,ur,up,xs,us);
            }
            int mr=task/args.ntiles*(8*R)+group*4*R,nc=task%args.ntiles*C;
            // Reuse one BF16 shared output tile per consumer WG. Auxiliary
            // stores serialize shared-buffer reuse, without allocating two
            // additional full output tiles or rounding the fused intermediates.
            if constexpr(Aux) {
                kt::rt_bf<R,C> full_gate;
#pragma unroll
                for(int r=0;r<gate.height;++r) {
#pragma unroll
                    for(int c=0;c<gate.width;++c) {
#pragma unroll
                        for(int i=0;i<gate.packed_per_tile;++i) {
                            float2 g=gate.tiles[r][c].data[i];
                            full_gate.tiles[r][c].data[i]=__floats2bfloat162_rn(2.f*g.x,2.f*g.y);
                        }
                    }
                }
                if(writer) kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                kt::warpgroup::store(output_tile,full_gate);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.gate_out,output_tile,{0,0,mr,nc});
                    kt::tma::store_commit_group();kt::tma::store_async_read_wait<0>();
                }
                kt::warpgroup::sync(group+1);
                kt::warpgroup::store(output_tile,up);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.up_out,output_tile,{0,0,mr,nc});
                    kt::tma::store_commit_group();
                }
            }
#pragma unroll
            for(int r=0;r<gate.height;++r) {
#pragma unroll
                for(int c=0;c<gate.width;++c) {
#pragma unroll
                    for(int i=0;i<gate.packed_per_tile;++i) {
                        float2 g=gate.tiles[r][c].data[i],u=up.tiles[r][c].data[i];
                        gate.tiles[r][c].data[i]={activate(g.x,u.x),activate(g.y,u.y)};
                    }
                }
            }
            if(writer) kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::warpgroup::store(output_tile,gate);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.out,output_tile,{0,0,mr,nc});
                kt::tma::store_commit_group();
            }
        }
        if(writer) kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
struct PlanBase {virtual ~PlanBase()=default;virtual void launch(cudaStream_t)=0;virtual void info(int*)=0;};
template<int R,int C,int BK,int S,bool Aux,bool SaveX=false> struct Plan final:PlanBase {
    using CF=Config<R,C,BK,S,SaveX>;
    typename CF::Args args;
    int grid,regs;
    Plan(void*x,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,int m,int n,int k,int sms,void*cached=nullptr,void*cached_scale=nullptr,int padded_m=0):args{
        Global<typename CF::A>(static_cast<kt::bf16*>(x),1,1,m,k),
        Global<typename CF::B>(static_cast<kt::fp8e4m3*>(g),1,1,n,k),
        Global<typename CF::B>(static_cast<kt::fp8e4m3*>(u),1,1,n,k),
        Global<typename CF::O>(static_cast<kt::bf16*>(o),1,1,m,n),
        Global<typename CF::O>(static_cast<kt::bf16*>(Aux?go:o),1,1,m,n),
        Global<typename CF::O>(static_cast<kt::bf16*>(Aux?uo:o),1,1,m,n),
        static_cast<uint8_t*>(gs),static_cast<uint8_t*>(us),
        (n+C-1)/C,((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,(n+31)/32,(k+31)/32,
        Global<typename CF::Cache>(static_cast<kt::fp8e4m3*>(SaveX?cached:o),1,1,k,
                                   SaveX?padded_m:((m+127)/128*128)),
        static_cast<uint8_t*>(cached_scale),(SaveX?padded_m:((m+127)/128*128))/32} {
        if(k%CF::B::swizzle_bytes || n%(CF::O::swizzle_bytes/2))
            throw std::runtime_error("K/N must align to TMA swizzle panels");
        int dev;check(cudaGetDevice(&dev));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        check(cudaFuncSetAttribute(persistent_swiglu<R,C,BK,S,Aux,SaveX>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,persistent_swiglu<R,C,BK,S,Aux,SaveX>));
        regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes) throw std::runtime_error("Insufficient WASP register pool or kernel spills");
    }
    void launch(cudaStream_t s) override {
        persistent_swiglu<R,C,BK,S,Aux,SaveX><<<grid,384,CF::smem,s>>>(args);check(cudaGetLastError());
    }
    void info(int*d) override {d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;d[7]=Aux;d[8]=SaveX;d[9]=args.cached_scale_cols*32;}
};
static thread_local std::string error;
template<bool Aux,bool SaveX=false> PlanBase*create(void*x,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,int m,int n,int k,int cfg,int sms,void*cached=nullptr,void*cached_scale=nullptr,int padded_m=0) {
#define CFG(ID,R,C,BK,S) case ID:return new Plan<R,C,BK,S,Aux,SaveX>(x,g,gs,u,us,o,go,uo,m,n,k,sms,cached,cached_scale,padded_m)
    switch(cfg) {
        CFG(0,16,32,64,2);CFG(1,16,64,64,2);CFG(2,16,128,32,2);CFG(3,16,128,64,2);
        CFG(4,32,64,32,2);CFG(5,32,32,64,2);CFG(6,16,64,64,3);CFG(7,16,32,128,2);
        default:throw std::runtime_error("Unknown configuration");
    }
#undef CFG
}
}
extern "C" const char*tk_swiglu_error(){return fp8gemm::swiglu::error.c_str();}
extern "C" void*tk_swiglu_create(void*x,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,int m,int n,int k,int cfg,int sms){
    using namespace fp8gemm::swiglu;
    try {
        if(m<=0 || n<=0 || k<=0 || !x || !g || !gs || !u || !us || !o || bool(go)!=bool(uo))
            throw std::runtime_error("Invalid shape or pointers");
        return go?create<true>(x,g,gs,u,us,o,go,uo,m,n,k,cfg,sms):create<false>(x,g,gs,u,us,o,go,uo,m,n,k,cfg,sms);
    }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void*tk_swiglu_create_cached(void*x,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,
    void*cached,void*cached_scale,int m,int n,int k,int cfg,int sms,int padded_m) {
    using namespace fp8gemm::swiglu;
    try {
        if(m<=0 || n<=0 || k<=0 || !x || !g || !gs || !u || !us || !o || bool(go)!=bool(uo)
           || !cached || !cached_scale || padded_m!=(m+127)/128*128)
            throw std::runtime_error("Invalid shape or activation cache buffers");
        return go?create<true,true>(x,g,gs,u,us,o,go,uo,m,n,k,cfg,sms,cached,cached_scale,padded_m)
                 :create<false,true>(x,g,gs,u,us,o,go,uo,m,n,k,cfg,sms,cached,cached_scale,padded_m);
    }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" int tk_swiglu_launch(void*p,void*s){
    try{static_cast<fp8gemm::swiglu::PlanBase*>(p)->launch(static_cast<cudaStream_t>(s));return 0;}
    catch(const std::exception&e){fp8gemm::swiglu::error=e.what();return -1;}
}
extern "C" void tk_swiglu_info(void*p,int*d){static_cast<fp8gemm::swiglu::PlanBase*>(p)->info(d);}
extern "C" void tk_swiglu_destroy(void*p){delete static_cast<fp8gemm::swiglu::PlanBase*>(p);}
