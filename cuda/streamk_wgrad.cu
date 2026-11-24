// Isolated config-12 hybrid Stream-K prototype. Torch-free translation unit.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

#ifndef STREAMK_ATOMIC
#define STREAMK_ATOMIC 0
#endif

namespace fp8gemm::streamk {
constexpr int R=16,C=128,BM=128,S=2;
constexpr bool atomic_reduce=STREAMK_ATOMIC;
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;
using A=kt::st_fp8e4m3<4*R,BM>;
using B=kt::st_fp8e4m3<BM,C>;
using O=kt::st_bf<4*R,C>;
using P=kt::st_fl<4*R,C/2>;
struct Work {int nr,kc,begin,end,slot;};
struct Fixup {int nr,kc,first,count;};
struct Packet {A a[2];B b;uint32_t scale_a[8*R],scale_b[C];};
// Two FP32 column panels reuse the same bytes as one BF16 output tile.
union Output {O bf;P fp;};
struct Shared {Packet input[S];Output output[2];};
constexpr int smem=1024+sizeof(Shared);
static_assert(smem==101376);
struct Args {
    Global<A> a;Global<B> b;Global<O> out;Global<P> partial;
    const uint32_t *as,*bs;
    const Work *work;
    const int *offsets;
    int n,k,m,as_stride,bs_stride;
};

template<bool Atomic=atomic_reduce>
__global__ __launch_bounds__(384,1) void hybrid_streamk_wgrad(
    const __grid_constant__ Args args) {
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=kt::laneid();
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto &shared=allocator.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            int first=args.offsets[blockIdx.x],end=args.offsets[blockIdx.x+1];
            for(int wi=first;wi<end;wi++) {
                const Work w=args.work[wi];
                for(int step=w.begin;step<w.end;step++) {
                    uint32_t pa[8*R/32],pb[C/32];
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        int row=w.nr+lane+i*32;
                        pa[i]=row<args.n?args.as[size_t(step)*args.as_stride+row]:0;
                    }
#pragma unroll
                    for(int i=0;i<C/32;i++) {
                        int col=w.kc+lane+i*32;
                        pb[i]=col<args.k?args.bs[size_t(step)*args.bs_stride+col]:0;
                    }
                    auto p=pipe.waitBuffer(0,shared.input);auto &tile=p.template get<0>();
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BM);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,w.nr,step*BM},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,w.nr+4*R,step*BM},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,step*BM,w.kc},bar);
                    }
#pragma unroll
                    for(int i=0;i<8*R/32;i++)tile.scale_a[lane+i*32]=pa[i];
#pragma unroll
                    for(int i=0;i<C/32;i++)tile.scale_b[lane+i*32]=pb[i];
                    last=p.SlotIdx;p.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            if(first<end)pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        int first=args.offsets[blockIdx.x],end=args.offsets[blockIdx.x+1];
        for(int wi=first;wi<end;wi++) {
            const Work w=args.work[wi];
            kt::rt_fl<R,C> accum;kt::warp::rt_maps::zero(accum);
            for(int step=w.begin;step<w.end;step++) {
                auto p=pipe.waitBuffer(2,shared.input);auto &tile=p.template get<0>();
                uint32_t sa=tile.scale_a[group*4*R+warp*R+lane/4+8*(lane&1)];
                uint32_t sb[C/16];
#pragma unroll
                for(int j=0;j<C/16;j++)sb[j]=tile.scale_b[j*16+lane/4+8*(lane&1)];
                kt::rt_fp8e4m3<R,BM> ar;
                kt::rt<kt::fp8e4m3,BM,C,kt::ducks::rt_layout::col> br;
                kt::mxfp8_scale<R,BM> as;kt::mxfp8_scale<C,BM> bs;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_col_factored(br,tile.b);
#pragma unroll
                for(int t=0;t<BM/32;t++)as.data[0][t]=sa>>(8*t);
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int t=0;t<BM/32;t++)bs.data[j][t]=sb[j]>>(8*t);
                p.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_AB(accum,ar,br,accum,as,bs);
            }
            if(w.slot<0) {
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                kt::warpgroup::store(shared.output[group].bf,accum);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.out,shared.output[group].bf,
                                        {0,0,w.nr+group*4*R,w.kc});
                    kt::tma::store_commit_group();
                }
            } else {
#pragma unroll
                for(int panel=0;panel<2;panel++) {
                    kt::rt_fl<R,C/2> part;
#pragma unroll
                    for(int j=0;j<C/32;j++)
#pragma unroll
                        for(int p=0;p<4;p++)
                            part.tiles[0][j].data[p]=accum.tiles[0][j+panel*(C/32)].data[p];
                    if(writer)kt::tma::store_async_read_wait<0>();
                    kt::warpgroup::sync(group+1);
                    kt::warpgroup::store(shared.output[group].fp,part);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    kt::warpgroup::sync(group+1);
                    if(writer) {
                        if constexpr(Atomic)
                            kt::tma::store_add_async(args.partial,shared.output[group].fp,
                                {0,0,w.slot*(8*R)+group*4*R,panel*(C/2)});
                        else
                            kt::tma::store_async(args.partial,shared.output[group].fp,
                                {0,0,w.slot*(8*R)+group*4*R,panel*(C/2)});
                        kt::tma::store_commit_group();
                    }
                }
            }
        }
        // Full destination completion, not only shared-source reuse, precedes
        // kernel completion and the stream-ordered fixup launch.
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

template<bool Atomic=atomic_reduce>
__global__ __launch_bounds__(256) void streamk_wgrad_fixup(const float *__restrict__ partial,
    kt::bf16 *__restrict__ out,const Fixup *__restrict__ fixups,
    const int *__restrict__ contributors,int n,int k) {
    Fixup f=fixups[blockIdx.x];
    for(int index=threadIdx.x*4;index<8*R*C;index+=256*4) {
        float4 sum=make_float4(0,0,0,0);
        if constexpr(Atomic) {
            sum=*reinterpret_cast<const float4*>(partial+size_t(blockIdx.x)*(8*R*C)+index);
        } else {
        // The host lists fragments in increasing K order; no global atomics.
        for(int i=0;i<f.count;i++) {
            int slot=contributors[f.first+i];
            float4 x=*reinterpret_cast<const float4*>(partial+size_t(slot)*(8*R*C)+index);
            sum.x+=x.x;sum.y+=x.y;sum.z+=x.z;sum.w+=x.w;
        }
        }
        int row=f.nr+index/C,col=f.kc+index%C;
        if(row<n && col+3<k) {
            auto *dst=reinterpret_cast<__nv_bfloat162*>(out+size_t(row)*k+col);
            dst[0]=__float22bfloat162_rn(make_float2(sum.x,sum.y));
            dst[1]=__float22bfloat162_rn(make_float2(sum.z,sum.w));
        }
    }
}

void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Plan {
    Args args;
    const Fixup *fixups;
    const int *contributors;
    int grid,nfix,regs;
    Plan(void *a,void *b,void *as,void *bs,void *out,void *partial,void *work,
         void *offsets,void *fix,void *ids,int n,int k,int m,int grid_,int slots,int nf):
        args{Global<A>((kt::fp8e4m3*)a,1,1,n,m),Global<B>((kt::fp8e4m3*)b,1,1,m,k),
             Global<O>((kt::bf16*)out,1,1,n,k),Global<P>((float*)partial,1,1,std::max(1,slots)*(8*R),C),
             (uint32_t*)as,(uint32_t*)bs,(Work*)work,(int*)offsets,n,k,m,(n+15)/16*16,(k+15)/16*16},
        fixups((Fixup*)fix),contributors((int*)ids),grid(grid_),nfix(nf) {
        if(n<=0 || k<=0 || m<=0 || m%BM || k%C || grid<=0)
            throw std::runtime_error("Invalid config-12 shape/grid");
        check(cudaFuncSetAttribute(hybrid_streamk_wgrad<>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
        cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,hybrid_streamk_wgrad<>));regs=attr.numRegs;
        if(attr.localSizeBytes || regs<168)throw std::runtime_error("Unexpected register pool/local memory");
    }
    void launch(cudaStream_t stream,int phase) {
        // Clearing is stream ordered before ALL contributors, on every replay.
        // Never mix an uncoordinated first-writer store with later atomic adds.
        if constexpr(atomic_reduce) {
            if(phase!=2 && nfix)
                check(cudaMemsetAsync(args.partial.raw_ptr,0,size_t(nfix)*(8*R*C)*sizeof(float),stream));
        }
        if(phase==3)return;
        if(phase!=2) {
            hybrid_streamk_wgrad<><<<grid,384,smem,stream>>>(args);check(cudaGetLastError());
        }
        if(phase!=1 && nfix) {
            streamk_wgrad_fixup<><<<nfix,256,0,stream>>>(args.partial.raw_ptr,args.out.raw_ptr,
                                                    fixups,contributors,args.n,args.k);
            check(cudaGetLastError());
        }
    }
};
static thread_local std::string error;
}
extern "C" const char *tk_sk_error(){return fp8gemm::streamk::error.c_str();}
extern "C" void *tk_sk_create(void *a,void *b,void *as,void *bs,void *out,void *partial,
    void *work,void *offsets,void *fix,void *ids,int n,int k,int m,int grid,int slots,int nfix) {
    try {return new fp8gemm::streamk::Plan(a,b,as,bs,out,partial,work,offsets,fix,ids,n,k,m,grid,slots,nfix);}
    catch(const std::exception &e){fp8gemm::streamk::error=e.what();return nullptr;}
}
extern "C" int tk_sk_launch(fp8gemm::streamk::Plan *p,void *stream,int phase) {
    try {p->launch((cudaStream_t)stream,phase);return 0;}
    catch(const std::exception &e){fp8gemm::streamk::error=e.what();return -1;}
}
extern "C" void tk_sk_info(fp8gemm::streamk::Plan *p,int *info) {
    info[0]=fp8gemm::streamk::smem;info[1]=p->regs;
}
extern "C" void tk_sk_destroy(fp8gemm::streamk::Plan *p){delete p;}
