// Stream-K adaptation of persistent_mxfp8_dgrad.cu at 85d003d.
// dX[M,K] = dY[M,N] @ W[N,K], computed as (W^T @ dY^T)^T.
// Prequantized E4M3 dY uses per-row/N32 scales; W uses existing 32x32 scales.
// Native transposing LDSM feeds QMMA A directly; row LDSM feeds B directly.
// FP32 accumulation, BF16 output only; no global transpose or online quantization.
#include "kittens.cuh"
#include "mxfp8_streamk.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::streamk_dgrad {
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;
__device__ __forceinline__ int2 raster(int task,int columns,int rows,int group) {
    int span=group*columns,first=task/span*group,count=min(group,rows-first),local=task%span;
    return make_int2(first+local%count,local/count);
}
template<int R,int C,int BN,int S> struct Config {
    using A=kt::st_fp8e4m3<4*R,BN>;
    using B=kt::st_fp8e4m3<BN,C>;
    using O=kt::st_bf<4*R,C>;
    using Partial=kt::st_fl<4*R,C/2>;
    union Output {O bf;Partial fp;};
    struct Packet {
        A a[2]; B b;
        uint32_t scale_a[8*R], scale_b[(BN/32)*(C/32)];
    };
    struct Shared {Packet input[S];Output output[2];};
    struct Args {
        Global<A> a;Global<B> b;Global<O> out;
        // dY: packed [ceil(N/128),roundup(M,16)]; W: bytes [ceil(N/32),K/32].
        const uint32_t *as;
        const uint8_t *bs;
        int m,k,n,nrows,ncols,total,steps,group,as_stride,bs_stride;
        sk::Params<R,C,1> sk;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    static_assert((BN/32)*(C/32)<=32);
    struct ScaleValues {
        uint32_t a[R/16],b;
    };
};

template<int R,int C,int BN,int S,bool Atomic=sk::atomic>
__global__ __launch_bounds__(384,1) void persistent_mxfp8_dgrad(
    const __grid_constant__ typename Config<R,C,BN,S>::Args args) {
    using CF=Config<R,C,BN,S>;using Shared=typename CF::Shared;
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
            for(int wi=args.sk.offsets[blockIdx.x];wi<args.sk.offsets[blockIdx.x+1];wi++) {
                const sk::Work work=args.sk.work[wi];
                int2 coord=make_int2(work.nr/(8*R),work.kc/C);
                int nr=coord.x*8*R,kc=coord.y*C;
                for(int step=work.begin;step<work.end;step++) {
                    uint32_t prefetched_a[8*R/32], prefetched_b=0;
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        int global_row=nr+lane+i*32;
                        prefetched_a[i]=global_row<args.m
                            ?args.as[size_t(step*(BN/32)/4)*args.as_stride+global_row]:0;
                    }
                    if(lane<(BN/32)*(C/32)) {
                        int row=step*(BN/32)+lane/(C/32), col=kc/32+lane%(C/32);
                        prefetched_b=(row<(args.n+31)/32 && col<args.bs_stride)
                            ?args.bs[size_t(row)*args.bs_stride+col]:0;
                    }
                    auto p=pipe.waitBuffer(0,shared.input);auto &tile=p.template get<0>();
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BN);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,step*BN,kc},bar);
                    }
                    // All producer lanes publish their scale stores through the same
                    // release/acquire generation as the TMA matrix tile.
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        tile.scale_a[lane+i*32]=prefetched_a[i];
                    }
                    if(lane<(BN/32)*(C/32)) {
                        tile.scale_b[lane]=prefetched_b;
                    }
                    last=p.SlotIdx;p.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int wi=args.sk.offsets[blockIdx.x];wi<args.sk.offsets[blockIdx.x+1];wi++) {
                const sk::Work work=args.sk.work[wi];
            int2 coord=make_int2(work.nr/(8*R),work.kc/C);
            int nr=coord.x*8*R+group*4*R+warp*R,kc=coord.y*C;
            kt::rt_fl<C,R> accum;kt::warp::rt_maps::zero(accum);
            typename CF::ScaleValues current;
            // Scales arrive in the acquired matrix packet.
            for(int step=work.begin;step<work.end;step++) {
                auto p=pipe.waitBuffer(2,shared.input);auto &tile=p.template get<0>();
                // Read every scale before releasing this input generation.
#pragma unroll
                for(int i=0;i<R/16;i++)
                    current.a[i]=tile.scale_a[group*4*R+warp*R+i*16+lane/4+8*(lane&1)];
                current.b=tile.scale_b[lane%((BN/32)*(C/32))];
                kt::rt_fp8e4m3<R,BN> ar;
                kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> br;
                kt::mxfp8_scale<R,BN> as;kt::mxfp8_scale<C,BN> bs;
                load_fp8_factored<4,true>(ar,tile.a[group]);
                // Native transposing LDSM is already in QMMA A register order.
                load_fp8_col_factored(br,tile.b);
#pragma unroll
                for(int i=0;i<R/16;i++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
                        as.data[i][t]=current.a[i]>>(8*((step*(BN/32)+t)%4));
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
                        bs.data[j][t]=__shfl_sync(0xffffffff,current.b,t*(C/32)+j/2);
                // No consumer global scale prefetch.
                p.submitToNextAndTrigger();pipe.moveNext();
                kt::rt_fp8e4m3<C,BN> weight_a;
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
#pragma unroll
                        for(int v=0;v<4;v++)weight_a.tiles[j][t].data[v]=br.tiles[t][j].data[v];
                kt::warp::wmma::mma_ABt_block_scaled(accum,weight_a,ar,accum,bs,as);
            }
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            // Logical transpose plus BF16 conversion/STSM.trans restores [M,K].
            kt::rt<float,R,C,kt::ducks::rt_layout::col> transposed;
#pragma unroll
            for(int i=0;i<R/16;i++)
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int v=0;v<4;v++)transposed.tiles[i][j].data[v]=accum.tiles[j][i].data[v];
            if(work.slot>=0) {
                sk::store_partial<R,C,1>(shared.output[group].fp,transposed,args.sk,work.slot,0,group,writer);
                continue;
            }
            kt::warpgroup::store(shared.output[group].bf,transposed);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.out,shared.output[group].bf,
                                    {0,0,coord.x*8*R+group*4*R,kc});
                kt::tma::store_commit_group();
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}
}
using CF=fp8gemm::streamk_dgrad::Config<16,128,128,2>;
#define SK_KERNEL fp8gemm::streamk_dgrad::persistent_mxfp8_dgrad<16,128,128,2>
#define SK_R 16
#define SK_C 128
#define SK_COMP 1
static CF::Args make_args(void **p,int m,int n,int k,int slots) {
    return {fp8gemm::sk::Global<CF::A>((kt::fp8e4m3*)p[0],1,1,m,k),
        fp8gemm::sk::Global<CF::B>((kt::fp8e4m3*)p[1],1,1,k,n),
        fp8gemm::sk::Global<CF::O>((kt::bf16*)p[8],1,1,m,n),
        (uint32_t*)p[2],(uint8_t*)p[3],m,n,k,(m+127)/128,(n+127)/128,
        ((m+127)/128)*((n+127)/128),k/128,1,(m+15)/16*16,n/32,{fp8gemm::sk::Global<CF::Partial>((float*)p[11],1,1,std::max(1,slots)*SK_COMP*128,SK_C),
        (fp8gemm::sk::Work*)p[12],(int*)p[13]}};
}
#include "mxfp8_streamk_host.cuh"
