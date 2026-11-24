// Stream-K adaptation of persistent_mxfp8_swiglu_dgrad.cu at 85d003d.
// Fused dX[M,K] = dG[M,N] @ Wg[N,K] + dU[M,N] @ Wu[N,K].
// Both gradients use packed per-row/N32 E8M0 scales; both weights use the
// forward 32x32 scale grid. The swapAB path shares one FP32 accumulator and
// writes dX once as BF16 through TMA.
#include "kittens.cuh"
#include "mxfp8_streamk.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::streamk_swd {
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
        A gate[2],up[2];
        B weight_gate,weight_up;
    };
    // Compact metadata uses the same acquire/publish/release as input[slot].
    using AScale=std::conditional_t<BN==32,uint8_t,
                 std::conditional_t<BN==64,uint16_t,uint32_t>>;
    struct ScalePacket {
        AScale gate[8*R],up[8*R];
        uint8_t wg[(BN/32)*(C/32)],wu[(BN/32)*(C/32)];
    };
    struct Shared {
        Packet input[S];
        ScalePacket scales[S];
        Output output[2];
    };
    struct Args {
        Global<A> gate,up;
        Global<B> weight_gate,weight_up;
        Global<O> out;
        const uint32_t *gate_scale,*up_scale;
        const uint8_t *weight_gate_scale,*weight_up_scale;
        int m,k,n,nrows,ncols,total,steps,group,as_stride,bs_stride;
        sk::Params<R,C,1> sk;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    static_assert((BN/32)*(C/32)<=32);

    struct ScaleValues {
        uint32_t gate[R/16],up[R/16],weight_gate,weight_up;
    };
};

template<int R,int C,int BN>
__device__ __forceinline__ void transpose_weight(
    kt::rt_fp8e4m3<C,BN> &weight_a,
    const kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> &weight_b) {
#pragma unroll
    for(int j=0;j<C/16;j++)
#pragma unroll
        for(int t=0;t<BN/32;t++)
#pragma unroll
            for(int v=0;v<4;v++)
                weight_a.tiles[j][t].data[v]=weight_b.tiles[t][j].data[v];
}

template<int R,int C,int BN,int S,bool Atomic=sk::atomic>
__global__ __launch_bounds__(384,1) void persistent_swiglu_dgrad(
    const __grid_constant__ typename Config<R,C,BN,S>::Args args) {
    using CF=Config<R,C,BN,S>;
    using Shared=typename CF::Shared;
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
                    typename CF::AScale pg[8*R/32],pu[8*R/32];
                    uint32_t pwg=0,pwu=0;
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        int row=nr+lane+i*32;
                        size_t index=size_t(step*(BN/32)/4)*args.as_stride+row;
                        pg[i]=(row<args.m?args.gate_scale[index]:0)>>(8*((step*(BN/32))%4));
                        pu[i]=(row<args.m?args.up_scale[index]:0)>>(8*((step*(BN/32))%4));
                    }
                    if(lane<(BN/32)*(C/32)) {
                        int row=step*(BN/32)+lane/(C/32),col=kc/32+lane%(C/32);
                        bool valid=row<(args.n+31)/32 && col<args.bs_stride;
                        size_t index=size_t(row)*args.bs_stride+col;
                        pwg=valid?args.weight_gate_scale[index]:0;
                        pwu=valid?args.weight_up_scale[index]:0;
                    }
                    auto p=pipe.waitBuffer(0,shared.input);
                    auto &tile=p.template get<0>();
                    auto &scale=shared.scales[p.SlotIdx];
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,(16*R+2*C)*BN);
                        kt::tma::load_async(tile.gate[0],args.gate,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.gate[1],args.gate,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.up[0],args.up,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.up[1],args.up,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.weight_gate,args.weight_gate,
                                            {0,0,step*BN,kc},bar);
                        kt::tma::load_async(tile.weight_up,args.weight_up,
                                            {0,0,step*BN,kc},bar);
                    }
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        scale.gate[lane+i*32]=pg[i];
                        scale.up[lane+i*32]=pu[i];
                    }
                    if(lane<(BN/32)*(C/32)) {
                        scale.wg[lane]=pwg;scale.wu[lane]=pwu;
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
            kt::rt_fl<C,R> accum;
            kt::warp::rt_maps::zero(accum);
            for(int step=work.begin;step<work.end;step++) {
                auto p=pipe.waitBuffer(2,shared.input);
                auto &tile=p.template get<0>();
                const auto &scale=shared.scales[p.SlotIdx];
                typename CF::ScaleValues current;
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    int row=group*4*R+warp*R+i*16+lane/4+8*(lane&1);
                    current.gate[i]=scale.gate[row];current.up[i]=scale.up[row];
                }
                current.weight_gate=scale.wg[lane%((BN/32)*(C/32))];
                current.weight_up=scale.wu[lane%((BN/32)*(C/32))];
                {
                    kt::rt_fp8e4m3<R,BN> a;
                    kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> b;
                    kt::mxfp8_scale<R,BN> as;
                    kt::mxfp8_scale<C,BN> bs;
                    load_fp8_factored<4,true>(a,tile.gate[group]);
                    load_fp8_col_factored(b,tile.weight_gate);
#pragma unroll
                    for(int i=0;i<R/16;i++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            as.data[i][t]=
                                current.gate[i]>>(8*t);
#pragma unroll
                    for(int j=0;j<C/16;j++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            bs.data[j][t]=__shfl_sync(
                                0xffffffff,current.weight_gate,t*(C/32)+j/2);
                    kt::rt_fp8e4m3<C,BN> weight_a;
                    transpose_weight<R,C,BN>(weight_a,b);
                    kt::warp::wmma::mma_ABt_block_scaled(
                        accum,weight_a,a,accum,bs,as);
                }
                {
                    kt::rt_fp8e4m3<R,BN> a;
                    kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> b;
                    kt::mxfp8_scale<R,BN> as;
                    kt::mxfp8_scale<C,BN> bs;
                    load_fp8_factored<4,true>(a,tile.up[group]);
                    load_fp8_col_factored(b,tile.weight_up);
#pragma unroll
                    for(int i=0;i<R/16;i++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            as.data[i][t]=
                                current.up[i]>>(8*t);
#pragma unroll
                    for(int j=0;j<C/16;j++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            bs.data[j][t]=__shfl_sync(
                                0xffffffff,current.weight_up,t*(C/32)+j/2);
                    kt::rt_fp8e4m3<C,BN> weight_a;
                    transpose_weight<R,C,BN>(weight_a,b);
                    p.submitToNextAndTrigger();pipe.moveNext();
                    kt::warp::wmma::mma_ABt_block_scaled(
                        accum,weight_a,a,accum,bs,as);
                }
            }
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::rt<float,R,C,kt::ducks::rt_layout::col> transposed;
#pragma unroll
            for(int i=0;i<R/16;i++)
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int v=0;v<4;v++)
                        transposed.tiles[i][j].data[v]=accum.tiles[j][i].data[v];
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
using CF=fp8gemm::streamk_swd::Config<16,64,64,2>;
#define SK_KERNEL fp8gemm::streamk_swd::persistent_swiglu_dgrad<16,64,64,2>
#define SK_R 16
#define SK_C 64
#define SK_COMP 1
static CF::Args make_args(void **p,int m,int n,int k,int slots) {
    return {fp8gemm::sk::Global<CF::A>((kt::fp8e4m3*)p[0],1,1,m,k),
        fp8gemm::sk::Global<CF::A>((kt::fp8e4m3*)p[1],1,1,m,k),
        fp8gemm::sk::Global<CF::B>((kt::fp8e4m3*)p[2],1,1,k,n),
        fp8gemm::sk::Global<CF::B>((kt::fp8e4m3*)p[3],1,1,k,n),
        fp8gemm::sk::Global<CF::O>((kt::bf16*)p[8],1,1,m,n),
        (uint32_t*)p[4],(uint32_t*)p[5],(uint8_t*)p[6],(uint8_t*)p[7],
        m,n,k,(m+127)/128,(n+63)/64,((m+127)/128)*((n+63)/64),k/64,1,(m+15)/16*16,n/32,{fp8gemm::sk::Global<CF::Partial>((float*)p[11],1,1,std::max(1,slots)*SK_COMP*128,SK_C),
        (fp8gemm::sk::Work*)p[12],(int*)p[13]}};
}
#include "mxfp8_streamk_host.cuh"
