// Stream-K adaptation of persistent_mxfp8_swiglu.cu at 85d003d.
// SM120a persistent fused up(x) * SiLU(gate(x)) with prequantized MXFP8
// operands. Both projections share the X tile and keep independent FP32
// accumulators; the activation and BF16 conversion stay in registers.
#include "kittens.cuh"
#include "mxfp8_streamk.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::streamk_swiglu {
namespace kt=kittens;

template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

__device__ __forceinline__ int2 raster(int task,int ntiles,int mtiles,int group) {
    if(group==1)return make_int2(task/ntiles,task%ntiles);
    int span=group*ntiles,first=task/span*group,rows=min(group,mtiles-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}

// The gate accumulator has already been halved through its E8M0 block scale:
// up * gate * sigmoid(gate) = up * gate_half*(1+tanh(gate_half)).
__device__ __forceinline__ float swiglu(float gate_half,float up) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(gate_half));
    return fmaf(gate_half,t,gate_half)*up;
}

template<int R,int C,int BK,int S,bool Aux> struct Config {
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using OTile=kt::st_bf<4*R,C>;
    using Partial=kt::st_fl<4*R,C/2>;
    union Output {OTile bf;Partial fp;};
    struct Packet {
        ATile a[2];
        BTile gate,up;
    };
    // Metadata follows the input ring's slot lifetime. Keep it outside Packet
    // so tile alignment does not pad every metadata entry by 256--1024 bytes.
    using AScale=std::conditional_t<BK==32,uint8_t,
                 std::conditional_t<BK==64,uint16_t,uint32_t>>;
    struct ScalePacket {
        AScale a[8*R];
        uint8_t gate[((C+31)/32)*(BK/32)],up[((C+31)/32)*(BK/32)];
    };
    static constexpr int max_smem=101376;
    // Preserve the existing one/two auxiliary-buffer choice. Compact scale
    // metadata fits beside it for every configuration (checked below).
    static constexpr int output_buffers=
        Aux && 1024+S*sizeof(Packet)+2*sizeof(OTile)+2*sizeof(OTile)<=max_smem
        ?2:1;
    struct Shared {
        Packet input[S];
        ScalePacket scales[S];
        Output output[2][output_buffers];
    };
    struct Args {
        Global<ATile> a;
        Global<BTile> gate,up;
        Global<OTile> out,gate_half,up_linear;
        const uint32_t *as; // [ceil(K/128),roundup(M,16)] packed E8M0 words
        const uint8_t *gs,*us; // [ceil(N/32),ceil(K/32)] E8M0 bytes
        int m,n,sc,ntiles,mtiles,total,ktiles,raster_group;
        uint32_t as_stride;
        sk::Params<R,C,2> sk;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=max_smem);
};

template<int R,int C,int BK,int S,bool Aux,bool Atomic=sk::atomic>
__global__ __launch_bounds__(384,1) void persistent_swiglu(
    const __grid_constant__ typename Config<R,C,BK,S,Aux>::Args args) {
    using CF=Config<R,C,BK,S,Aux>;
    using Shared=typename CF::Shared;
    constexpr int SR=(C+31)/32,SC=BK/32;
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
            for(int wi=args.sk.offsets[blockIdx.x];wi<args.sk.offsets[blockIdx.x+1];wi++) {
                const sk::Work work=args.sk.work[wi];
                int2 coord=make_int2(work.nr/(8*R),work.kc/C);
                int mr=coord.x*(8*R),nc=coord.y*C;
                for(int k=work.begin;k<work.end;k++) {
                    // Prefetch into registers before acquiring the shared ring slot.
                    typename CF::AScale pa[8*R/32];
                    constexpr int scale_count=((C+31)/32)*(BK/32);
                    uint32_t pb[(scale_count+31)/32],pc[(scale_count+31)/32];
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        int row=mr+lane+i*32;
                        uint32_t word=row<args.m
                            ?args.as[size_t(k*(BK/32)/4)*args.as_stride+row]:0;
                        pa[i]=word>>(8*((k*(BK/32))%4));
                    }
#pragma unroll
                    for(int i=0;i<(scale_count+31)/32;i++) {
                        int index=lane+i*32;
                        if(index<scale_count) {
                            int row=nc/32+index/(BK/32),col=k*(BK/32)+index%(BK/32);
                            bool valid=row<(args.n+31)/32 && col<args.sc;
                            pb[i]=valid?args.gs[size_t(row)*args.sc+col]:0;
                            pc[i]=valid?args.us[size_t(row)*args.sc+col]:0;
                        }
                    }
                    auto packet=pipe.waitBuffer(0,shared.input);
                    auto &tile=packet.template get<0>();
                    auto &scale=shared.scales[packet.SlotIdx];
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        // Four FP8 TMA loads: two half-M X panels and one N
                        // panel for each of the gate/up projections.
                        kt::tma::expect_bytes(bar,(8*R+2*C)*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.gate,args.gate,{0,0,nc,k*BK},bar);
                        kt::tma::load_async(tile.up,args.up,{0,0,nc,k*BK},bar);
                    }
#pragma unroll
                    for(int i=0;i<8*R/32;i++)scale.a[lane+i*32]=pa[i];
#pragma unroll
                    for(int i=0;i<(scale_count+31)/32;i++) {
                        int index=lane+i*32;
                        if(index<scale_count) {
                            scale.gate[index]=pb[i];
                            scale.up[index]=pc[i];
                        }
                    }
                    last=packet.SlotIdx;packet.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int wi=args.sk.offsets[blockIdx.x];wi<args.sk.offsets[blockIdx.x+1];wi++) {
                const sk::Work work=args.sk.work[wi];
            kt::rt_fl<R,C> gate_acc,up_acc;
            kt::warp::rt_maps::zero(gate_acc);
            kt::warp::rt_maps::zero(up_acc);
            int2 coord=make_int2(work.nr/(8*R),work.kc/C);
            int mr=coord.x*8*R+group*4*R+warp*R,nc=coord.y*C;
            for(int k=work.begin;k<work.end;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);
                auto &tile=packet.template get<0>();
                const auto &scale=shared.scales[packet.SlotIdx];
                uint32_t gate_scale=scale.gate[lane%(SR*SC)];
                uint32_t up_scale=scale.up[lane%(SR*SC)];
                kt::rt_fp8e4m3<R,BK> ar;
                kt::rt_fp8e4m3<C,BK> gr,ur;
                kt::mxfp8_scale<R,BK> as,half_as;
                kt::mxfp8_scale<C,BK> gs,us;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_factored<1,true>(gr,tile.gate);
                load_fp8_factored<1,true>(ur,tile.up);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    uint32_t word=uint32_t(scale.a[group*4*R+warp*R+i*16+lane/4+8*(lane&1)]);
#pragma unroll
                    for(int kk=0;kk<SC;kk++) {
                        as.data[i][kk]=word >> (8*kk);
                        // Decrementing E8M0 by one halves the entire gate
                        // projection. Keep the unmodified scale for up.
                        half_as.data[i][kk]=max(int(as.data[i][kk]&255)-1,0);
                    }
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
#pragma unroll
                    for(int kk=0;kk<SC;kk++) {
                        int source=(i/2)*SC+kk;
                        gs.data[i][kk]=__shfl_sync(0xffffffff,gate_scale,source);
                        us.data[i][kk]=__shfl_sync(0xffffffff,up_scale,source);
                    }
                }
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(gate_acc,ar,gr,gate_acc,half_as,gs);
                kt::warp::wmma::mma_ABt_block_scaled(up_acc,ar,ur,up_acc,as,us);
            }
            if(work.slot>=0) {
                sk::store_partial<R,C,2>(shared.output[group][0].fp,gate_acc,args.sk,work.slot,0,group,writer);
                sk::store_partial<R,C,2>(shared.output[group][0].fp,up_acc,args.sk,work.slot,1,group,writer);
                continue;
            }
            kt::rt_bf<R,C> rounded;
            if constexpr(Aux) {
                // Save the half-gate used by the activation directly. This
                // avoids multiplying every gate value by two only to halve it
                // again in backward.
                kt::rt_bf<R,C> gate_half;
                kt::warp::copy(gate_half,gate_acc);
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                kt::warpgroup::store(shared.output[group][0].bf,gate_half);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.gate_half,shared.output[group][0].bf,
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }

                kt::rt_bf<R,C> up_linear;
                kt::warp::copy(up_linear,up_acc);
                if constexpr(CF::output_buffers==1) {
                    if(writer)kt::tma::store_async_read_wait<0>();
                    kt::warpgroup::sync(group+1);
                }
                kt::warpgroup::store(shared.output[group][CF::output_buffers-1].bf,up_linear);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.up_linear,
                                        shared.output[group][CF::output_buffers-1].bf,
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }
            }

#pragma unroll
            for(int r=0;r<gate_acc.height;r++) {
#pragma unroll
                for(int c=0;c<gate_acc.width;c++) {
#pragma unroll
                    for(int i=0;i<gate_acc.packed_per_tile;i++) {
                        float2 g=gate_acc.tiles[r][c].data[i];
                        float2 u=up_acc.tiles[r][c].data[i];
                        gate_acc.tiles[r][c].data[i]={swiglu(g.x,u.x),swiglu(g.y,u.y)};
                    }
                }
            }
            kt::warp::copy(rounded,gate_acc);
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::warpgroup::store(shared.output[group][0].bf,rounded);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.out,shared.output[group][0].bf,
                                    {0,0,coord.x*8*R+group*4*R,nc});
                kt::tma::store_commit_group();
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

}
using CF=fp8gemm::streamk_swiglu::Config<16,64,64,2,true>;
#define SK_KERNEL fp8gemm::streamk_swiglu::persistent_swiglu<16,64,64,2,true>
#define SK_R 16
#define SK_C 64
#define SK_COMP 2
static CF::Args make_args(void **p,int m,int n,int k,int slots) {
    return {fp8gemm::sk::Global<CF::ATile>((kt::fp8e4m3*)p[0],1,1,m,k),
        fp8gemm::sk::Global<CF::BTile>((kt::fp8e4m3*)p[2],1,1,n,k),
        fp8gemm::sk::Global<CF::BTile>((kt::fp8e4m3*)p[4],1,1,n,k),
        fp8gemm::sk::Global<CF::OTile>((kt::bf16*)p[8],1,1,m,n),
        fp8gemm::sk::Global<CF::OTile>((kt::bf16*)p[9],1,1,m,n),
        fp8gemm::sk::Global<CF::OTile>((kt::bf16*)p[10],1,1,m,n),
        (uint32_t*)p[1],(uint8_t*)p[3],(uint8_t*)p[5],m,n,k/32,(n+63)/64,(m+127)/128,
        ((m+127)/128)*((n+63)/64),k/64,1,uint32_t((m+15)/16*16),{fp8gemm::sk::Global<CF::Partial>((float*)p[11],1,1,std::max(1,slots)*SK_COMP*128,SK_C),
        (fp8gemm::sk::Work*)p[12],(int*)p[13]}};
}
#include "mxfp8_streamk_host.cuh"
