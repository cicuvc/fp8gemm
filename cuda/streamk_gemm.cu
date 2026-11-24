// Stream-K adaptation of persistent_mxfp8_gemm.cu at 85d003d.
// Persistent MXFP8 GEMM with prequantized A/B operands.
#include "kittens.cuh"
#include "mxfp8_streamk.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include "mxfp8_epilogue.cuh"
#include "mxfp8_lse.cuh"
#include <stdexcept>
#include <cmath>
#include <string>
#ifndef FP8GEMM_FACTORED_STORES
#define FP8GEMM_FACTORED_STORES 0
#endif
namespace fp8gemm::streamk_gemm {
// Scale tensors may have a byte-aligned storage offset. Use one load for the
// naturally aligned production layout, retaining the unaligned API contract.
template<int Bytes>
__device__ __forceinline__ uint32_t mx_load_row_scale(const uint8_t *ptr) {
    static_assert(Bytes==1 || Bytes==2 || Bytes==4);
    if constexpr(Bytes==1)return *ptr;
    if((reinterpret_cast<uintptr_t>(ptr)&(Bytes-1))==0) {
        uint32_t word;
        if constexpr(Bytes==2)asm("ld.global.u16 %0, [%1];" : "=r"(word) : "l"(ptr));
        else asm("ld.global.u32 %0, [%1];" : "=r"(word) : "l"(ptr));
        return word;
    }
    uint32_t word=0;
#pragma unroll
    for(int j=0;j<Bytes;j++)word|=uint32_t(ptr[j])<<(8*j);
    return word;
}
template<class ST,class RT>
__device__ __forceinline__ void mx_store_output(ST &dst,const RT &src) {
#if FP8GEMM_FACTORED_STORES
    store_factored<4>(dst,src);
#else
    kt::warpgroup::store(dst,src);
#endif
}
template<class T> using MXGlobal=kt::gl<typename T::dtype,-1,-1,-1,-1,T>;
__device__ __forceinline__ int2 mx_raster(int task,int ntiles,int mtiles,int g) {
    if(g==1)return make_int2(task/ntiles,task%ntiles);
    int span=g*ntiles,first=task/span*g,rows=min(g,mtiles-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}
template<int R,int C,int BK,int S,bool SaveQ=false,bool OnlyQ=false,bool SaveLSE=false,bool Softcap=false,bool RowwiseB=false> struct MXConfig {
    static_assert(!OnlyQ || SaveQ);
    static_assert(!Softcap || SaveLSE);
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using OTile=kt::st_bf<4*R,C>;
    using QTile=kt::st_fp8e4m3<4*R,(C+31)/32*32>;
    using Partial=kt::st_fl<4*R,C/2>;
    union OutputStorage {OTile bf;QTile q;Partial fp;};
    struct QuantizedArgs {MXGlobal<QTile> out;uint32_t *scales;};
    struct NoQuantizedArgs {};
    struct LSEArgs {float *data;float softcap;};
    static constexpr int SC=BK/32,SR=(C+31)/32;
    struct Packet {
        ATile a[2];BTile b;
    };
    // Metadata follows the input ring's slot lifetime. Keep it outside Packet
    // so tile alignment does not pad every metadata entry by 256--1024 bytes.
    using AScale=std::conditional_t<BK==32,uint8_t,
                 std::conditional_t<BK==64,uint16_t,uint32_t>>;
    using BScale=std::conditional_t<RowwiseB,AScale,uint8_t>;
    struct ScalePacket {
        AScale a[8*R];
        BScale b[RowwiseB?C:SR*SC];
    };
    struct Shared {
        Packet input[S];
        ScalePacket scales[S];
        std::conditional_t<OnlyQ,QTile,OutputStorage> output[2];
    };
    struct Args {
        MXGlobal<ATile> a;MXGlobal<BTile> b;
        [[no_unique_address]] std::conditional_t<OnlyQ,NoQuantizedArgs,MXGlobal<OTile>> out;
        const uint32_t *as; // packed [ceil(K/128),roundup(M,16)] uint32 words
        const uint8_t *bs; // blockwise [ceil(N/32),K/32] or rowwise [N,K/32]
        int m,n,sc,ntiles,mtiles,total,ktiles,raster_group;
        uint32_t as_stride;
        [[no_unique_address]] std::conditional_t<SaveLSE,LSEArgs,NoQuantizedArgs> lse;
        [[no_unique_address]] std::conditional_t<SaveQ,QuantizedArgs,NoQuantizedArgs> q;
        sk::Params<R,C,1> sk;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
};
template<int R,int C,int BK,int S,bool SaveQ=false,bool OnlyQ=false,bool SaveLSE=false,bool Softcap=false,bool RowwiseB=false,bool Atomic=sk::atomic>
__global__ __launch_bounds__(384,1) void persistent_mxfp8(
    const __grid_constant__ typename MXConfig<R,C,BK,S,SaveQ,OnlyQ,SaveLSE,Softcap,RowwiseB>::Args args) {
    using CF=MXConfig<R,C,BK,S,SaveQ,OnlyQ,SaveLSE,Softcap,RowwiseB>;using Shared=typename CF::Shared;
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
                    constexpr int scale_count=(RowwiseB?C:CF::SR)*CF::SC;
                    uint32_t pb=0;
#pragma unroll
                    for(int i=0;i<8*R/32;i++) {
                        int row=mr+lane+i*32;
                        uint32_t word=row<args.m
                            ?args.as[size_t(k*(BK/32)/4)*args.as_stride+row]:0;
                        pa[i]=word>>(8*((k*(BK/32))%4));
                    }
                    if constexpr(!RowwiseB) {
                        if(lane<scale_count) {
                            int row=nc/32+lane/(BK/32),col=k*(BK/32)+lane%(BK/32);
                            bool valid=row<(args.n+31)/32 && col<args.sc;
                            pb=valid?args.bs[size_t(row)*args.sc+col]:0;
                        }
                    }
                    auto packet=pipe.waitBuffer(0,shared.input);auto &tile=packet.template get<0>();
                    auto &scale=shared.scales[packet.SlotIdx];
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                    }
#pragma unroll
                    for(int i=0;i<8*R/32;i++)scale.a[lane+i*32]=pa[i];
                    if constexpr(RowwiseB) {
                        // CE has many more B exponents than blockwise FFN weights.
                        // Stage them after acquisition, while operand TMA is in flight,
                        // to avoid carrying the entire set across the slot wait.
#pragma unroll
                        for(int i=0;i<C/32;i++) {
                            int index=lane+i*32,row=nc+index,col=k*CF::SC;
                            scale.b[index]=(row<args.n && col<args.sc)
                                ?mx_load_row_scale<CF::SC>(args.bs+size_t(row)*args.sc+col):0;
                        }
                    } else if(lane<scale_count) {
                        scale.b[lane]=pb;
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
            kt::rt_fl<R,C> accum;kt::warp::rt_maps::zero(accum);
            int2 coord=make_int2(work.nr/(8*R),work.kc/C);
            int mr=coord.x*8*R+group*4*R+warp*R,nc=coord.y*C;
            for(int k=work.begin;k<work.end;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);auto &tile=packet.template get<0>();
                const auto &scale=shared.scales[packet.SlotIdx];
                uint32_t block_scale=0;
                if constexpr(!RowwiseB)block_scale=scale.b[lane%(CF::SR*CF::SC)];
                kt::rt_fp8e4m3<R,BK> ar;kt::rt_fp8e4m3<C,BK> br;
                kt::mxfp8_scale<R,BK> as;kt::mxfp8_scale<C,BK> bs;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_factored<1,true>(br,tile.b);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    uint32_t word=uint32_t(scale.a[group*4*R+warp*R+i*16+lane/4+8*(lane&1)]);
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)as.data[i][kk]=word >> (8*kk);
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++) {
                        if constexpr(RowwiseB)bs.data[i][kk]=uint32_t(scale.b[i*16+lane/4+8*(lane&1)])>>(8*kk);
                        else bs.data[i][kk]=__shfl_sync(0xffffffff,block_scale,(i/2)*CF::SC+kk);
                    }
                }
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(accum,ar,br,accum,as,bs);
            }
            if(work.slot>=0) {
                sk::store_partial<R,C,1>(shared.output[group].fp,accum,args.sk,work.slot,0,group,writer);
            } else {
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                mx_store_output(shared.output[group].bf,accum);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.out,shared.output[group].bf,{0,0,work.nr+group*4*R,work.kc});
                    kt::tma::store_commit_group();
                }
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}
}
using CF=fp8gemm::streamk_gemm::MXConfig<16,128,128,2>;
#define SK_KERNEL fp8gemm::streamk_gemm::persistent_mxfp8<16,128,128,2>
#define SK_R 16
#define SK_C 128
#define SK_COMP 1
static CF::Args make_args(void **p,int m,int n,int k,int slots) {
    return {fp8gemm::sk::Global<CF::ATile>((kt::fp8e4m3*)p[0],1,1,m,k),
        fp8gemm::sk::Global<CF::BTile>((kt::fp8e4m3*)p[1],1,1,n,k),
        fp8gemm::sk::Global<CF::OTile>((kt::bf16*)p[8],1,1,m,n),
        (uint32_t*)p[2],(uint8_t*)p[3],m,n,k/32,(n+127)/128,(m+127)/128,
        ((m+127)/128)*((n+127)/128),k/128,1,uint32_t((m+15)/16*16),{},{},{fp8gemm::sk::Global<CF::Partial>((float*)p[11],1,1,std::max(1,slots)*SK_COMP*128,SK_C),
        (fp8gemm::sk::Work*)p[12],(int*)p[13]}};
}
#include "mxfp8_streamk_host.cuh"
