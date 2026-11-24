#pragma once
#include "kittens.cuh"
#include <stdexcept>
#include <string>
namespace kt=kittens;
#ifndef STREAMK_ATOMIC
#define STREAMK_ATOMIC 1
#endif
namespace fp8gemm::sk {
constexpr bool atomic=STREAMK_ATOMIC;
struct Work {int nr,kc,begin,end,slot;};
struct Fixup {int nr,kc,first,count;};
template<class T> using Global=kt::gl<typename T::dtype,-1,-1,-1,-1,T>;
template<int R,int C,int Components=1> struct Params {
    using Partial=kt::st_fl<4*R,C/2>;
    Global<Partial> partial;
    const Work *work;
    const int *offsets;
};
template<int R,int C,int Components,class RT>
__device__ __forceinline__ void store_partial(typename Params<R,C,Components>::Partial &shared,
    const RT &accum,const Params<R,C,Components> &args,int slot,int component,int group,bool writer) {
#pragma unroll
    for(int panel=0;panel<2;panel++) {
        kt::rt<float,R,C/2,typename RT::layout> part;
#pragma unroll
        for(int i=0;i<R/16;i++)
#pragma unroll
            for(int j=0;j<C/32;j++)
#pragma unroll
                for(int p=0;p<4;p++)part.tiles[i][j].data[p]=accum.tiles[i][j+panel*(C/32)].data[p];
        if(writer)kt::tma::store_async_read_wait<0>();
        kt::warpgroup::sync(group+1);
        kt::warpgroup::store(shared,part);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        kt::warpgroup::sync(group+1);
        if(writer) {
            kt::coord<> dst={0,0,(slot*Components+component)*(8*R)+group*4*R,panel*(C/2)};
            if constexpr(atomic)kt::tma::store_add_async(args.partial,shared,dst);
            else kt::tma::store_async(args.partial,shared,dst);
            kt::tma::store_commit_group();
        }
    }
}
__device__ __forceinline__ float activate(float half,float up) {
    float t;asm("tanh.approx.f32 %0, %1;":"=f"(t):"f"(half));
    return fmaf(half,t,half)*up;
}
template<int R,int C,int Components,bool Atomic=atomic>
__global__ __launch_bounds__(256) void fixup(const float *partial,kt::bf16 *out,kt::bf16 *gate,kt::bf16 *up,
    const Fixup *fix,const int *ids,int m,int n) {
    auto f=fix[blockIdx.x];
    for(int index=threadIdx.x*4;index<8*R*C;index+=256*4) {
        float4 sums[Components];
#pragma unroll
        for(int comp=0;comp<Components;comp++) {
            float4 v=make_float4(0,0,0,0);
            if constexpr(Atomic) v=*reinterpret_cast<const float4*>(
                partial+(size_t(blockIdx.x)*Components+comp)*(8*R*C)+index);
            else for(int i=0;i<f.count;i++) {
                float4 x=*reinterpret_cast<const float4*>(
                    partial+(size_t(ids[f.first+i])*Components+comp)*(8*R*C)+index);
                v.x+=x.x;v.y+=x.y;v.z+=x.z;v.w+=x.w;
            }
            sums[comp]=v;
        }
        int row=f.nr+index/C,col=f.kc+index%C;
        if(row<m && col+3<n) {
            size_t offset=size_t(row)*n+col;
            auto store=[&](kt::bf16 *dst,float4 v) {
                auto *p=reinterpret_cast<__nv_bfloat162*>(dst+offset);
                p[0]=__float22bfloat162_rn(make_float2(v.x,v.y));
                p[1]=__float22bfloat162_rn(make_float2(v.z,v.w));
            };
            if constexpr(Components==2) {
                if(gate)store(gate,sums[0]);
                if(up)store(up,sums[1]);
                float4 g=sums[0],u=sums[1];
                store(out,make_float4(activate(g.x,u.x),activate(g.y,u.y),activate(g.z,u.z),activate(g.w,u.w)));
            } else store(out,sums[0]);
        }
    }
}
inline void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
