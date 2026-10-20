// Fused BF16 SwiGLU backward and two-direction MXFP8 quantization.
// Each warp owns one 32x32 element tile. Lane r holds the 32 BF16 values of
// row r: the 1x32 scale is reduced entirely in-thread, while each 32x1 scale
// uses one redux.sync.max.u32 across the warp. No BF16 gradients are written
// to global memory; only E4M3 data and E8M0 scale bytes are stored.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "silu_bwd_common.cuh"

namespace fp8gemm::silu_bwd_quantize {
using fp8gemm::silu_bwd::PairGrad;
using fp8gemm::silu_bwd::backward_pair;
using fp8gemm::silu_bwd::bf16x2_scale;
using fp8gemm::silu_bwd::cvt_e4m3x2_bf16x2;

static thread_local std::string error;

struct Args {
    const uint16_t *dy,*half_gate,*up;
    uint8_t *gate_row,*up_row,*gate_col,*up_col;
    uint8_t *gate_row_scale,*up_row_scale,*gate_col_scale,*up_col_scale;
    int m,n,mp,mp16,nblocks,tile_columns,total_tiles;
};

__device__ __forceinline__ uint32_t max_exponent(
    uint32_t values,uint32_t current=0) {
    current=max(current,values&0x7f80u);
    return max(current,(values>>16)&0x7f80u);
}

__global__ __launch_bounds__(128) void silu_bwd_quantize_kernel(Args args) {
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int tile=blockIdx.x*4+warp;
    if(tile>=args.total_tiles)return;
    int tile_col=tile%args.tile_columns,tile_row=tile/args.tile_columns;
    int row=tile_row*32+lane,col=tile_col*32;

    uint32_t dy_values[16]={0},half_gate_values[16]={0},up_values_in[16]={0};
    if(row<args.m) {
        auto load16=[&](const uint16_t *source,uint32_t (&dst)[16]) {
            const uint4 *src=reinterpret_cast<const uint4*>(
                source+size_t(row)*args.n+col);
            uint4 a=src[0],b=src[1],c=src[2],d=src[3];
            uint32_t values[16]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,
                                 c.x,c.y,c.z,c.w,d.x,d.y,d.z,d.w};
#pragma unroll
            for(int i=0;i<16;i++)dst[i]=values[i];
        };
        load16(args.dy,dy_values);
        load16(args.half_gate,half_gate_values);
        load16(args.up,up_values_in);
    }

    uint32_t gate_values[16],up_values[16];
    uint32_t gate_exponent=0,up_exponent=0;
#pragma unroll
    for(int pair=0;pair<16;pair++) {
        PairGrad grad=backward_pair(
            dy_values[pair],half_gate_values[pair],up_values_in[pair]);
        gate_values[pair]=grad.gate;
        up_values[pair]=grad.up;
        if(row<args.m) {
            gate_exponent=max_exponent(grad.gate,gate_exponent);
            up_exponent=max_exponent(grad.up,up_exponent);
        }
    }

    uint8_t gate_row_exp=uint8_t(gate_exponent>>7);
    uint8_t up_row_exp=uint8_t(up_exponent>>7);
    if(row<args.m) {
        uint16_t gate_out[16],up_out[16];
#pragma unroll
        for(int pair=0;pair<16;pair++) {
            gate_out[pair]=cvt_e4m3x2_bf16x2(
                bf16x2_scale(gate_values[pair],gate_row_exp));
            up_out[pair]=cvt_e4m3x2_bf16x2(
                bf16x2_scale(up_values[pair],up_row_exp));
        }
        auto *gate_dst=reinterpret_cast<uint4*>(
            args.gate_row+size_t(row)*args.n+col);
        auto *up_dst=reinterpret_cast<uint4*>(
            args.up_row+size_t(row)*args.n+col);
#pragma unroll
        for(int vector=0;vector<2;vector++) {
            int pair=vector*8;
            gate_dst[vector]={uint32_t(gate_out[pair])|(uint32_t(gate_out[pair+1])<<16),
                              uint32_t(gate_out[pair+2])|(uint32_t(gate_out[pair+3])<<16),
                              uint32_t(gate_out[pair+4])|(uint32_t(gate_out[pair+5])<<16),
                              uint32_t(gate_out[pair+6])|(uint32_t(gate_out[pair+7])<<16)};
            up_dst[vector]={uint32_t(up_out[pair])|(uint32_t(up_out[pair+1])<<16),
                            uint32_t(up_out[pair+2])|(uint32_t(up_out[pair+3])<<16),
                            uint32_t(up_out[pair+4])|(uint32_t(up_out[pair+5])<<16),
                            uint32_t(up_out[pair+6])|(uint32_t(up_out[pair+7])<<16)};
        }
    }

    int block_row=tile_row*32;
    if(row<args.mp16) {
        size_t offset=size_t(tile_col/4)*args.mp16*4+size_t(row)*4+(tile_col&3);
        args.gate_row_scale[offset]=gate_row_exp;
        args.up_row_scale[offset]=up_row_exp;
    }

    // One redux per feature produces the 32x1 scale. Even lanes then pair
    // adjacent rows and use one packed BF16 multiply + FP8 conversion.
#pragma unroll
    for(int pair=0;pair<16;pair++) {
#pragma unroll
        for(int half=0;half<2;half++) {
            uint32_t gate_value=(gate_values[pair]>>(16*half))&0xffffu;
            uint32_t up_value=(up_values[pair]>>(16*half))&0xffffu;
            uint32_t gate_exp=__reduce_max_sync(
                0xffffffffu,gate_value&0x7f80u);
            uint32_t up_exp=__reduce_max_sync(
                0xffffffffu,up_value&0x7f80u);
            uint32_t gate_next=__shfl_down_sync(0xffffffffu,gate_value,1);
            uint32_t up_next=__shfl_down_sync(0xffffffffu,up_value,1);
            int feature=col+pair*2+half;
            if(lane==0) {
                size_t offset=
                    size_t(tile_row/4)*args.n*4+size_t(feature)*4+(tile_row&3);
                args.gate_col_scale[offset]=uint8_t(gate_exp>>7);
                args.up_col_scale[offset]=uint8_t(up_exp>>7);
            }
            if((lane&1)==0) {
                uint32_t gate_pair=gate_value|(gate_next<<16);
                uint32_t up_pair=up_value|(up_next<<16);
                uint16_t gate_q=cvt_e4m3x2_bf16x2(
                    bf16x2_scale(gate_pair,uint8_t(gate_exp>>7)));
                uint16_t up_q=cvt_e4m3x2_bf16x2(
                    bf16x2_scale(up_pair,uint8_t(up_exp>>7)));
                size_t index=size_t(feature)*args.mp+block_row+lane;
                *reinterpret_cast<uint16_t*>(args.gate_col+index)=gate_q;
                *reinterpret_cast<uint16_t*>(args.up_col+index)=up_q;
            }
        }
    }
}

static void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

struct Plan {
    Args args;
    int blocks;
    Plan(
        void*dy,void*half_gate,void*up,
        void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
        void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale,
        int m,int n):
        args{
            static_cast<const uint16_t*>(dy),
            static_cast<const uint16_t*>(half_gate),
            static_cast<const uint16_t*>(up),
            static_cast<uint8_t*>(gate_row),
            static_cast<uint8_t*>(up_row),
            static_cast<uint8_t*>(gate_col),
            static_cast<uint8_t*>(up_col),
            static_cast<uint8_t*>(gate_row_scale),
            static_cast<uint8_t*>(up_row_scale),
            static_cast<uint8_t*>(gate_col_scale),
            static_cast<uint8_t*>(up_col_scale),
            m,n,(m+127)/128*128,(m+15)/16*16,n/32,n/32,
            (n/32)*((m+127)/128*128/32)},
        blocks{(args.total_tiles+3)/4} {}
    void launch(cudaStream_t stream) {
        silu_bwd_quantize_kernel<<<blocks,128,0,stream>>>(args);
    }
};

}

extern "C" const char *tk_silu_bwd_q_error() {
    return fp8gemm::silu_bwd_quantize::error.c_str();
}

extern "C" void *tk_silu_bwd_q_create(
    void*dy,void*half_gate,void*up,
    void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
    void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale,
    int m,int n) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        if(!dy||!half_gate||!up||!gate_row||!gate_row_scale||
           !up_row||!up_row_scale||!gate_col||!gate_col_scale||
           !up_col||!up_col_scale||m<=0||n<=0||n%32)
            throw std::runtime_error("Invalid arguments or N not multiple of 32");
        return new Plan(
            dy,half_gate,up,
            gate_row,gate_row_scale,up_row,up_row_scale,
            gate_col,gate_col_scale,up_col,up_col_scale,m,n);
    }catch(const std::exception &exception) {
        error=exception.what();return nullptr;
    }
}

extern "C" int tk_silu_bwd_q_launch(void*opaque,void*stream) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        static_cast<Plan*>(opaque)->launch(static_cast<cudaStream_t>(stream));
        check(cudaGetLastError());
        return 0;
    }catch(const std::exception &exception) {
        error=exception.what();return -1;
    }
}

extern "C" void tk_silu_bwd_q_info(void*opaque,int*out) {
    auto *plan=static_cast<fp8gemm::silu_bwd_quantize::Plan*>(opaque);
    out[0]=plan->blocks;out[1]=128;
}

extern "C" void tk_silu_bwd_q_destroy(void*opaque) {
    delete static_cast<fp8gemm::silu_bwd_quantize::Plan*>(opaque);
}
