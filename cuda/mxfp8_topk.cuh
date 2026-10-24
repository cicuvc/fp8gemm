#pragma once
#include "kittens.cuh"
#include <stdexcept>

namespace fp8gemm {
// Adapted from dism_v3/include/summary/primitives.cuh's
// KPermutationSharedBuffer: transpose the [Rows/8,8] row factors during TMA.
// Only complete N tiles use this descriptor; ordinary TMA handles the tail.
template<int Rows,int Cols> struct alignas(1024) TopKPermutationBuffer {
    using dtype=kt::fp8e4m3;
    using tile_type=kt::st_fp8e4m3<Rows,Cols>;
    static constexpr int required_alignment=1024;
    tile_type payload;
};

template<int C> __host__ __device__ constexpr int topk_permute_n(int physical) {
    return (physical%8)*(C/8)+physical/8;
}
}

namespace kittens::tma {
template<int Rows,int Cols>
struct buffer_traits<fp8gemm::TopKPermutationBuffer<Rows,Cols>> {
    using buffer_type=fp8gemm::TopKPermutationBuffer<Rows,Cols>;
    using dtype=typename buffer_type::dtype;
    using coord_type=kittens::coord<>;
    static constexpr bool default_swizzle=true;
    static constexpr int rank=5, SB=buffer_type::tile_type::swizzle_bytes;
    static constexpr uint32_t transfer_bytes=Rows*Cols;
    template<bool EnableSwizzle,class T>
    __host__ static void encode(CUtensorMap *map,const global_tensor_view<T>& view) {
        static_assert(EnableSwizzle && std::is_same_v<T,dtype>);
        if(view.stride[3]!=1 || view.shape[3]%SB)
            throw std::runtime_error("TopK TMA requires aligned contiguous H");
        uint64_t h=view.shape[3], n=view.shape[2];
        uint64_t shape[5]={SB,8,Rows/8,std::max(uint64_t(1),n/Rows),h/SB};
        uint64_t stride[4]={(Rows/8)*h,h,Rows*h,SB};
        uint32_t box[5]={SB,8,Rows/8,1,Cols/SB}, step[5]={1,1,1,1,1};
        auto swizzle=SB==128 ? CU_TENSOR_MAP_SWIZZLE_128B :
                     SB==64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B;
        auto result=encode_tiled(map,CU_TENSOR_MAP_DATA_TYPE_UINT8,rank,view.data,
            shape,stride,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(result!=CUDA_SUCCESS)throw std::runtime_error("Invalid topK TMA descriptor");
    }
    __device__ static void *shared_address(buffer_type& t){return &t.payload;}
    __device__ static const void *shared_address(const buffer_type& t){return &t.payload;}
    template<class Coord,class GL>
    __device__ static int5 coordinates(const Coord& idx,const GL&) {
        return {0,0,0,idx.r/Rows,idx.c/SB};
    }
};
}

namespace fp8gemm {
// BF16 numeric order followed by ascending original column ID. Normalize
// signed zero so equal scores use the same tie rule as floating comparisons.
__device__ __forceinline__ uint32_t topk_key(float value,int col,bool valid) {
    uint16_t bits=__bfloat16_as_ushort(__float2bfloat16_rn(value));
    if((bits&0x7fff)==0)bits=0;
    uint32_t order=(bits&0x8000) ? uint16_t(~bits) : (bits^0x8000);
    return valid ? (order<<16)|(0xffffu-uint32_t(col)) : 0;
}

template<int Size,int Distance,int N>
__device__ __forceinline__ void topk_sort_network(uint32_t (&v)[N]) {
#pragma unroll
    for(int i=0;i<N;i++) {
        if((i^Distance)>i) {
            uint32_t a=v[i],b=v[i^Distance];
            v[i]=(i&Size) ? min(a,b) : max(a,b);
            v[i^Distance]=(i&Size) ? max(a,b) : min(a,b);
        }
    }
    if constexpr(Distance>1)topk_sort_network<Size,Distance/2>(v);
    else if constexpr(Size<N)topk_sort_network<Size*2,Size>(v);
}

template<int Distance,int K>
__device__ __forceinline__ void topk_merge_network(uint32_t (&v)[K]) {
#pragma unroll
    for(int i=0;i<K;i++) {
        if((i^Distance)>i) {
            uint32_t a=v[i],b=v[i^Distance];
            v[i]=max(a,b);v[i^Distance]=min(a,b);
        }
    }
    if constexpr(Distance>1)topk_merge_network<Distance/2>(v);
}

template<int R,int C,int K>
__device__ __forceinline__ void store_topk(kt::rt_fl<R,C>& z,uint64_t *out,
                                         int mr,int nc,int m,int n,bool permuted) {
    constexpr int L=C/4, Local=L>K ? L:K;
    int lane=kt::laneid(), l=lane%4, parts=(n+C-1)/C;
    __syncwarp();
#pragma unroll
    for(int i=0;i<R/16;i++) {
#pragma unroll
        for(int r=0;r<2;r++) {
            uint32_t v[Local];
#pragma unroll
            for(int t=0;t<Local;t++)v[t]=0;
#pragma unroll
            for(int j=0;j<C/16;j++) {
#pragma unroll
                for(int half=0;half<2;half++) {
                    auto value=z.tiles[i][j].data[2*half+r];
                    int p=j*16+2*l+8*half;
                    int cx=permuted ? topk_permute_n<C>(p):p;
                    int cy=permuted ? topk_permute_n<C>(p+1):p+1;
                    // The permuted layout makes these register positions a
                    // contiguous logical interval. Ordinary tails use the same
                    // network but carry the original, strided column IDs.
                    v[2*j+half]=topk_key(value.x,cx,nc+cx<n);
                    v[C/8+2*j+half]=topk_key(value.y,cy,nc+cy<n);
                }
            }
            uint32_t best[K];
            if constexpr(K==1) {
                uint32_t maximum=0;
#pragma unroll
                for(int t=0;t<L;t++)maximum=max(maximum,v[t]);
                best[0]=maximum;
            } else {
                topk_sort_network<2,1>(v);
#pragma unroll
                for(int t=0;t<K;t++)best[t]=v[t];
            }
            // Merge two descending K-lists, retaining only the upper half.
            // Both lanes in a pair get the same result; the second stage merges
            // disjoint pairs. All four lanes then own the row's sorted top K.
#pragma unroll
            for(int delta=1;delta<=2;delta*=2) {
                uint32_t other[K];
#pragma unroll
                for(int t=0;t<K;t++)
                    other[t]=__shfl_xor_sync(0xffffffff,best[K-1-t],delta);
#pragma unroll
                for(int t=0;t<K;t++)best[t]=max(best[t],other[t]);
                if constexpr(K>1)topk_merge_network<K/2>(best);
            }
            int row=mr+i*16+lane/4+r*8;
            if(l==0 && row<m) {
#pragma unroll
                for(int t=0;t<K;t++) {
                    uint32_t key=best[t];
                    uint32_t col=nc+(0xffffu-(key&0xffffu));
                    uint64_t global=key ? (uint64_t(key>>16)<<32)|(0xffffffffu-col):0;
                    out[(size_t(row)*parts+nc/C)*K+t]=global;
                }
            }
        }
    }
}
}
