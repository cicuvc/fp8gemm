// Layout preparation for persistent FP8 linear backward. GEMMs remain in TK.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// A padded copy for dX and a transposed copy for dW share one global read.
// Bias partials sum the BF16 gradient, matching FP8Linear's existing contract.
template<bool Bias>
__global__ void prepare_dy(const __nv_bfloat16* src, __nv_bfloat16* dy,
                          __nv_bfloat16* dyt, float* partial,
                          int m,int n,int mp,int np,int64_t s0,int64_t s1) {
    __shared__ __nv_bfloat16 tile[32][33];
    int tx=threadIdx.x,ty=threadIdx.y;
    int r0=blockIdx.y*32,c0=blockIdx.x*32;
#pragma unroll
    for(int j=0;j<32;j+=8) {
        int r=r0+ty+j,c=c0+tx;
        auto v=(r<m && c<n)?src[int64_t(r)*s0+int64_t(c)*s1]:__float2bfloat16(0.f);
        tile[ty+j][tx]=v;
        if(r<m)dy[int64_t(r)*np+c]=v;
    }
    __syncthreads();
#pragma unroll
    for(int j=0;j<32;j+=8)
        dyt[int64_t(c0+ty+j)*mp+r0+tx]=tile[tx][ty+j];
    if constexpr(Bias) {
        if(ty==0) {
            float sum=0;
#pragma unroll
            for(int r=0;r<32;++r)sum+=__bfloat162float(tile[r][tx]);
            partial[int64_t(blockIdx.y)*np+c0+tx]=sum;
        }
    }
}

// Lossless transpose of the already-quantized forward weight. Its 32x32
// scale blocks transpose with it; no second weight rounding is introduced.
__global__ void transpose_weight(const uint8_t* w,const uint8_t* scales,
                                 uint8_t* wt,uint8_t* st,int n,int k,int np,int kp) {
    __shared__ uint8_t tile[32][33];
    int tx=threadIdx.x,ty=threadIdx.y;
    int r0=blockIdx.y*32,c0=blockIdx.x*32;
#pragma unroll
    for(int j=0;j<32;j+=8) {
        int r=r0+ty+j,c=c0+tx;
        tile[ty+j][tx]=(r<n && c<k)?w[int64_t(r)*k+c]:0;
    }
    __syncthreads();
#pragma unroll
    for(int j=0;j<32;j+=8)
        wt[int64_t(c0+ty+j)*np+r0+tx]=tile[tx][ty+j];
    if(tx==0 && ty==0)
        st[int64_t(blockIdx.x)*(np/32)+blockIdx.y]=
            (r0<n && c0<k)?scales[int64_t(blockIdx.y)*((k+31)/32)+blockIdx.x]:0;
}

__global__ void bias_reduce(const float* partial,float* db,int n,int np,int chunks) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<n) {
        float sum=0;
        for(int r=0;r<chunks;++r)sum+=partial[int64_t(r)*np+c];
        db[c]=sum;
    }
}

template<class T>
__global__ void crop(const T* src,T* dst,int rows,int cols,int stride) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(rows)*cols)dst[i]=src[(i/cols)*stride+i%cols];
}

__global__ void pad_weight_rows(const uint8_t* src,uint8_t* dst,int n,int k,int stride) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(n)*stride) {
        int c=i%stride;
        dst[i]=c<k?src[(i/stride)*k+c]:0;
    }
}

extern "C" int linear_pad_weight(void* w,void* padded,int n,int k,int stride,void* stream) {
    pad_weight_rows<<<(int64_t(n)*stride+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(w),static_cast<uint8_t*>(padded),n,k,stride);
    return int(cudaGetLastError());
}
extern "C" int linear_prepare_dy(void* src,void* dy,void* dyt,void* partial,
    int m,int n,int mp,int np,int64_t s0,int64_t s1,int bias,void* stream) {
    dim3 grid(np/32,mp/32),block(32,8);
    auto s=static_cast<cudaStream_t>(stream);
    if(bias)prepare_dy<true><<<grid,block,0,s>>>(static_cast<__nv_bfloat16*>(src),
        static_cast<__nv_bfloat16*>(dy),static_cast<__nv_bfloat16*>(dyt),static_cast<float*>(partial),m,n,mp,np,s0,s1);
    else prepare_dy<false><<<grid,block,0,s>>>(static_cast<__nv_bfloat16*>(src),
        static_cast<__nv_bfloat16*>(dy),static_cast<__nv_bfloat16*>(dyt),nullptr,m,n,mp,np,s0,s1);
    return int(cudaGetLastError());
}
extern "C" int linear_transpose_weight(void* w,void* s,void* wt,void* st,
    int n,int k,int np,int kp,void* stream) {
    transpose_weight<<<dim3(kp/32,np/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(w),static_cast<uint8_t*>(s),static_cast<uint8_t*>(wt),static_cast<uint8_t*>(st),n,k,np,kp);
    return int(cudaGetLastError());
}
extern "C" int linear_finish(void* dxp,void* dwp,void* partial,void* dx,void* dw,void* db,
    int m,int n,int k,int mp,int np,int kp,int dx_stride,int bias,void* stream) {
    auto s=static_cast<cudaStream_t>(stream);
    if(dxp!=dx)crop<<<(int64_t(m)*k+255)/256,256,0,s>>>(static_cast<__nv_bfloat16*>(dxp),static_cast<__nv_bfloat16*>(dx),m,k,dx_stride);
    if(dwp!=dw)crop<<<(int64_t(n)*k+255)/256,256,0,s>>>(static_cast<float*>(dwp),static_cast<float*>(dw),n,k,kp);
    if(bias)bias_reduce<<<(n+255)/256,256,0,s>>>(static_cast<float*>(partial),static_cast<float*>(db),n,np,mp/32);
    return int(cudaGetLastError());
}
