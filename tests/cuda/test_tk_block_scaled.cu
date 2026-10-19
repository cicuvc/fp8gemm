// Fragment/tile tests only: no shared-memory pipeline or full GEMM kernel.
#include "kittens.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace kittens;

#define CUDA_CHECK(call) do { auto e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); \
    std::exit(2); } } while(0)

template<typename T> struct buffer {
    T *p;
    explicit buffer(const std::vector<T> &v) {
        CUDA_CHECK(cudaMalloc(&p,v.size()*sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~buffer() { cudaFree(p); }
    void read(std::vector<T> &v) { CUDA_CHECK(cudaMemcpy(v.data(),p,v.size()*sizeof(T),cudaMemcpyDeviceToHost)); }
};

// Explicit PTX lane mapping, independent of the MMA wrapper's indexing.
template<typename T, int M, int K>
__device__ void load_fp8(rt<T,M,K> &r, const T *x) {
    int lane=threadIdx.x%32, row=lane/4, col=4*(lane%4);
#pragma unroll
    for(int i=0;i<M/16;++i) {
#pragma unroll
        for(int j=0;j<K/32;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                const T *src=x+(i*16+row+(p%2)*8)*K+j*32+col+(p/2)*16;
                r.tiles[i][j].data[p]=std::bit_cast<typename rt<T,M,K>::T2>(
                    *reinterpret_cast<const uint32_t*>(src));
            }
        }
    }
}

template<int M,int N>
__device__ void load_c(rt_fl<M,N> &r,const float *x) {
    int lane=threadIdx.x%32;
#pragma unroll
    for(int i=0;i<M/16;++i) {
#pragma unroll
        for(int j=0;j<N/16;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                int row=i*16+lane/4+(p%2)*8;
                int col=j*16+2*(lane%4)+(p/2)*8;
                r.tiles[i][j].data[p]=*reinterpret_cast<const float2*>(x+row*N+col);
            }
        }
    }
}

template<int M,int N>
__device__ void store_d(float *x,const rt_fl<M,N> &r) {
    int lane=threadIdx.x%32;
#pragma unroll
    for(int i=0;i<M/16;++i) {
#pragma unroll
        for(int j=0;j<N/16;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                int row=i*16+lane/4+(p%2)*8;
                int col=j*16+2*(lane%4)+(p/2)*8;
                *reinterpret_cast<float2*>(x+row*N+col)=r.tiles[i][j].data[p];
            }
        }
    }
}

template<int Rows,int K>
__device__ void load_scales(mxfp8_scale<Rows,K> &s,const uint8_t *x) {
    int lane=threadIdx.x%32;
#pragma unroll
    for(int i=0;i<Rows/16;++i) {
#pragma unroll
        for(int k=0;k<K/32;++k)
            s.data[i][k]=0xffffff00u | x[(i*16+lane/4+8*(lane%2))*(K/32)+k];
    }
}

template<typename TA,typename TB,int M,int N,int K>
__global__ void tile_test(const TA *a,const TB *b,const uint8_t *sa,const uint8_t *sb,
                          const float *c,float *out,int batches,bool alias,int repeats) {
    int batch=blockIdx.x*4+threadIdx.x/32;
    if(batch>=batches) return; // whole warps only
    rt<TA,M,K> ar;
    rt<TB,N,K> br;
    rt_fl<M,N> cr,dr;
    mxfp8_scale<M,K> as;
    mxfp8_scale<N,K> bs;
    load_fp8(ar,a+batch*M*K); load_fp8(br,b+batch*N*K);
    load_c(cr,c+batch*M*N);
    load_scales(as,sa+batch*M*K/32); load_scales(bs,sb+batch*N*K/32);
    if(alias) {
        for(int i=0;i<repeats;++i) warp::wmma::mma_ABt_block_scaled(cr,ar,br,cr,as,bs);
        store_d(out+batch*M*N,cr);
    } else {
        warp::wmma::mma_ABt_block_scaled(dr,ar,br,cr,as,bs);
        for(int i=1;i<repeats;++i) warp::wmma::mma_ABt_block_scaled(dr,ar,br,dr,as,bs);
        store_d(out+batch*M*N,dr);
    }
}

// Exercise nonzero scale byte/thread selectors with NaN poison elsewhere.
template<int BA,int TA,int BB,int TB>
__global__ void selector_test(const fp8e4m3 *a,const fp8e4m3 *b,const uint8_t *sa,
                              const uint8_t *sb,const float *c,float *out) {
    int lane=threadIdx.x%32;
    rt_fp8e4m3<16,32> ar;
    load_fp8(ar,a);
    auto b0=std::bit_cast<fp8e4m3_4>(*reinterpret_cast<const uint32_t*>(b+(lane/4)*32+4*(lane%4)));
    auto b1=std::bit_cast<fp8e4m3_4>(*reinterpret_cast<const uint32_t*>(b+(lane/4)*32+4*(lane%4)+16));
    uint32_t as=0xffffffffu,bs=0xffffffffu;
    if((lane%4)/2==TA)
        as=(as & ~(0xffu<<(8*BA))) | (uint32_t(sa[lane/4+8*(lane%2)])<<(8*BA));
    if(lane%4==TB)
        bs=(bs & ~(0xffu<<(8*BB))) | (uint32_t(sb[lane/4])<<(8*BB));
    float2 c0=*reinterpret_cast<const float2*>(c+(lane/4)*8+2*(lane%4));
    float2 c1=*reinterpret_cast<const float2*>(c+(lane/4+8)*8+2*(lane%4));
    float2 d0,d1;
    mma16832_block_scaled<BA,TA,BB,TB>(d0,d1,ar.tiles[0][0].data[0],ar.tiles[0][0].data[1],
        ar.tiles[0][0].data[2],ar.tiles[0][0].data[3],b0,b1,c0,c1,as,bs);
    *reinterpret_cast<float2*>(out+(lane/4)*8+2*(lane%4))=d0;
    *reinterpret_cast<float2*>(out+(lane/4+8)*8+2*(lane%4))=d1;
}

static int cases=0;
static size_t checked_values=0;

template<typename T> float decoded(T x) { return static_cast<float>(x); }

template<typename TA,typename TB>
void check(const char *label,int M,int N,int K,int batches,int repeats,
           const std::vector<TA> &a,const std::vector<TB> &b,
           const std::vector<uint8_t> &sa,const std::vector<uint8_t> &sb,
           const std::vector<float> &c,const std::vector<float> &d) {
    double worst=0;
    for(int batch=0;batch<batches;++batch) for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        double sum=0;
        for(int k=0;k<K;++k) {
            double av=std::ldexp(double(decoded(a[(batch*M+m)*K+k])),int(sa[(batch*M+m)*(K/32)+k/32])-127);
            double bv=std::ldexp(double(decoded(b[(batch*N+n)*K+k])),int(sb[(batch*N+n)*(K/32)+k/32])-127);
            sum+=av*bv;
        }
        size_t offset=(batch*M+m)*N+n;
        float ref=float(double(c[offset])+repeats*sum);
        double error=std::abs(double(d[offset])-ref);
        worst=std::max(worst,error);
        // Inputs are dyadic, bounded so every intermediate sum is exact in FP32.
        if(!std::isfinite(d[offset]) || d[offset]!=ref) {
            std::fprintf(stderr,"FAIL %s M%d N%d K%d batch%d [%d,%d] actual %.9g expected %.9g\n",
                         label,M,N,K,batch,m,n,d[offset],ref);
            std::exit(1);
        }
        ++checked_values;
    }
    ++cases;
    std::fprintf(stdout,"PASS %s M%d N%d K%d batch%d repeats%d max_abs=%.1f\n",label,M,N,K,batches,repeats,worst);
}

template<typename TA,typename TB>
void make_data(int M,int N,int K,int batches,int pattern,std::vector<TA> &a,std::vector<TB> &b,
               std::vector<uint8_t> &sa,std::vector<uint8_t> &sb,std::vector<float> &c) {
    std::mt19937 rng(1234+pattern+M+N+K);
    for(auto &v:a) v=TA(float(int(rng()%17)-8)*0.25f);
    for(auto &v:b) v=TB(float(int(rng()%17)-8)*0.25f);
    for(auto &v:c) v=float(int(rng()%17)-8)*0.125f;
    for(size_t i=0;i<sa.size();++i) sa[i]=pattern==0?127:125+(i*7+i/(K/32))%5;
    for(size_t i=0;i<sb.size();++i) sb[i]=pattern==0?127:125+(i*3+i/(K/32))%5;
    if(pattern==2 || pattern==4) { // One active K position per A row.
        for(int batch=0;batch<batches;++batch) for(int m=0;m<M;++m) for(int k=0;k<K;++k)
            a[(batch*M+m)*K+k]=TA(float(k==(batch*M+m)%K));
    }
    if(pattern==3) std::fill(a.begin(),a.end(),TA(0.f)); // D must preserve C, not scale it.
    if(pattern==4 || pattern==5) {
        // Sweep finite FP8 encodings, including subnormals/max finite, against
        // one-hot operands. One nonzero product keeps the FP64 oracle exact.
        auto sweep=[](auto &values) {
            using T=typename std::remove_reference_t<decltype(values)>::value_type;
            static_assert(sizeof(T)==1);
            for(size_t i=0;i<values.size();++i) {
                reinterpret_cast<unsigned char*>(&values[i])[0]=i%256;
                if(!std::isfinite(decoded(values[i]))) values[i]=T(0.f);
            }
        };
        if(pattern==4) sweep(b);
        else {
            sweep(a);
            for(int batch=0;batch<batches;++batch) for(int n=0;n<N;++n) for(int k=0;k<K;++k)
                b[(batch*N+n)*K+k]=TB(float(k==(batch*N+n)%K));
        }
        std::fill(c.begin(),c.end(),0.f);
    }
}

template<typename TA,typename TB,int M,int N,int K>
void run_tile(const char *label) {
    constexpr int batches=13;
    std::vector<TA>a(batches*M*K); std::vector<TB>b(batches*N*K);
    std::vector<uint8_t>sa(batches*M*K/32),sb(batches*N*K/32);
    std::vector<float>c(batches*M*N),d(c.size(),NAN);
    for(int pattern=0;pattern<6;++pattern) {
        make_data(M,N,K,batches,pattern,a,b,sa,sb,c);
        buffer<TA>da(a);buffer<TB>db(b);buffer<uint8_t>dsa(sa),dsb(sb);buffer<float>dc(c),dd(d);
        for(bool alias:{false,true}) {
            int repeats=(alias && pattern<4)?3:1;
            tile_test<TA,TB,M,N,K><<<(batches+3)/4,128>>>(da.p,db.p,dsa.p,dsb.p,dc.p,dd.p,batches,alias,repeats);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); dd.read(d);
            check(label,M,N,K,batches,repeats,a,b,sa,sb,c,d);
        }
    }
}

template<int S> void run_selector() {
    constexpr int BA=S%4,TA=S/4,BB=(S+1)%4,TB=(S/2)%4;
    std::vector<fp8e4m3>a(16*32),b(8*32);
    std::vector<uint8_t>sa(16),sb(8);std::vector<float>c(16*8),d(c.size(),NAN);
    make_data(16,8,32,1,1,a,b,sa,sb,c);
    buffer<fp8e4m3>da(a),db(b);buffer<uint8_t>dsa(sa),dsb(sb);buffer<float>dc(c),dd(d);
    selector_test<BA,TA,BB,TB><<<1,32>>>(da.p,db.p,dsa.p,dsb.p,dc.p,dd.p);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); dd.read(d);
    check("selectors",16,8,32,1,1,a,b,sa,sb,c,d);
}

int main() {
    cudaDeviceProp props;CUDA_CHECK(cudaGetDeviceProperties(&props,0));
    std::fprintf(stdout,"GPU: %s; SM %d.%d\n",props.name,props.major,props.minor);
    if(props.major!=12) { std::fprintf(stderr,"This test targets SM120\n");return 2; }
    run_tile<fp8e4m3,fp8e4m3,16,16,32>("e4m3/e4m3");
    run_tile<fp8e4m3,fp8e4m3,16,32,64>("e4m3/e4m3");
    run_tile<fp8e4m3,fp8e4m3,32,16,96>("e4m3/e4m3");
    run_tile<fp8e4m3,fp8e4m3,32,32,128>("e4m3/e4m3");
    run_tile<fp8e4m3,fp8e4m3,16,64,64>("e4m3/e4m3");
    run_tile<fp8e4m3,fp8e4m3,64,16,32>("e4m3/e4m3");
    run_tile<fp8e5m2,fp8e5m2,32,32,64>("e5m2/e5m2");
    run_tile<fp8e4m3,fp8e5m2,32,32,64>("e4m3/e5m2");
    run_tile<fp8e5m2,fp8e4m3,32,32,64>("e5m2/e4m3");
    run_selector<0>();run_selector<1>();run_selector<2>();run_selector<3>();
    run_selector<4>();run_selector<5>();run_selector<6>();run_selector<7>();
    std::fprintf(stdout,"PASS ALL: %d cases, %zu values, exact FP32 agreement with CPU FP64 oracle\n",cases,checked_values);
}
