// Standalone file-driven fragment validation; no Torch headers or full GEMM.
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
using namespace kittens;
#define CHECK(c) do { auto e=(c); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#c,cudaGetErrorString(e)); std::exit(2); } } while(0)

template<typename F8, int M, int K, bool Gram>
__global__ void quantize(const bf16 *x, F8 *q, uint8_t *scales, float *gram, int batches) {
    int batch=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32;
    if(batch>=batches) return;
    x+=batch*M*K; q+=batch*M*K; scales+=batch*M*K/32;
    rt<bf16,M,K> src;
    rt<F8,M,K> dst;
    mxfp8_scale<M,K> sf;
#pragma unroll
    for(int i=0;i<M/16;++i) {
#pragma unroll
        for(int j=0;j<K/16;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                int row=i*16+lane/4+8*(p%2), col=j*16+2*(lane%4)+8*(p/2);
                src.tiles[i][j].data[p]=*reinterpret_cast<const bf16_2*>(x+row*K+col);
            }
        }
    }
    online_quantize_mxfp8(dst,sf,src);
#pragma unroll
    for(int i=0;i<M/16;++i) {
#pragma unroll
        for(int j=0;j<K/32;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                int row=i*16+lane/4+8*(p%2), col=j*32+4*(lane%4)+16*(p/2);
                *reinterpret_cast<uint32_t*>(q+row*K+col)=std::bit_cast<uint32_t>(dst.tiles[i][j].data[p]);
            }
            if(lane%4<2) scales[(i*16+lane/4+8*(lane%2))*(K/32)+j]=sf.data[i][j];
        }
    }
    if constexpr(Gram) {
        rt_fl<M,M> c,d;
        warp::rt_maps::zero(c);
        warp::wmma::mma_ABt_block_scaled(d,dst,dst,c,sf,sf);
        gram+=batch*M*M;
#pragma unroll
        for(int i=0;i<M/16;++i) {
#pragma unroll
            for(int j=0;j<M/16;++j) {
#pragma unroll
                for(int p=0;p<4;++p) {
                    int row=i*16+lane/4+8*(p%2), col=j*16+2*(lane%4)+8*(p/2);
                    *reinterpret_cast<float2*>(gram+row*M+col)=d.tiles[i][j].data[p];
                }
            }
        }
    }
}

template<typename F8,int M,int K>
void run(const char *input,const char *output,int batches,bool gram) {
    size_t count=size_t(batches)*M*K;
    std::vector<bf16> x(count); std::vector<F8> q(count);
    std::vector<uint8_t> s(count/32); std::vector<float> g(gram ? batches*M*M : 0);
    std::ifstream in(input,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(x.data()),count*2) || in.peek()!=EOF) std::exit(3);
    bf16 *dx; F8 *dq; uint8_t *ds; float *dg=nullptr;
    CHECK(cudaMalloc(&dx,count*2)); CHECK(cudaMalloc(&dq,count)); CHECK(cudaMalloc(&ds,count/32));
    if(gram) CHECK(cudaMalloc(&dg,g.size()*4));
    CHECK(cudaMemcpy(dx,x.data(),count*2,cudaMemcpyHostToDevice));
    if(gram) quantize<F8,M,K,true><<<(batches+3)/4,128>>>(dx,dq,ds,dg,batches);
    else quantize<F8,M,K,false><<<(batches+3)/4,128>>>(dx,dq,ds,dg,batches);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(q.data(),dq,count,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(s.data(),ds,count/32,cudaMemcpyDeviceToHost));
    if(gram) CHECK(cudaMemcpy(g.data(),dg,g.size()*4,cudaMemcpyDeviceToHost));
    std::ofstream out(output,std::ios::binary);
    out.write(reinterpret_cast<char*>(q.data()),count);
    out.write(reinterpret_cast<char*>(s.data()),count/32);
    if(gram) out.write(reinterpret_cast<char*>(g.data()),g.size()*4);
    if(!out) std::exit(4);
    CHECK(cudaFree(dx)); CHECK(cudaFree(dq)); CHECK(cudaFree(ds)); if(gram) CHECK(cudaFree(dg));
    std::fprintf(stdout,"PASS launch M=%d K=%d batches=%d format=%s gram=%d\n",M,K,batches,
                 std::is_same_v<F8,fp8e4m3> ? "e4m3" : "e5m2",gram);
}
template<typename F8>
void dispatch(int m,int k,const char *in,const char *out,int batches,bool gram) {
#define SHAPE(M,K) if(m==M && k==K) {run<F8,M,K>(in,out,batches,gram); return;}
    SHAPE(16,32) SHAPE(16,64) SHAPE(32,64) SHAPE(32,96) SHAPE(64,128)
#undef SHAPE
    std::exit(5);
}
int main(int argc,char **argv) {
    if(argc!=8) {std::fprintf(stderr,"usage: test M K batches e4|e5 gram input output\n"); return 1;}
    int m=std::atoi(argv[1]),k=std::atoi(argv[2]),b=std::atoi(argv[3]);
    if(b<=0) return 1;
    if(std::string(argv[4])=="e4") dispatch<fp8e4m3>(m,k,argv[6],argv[7],b,std::atoi(argv[5]));
    else if(std::string(argv[4])=="e5") dispatch<fp8e5m2>(m,k,argv[6],argv[7],b,std::atoi(argv[5]));
    else return 1;
}
