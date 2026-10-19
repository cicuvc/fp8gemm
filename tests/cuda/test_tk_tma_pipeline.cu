#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace kt=kittens;
using namespace fp8gemm;
#define CHECK(c) do { auto e=(c); if(e!=cudaSuccess) {std::fprintf(stderr,"%s: %s\n",#c,cudaGetErrorString(e)); std::exit(2);} } while(0)

// Same full producer-WG / consumer-WG split and ring protocol as forward.cu.
// Only the first producer warp publishes. All consumer threads release.
// Register redistribution is unnecessary for these small validation kernels.
template<class Tile,int Stages> struct RingShared { Tile tiles[Stages]; };
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

struct Schedule {
    int steps,stress;
    __host__ __device__ int length(int task) const {return task==0 ? steps : task==1 ? 0 : task==2 ? 1 : steps+2;}
    __host__ __device__ int total() const {return 2*steps+3;}
};

template<class T,int R,int C>
__device__ __forceinline__ void store_rt(T *out,const kt::rt<T,R,C>& x,int row_base,int stride) {
    int lane=threadIdx.x%32;
    constexpr int pack=sizeof(T)==1 ? 4 : 2;
    constexpr int tile_cols=sizeof(T)==1 ? 32 : 16;
#pragma unroll
    for(int i=0;i<R/16;++i) {
#pragma unroll
        for(int j=0;j<C/tile_cols;++j) {
#pragma unroll
            for(int p=0;p<4;++p) {
                int row=row_base+i*16+lane/4+8*(p%2);
                int col=j*tile_cols+pack*(lane%4)+(tile_cols/2)*(p/2);
                uint32_t bits;
                __builtin_memcpy(&bits,&x.tiles[i][j].data[p],4);
                *reinterpret_cast<uint32_t*>(out+row*stride+col)=bits;
            }
        }
    }
}
__device__ __forceinline__ void store_float(float *out,const kt::rt_fl<16,16>& x,int row_base) {
    int lane=threadIdx.x%32;
#pragma unroll
    for(int p=0;p<4;++p)
        *reinterpret_cast<float2*>(out+(row_base+lane/4+8*(p%2))*16+2*(lane%4)+8*(p/2))=x.tiles[0][0].data[p];
}

template<class T,int C,int CW,int Stages>
__global__ __launch_bounds__((CW+4)*32,1) void copy_pipe(
    const __grid_constant__ Global<kt::st<T,CW*16,C>> input,T *output,Schedule schedule) {
    using Tile=kt::st<T,CW*16,C>;
    using Shared=RingShared<Tile,Stages>;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> alloc(reinterpret_cast<int*>(storage));
    const int group=kt::warpgroup::groupid(), wid=threadIdx.x/32;
    const bool producer=wid>=CW;
    MbarrierRingPipe pipe(alloc,BufferSet{&Shared::tiles},int(producer),
                         ProducerWarp<true,1>,CommEndInfo<CW,false,0>{});
    auto &shared=alloc.allocate<Shared>();
    int batch=blockIdx.x/3,depth=blockIdx.x%3,base=0;
    if(producer) {
        if(kt::warpgroup::warpid()==0) {
            bool leader=kt::warp::elect_leader();
            for(int task=0;task<4;++task) {
                pipe.setup();
                int n=schedule.length(task),last=0;
                for(int it=0;it<n;++it) {
                    auto packet=pipe.waitBuffer(0,shared.tiles);
                    auto &tile=packet.template get<0>();
                    if(leader) {
                        if(schedule.stress && ((base+it)&3)==0) __nanosleep(1000);
                        kt::tma::expect_bytes(packet.getBarrier(),kt::tma::buffer_traits<Tile>::transfer_bytes);
                        kt::tma::load_async(tile,input,{batch,depth,(base+it)*Tile::rows,((base+it)%3)*C},packet.getBarrier());
                    }
                    last=packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    pipe.moveNext();
                }
                if(n) pipe.waitSlot(last); // drain before resetting the slot; retain all phases
                base+=n;
            }
        }
    } else {
        const int row_base=16*(wid/4+(wid%4)*(CW/4));
        for(int task=0;task<4;++task) {
            pipe.setup();
            int n=schedule.length(task);
            for(int it=0;it<n;++it) {
                auto packet=pipe.waitBuffer(1,shared.tiles);
                if(schedule.stress && ((wid+it)&3)==0) __nanosleep(2000);
                kt::rt<T,16,C> values;
                kt::group<CW>::load(values,packet.template get<0>());
                // All shared-memory reads complete before release. Subsequent
                // global stores use only registers, while producer reuses SMEM.
                packet.submitToNextAndTrigger();
                pipe.moveNext();
                T *dst=output+(size_t(blockIdx.x)*schedule.total()+base+it)*Tile::rows*C;
                store_rt(dst,values,row_base,C);
            }
            base+=n;
        }
    }
    kt::warpgroup::sync(group+1); // independent retirement for each WG
}

template<class F8,int C,int CW,int Stages> struct MixedShared {
    struct Packet {kt::st_bf<CW*16,C> a; kt::st<F8,16,C> b;};
    Packet tiles[Stages];
};

template<class F8,int C,int CW,int Stages>
__global__ __launch_bounds__((CW+4)*32,1) void mixed_pipe(
    const __grid_constant__ Global<kt::st_bf<CW*16,C>> a,
    const __grid_constant__ Global<kt::st<F8,16,C>> b,float *out,Schedule schedule) {
    using Shared=MixedShared<F8,C,CW,Stages>;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> alloc(reinterpret_cast<int*>(storage));
    int wid=threadIdx.x/32,group=kt::warpgroup::groupid();
    bool producer=wid>=CW;
    MbarrierRingPipe pipe(alloc,BufferSet{&Shared::tiles},int(producer),
                         ProducerWarp<true,1>,CommEndInfo<CW,false,0>{});
    auto &shared=alloc.allocate<Shared>();
    int batch=blockIdx.x/3,depth=blockIdx.x%3,base=0;
    if(producer) {
        if(kt::warpgroup::warpid()==0) {
            bool leader=kt::warp::elect_leader();
            for(int task=0;task<4;++task) {
                pipe.setup();
                int n=schedule.length(task),last=0;
                for(int it=0;it<n;++it) {
                    auto packet=pipe.waitBuffer(0,shared.tiles);
                    auto &tile=packet.template get<0>();
                    if(leader) {
                        if(schedule.stress && (it&3)==0) __nanosleep(1000);
                        kt::tma::expect_bytes(packet.getBarrier(),CW*16*C*2+16*C);
                        kt::tma::load_async(tile.a,a,{batch,depth,(base+it)*CW*16,0},packet.getBarrier());
                        kt::tma::load_async(tile.b,b,{batch,depth,(base+it)*16,0},packet.getBarrier());
                    }
                    last=packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    pipe.moveNext();
                }
                if(n) pipe.waitSlot(last);
                base+=n;
            }
        }
    } else {
        const int row_base=16*(wid/4+(wid%4)*(CW/4));
        for(int task=0;task<4;++task) {
            pipe.setup();
            int n=schedule.length(task);
            for(int it=0;it<n;++it) {
                auto packet=pipe.waitBuffer(1,shared.tiles);
                if(schedule.stress && ((wid+it)&3)==0) __nanosleep(2000);
                kt::rt_bf<16,C> ar;
                kt::rt<F8,16,C> br,aq;
                kt::group<CW>::load(ar,packet.template get<0>().a);
                kt::warp::load<true>(br,packet.template get<0>().b);
                packet.submitToNextAndTrigger();
                pipe.moveNext();
                kt::mxfp8_scale<16,C> as,bs;
                kt::online_quantize_mxfp8(aq,as,ar);
#pragma unroll
                for(int k=0;k<C/32;++k) bs.data[0][k]=127;
                kt::rt_fl<16,16> accum,result;
                kt::warp::rt_maps::zero(accum);
                kt::warp::wmma::mma_ABt_block_scaled(result,aq,br,accum,as,bs);
                float *dst=out+(size_t(blockIdx.x)*schedule.total()+base+it)*CW*16*16;
                store_float(dst,result,row_base);
            }
            base+=n;
        }
    }
    kt::warpgroup::sync(group+1);
}

template<class T> struct Device {
    T *p; size_t count;
    explicit Device(size_t n):count(n) {CHECK(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void upload(const std::vector<T>& x){CHECK(cudaMemcpy(p,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    void read(std::vector<T>& x){CHECK(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));}
};
static int cases=0; static size_t values=0;
template<class T> const char *name(){if constexpr(sizeof(T)==2) return "bf16"; else return std::is_same_v<T,kt::fp8e4m3> ? "e4m3" : "e5m2";}

// Explicit host logical indexing is independent of TK's descriptor/swizzle.
template<class T,int C,int CW,int S>
void check_copy(int steps,int stress,bool padded) {
    using Tile=kt::st<T,CW*16,C>; using Shared=RingShared<Tile,S>;
    Schedule sch{steps,stress}; int rows=sch.total()*Tile::rows-5;
    // Column extent is an exact swizzle-panel multiple: TK's standard 5D
    // descriptor rounds to panels, so arbitrary sub-panel tails need padding.
    int cols=C+Tile::swizzle_bytes/sizeof(T);
    size_t rs=cols+(padded ? 32/sizeof(T) : 0),ds=rs*rows+(padded ? 64/sizeof(T) : 0),bs=ds*3+(padded ? 128/sizeof(T) : 0);
    std::vector<T> input(bs*2),output(size_t(6)*sch.total()*Tile::rows*C);
    for(size_t i=0;i<input.size();++i) {
        uint32_t h=uint32_t(i)*1664525u+1013904223u; h^=h>>16;
        std::memcpy(&input[i],&h,sizeof(T)); // all bit patterns, including NaNs and signed zero
    }
    Device<T> di(input.size()),dout(output.size()); di.upload(input);
    CHECK(cudaMemset(dout.p,0xa5,output.size()*sizeof(T)));
    Global<Tile> gl(di.p,2,3,rows,cols,kt::gl_strides{bs,ds,rs});
    constexpr int smem=sizeof(Shared)+2048;
    CHECK(cudaFuncSetAttribute(copy_pipe<T,C,CW,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    copy_pipe<T,C,CW,S><<<6,(CW+4)*32,smem>>>(gl,dout.p,sch);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); dout.read(output);
    for(int bd=0;bd<6;++bd) for(int it=0;it<sch.total();++it)
        for(int r=0;r<Tile::rows;++r) for(int c=0;c<C;++c) {
            int gr=it*Tile::rows+r,gc=(it%3)*C+c;
            T expected{};
            if(gr<rows && gc<cols) expected=input[(bd/3)*bs+(bd%3)*ds+gr*rs+gc];
            size_t pos=(size_t(bd)*sch.total()+it)*Tile::rows*C+r*C+c;
            if(std::memcmp(&expected,&output[pos],sizeof(T))) {
                std::fprintf(stderr,"FAIL copy %s C%d CW%d S%d steps%d bd%d it%d r%d c%d\n",name<T>(),C,CW,S,steps,bd,it,r,c); std::exit(1);
            }
        }
    ++cases; values+=output.size();
    std::fprintf(stdout,"PASS copy %s C%d CW%d stages%d steps%d stress%d padded%d values%zu\n",name<T>(),C,CW,S,steps,stress,padded,output.size());
}

template<class F8,int C,int CW,int S>
void check_mixed(int steps,int stress) {
    using Shared=MixedShared<F8,C,CW,S>; Schedule sch{steps,stress};
    int rowsa=sch.total()*CW*16-3,rowsb=sch.total()*16-2;
    std::vector<kt::bf16> a(size_t(6)*rowsa*C);
    std::vector<F8> b(size_t(6)*rowsb*C);
    std::vector<float> result(size_t(6)*sch.total()*CW*16*16);
    for(size_t i=0;i<a.size();++i) {
        // BF16 mantissas exceed FP8 precision. Row/K32 scale changes, with
        // exact dyadic output sums so no tolerance hides layout errors.
        int exponent=int((i/C+i/32)%5)-2;
        a[i]=kt::bf16(std::ldexp(float(int((i*73+17)%255)-127)/64.f,exponent));
    }
    for(size_t i=0;i<b.size();++i) b[i]=F8(float(int((i*29+11)%17)-8)/4.f);
    Device<kt::bf16> da(a.size()); Device<F8> db(b.size()); Device<float> dout(result.size());
    da.upload(a); db.upload(b); CHECK(cudaMemset(dout.p,0xa5,result.size()*4));
    Global<kt::st_bf<CW*16,C>> ga(da.p,2,3,rowsa,C);
    Global<kt::st<F8,16,C>> gb(db.p,2,3,rowsb,C);
    constexpr int smem=sizeof(Shared)+2048;
    CHECK(cudaFuncSetAttribute(mixed_pipe<F8,C,CW,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    mixed_pipe<F8,C,CW,S><<<6,(CW+4)*32,smem>>>(ga,gb,dout.p,sch);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); dout.read(result);
    for(int bd=0;bd<6;++bd) for(int it=0;it<sch.total();++it) for(int r=0;r<CW*16;++r) {
        float q[C]; int gr=it*CW*16+r;
        for(int k=0;k<C;k+=32) {
            int e=0;
            if(gr<rowsa) for(int j=0;j<32;++j) {
                uint16_t raw; std::memcpy(&raw,&a[(size_t(bd)*rowsa+gr)*C+k+j],2);
                e=std::max(e,int(raw>>7)&255);
            }
            for(int j=0;j<32;++j) {
                float x=gr<rowsa ? float(a[(size_t(bd)*rowsa+gr)*C+k+j]) : 0.f;
                q[k+j]=std::ldexp(float(F8(std::ldexp(x,127-e))),e-127);
            }
        }
        for(int n=0;n<16;++n) {
            double ref=0; int br=it*16+n;
            if(br<rowsb) for(int k=0;k<C;++k) ref+=double(q[k])*float(b[(size_t(bd)*rowsb+br)*C+k]);
            size_t pos=((size_t(bd)*sch.total()+it)*CW*16+r)*16+n;
            if(result[pos]!=float(ref)) {
                std::fprintf(stderr,"FAIL mixed %s C%d CW%d S%d bd%d it%d r%d n%d got%.9g ref%.9g\n",name<F8>(),C,CW,S,bd,it,r,n,result[pos],float(ref)); std::exit(1);
            }
        }
    }
    ++cases; values+=result.size();
    std::fprintf(stdout,"PASS mixed %s C%d CW%d stages%d steps%d stress%d values%zu\n",name<F8>(),C,CW,S,steps,stress,result.size());
}

template<class T> void copies(bool quick) {
    for(int n : (quick ? std::vector<int>{5} : std::vector<int>{0,1,2,3,7,33})) {
        check_copy<T,32,4,1>(n,1,false);
        check_copy<T,64,4,2>(n,1,true);
        check_copy<T,128,4,3>(n,1,true);
        check_copy<T,64,8,3>(n,1,false);
    }
    check_copy<T,256,4,2>(17,1,true);
}
template<class T> void mixed(bool quick) {
    for(int n : (quick ? std::vector<int>{5} : std::vector<int>{0,1,2,3,7,33})) {
        check_mixed<T,32,4,1>(n,1);
        check_mixed<T,64,4,3>(n,1);
        check_mixed<T,128,8,2>(n,1);
    }
}
int main(int argc,char **argv) {
    bool quick=argc>1 && std::string(argv[1])=="--quick";
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    std::fprintf(stdout,"GPU %s SM%d.%d shared_optin=%zu\n",prop.name,prop.major,prop.minor,prop.sharedMemPerBlockOptin);
    copies<kt::bf16>(quick); copies<kt::fp8e4m3>(quick); copies<kt::fp8e5m2>(quick);
    check_copy<kt::bf16,16,4,3>(7,1,true);
    mixed<kt::fp8e4m3>(quick); mixed<kt::fp8e5m2>(quick);
    std::fprintf(stdout,"PASS ALL: %d cases, %zu values\n",cases,values);
}
