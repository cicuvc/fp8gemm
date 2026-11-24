// Included after the family defines CF, SK_KERNEL, make_args, SK_R/C/COMP.
namespace fp8gemm::sk_host {
namespace {
struct Plan {
    CF::Args args;
    void *pointers[16];
    int m,n,k,grid,slots,nfix,regs;
    Plan(void **p,int m_,int n_,int k_,int grid_,int slots_,int nf):
        args(make_args(p,m_,n_,k_,slots_)),m(m_),n(n_),k(k_),grid(grid_),slots(slots_),nfix(nf) {
        for(int i=0;i<16;i++)pointers[i]=p[i];
        if(m<=0||n<=0||k<=0||grid<=0||slots<0||nfix<0)throw std::runtime_error("Invalid stream-K dimensions");
        sk::check(cudaFuncSetAttribute(SK_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes a;sk::check(cudaFuncGetAttributes(&a,SK_KERNEL));regs=a.numRegs;
        if(a.localSizeBytes||regs<168)throw std::runtime_error("Unexpected register pool/local memory");
    }
    void launch(cudaStream_t stream,int phase) {
        if constexpr(sk::atomic) {
            if(phase!=2 && nfix)sk::check(cudaMemsetAsync(pointers[11],0,
                size_t(slots)*SK_COMP*8*SK_R*SK_C*sizeof(float),stream));
        }
        if(phase==3)return;
        if(phase!=2) {SK_KERNEL<<<grid,384,CF::smem,stream>>>(args);sk::check(cudaGetLastError());}
        if(phase!=1 && nfix) {
            sk::fixup<SK_R,SK_C,SK_COMP><<<nfix,256,0,stream>>>((float*)pointers[11],
                (kt::bf16*)pointers[8],(kt::bf16*)pointers[9],(kt::bf16*)pointers[10],
                (sk::Fixup*)pointers[14],(int*)pointers[15],m,n);
            sk::check(cudaGetLastError());
        }
    }
};
static thread_local std::string error;
}
}
extern "C" const char *tk_ffsk_error(){return fp8gemm::sk_host::error.c_str();}
extern "C" void *tk_ffsk_create(void **p,int m,int n,int k,int grid,int slots,int nfix) {
    try{return new fp8gemm::sk_host::Plan(p,m,n,k,grid,slots,nfix);}
    catch(const std::exception &e){fp8gemm::sk_host::error=e.what();return nullptr;}
}
extern "C" int tk_ffsk_launch(void *handle,void *s,int phase) {
    auto *p=static_cast<fp8gemm::sk_host::Plan*>(handle);
    try{p->launch((cudaStream_t)s,phase);return 0;}
    catch(const std::exception &e){fp8gemm::sk_host::error=e.what();return -1;}
}
extern "C" void tk_ffsk_info(void *handle,int *info){auto *p=static_cast<fp8gemm::sk_host::Plan*>(handle);info[0]=CF::smem;info[1]=p->regs;}
extern "C" void tk_ffsk_destroy(void *handle){delete static_cast<fp8gemm::sk_host::Plan*>(handle);}
