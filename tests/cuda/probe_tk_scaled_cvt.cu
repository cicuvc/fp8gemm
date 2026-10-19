// Compile-only capability probe, intentionally rejected for SM120a by CUDA
// 13.4. PTX 9.4 limits scaled::n1::ue8m0 down-conversion to the SM107 family.
// The b8 scale is declared explicitly so the diagnostic is about the target,
// rather than a mismatched inline-asm operand type. No kernel is launched.
#include <cuda_runtime.h>
#ifdef PROBE_E5
#define PROBE_FORMAT "e5m2x2"
#else
#define PROBE_FORMAT "e4m3x2"
#endif
__global__ void probe_scaled_cvt(unsigned *out, unsigned bf16x2, unsigned scale) {
    unsigned short d;
    asm("{ .reg .b8 sf; mov.b32 {sf,_,_,_}, %2; "
        "cvt.rn.satfinite.scaled::n1::ue8m0." PROBE_FORMAT ".bf16x2 %0, %1, sf; }"
        : "=h"(d) : "r"(bf16x2), "r"(scale));
    *out=d;
}
