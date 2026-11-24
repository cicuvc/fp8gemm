/** Host-side tiled tensor maps for MXFP8 kernels.
 * CUtensorMap is opaque: use the driver encoder while constructing/rebinding
 * plans, before graph capture. The retired manual encoding set word 2 bit 21
 * incorrectly for some UINT8 shapes (e.g. 129x640), and could fault in TMA loads
 * after allocation reuse. Do not infer these flags from tensor element count.
 */
#pragma once
#include <cuda.h>

namespace kittens::tma {
__host__ static inline CUresult
encode_tiled(CUtensorMap *tensor_map, CUtensorMapDataType data_type, uint32_t rank, void *global_address,
             const uint64_t *global_dim, const uint64_t *global_stride, const uint32_t *box_dim,
             const uint32_t *element_stride, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
             CUtensorMapL2promotion l2_promotion, CUtensorMapFloatOOBfill oob_fill) noexcept {
    return cuTensorMapEncodeTiled(tensor_map, data_type, rank, global_address, global_dim, global_stride,
                                 box_dim, element_stride, interleave, swizzle, l2_promotion, oob_fill);
}
} // namespace kittens::tma
