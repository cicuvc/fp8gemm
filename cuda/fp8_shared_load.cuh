#pragma once
#include "kittens.cuh"

namespace fp8gemm {
// Complete FP8 shared allocations; optionally read a slice starting at a K32
// tile index. The caller must keep the slice within ST::cols. The allocation
// must honor ST's 8*swizzle_bytes alignment, keeping the base outside the XOR.
template<int GW, bool Swap, class RT, class ST>
__device__ inline void load_fp8_factored(RT &dst, const ST &src, int k32_start=0) {
    namespace kt = kittens;
    static_assert(sizeof(typename ST::dtype) == 1);
    static_assert(std::is_same_v<typename RT::layout, kt::ducks::rt_layout::row>);
    static_assert(ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols);
    static_assert(ST::cols >= RT::cols && ST::rows % (GW * RT::rows) == 0);
    constexpr int SB = ST::swizzle_bytes;
    static_assert(alignof(ST) >= 8 * SB && ST::required_alignment >= 8 * SB);
    constexpr int panel_tiles = SB / RT::tile_size_col;
    int lane = kt::laneid(), warp = kt::group<GW>::warpid();
    if constexpr (GW % 4 == 0) warp = warp / 4 + (warp % 4) * (GW / 4);
    int row = Swap ? (lane % 8) + (lane / 16) * 8 : lane % 16;
    int col = (Swap ? (lane / 8) % 2 : lane / 16) * 16;
    // Disjoint row-group/lane bits: XOR prevents ptxas from reassociating
    // the warp offset into the changing uniform ring-buffer base.
    uint32_t lane_offset = ST::idx(0u, {row, col}) ^ (warp * RT::rows * SB);
    uint32_t shared_base = __cvta_generic_to_shared(src.data);
#pragma unroll
    for (int i = 0; i < RT::height; ++i) {
#pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            int col_tile = k32_start + j;
            uint32_t offset = lane_offset ^ ((col_tile % panel_tiles) * RT::tile_size_col);
            uint32_t addr = shared_base + offset + i * RT::tile_size_row * SB
                + (col_tile / panel_tiles) * ST::rows * SB;
            auto &d = dst.tiles[i][j].data;
            using U2 = typename kt::base_types::packing<typename ST::dtype>::packed_type;
            if constexpr (Swap) kt::move<U2>::ldsm4(d[0], d[2], d[1], d[3], addr);
            else kt::move<U2>::ldsm4(d[0], d[1], d[2], d[3], addr);
        }
    }
}
}

namespace fp8gemm {
// Full, aligned shared tiles only; preserve the native FP8 transposing LDSM
// fragment order. Row-tile advances are whole swizzle periods.
template<class RT, class ST>
__device__ __forceinline__ void load_fp8_col_factored(RT &dst, const ST &src) {
    static_assert(sizeof(typename ST::dtype) == 1);
    static_assert(std::is_same_v<typename RT::layout, kittens::ducks::rt_layout::col>);
    static_assert(ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols);
    static_assert(ST::rows == RT::rows && ST::cols == RT::cols);
    constexpr int SB = ST::swizzle_bytes;
    static_assert(alignof(ST) >= 8 * SB && ST::required_alignment >= 8 * SB);
    static_assert(RT::tile_size_row == 32 && RT::tile_size_col == 16);
    constexpr int panel_tiles = SB / 16;
    const uint32_t lane_offset = ST::idx(0u, {kittens::laneid(), 0});
    const uint32_t shared_base = __cvta_generic_to_shared(src.data);
#pragma unroll
    for (int i = 0; i < RT::height; ++i) {
#pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const uint32_t addr = shared_base + (lane_offset ^ ((j % panel_tiles) * 16))
                + i * 32 * SB + (j / panel_tiles) * ST::rows * SB;
            auto &d = dst.tiles[i][j].data;
            asm volatile(
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&d[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&d[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&d[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&d[3]))
                : "r"(addr));
        }
    }
}

}
