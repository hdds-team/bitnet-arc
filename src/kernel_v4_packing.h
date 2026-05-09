/*
 * bitnet-arc v4 packing helpers: TQ2_0 -> INT2 + FP16 -> INT2 + VNNI.
 *
 * Pure C++ functions, host-testable AND device-callable. No SYCL
 * types here (no esimd::simd<>) so we can unit-test on CPU before
 * embedding in the ESIMD kernel.
 *
 * Per @naskel design v4 review fold #2: "fais deux petites
 * fonctions pures très testables : pack_ternary_to_vnni_int2() +
 * quant_fp16_to_vnni_int2(). Avec des tests unitaires sur fixture."
 *
 * VNNI INT2 layout (per design-v4.md sec3.5):
 *   For a fragment of shape K=64 × N=16 INT2 elements (= 1024 bits
 *   = 64 dwords), each dword packs 16 K-positions × 2 bits:
 *     dword[d] holds K positions {d*16, d*16+1, ..., d*16+15}
 *     bit position of element k within dword = (k % 16) * 2
 *
 *   Storage order across the (d, n) plane: dword-major then N-col:
 *     buf[d * 16 + n] for d in 0..3, n in 0..15
 *
 *   For A fragment of shape M=8 × K=64: similar layout but with M
 *   rows instead of N cols:
 *     buf[d * 8 + m] for d in 0..3, m in 0..7
 *
 * INT2 signed encoding:
 *   value -1 → bit pattern 11
 *   value  0 → bit pattern 00
 *   value +1 → bit pattern 01
 *   value -2 → bit pattern 10  (unused for ternary, leave clamped)
 *
 * Phase 1 v4 W1 fixtures must use heterogeneous data to validate
 * this layout actually matches what xmx::dpas expects. The probe
 * 0x55555555 (uniform-1) was layout-invariant.
 */

#ifndef BITNET_ARC_SRC_KERNEL_V4_PACKING_H
#define BITNET_ARC_SRC_KERNEL_V4_PACKING_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../oracle/fp16.h"
#include "../oracle/tq2_0.h"

namespace bitnet_arc {
namespace kv4 {

/* --- INT2 signed encoding -------------------------------------------- */

/* Map a ternary value in {-1, 0, +1} to its INT2 signed 2-bit pattern.
 * Returns one of {0b00, 0b01, 0b11} (low 2 bits valid).
 *
 * Note: bitwise mask 0x3 is applied at the call site when packing
 * into a dword, so we don't enforce it here.
 */
inline std::uint32_t ternary_to_int2_signed(std::int8_t v) {
    /* +1 → 01,  0 → 00, -1 → 11 (= 2's complement of 1 in 2 bits) */
    return (v < 0) ? 0x3u : ((v == 0) ? 0x0u : 0x1u);
}

/* Inverse mapping: INT2 signed bit pattern → ternary value. Used by
 * the host-side W1 oracle to simulate what DPAS will compute. */
inline std::int8_t int2_signed_to_ternary(std::uint32_t bits) {
    bits &= 0x3u;
    if (bits == 0x0u) return  0;
    if (bits == 0x1u) return  1;
    if (bits == 0x3u) return -1;
    return -2;  /* unused */
}

/* --- TQ2_0 -> ternary unpack ----------------------------------------- *
 *
 * Decode a single TQ2_0 block into an array of 256 INT8 ternary
 * values in {-1, 0, +1}. Same code mapping as the oracle:
 *   stored_code 00 → -1, 01 → 0, 10 → +1
 *   (per oracle/tq2_0.c convention).
 */
inline void unpack_tq2_0_block_to_ternary(
    const bitnet_arc_tq2_0_block& blk,
    std::int8_t out_ternary[256])
{
    for (unsigned k = 0; k < 256u; ++k) {
        const unsigned byte_idx = (k >> 7u) * 32u + (k & 31u);
        const unsigned shift    = ((k >> 5u) & 3u) * 2u;
        const std::uint8_t code = (blk.qs[byte_idx] >> shift) & 0x3u;
        /* code: 0->-1, 1->0, 2->+1 (per oracle/tq2_0.c) */
        out_ternary[k] = static_cast<std::int8_t>(
            (code == 0u) ? -1 : ((code == 1u) ? 0 : 1));
    }
}

/* --- Compute per-row max-abs scale for activation quant -------------- */

inline float compute_a_row_max_abs(
    const std::uint16_t* a_fp16_row, std::size_t K)
{
    float mx = 0.0f;
    for (std::size_t k = 0; k < K; ++k) {
        const float v = bitnet_arc_fp16_to_fp32(a_fp16_row[k]);
        const float a = std::fabs(v);
        if (a > mx) mx = a;
    }
    /* Avoid divide-by-zero on all-zero rows. */
    return (mx > 0.0f) ? mx : 1.0f;
}

/* --- VNNI INT2 packing for B fragment ------------------------------- *
 *
 * Input: ternary_K64xN16 = (K=64, N=16) row-major INT8 buffer with
 *        values in {-1, 0, +1}.
 * Output: out[64] = 64 dwords in VNNI layout
 *         out[d * 16 + n] holds bits for K positions {d*16..d*16+15}
 *         of column n, packed at (k%16)*2 bit positions.
 *
 * Each output dword contains 16 INT2 elements (= 32 bits = 16 × 2).
 *
 * Test invariant: input all = 0 → output all dwords = 0.
 *                 input all = +1 → output all dwords = 0x55555555
 *                 (= 0b01_01...01, 16 reps of '01').
 *                 input all = -1 → output all dwords = 0xFFFFFFFF
 *                 (= 0b11_11...11, 16 reps of '11').
 */
inline void pack_b_fragment_vnni_int2(
    const std::int8_t* ternary_K64xN16,
    std::uint32_t out[64])
{
    std::memset(out, 0, sizeof(std::uint32_t) * 64);
    for (unsigned k = 0; k < 64u; ++k) {
        const unsigned d        = k >> 4u;       /* dword index 0..3 */
        const unsigned bit_pos  = (k & 15u) * 2u; /* bit pos in dword */
        for (unsigned n = 0; n < 16u; ++n) {
            const std::int8_t v = ternary_K64xN16[k * 16u + n];
            const std::uint32_t bits = ternary_to_int2_signed(v);
            out[d * 16u + n] |= (bits & 0x3u) << bit_pos;
        }
    }
}

/* --- FP16 → INT2 quant + VNNI packing for A fragment ---------------- *
 *
 * Input: a_fp16_M8xK64 = (M=8, K=64) row-major FP16 (uint16_t bits)
 *        s_a_per_row[8] = pre-computed per-row max-abs scales
 * Output: out[32] = 32 dwords in VNNI layout
 *         out[d * 8 + m] holds bits for K positions {d*16..d*16+15}
 *         of row m, packed at (k%16)*2 bit positions.
 *
 * Quantization per-row symmetric: a_q[m, k] = round(a/s_a) clamped
 * to {-1, 0, +1}.
 */
inline void pack_a_fragment_vnni_int2(
    const std::uint16_t* a_fp16_M8xK64,
    const float* s_a_per_row,
    std::uint32_t out[32])
{
    std::memset(out, 0, sizeof(std::uint32_t) * 32);
    for (unsigned k = 0; k < 64u; ++k) {
        const unsigned d       = k >> 4u;
        const unsigned bit_pos = (k & 15u) * 2u;
        for (unsigned m = 0; m < 8u; ++m) {
            const float a = bitnet_arc_fp16_to_fp32(
                a_fp16_M8xK64[m * 64u + k]);
            const float s = s_a_per_row[m];
            const float q_f = (s > 0.0f) ? (a / s) : 0.0f;
            /* Round-to-nearest, clamp to {-1, 0, +1}. */
            int q = static_cast<int>(std::lround(q_f));
            if (q < -1) q = -1;
            if (q >  1) q =  1;
            const std::uint32_t bits =
                ternary_to_int2_signed(static_cast<std::int8_t>(q));
            out[d * 8u + m] |= (bits & 0x3u) << bit_pos;
        }
    }
}

} /* namespace kv4 */
} /* namespace bitnet_arc */

#endif /* BITNET_ARC_SRC_KERNEL_V4_PACKING_H */
