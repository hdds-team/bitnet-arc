/*
 * IEEE-754 binary16 (FP16) <-> FP32 conversion helpers.
 *
 * Header-only, inline. No platform _Float16 dependency: pure bit math
 * on uint16_t / uint32_t to keep the oracle reproducible across
 * toolchains (gcc, clang, MSVC, oneAPI).
 *
 * Round-to-nearest-even is approximated via truncation on the FP32 ->
 * FP16 path; this is acceptable for the oracle because:
 *   - Input activations in v0 are already valid FP16 (round-trip safe)
 *   - The oracle reduction is exact for ternary x FP16 -> FP32 anyway
 *
 * If we later need IEEE-correct rounding for FP32 -> FP16, swap the
 * truncation with proper round-to-nearest-even in fp32_to_fp16(); the
 * inverse fp16_to_fp32() is already exact.
 */

#ifndef BITNET_ARC_ORACLE_FP16_H
#define BITNET_ARC_ORACLE_FP16_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint16_t bitnet_arc_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) {
        /* Inf or NaN. Preserve NaN-ness via mantissa MSB. */
        return (uint16_t)((sign << 15) | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    }
    if (exp > 15) {
        /* Overflow -> Inf. */
        return (uint16_t)((sign << 15) | (0x1Fu << 10));
    }
    if (exp < -14) {
        /* Subnormal range or underflow to zero (truncation). */
        if (exp < -24) return (uint16_t)(sign << 15);
        uint32_t sub = (mant | 0x800000u) >> (-exp - 14 + 13);
        return (uint16_t)((sign << 15) | sub);
    }
    /* Normal. */
    return (uint16_t)((sign << 15)
                    | ((uint32_t)(exp + 15) << 10)
                    | (mant >> 13));
}

static inline float bitnet_arc_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)((h >> 15) & 0x1u);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            /* Subnormal: normalize. */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31)
                 | ((uint32_t)(127 - 15 - e) << 23)
                 | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        /* Inf or NaN. */
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        /* Normal. */
        bits = (sign << 31)
             | ((exp + (127 - 15)) << 23)
             | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_ORACLE_FP16_H */
