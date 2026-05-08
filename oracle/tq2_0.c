/*
 * TQ2_0 quantize / dequantize implementation.
 *
 * Reference port of upstream llama.cpp ggml/src/ggml-quants.c
 * (quantize_row_tq2_0_ref + dequantize_row_tq2_0). Codebook and packing
 * layout match upstream verbatim so that GGUF blobs produced by either
 * side dequantize bit-identically. Validated by the one-shot gating
 * test in oracle/upstream_gating/.
 *
 * Codebook (from upstream dequantize_row_tq2_0):
 *
 *     stored_code = 0  ->  -1 * d
 *     stored_code = 1  ->   0 * d
 *     stored_code = 2  ->  +1 * d
 *     stored_code = 3  ->  invalid (must not be emitted by quantize)
 *
 * Equivalently: dequantized_value = (stored_code - 1) * d.
 *
 * Packing layout (from upstream quantize_row_tq2_0_ref):
 *
 *   - 256 weights per block, viewed as two halves of 128 weights.
 *   - Each half is packed into 32 bytes (qs[0..31] for the first half,
 *     qs[32..63] for the second).
 *   - Within a half, byte qs[base + m] (m in [0, 32)) holds 4 weights
 *     at positions  m + n*32 for n in {0, 1, 2, 3}, where n*2 is the
 *     bit-shift inside the byte:
 *
 *       bits 0..1 -> weight at position base*4 + m + 0*32
 *       bits 2..3 -> weight at position base*4 + m + 1*32
 *       bits 4..5 -> weight at position base*4 + m + 2*32
 *       bits 6..7 -> weight at position base*4 + m + 3*32
 *
 *     where base in {0, 32}; base*4 in {0, 128} is the half-block start.
 *
 * Convenience formula used below:
 *
 *     for weight at block-relative position i in [0, 256):
 *         byte_index = (i >> 7) * 32 + (i & 31)
 *         bit_shift  = ((i >> 5) & 3) * 2
 */

#include "tq2_0.h"
#include "fp16.h"

#include <assert.h>
#include <string.h>

/* Code -> ternary value, matching upstream's `(q - 1) * d` rule.
 * Index 3 is reserved; we map it to 0 defensively (must never appear
 * in a valid TQ2_0 stream). */
static const int8_t k_code_to_ternary[4] = { -1, 0, 1, 0 };

static inline uint8_t ternary_to_code(int8_t v) {
    /* Inverse of `(code - 1)`: ternary value v -> code (v + 1). */
    switch (v) {
        case -1: return 0u;
        case  0: return 1u;
        case  1: return 2u;
        default:
            assert(0 && "ternary_to_code: input out of {-1, 0, +1}");
            return 1u;
    }
}

/* Map block-relative weight position to (byte_index, bit_shift) in qs. */
static inline void tq2_0_pos_to_byte_shift(size_t i,
                                           size_t  *byte_out,
                                           unsigned *shift_out)
{
    *byte_out  = (i >> 7) * 32u + (i & 31u);
    *shift_out = (unsigned)(((i >> 5) & 3u) * 2u);
}

void bitnet_arc_quantize_row_tq2_0(const int8_t *src,
                                   bitnet_arc_tq2_0_block *dst,
                                   size_t n,
                                   const float *scale_per_block)
{
    const size_t B = BITNET_ARC_TQ2_0_BLOCK_SIZE;
    assert((n % B) == 0 && "n must be a multiple of block size");

    const size_t nblocks = n / B;
    for (size_t b = 0; b < nblocks; ++b) {
        bitnet_arc_tq2_0_block *blk = &dst[b];
        const int8_t *w = &src[b * B];

        memset(blk->qs, 0, sizeof(blk->qs));
        const float scale = scale_per_block ? scale_per_block[b] : 1.0f;
        blk->d = bitnet_arc_fp32_to_fp16(scale);

        /* Upstream layout: stride-32 interleaving inside each 128-weight
         * half. See header comment for derivation. */
        for (size_t i = 0; i < B; ++i) {
            const uint8_t code = ternary_to_code(w[i]);
            size_t   byte;
            unsigned shift;
            tq2_0_pos_to_byte_shift(i, &byte, &shift);
            blk->qs[byte] |= (uint8_t)(code << shift);
        }
    }
}

void bitnet_arc_dequantize_row_tq2_0(const bitnet_arc_tq2_0_block *src,
                                     float *dst,
                                     size_t n)
{
    const size_t B = BITNET_ARC_TQ2_0_BLOCK_SIZE;
    assert((n % B) == 0 && "n must be a multiple of block size");

    const size_t nblocks = n / B;
    for (size_t b = 0; b < nblocks; ++b) {
        const bitnet_arc_tq2_0_block *blk = &src[b];
        const float scale = bitnet_arc_fp16_to_fp32(blk->d);
        float *out = &dst[b * B];

        for (size_t i = 0; i < B; ++i) {
            size_t   byte;
            unsigned shift;
            tq2_0_pos_to_byte_shift(i, &byte, &shift);
            const uint8_t code = (uint8_t)((blk->qs[byte] >> shift) & 0x3u);
            out[i] = scale * (float)k_code_to_ternary[code];
        }
    }
}
