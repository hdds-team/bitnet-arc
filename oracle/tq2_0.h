/*
 * TQ2_0 ternary block format (oracle reference, port of upstream).
 *
 * Format spec (from llama.cpp upstream, ggml/src/ggml-quants.c
 * dequantize_row_tq2_0 + quantize_row_tq2_0_ref):
 *   - 256 weights per block (QK_K = 256)
 *   - 64 bytes for 2-bit codes, organized as two 32-byte halves with
 *     stride-32 interleaving inside each half (see tq2_0.c for the
 *     exact byte/shift formula)
 *   - 2 bytes FP16 scale (IEEE-754 binary16)
 *   - Total: 66 bytes per 256 weights = 2.0625 bpw exact
 *
 * Codebook (verified against upstream by @claude-opus):
 *   00 -> -1 * d
 *   01 ->  0 * d
 *   10 -> +1 * d
 *   11 -> reserved (must not appear in valid TQ2_0 streams)
 *
 * Equivalently: dequantized_value = (stored_code - 1) * d.
 *
 * Bit-identical round-trip with upstream is CONFIRMED by the one-shot
 * gating test in oracle/upstream_gating/ -- 10000/10000 tensors at
 * lengths {256, 512, 1024, 2048, 4096} passed on llama.cpp commit
 * deab41ec6 (run by @claude-opus). If gating ever regresses, fix the
 * mismatch here (codebook and/or packing formula in tq2_0.c) -- never
 * patch the gating test.
 */

#ifndef BITNET_ARC_ORACLE_TQ2_0_H
#define BITNET_ARC_ORACLE_TQ2_0_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BITNET_ARC_TQ2_0_BLOCK_SIZE       256
#define BITNET_ARC_TQ2_0_BYTES_PER_BLOCK  66 /* 64 codes + 2 scale */

/* Pack the block to defend against compiler-inserted padding. The
 * upstream block_tq2_0 struct is 66 bytes exact (qs[64] + ggml_half),
 * and we want our struct to round-trip byte-identically with files
 * produced by upstream. */
#if defined(__GNUC__) || defined(__clang__)
#  define BITNET_ARC_TQ2_0_PACKED __attribute__((packed))
#elif defined(_MSC_VER)
#  pragma pack(push, 1)
#  define BITNET_ARC_TQ2_0_PACKED
#else
#  define BITNET_ARC_TQ2_0_PACKED
#endif

typedef struct BITNET_ARC_TQ2_0_PACKED {
    uint8_t  qs[64]; /* 256 weights x 2 bits, stride-32 interleaved */
    uint16_t d;      /* FP16 scale (IEEE-754 binary16) */
} bitnet_arc_tq2_0_block;

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif

/* Compile-time assertion: block must be exactly 66 bytes. Fires at
 * compile if a toolchain or struct edit accidentally inserts padding. */
typedef char bitnet_arc_tq2_0_block_size_check[
    sizeof(bitnet_arc_tq2_0_block) == BITNET_ARC_TQ2_0_BYTES_PER_BLOCK
        ? 1 : -1];

/*
 * Quantize n ternary weights (each int8 in {-1, 0, +1}) into TQ2_0 blocks.
 *
 *   src              : n int8 values, each in {-1, 0, +1}
 *   dst              : (n / 256) blocks of bitnet_arc_tq2_0_block
 *   n                : must be a multiple of BITNET_ARC_TQ2_0_BLOCK_SIZE
 *   scale_per_block  : caller-provided FP32 scale, one per block.
 *                      If NULL, scale defaults to 1.0f (valid for
 *                      already-ternary input).
 *
 * Asserts on inputs out of {-1, 0, +1} or n not a multiple of 256.
 */
void bitnet_arc_quantize_row_tq2_0(const int8_t *src,
                                   bitnet_arc_tq2_0_block *dst,
                                   size_t n,
                                   const float *scale_per_block);

/*
 * Dequantize n weights from TQ2_0 blocks to FP32.
 *
 * Output is exactly representable as FP32 since values are
 * scale_fp32 * {-1, 0, +1}, where scale_fp32 = fp16_to_fp32(blk->d).
 *
 *   src : (n / 256) blocks
 *   dst : n FP32 values
 *   n   : must be a multiple of BITNET_ARC_TQ2_0_BLOCK_SIZE
 */
void bitnet_arc_dequantize_row_tq2_0(const bitnet_arc_tq2_0_block *src,
                                     float *dst,
                                     size_t n);

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_ORACLE_TQ2_0_H */
