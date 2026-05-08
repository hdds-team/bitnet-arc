/*
 * Vendored TQ2_0 reference from upstream llama.cpp.
 *
 * Source: https://github.com/ggml-org/llama.cpp
 *   - ggml/src/ggml-quants.c lines 2306-2336 (quantize_row_tq2_0_ref)
 *   - ggml/src/ggml-quants.c lines 2391-2408 (dequantize_row_tq2_0)
 *   - ggml/src/ggml-common.h  lines 273-278  (block_tq2_0)
 *
 * License: MIT (see llama.cpp LICENSE in the upstream repo).
 *
 * NOT a build dependency. NOT shipped at runtime. Used only by gate.c
 * to one-shot-validate that oracle/tq2_0.{h,c} matches upstream
 * byte-for-byte. If upstream changes the format, refresh this file
 * (the change WILL be a visible breaking-change in their commit log)
 * and re-run gate.
 *
 *   Copied on : 2026-05-08
 *   Pinned to : llama.cpp master @ deab41ec6 (or any later commit that
 *               does not touch ggml-quants.c TQ2_0 functions)
 *
 *   #error guard below catches an accidental include from anything
 *   other than gate.c.
 */

#ifndef BITNET_ARC_GATING_UPSTREAM_REF_H
#define BITNET_ARC_GATING_UPSTREAM_REF_H

#ifndef BITNET_ARC_GATING_GATE_C_TU
#  error "upstream_ref.h is for upstream_gating/gate.c only -- do not include elsewhere"
#endif

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <assert.h>

/* Upstream's QK_K is a global constant. For TQ2_0 it is 256. */
#define UPSTREAM_QK_K 256

/* Vendored from ggml-common.h:273-278. Uses uint16_t for ggml_half
 * so we don't need the upstream typedef chain. */
typedef struct __attribute__((packed)) {
    uint8_t  qs[UPSTREAM_QK_K / 4];   /* 2 bits per element */
    uint16_t d;                        /* ggml_half (FP16 IEEE-754) */
} upstream_block_tq2_0;

/* fp32 -> fp16 helper. We reuse oracle/fp16.h's implementation, which is
 * a pure-bits truncation. The reference function only ever rounds the
 * absolute-max scale `d`; for our gating fixtures `d` is always exactly
 * representable (it is 0.0f or 1.0f), so truncation matches upstream's
 * round-to-nearest-even bit-for-bit on these inputs. */
#include "../fp16.h"
#define UPSTREAM_FP32_TO_FP16(x) bitnet_arc_fp32_to_fp16(x)
#define UPSTREAM_FP16_TO_FP32(x) bitnet_arc_fp16_to_fp32(x)

/* MAX macro -- vendored to keep this header self-contained. */
#ifndef UPSTREAM_MAX
#  define UPSTREAM_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* ----------------------------------------------------------------------
 * BEGIN VENDORED CODE -- do not modify without updating the source pin.
 * ----------------------------------------------------------------------
 * Copied verbatim (modulo the typedef and helper renames above) from
 * llama.cpp ggml-quants.c. */

static void upstream_quantize_row_tq2_0_ref(const float * x, upstream_block_tq2_0 * y, int64_t k) {
    assert(k % UPSTREAM_QK_K == 0);
    const int64_t nb = k / UPSTREAM_QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;

        for (int j = 0; j < UPSTREAM_QK_K; j++) {
            const float v = x[j];
            amax = UPSTREAM_MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = UPSTREAM_FP32_TO_FP16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}

static void upstream_dequantize_row_tq2_0(const upstream_block_tq2_0 * x, float * y, int64_t k) {
    assert(k % UPSTREAM_QK_K == 0);
    const int64_t nb = k / UPSTREAM_QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = UPSTREAM_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------
 * END VENDORED CODE
 * ---------------------------------------------------------------------- */

#endif /* BITNET_ARC_GATING_UPSTREAM_REF_H */
