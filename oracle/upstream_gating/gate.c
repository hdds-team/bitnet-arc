/*
 * One-shot bit-identical gating test.
 *
 * For each ternary input file produced by gen_inputs.py:
 *   1. Read int8 array (length must be a multiple of 256).
 *   2. Quantize via our oracle -> blocks A.
 *   3. Convert the same int8 input to FP32 and quantize via the
 *      vendored upstream reference -> blocks B.
 *   4. memcmp(A, B). Must be byte-equal.
 *   5. Dequantize both A and B back to FP32. Must be byte-equal.
 *
 * Exit 0 on full pass, non-zero on first mismatch (with diagnostics).
 *
 * Build:
 *     make gate
 * Run:
 *     ./gate inputs/inputs_NNN.bin ...
 */

#define BITNET_ARC_GATING_GATE_C_TU 1

#include "../tq2_0.h"
#include "upstream_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#define BLOCK_SIZE 256

static int file_size_int8(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); fclose(f); return -1; }
    long sz = ftell(f);
    fclose(f);
    if (sz < 0) return -1;
    *out_n = (size_t)sz;
    return 0;
}

static int read_int8(const char *path, int8_t *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    size_t r = fread(buf, 1, n, f);
    fclose(f);
    if (r != n) {
        fprintf(stderr, "%s: short read (%zu of %zu bytes)\n", path, r, n);
        return -1;
    }
    return 0;
}

/* Compare two byte buffers, print the first divergence and a small
 * window around it on failure. Returns 0 on equal, non-zero on diff. */
static int diff_bytes(const char *label,
                      const uint8_t *a,
                      const uint8_t *b,
                      size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  %s: first divergence at byte %zu / %zu\n",
                    label, i, n);
            size_t lo = i >= 4 ? i - 4 : 0;
            size_t hi = i + 4 < n ? i + 4 : n - 1;
            fprintf(stderr, "    ours    [%zu..%zu]: ", lo, hi);
            for (size_t k = lo; k <= hi; ++k) fprintf(stderr, "%02x ", a[k]);
            fprintf(stderr, "\n    upstream[%zu..%zu]: ", lo, hi);
            for (size_t k = lo; k <= hi; ++k) fprintf(stderr, "%02x ", b[k]);
            fprintf(stderr, "\n");
            return 1;
        }
    }
    return 0;
}

static int gate_one(const char *path) {
    size_t nbytes;
    if (file_size_int8(path, &nbytes) != 0) return 1;
    if (nbytes == 0 || (nbytes % BLOCK_SIZE) != 0) {
        fprintf(stderr, "%s: size %zu is not a multiple of %d\n",
                path, nbytes, BLOCK_SIZE);
        return 1;
    }
    const size_t n = nbytes;
    const size_t nblocks = n / BLOCK_SIZE;

    int8_t *src_i8 = malloc(n);
    float  *src_f  = malloc(n * sizeof(float));
    bitnet_arc_tq2_0_block *ours    = malloc(nblocks * sizeof(*ours));
    upstream_block_tq2_0   *upstr   = malloc(nblocks * sizeof(*upstr));
    float *deq_ours  = malloc(n * sizeof(float));
    float *deq_upstr = malloc(n * sizeof(float));
    float *scales    = malloc(nblocks * sizeof(float));
    if (!src_i8 || !src_f || !ours || !upstr || !deq_ours || !deq_upstr || !scales) {
        fprintf(stderr, "%s: allocation failed\n", path);
        return 1;
    }

    if (read_int8(path, src_i8, n) != 0) return 1;

    /* int8 -> float; also compute per-block amax to feed our quantize.
     * Upstream derives the scale from amax internally; we accept it as
     * a parameter, so we mirror upstream's policy for fair comparison. */
    for (size_t b = 0; b < nblocks; ++b) {
        int8_t bm = 0;
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            int8_t v = src_i8[b * BLOCK_SIZE + i];
            src_f[b * BLOCK_SIZE + i] = (float)v;
            int8_t av = v < 0 ? (int8_t)-v : v;
            if (av > bm) bm = av;
        }
        scales[b] = (float)bm;
    }

    bitnet_arc_quantize_row_tq2_0(src_i8, ours, n, scales);
    upstream_quantize_row_tq2_0_ref(src_f, upstr, (int64_t)n);

    /* Sanity: both block sizes are 66 bytes packed. */
    if (sizeof(*ours) != 66 || sizeof(*upstr) != 66) {
        fprintf(stderr, "%s: struct sizes not 66 (ours=%zu, upstream=%zu)\n",
                path, sizeof(*ours), sizeof(*upstr));
        return 1;
    }

    if (diff_bytes("quantize",
                   (const uint8_t *)ours,
                   (const uint8_t *)upstr,
                   nblocks * sizeof(*ours)) != 0) {
        fprintf(stderr, "%s: QUANTIZE MISMATCH\n", path);
        return 1;
    }

    bitnet_arc_dequantize_row_tq2_0(ours,  deq_ours,  n);
    upstream_dequantize_row_tq2_0  (upstr, deq_upstr, (int64_t)n);

    if (diff_bytes("dequantize",
                   (const uint8_t *)deq_ours,
                   (const uint8_t *)deq_upstr,
                   n * sizeof(float)) != 0) {
        fprintf(stderr, "%s: DEQUANTIZE MISMATCH\n", path);
        return 1;
    }

    free(src_i8); free(src_f); free(ours); free(upstr);
    free(deq_ours); free(deq_upstr); free(scales);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.bin> [more.bin ...]\n", argv[0]);
        return 2;
    }

    size_t pass = 0, fail = 0;
    for (int i = 1; i < argc; ++i) {
        if (gate_one(argv[i]) == 0) {
            ++pass;
        } else {
            ++fail;
            fprintf(stderr, "  -> %s FAILED\n", argv[i]);
        }
    }

    fprintf(stdout, "gate: %zu pass, %zu fail (%d total)\n",
            pass, fail, argc - 1);
    return fail == 0 ? 0 : 1;
}
