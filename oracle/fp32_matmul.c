/*
 * Naive FP32 reference matmul implementation.
 * See fp32_matmul.h for contract.
 */

#include "fp32_matmul.h"
#include "fp16.h"

void bitnet_arc_oracle_fp32_matmul(size_t M,
                                   size_t N,
                                   size_t K,
                                   const uint16_t *A_fp16,
                                   const float    *B_dequant,
                                   float          *C)
{
    /* Canonical loop order: m, n, k. Reduction direction = k increasing.
     * Accumulator type is explicit FP32. No fused ops. */
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                const float a = bitnet_arc_fp16_to_fp32(A_fp16[m * K + k]);
                const float w = B_dequant[k * N + n];
                /* Two-step to keep mul and add unfused at the source level.
                 * Compilers may still contract this; if a future toolchain
                 * does FMA contraction here, add -ffp-contract=off in the
                 * oracle build flags. */
                const float prod = a * w;
                acc = acc + prod;
            }
            C[m * N + n] = acc;
        }
    }
}
