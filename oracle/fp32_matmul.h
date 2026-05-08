/*
 * Naive FP32 reference matmul for ternary x FP16 -> FP32.
 *
 *     C[M, N] = A[M, K] (FP16) * B[K, N] (FP32, dequantized from TQ2_0)
 *
 *   - Reduction order: canonical m-n-k, increasing k.
 *   - Accumulator: FP32, declared explicitly.
 *   - No fused-multiply-add, no SIMD intrinsics, no reordering.
 *
 * For ternary B (entries are scale_fp16 * {-1, 0, +1}), the inner-loop
 * arithmetic is well-defined and bounded. Cross-check tolerances are
 * defined in oracle/tolerance.h:
 *
 *   - voice 1 (this) vs voice 2 (ggml-style FP32): TOL_FP32_VS_FP32 = 0
 *     (bit-equal, same canonical loop, only weight source differs)
 *   - voice 1 (this) vs SYCL kernel: TOL_SYCL_VS_FP32REF ~ 1e-2 rel
 *     (different accumulator type: FP32 here, FP16 in SYCL)
 *   - voice 1 (this) vs voice 3 (numpy float64 BLAS):
 *     TOL_FP32REF_VS_NUMPY_F64 ~ 1e-3 rel (different reduction order)
 *
 * NB: must be built with -ffp-contract=off (see oracle/Makefile) to
 * keep the (a * w) + acc pair from being fused into a single rounded
 * FMA at compile time, which would break voice 1 vs voice 2
 * bit-equality.
 *
 * Complexity: O(M * N * K). Use only as oracle, not for performance.
 */

#ifndef BITNET_ARC_ORACLE_FP32_MATMUL_H
#define BITNET_ARC_ORACLE_FP32_MATMUL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void bitnet_arc_oracle_fp32_matmul(size_t M,
                                   size_t N,
                                   size_t K,
                                   const uint16_t *A_fp16,    /* row-major M x K */
                                   const float    *B_dequant, /* row-major K x N, FP32 from TQ2_0 */
                                   float          *C);        /* row-major M x N, output */

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_ORACLE_FP32_MATMUL_H */
