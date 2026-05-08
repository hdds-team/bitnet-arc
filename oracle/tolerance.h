/*
 * Cross-check tolerance constants for the bitnet-arc oracle.
 *
 * Single source of truth shared by the C/C++ harness
 * (bench/harness_3way.cpp) and the Python xcheck (numpy_xcheck.py,
 * which parses this header at import time -- see _load_tolerances()).
 *
 * If you tighten or loosen a bar here, both consumers pick it up
 * without manual sync. Do NOT duplicate these values in docstrings or
 * source comments -- always reference the macro by name.
 *
 * Rationale (per oracle/README.md tolerance model, set during
 * design v0 review #58 by @sonnet + @theta):
 *
 *   voice 1 = maison FP32 matmul (oracle/fp32_matmul.c)
 *   voice 2 = ggml-style dequant + maison FP32 matmul
 *   voice 3 = numpy float64 BLAS dgemm (oracle/numpy_xcheck.py)
 *   SYCL    = the production kernel (W1+)
 *
 * Pair-wise gates and tolerance bars below.
 */

#ifndef BITNET_ARC_ORACLE_TOLERANCE_H
#define BITNET_ARC_ORACLE_TOLERANCE_H

/*
 * voice 1 vs voice 2 -- same canonical reduction loop, only the weight
 * source differs (integer ternary vs dequantized FP32). Must be
 * bit-equal at FP32. Catches dequant / unpack bugs.
 *
 * Use this as: assert |a - b| == 0.0f when comparing voice1 and voice2.
 */
#define BITNET_ARC_TOL_FP32_VS_FP32           0.0f

/*
 * voice 1 vs voice 3 -- detects bugs in our maison FP32 reference.
 * NumPy float64 worst-case drift at K=14336 is ~3e-12 relative
 * (negligible). Our FP32 ref worst-case drift at K=14336 is ~1.7e-3
 * relative. Bar set generously above the expected drift.
 */
#define BITNET_ARC_TOL_FP32REF_VS_NUMPY_F64   1e-3f

/*
 * SYCL kernel vs voice 3 -- the headline correctness gate for the
 * production kernel. SYCL accumulates in FP16 internally (tile-local),
 * which can drift up to ~5-10% relative at K=14336. Anything beyond
 * this bar is a real bug in the kernel, not just precision loss.
 */
#define BITNET_ARC_TOL_SYCL_VS_NUMPY_F64      1e-1f

/*
 * SYCL kernel vs voice 1 -- intermediate drift detection. Both share
 * the kernel topology (same dispatch, same tiling) but differ on
 * accumulator type (FP16 in SYCL, FP32 in the maison ref). Used to
 * spot-check progressive degradation as we tune the kernel.
 */
#define BITNET_ARC_TOL_SYCL_VS_FP32REF        1e-2f

#endif /* BITNET_ARC_ORACLE_TOLERANCE_H */
