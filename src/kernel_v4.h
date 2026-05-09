/*
 * bitnet-arc v4 ESIMD kernel: Path 9 INT2 DPAS native silicon path.
 *
 * Design contract: docs/design-v4.md (ratified 2026-05-09 by @naskel
 * after 2-voice spec-vs-code review).
 *
 * Path: SYCL ESIMD via sycl::ext::intel::esimd::xmx::dpas with
 *       dpas_argument_type::s2 (signed 2-bit). Empirically validated
 *       at 3.76x FP16 wrapper baseline (commit 5777078,
 *       bench/probe_esimd_int2.cpp = 351.6 GOps/s on Arc B60).
 *
 * v4 vs v0..v3 (compact):
 *   - ESIMD programming model: single SIMD16 ESIMD thread per output
 *     tile. NO sub-group cooperation (different from kv0..v3).
 *   - INT2 ternary path: ternary {-1, 0, +1} fits in INT2 signed
 *     {00=0, 01=+1, 11=-1, 10=-2 unused}. Zero storage waste.
 *   - Activation INT2 quant (more aggressive than BitNet W1.58A8
 *     reference). Compile-time fallback to INT4 quant via
 *     KV4_USE_ACT_INT2 = false if W1 precision blows.
 *   - DPAS shape: M=8 (RepeatCount), N=16 (ExecutionSize Xe2),
 *     K=64 (SystolicDepth=8 * OpsPerChannel=8 for INT2).
 *     ops/MMA = 8*16*64*2 = 16384.
 *   - K_CHUNK = 256 = 4 fragments per chunk (one TQ2_0 block).
 *   - Per-chunk d fold register-only via mC_fp32 simd<float, 128>
 *     accumulator across chunks (no SLM round-trip, no barriers).
 *
 * Input contracts (Risk #7 per design v4 sec5):
 *   - K % K_CHUNK (= 256) == 0  (TQ2_0 block-aligned)
 *   - M % TILE_M (= 8) == 0     (DPAS RepeatCount)
 *   - N % TILE_N (= 16) == 0    (DPAS ExecutionSize Xe2)
 *
 * Header path note: ESIMD lives at sycl/ext/intel/esimd.hpp. xmx::dpas
 * + dpas_argument_type live under sycl/ext/intel/esimd/xmx/.
 */

#ifndef BITNET_ARC_SRC_KERNEL_V4_H
#define BITNET_ARC_SRC_KERNEL_V4_H

#include <cstddef>
#include <cstdint>

#include "../oracle/tq2_0.h"
#include "kernel_v0.h"  /* sycl_queue_handle */

#ifdef __cplusplus
extern "C++" {
#endif

namespace bitnet_arc {

/*
 * Compile-time activation precision flag.
 * Default true = ship aggressive W1.58A2 (INT2 activation quant).
 * If W1 max_rel_err > 1e-2 fails on real ternary data, override to
 * false to use INT4 activation quant (less aggressive, ~2.6x FP16
 * compute path instead of 3.76x). Both variants ship as separate
 * registered variants in kv4_variants[].
 */
#ifndef KV4_USE_ACT_INT2
#define KV4_USE_ACT_INT2 1
#endif

/*
 * v4 fixed parameters (Phase 1 ships exactly one configuration per
 * activation precision flag).
 */
constexpr unsigned KV4_TILE_M    = 8u;    /* DPAS RepeatCount */
constexpr unsigned KV4_TILE_N    = 16u;   /* DPAS ExecutionSize Xe2 */
constexpr unsigned KV4_TILE_K    = 64u;   /* DPAS K dim per fragment */
constexpr unsigned KV4_K_CHUNK   = 256u;  /* TQ2_0 block size */
constexpr unsigned KV4_FRAGS_PER_CHUNK = KV4_K_CHUNK / KV4_TILE_K; /* 4 */

/* Compile-time invariants. */
static_assert(KV4_K_CHUNK == 256u,
              "kv4: K_CHUNK locked at TQ2_0 block size 256");
static_assert(KV4_TILE_K == 64u,
              "kv4: TILE_K locked at SystolicDepth*OpsPerChannel=64 for INT2");
static_assert(KV4_TILE_M == 8u,
              "kv4: TILE_M locked at DPAS RepeatCount=8");
static_assert(KV4_TILE_N == 16u,
              "kv4: TILE_N locked at Xe2 ExecutionSize=16");
static_assert(KV4_K_CHUNK % KV4_TILE_K == 0u,
              "kv4: K_CHUNK must be multiple of TILE_K");

/*
 * Run kv4 kernel. Same external interface as kv0/v1/v2/v3:
 *   A_fp16   : USM device, M x K FP16 activations.
 *   B_blocks : USM device, (K/256) x N TQ2_0 blocks.
 *   C_fp16   : USM device, M x N FP16 output.
 *
 * Internally activations are quantized to INT2 (or INT4 fallback)
 * inside the kernel; weights are dequantized on-the-fly from TQ2_0
 * to INT2 via the VNNI layout pack helpers (kernel_v4_packing.h).
 *
 * Runtime asserts (Risk #7 per design v4):
 *   K % 256 == 0
 *   M % 8 == 0
 *   N % 16 == 0
 */
void run_kernel_v4(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16);

/* --- variant table -------------------------------------------------- */

/* Same shape as kv2/v3 launch_fn. Phase 1 ships 1 or 2 entries
 * (INT2-act primary, INT4-act fallback if KV4_USE_ACT_INT2 is set). */
typedef void (*kv4_launch_fn)(sycl_queue_handle& q_handle,
                              std::size_t M,
                              std::size_t N,
                              std::size_t K,
                              const std::uint16_t* A_fp16,
                              const bitnet_arc_tq2_0_block* B_blocks,
                              std::uint16_t* C_fp16);

struct kv4_variant_desc {
    unsigned       tile_M;
    unsigned       tile_N;
    unsigned       tile_K;
    unsigned       sg_size;    /* 16 for ESIMD Xe2 (kept for sweep_tile printf compat) */
    unsigned       k_chunk;
    bool           act_int2;   /* true = W1.58A2, false = W1.58A4 */
    const char*    name;       /* "v4_8x16x64_aint2_k256" etc. */
    kv4_launch_fn  launch;
};

extern const kv4_variant_desc kv4_variants[];
extern const std::size_t      kv4_variants_count;

} /* namespace bitnet_arc */

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_SRC_KERNEL_V4_H */
