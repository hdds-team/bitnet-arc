/*
 * bitnet-arc v2 SYCL kernel: XMX-accelerated ternary x FP16 -> FP16
 * matmul on Arc B60 (Xe2 / Battlemage), via the SYCL2020
 * joint_matrix extension.
 *
 * Design contract: docs/design-v2.md (review #73, ratified e3fb4f2)
 *                  docs/design-v2-phase-1.md (review #76, ratified e76bc98)
 *
 * v2 vs v1 (compact):
 *   - XMX path : joint_matrix MMA replaces the scalar inner K-walk.
 *     v1 was COMPUTE-BOUND on Arc B60 (~250x off scalar peak per
 *     bench/profile_v0_bl.md / commit a5d4283); v2 collapses the
 *     redundant per-output-element reduction into one fragment-MMA
 *     per K=16 step.
 *   - Operand / accumulator pair : FP16 x FP32 (Phase 0 chosen pair,
 *     commit e44c587 -- 4/4 combos green on Arc B60).
 *   - Cooperative TQ2_0 -> FP16 dequant in SLM (§3.1 of the Phase 1
 *     brief), then a sequence of joint_matrix_mad fragment ops over
 *     K_CHUNK / 16 = 16 fragment-Ks per K_CHUNK iteration.
 *   - Launch geometry : nd_range<1>({16}, {16}) per output tile so
 *     1 WG = 1 SG of 16 lanes -- exactly what the joint_matrix
 *     fragment expects, no race surface (lesson from probe review #75).
 *
 * Storage layout (A_fp16, B_blocks, C_fp16) is unchanged from v0/v1
 * -- the bench harness can swap kv2 in by index without touching
 * the fixture-generation path.
 *
 * Phase 1 scope (ratified) : single registered variant
 * (TILE_M=16, TILE_N=16, SG_SIZE=16, K_CHUNK=256). Multi-tile sweep
 * + perf gate are Phase 2 work (see brief §13 out-of-scope).
 */

#ifndef BITNET_ARC_SRC_KERNEL_V2_H
#define BITNET_ARC_SRC_KERNEL_V2_H

#include <cstddef>
#include <cstdint>

#include "../oracle/tq2_0.h"
#include "kernel_v0.h"  /* for sycl_queue_handle forward decl */

#ifdef __cplusplus
extern "C++" {
#endif

namespace bitnet_arc {

/*
 * Runtime configuration for run_kernel_v2(). Phase 1 fixes all four
 * fields; the struct exists for parity with v0/v1 dispatch and for
 * Phase 2 multi-tile sweep extension.
 *
 * Unsigned types encode the non-negative invariant at the type level
 * (per @codex review #60, carries through v1).
 */
struct kernel_v2_config {
    unsigned tile_M;     /* Phase 1: fixed 16                          */
    unsigned tile_N;     /* Phase 1: fixed 16                          */
    unsigned sg_size;    /* Phase 1: fixed 16 (joint_matrix on Xe2)    */
    unsigned k_chunk;    /* Phase 1: fixed 256 (TQ2_0 block size)      */
};

/* Phase 1 default == only registered variant. */
inline kernel_v2_config kernel_v2_config_default() {
    return kernel_v2_config{
        /* tile_M  */ 16u,
        /* tile_N  */ 16u,
        /* sg_size */ 16u,
        /* k_chunk */ 256u,
    };
}

/*
 * Run the v2 XMX-accelerated matmul. Same pointer / lifetime contract
 * as v0/v1 (USM device or shared, caller manages transfers, returns
 * after kernel submission, caller owns queue.wait() before reading
 * C_fp16).
 *
 * Asserts (host-side, before kernel submission):
 *   - K % 256 == 0
 *   - cfg.k_chunk > 0 && cfg.k_chunk % 256 == 0
 *   - K % cfg.k_chunk == 0
 *   - M % cfg.tile_M == 0
 *   - N % cfg.tile_N == 0
 *
 * Phase 1: cfg must match the single registered variant
 * (16, 16, 16, 256). Other configs fall back to that variant with a
 * stderr warning (Phase 2 will sweep over alternatives).
 */
void run_kernel_v2(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v2_config& cfg = kernel_v2_config_default());

/* --- variant table -------------------------------------------------- */

/* Type-erased launcher; same shape as kv0/kv1_launch_fn. The compile-
 * time template parameters (TILE_M, TILE_N, SG_SIZE, K_CHUNK) are
 * baked into the function pointed to here -- the bench harness selects
 * by index without knowing the SYCL types. */
typedef void (*kv2_launch_fn)(sycl_queue_handle& q_handle,
                              std::size_t M,
                              std::size_t N,
                              std::size_t K,
                              const std::uint16_t* A_fp16,
                              const bitnet_arc_tq2_0_block* B_blocks,
                              std::uint16_t* C_fp16);

/* Per @sonnet review #66 nit (re-applied to v2): separate descriptor
 * struct from kernel_variant_desc / kv1_variant_desc. v2 has no
 * inner_mode (BRANCHLESS-only) and the K_CHUNK + frag triple is
 * v2-specific in the variant naming. */
struct kv2_variant_desc {
    unsigned       tile_M;
    unsigned       tile_N;
    unsigned       sg_size;
    unsigned       k_chunk;
    const char*    name;     /* "v2_<TM>x<TN>_sg<SG>_k<KCHUNK>" */
    kv2_launch_fn  launch;
};

/* Registered variants. Phase 1 ships exactly one entry:
 *
 *   v2_16x16_sg16_k256
 *
 * Phase 2 will sweep tile/sg/K_CHUNK once Phase 1 W1 correctness is
 * locked. The sweep harness skips variants where the input shape is
 * incompatible (M % tile_M != 0, N % tile_N != 0, K % k_chunk != 0)
 * the same way it does for kv0/kv1.
 */
extern const kv2_variant_desc kv2_variants[];
extern const std::size_t      kv2_variants_count;

} /* namespace bitnet_arc */

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_SRC_KERNEL_V2_H */
