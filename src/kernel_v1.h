/*
 * bitnet-arc v1 SYCL kernel: SLM-tiled ternary x FP16 -> FP16 matmul.
 *
 * Design contract: docs/design-v1.md (review #66, approved 1/1).
 *
 * v1 vs v0 (compact):
 *   - SLM tiling: per-WG cooperative load of A and B slabs, inner
 *     K-walk reads from SLM only. Eliminates the 16x global-load
 *     redundancy v0 measured on Arc B60.
 *   - Coalesced loads: cooperative-load helpers stride by linear
 *     local_id across the WG, line-aligned where possible.
 *   - Inner mode: BRANCHLESS only (W1.5 finding: BL wins universally
 *     on GPU; review #65 baseline switched accordingly).
 *   - K_CHUNK: compile-time block-aligned multiple of 256 (TQ2_0).
 *
 * Storage layout, dispatch contract, and host-side preconditions are
 * inherited from v0 unchanged; see kernel_v0.h. The format on disk
 * (TQ2_0) is unchanged, the variant table here registers compile-time
 * (TILE_M, TILE_N, SG_SIZE, K_CHUNK) instantiations.
 *
 * Per @sonnet review #66 nit: kv1_variant_desc is a SEPARATE struct
 * from kernel_variant_desc (not an extension) -- the K_CHUNK field is
 * a v1-specific compile-time parameter and including it in the v0
 * runtime descriptor would pollute the v0 API.
 */

#ifndef BITNET_ARC_SRC_KERNEL_V1_H
#define BITNET_ARC_SRC_KERNEL_V1_H

#include <cstddef>
#include <cstdint>

#include "../oracle/tq2_0.h"
#include "kernel_v0.h"  /* for sycl_queue_handle forward decl */

#ifdef __cplusplus
extern "C++" {
#endif

namespace bitnet_arc {

/*
 * Runtime configuration handed to run_kernel_v1(). The k_chunk field
 * is the only addition vs kernel_v0_config; inner_mode is gone (v1
 * is BRANCHLESS-only per design v1 section 2.4).
 *
 * The unsigned types encode the non-negative invariant at the type
 * level (per @codex's v0 review #60 -- carries through to v1).
 */
struct kernel_v1_config {
    unsigned tile_M;
    unsigned tile_N;
    unsigned sg_size;
    unsigned k_chunk;   /* must be a positive multiple of 256 */
};

/* Sensible default for v1: 32x32 tile, sg=16, K_CHUNK=256. The 32x32
 * tile maximizes SLM-amortization of weight loads (32:1 reuse instead
 * of 16:1) within the WG <= 1024 budget. K_CHUNK=256 is the natural
 * TQ2_0 block alignment; 512/1024 are also registered variants. */
inline kernel_v1_config kernel_v1_config_default() {
    return kernel_v1_config{
        /* tile_M  */ 32u,
        /* tile_N  */ 32u,
        /* sg_size */ 16u,
        /* k_chunk */ 256u,
    };
}

/*
 * Run the v1 SLM-tiled matmul. Same pointer / lifetime contract as v0
 * (USM device or shared, caller manages transfers, returns after
 * kernel submission, caller owns queue.wait() before reading C_fp16).
 *
 * Asserts (host-side, before kernel submission):
 *   - K % 256 == 0
 *   - cfg.k_chunk > 0 && cfg.k_chunk % 256 == 0
 *   - K % cfg.k_chunk == 0
 *   - M % cfg.tile_M == 0
 *   - N % cfg.tile_N == 0
 *   - cfg.tile_M * cfg.tile_N <= 1024 (Xe2 max WG size)
 *
 * Falls back to the default 32x32 / sg16 / K_CHUNK=256 if cfg does not
 * match any registered variant.
 */
void run_kernel_v1(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v1_config& cfg = kernel_v1_config_default());

/* --- variant table -------------------------------------------------- */

/* Type-erased launcher; same shape as kv0_launch_fn. The compile-time
 * template parameters (TILE_M, TILE_N, SG_SIZE, K_CHUNK) are baked into
 * the function pointed to here -- the bench harness selects by index
 * without knowing the SYCL types. */
typedef void (*kv1_launch_fn)(sycl_queue_handle& q_handle,
                              std::size_t M,
                              std::size_t N,
                              std::size_t K,
                              const std::uint16_t* A_fp16,
                              const bitnet_arc_tq2_0_block* B_blocks,
                              std::uint16_t* C_fp16);

/* Per @sonnet review #66: separate descriptor struct (not extending
 * kernel_variant_desc). The k_chunk field plus the v1-only launcher
 * type live here; sweep_tile iterates kv0_variants[] and kv1_variants[]
 * as two independent lists. */
struct kv1_variant_desc {
    unsigned       tile_M;
    unsigned       tile_N;
    unsigned       sg_size;
    unsigned       k_chunk;
    const char*    name;     /* "v1_<TM>x<TN>_sg<SG>_k<KCHUNK>" */
    kv1_launch_fn  launch;
};

/* Registered variants. Phase 1 + 2a (12 entries):
 *
 *   K_CHUNK=256 (phase 1, commit b3179af):
 *     v1_16x16_sg16_k256
 *     v1_32x16_sg16_k256
 *     v1_16x32_sg16_k256
 *     v1_32x32_sg16_k256
 *     v1_32x32_sg32_k256
 *
 *   K_CHUNK=512 (phase 2a, task #153):
 *     v1_<TM>x<TN>_sg<SG>_k512   (5 entries, same tile/sg combos)
 *
 *   K_CHUNK=1024 (phase 2a, task #153, SLM-constrained):
 *     v1_16x16_sg16_k1024
 *     v1_16x32_sg16_k1024
 *
 * Phase 2a tests hypothesis H2 (barrier overhead): K_CHUNK=1024 has
 * 4x fewer barrier-pairs than K_CHUNK=256 at K=14336. If the gain is
 * insignificant, H2 is not the dominant factor and the bottleneck
 * lies in the cooperative-load access pattern (tested separately by
 * phase 2b / task #154 -- two-pass coalesced load).
 *
 * SLM budget note (per @theta review #68): TILE_M=32 with K_CHUNK=1024
 * makes A_slab alone fill the ~64 KB Xe2 per-WG SLM hard limit, which
 * would tank occupancy to 1 WG/Xe-core. Those 3 variants are therefore
 * not registered at K_CHUNK=1024 (32x16, 32x32_sg16, 32x32_sg32).
 * Revisitable at v1.5 if Xe3+ exposes >=128 KB SLM/WG or via FP8 A_slab
 * staging (design v2 territory).
 *
 * Variants with K not a multiple of K_CHUNK are skipped at sweep
 * runtime (e.g. K_CHUNK=512 and 1024 skip on shape (16,16,256)). */
extern const kv1_variant_desc kv1_variants[];
extern const std::size_t      kv1_variants_count;

} /* namespace bitnet_arc */

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_SRC_KERNEL_V1_H */
