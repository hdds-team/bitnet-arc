/*
 * bitnet-arc v0 SYCL kernel: ternary x FP16 -> FP16 matmul on Arc B60.
 *
 * Design v0 anchors (see /projects/bitnet-arc/docs/design-v0.md):
 *   - Storage:  TQ2_0 (oracle/tq2_0.h)
 *   - GPU path: ALU vectorial, native ternary add/sub/skip (no multiplies)
 *   - Scope:    standalone, synthetic weights, no llama.cpp plumbing
 *
 * v0 MVP target: compile + correct against the 3-way oracle harness.
 * Tile sweep instrumentation lives in #148 via the variant table below.
 *
 * Data layout (v0):
 *   A_fp16  : row-major (M, K), uint16_t bit-pattern of IEEE-754 binary16
 *   B_blocks: array of (N * K / 256) TQ2_0 blocks. Block at index
 *             (n * blocks_per_col + k_chunk) carries the 256 ternary
 *             weights at B[k_chunk*256 .. k_chunk*256 + 256, n].
 *   C_fp16  : row-major (M, N), uint16_t bit-pattern.
 *
 * K is required to be a multiple of 256 (TQ2_0 block size). M and N
 * are required to be multiples of the work-group tile dims.
 */

#ifndef BITNET_ARC_SRC_KERNEL_V0_H
#define BITNET_ARC_SRC_KERNEL_V0_H

#include <cstddef>
#include <cstdint>

#include "../oracle/tq2_0.h"

#ifdef __cplusplus
extern "C++" {
#endif

namespace bitnet_arc {

/* Inner-loop arithmetic mode for the ternary contraction.
 *
 *   BRANCHFUL: explicit if/else over codes {0,1,2} -> sub/skip/add.
 *              Matches the design v0 narrative (native ternary semantics).
 *   BRANCHLESS: (code - 1) cast to float, fused-friendly. Tile sweep
 *               #148 compares both paths.
 */
enum class kernel_v0_inner_mode {
    BRANCHFUL  = 0,
    BRANCHLESS = 1,
};

struct kernel_v0_config {
    /* Work-group tile dimensions. Default 16x16 per design v0 #4.
     *
     * Unsigned to make negative values impossible at the type level:
     * the kernel does `M % tile_M` after casting to size_t, which
     * would be UB for `tile_M == 0` and produce a huge effective
     * divisor for negative input. Asserts protect against zero only
     * in debug builds, so we encode the non-negative invariant in the
     * type instead (per @codex review #60). */
    unsigned tile_M;
    unsigned tile_N;
    /* Subgroup size. SYCL requires this be compile-time, so the
     * runtime field here is only used by run_kernel_v0() to *dispatch*
     * to one of the explicit variants below. Unsupported combinations
     * fall back to the 16x16/sg16/branchful baseline. */
    unsigned sg_size;
    /* See kernel_v0_inner_mode above. */
    kernel_v0_inner_mode inner_mode;
};

/* Sensible defaults for the v0 baseline (16x16 tile, sg=16, branchful
 * matching the design v0 "ternary add/sub/skip" narrative). */
inline kernel_v0_config kernel_v0_config_default() {
    return kernel_v0_config{
        /* tile_M     */ 16u,
        /* tile_N     */ 16u,
        /* sg_size    */ 16u,
        /* inner_mode */ kernel_v0_inner_mode::BRANCHFUL,
    };
}

/* Opaque handle wrapping sycl::queue, declared here so the public
 * header does not pull <sycl/sycl.hpp> into every translation unit
 * that just wants to call the kernel. Implementation in kernel_v0.cpp. */
class sycl_queue_handle;

sycl_queue_handle* make_default_queue_handle();
void               destroy_queue_handle(sycl_queue_handle* h);

/*
 * Run the v0 ternary matmul on a SYCL queue.
 *
 * Pointers are device-accessible (USM device or shared). Caller manages
 * allocation and host<->device transfers.
 *
 * Asserts (host-side, before kernel submission):
 *   - K % 256 == 0
 *   - M % cfg.tile_M == 0
 *   - N % cfg.tile_N == 0
 *
 * Returns after the kernel is submitted (does NOT wait). Caller is
 * responsible for queue.wait() before reading C_fp16.
 *
 * cfg selects one of the precompiled variants exposed in kv0_variants[]
 * below; if cfg does not match a registered variant, falls back to the
 * baseline (16x16 / sg16 / branchful).
 */
void run_kernel_v0(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v0_config& cfg = kernel_v0_config_default());

/* --- variant table for #148 tile sweep ------------------------------ */

/* Type-erased launcher. The compile-time template parameters
 * (TILE_M, TILE_N, SG_SIZE, MODE) are baked into the function pointed
 * to here; the bench harness picks one of these by index without
 * needing to know the SYCL types. */
typedef void (*kv0_launch_fn)(sycl_queue_handle& q_handle,
                              std::size_t M,
                              std::size_t N,
                              std::size_t K,
                              const std::uint16_t* A_fp16,
                              const bitnet_arc_tq2_0_block* B_blocks,
                              std::uint16_t* C_fp16);

struct kernel_variant_desc {
    unsigned             tile_M;
    unsigned             tile_N;
    unsigned             sg_size;
    kernel_v0_inner_mode inner_mode;
    const char*          name;
    kv0_launch_fn        launch;
};

/* Registered variants (smoke set, expanded in follow-up sweeps). */
extern const kernel_variant_desc kv0_variants[];
extern const std::size_t         kv0_variants_count;

} /* namespace bitnet_arc */

#ifdef __cplusplus
}
#endif

#endif /* BITNET_ARC_SRC_KERNEL_V0_H */
