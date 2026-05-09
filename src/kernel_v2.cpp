/*
 * bitnet-arc v2 SYCL kernel implementation -- XMX-accelerated ternary
 * x FP16 -> FP16 matmul on Arc B60 (Xe2 / Battlemage).
 *
 * See kernel_v2.h for the public contract and docs/design-v2-phase-1.md
 * for the implementation brief (review #76, ratified e76bc98).
 *
 * Diff vs v1 (kernel_v1.cpp) -- the relevant ones:
 *   - joint_matrix MMA replaces the scalar inner K-walk. v1's
 *     verdict was COMPUTE-BOUND (~250x off scalar peak per
 *     bench/profile_v0_bl.md / a5d4283); v2 uses the XMX engine.
 *   - Cooperative TQ2_0 -> FP16 dequant in SLM (per Phase 1 brief
 *     §3.1: byte-stripe load + sub-group barrier + code decode).
 *   - Inner MMA loop: 16 fragment-Ks per K_CHUNK (per §3.3), single
 *     accumulator fragment mC stays in registers across all
 *     K / K_CHUNK outer iterations.
 *   - Launch geometry: nd_range<1>({16}, {16}) per output tile = 1 WG
 *     = 1 SG of 16 lanes (per §4, lesson from probe review #75).
 *
 * Phase 1 scope: single registered variant (TILE_M=16, TILE_N=16,
 * SG_SIZE=16, K_CHUNK=256, fragment 16x16x16). Multi-tile sweep and
 * perf gate are Phase 2 scope (brief §13).
 *
 * Header path note (per Phase 0 record e44c587): joint_matrix lives
 * at sycl/ext/oneapi/matrix/matrix.hpp (no `experimental/` in path
 * on icpx 2025.3, even though symbols stay in the experimental
 * namespace).
 */

#include "kernel_v0_sycl.hpp"
#include "kernel_v2.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bitnet_arc {

namespace mx = sycl::ext::oneapi::experimental::matrix;

/* -- shared device helpers ------------------------------------------- *
 *
 * Same TQ2_0 unpack + FP16 conversion shape as kernel_v1.cpp; we
 * re-declare with kv2_ prefix to keep linkage clean and avoid pulling
 * v1's static helpers into the v2 translation unit.
 */

/* (kv2_unpack_code helper dropped per @beta review #77 nit -- the
 * kernel inlines the bit math directly on qs_local since the load
 * pattern differs from v1's per-call pattern.) */

static inline float kv2_fp16_to_fp32(std::uint16_t h) {
    std::uint32_t sign = (std::uint32_t)((h >> 15) & 0x1u);
    std::uint32_t exp  = (std::uint32_t)((h >> 10) & 0x1Fu);
    std::uint32_t mant = (std::uint32_t)(h & 0x3FFu);
    std::uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31)
                 | ((std::uint32_t)(127 - 15 - e) << 23)
                 | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31)
             | ((exp + (127 - 15)) << 23)
             | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline std::uint16_t kv2_fp32_to_fp16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    std::uint32_t sign = (x >> 31) & 0x1u;
    std::int32_t  exp  = (std::int32_t)((x >> 23) & 0xFFu) - 127;
    std::uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) {
        return (std::uint16_t)((sign << 15) | (0x1Fu << 10)
                              | (mant ? 0x200u : 0u));
    }
    if (exp > 15)  return (std::uint16_t)((sign << 15) | (0x1Fu << 10));
    if (exp < -14) {
        if (exp < -24) return (std::uint16_t)(sign << 15);
        std::uint32_t sub = (mant | 0x800000u) >> (-exp - 14 + 13);
        return (std::uint16_t)((sign << 15) | sub);
    }
    return (std::uint16_t)((sign << 15)
                          | ((std::uint32_t)(exp + 15) << 10)
                          | (mant >> 13));
}

/* -- templated kernel ------------------------------------------------- *
 *
 * Phase 1 ships a single instantiation: <16, 16, 16, 256>. The
 * template is kept generic for Phase 2 sweep extension; the
 * compile-time invariants below pin Phase 1's contract.
 *
 * SLM layout (per Phase 1 brief §5, post Phase 2b Lecture B switch):
 *   A_slab    : TILE_M  × K_CHUNK FP16  = 16 × 256 × 2 = 8 KB
 *   B_slab    : TILE_N  × K_CHUNK FP16  = 16 × 256 × 2 = 8 KB
 *   ----                                  ~16 KB total
 * Well below Xe2's 64 KB / WG hard limit (port v1's static_assert).
 * qs_local SLM scratch dropped in Lecture B (each lane reads its own
 * block's qs[] directly from global into registers).
 *
 * NOTE on §3.1 lane mapping interpretation: Phase 2b switches to
 * "Lecture B" (lane = column, all parallel). Each lane owns one
 * column n_local = lane and decodes all K_CHUNK K-positions for that
 * column from its own block (qs[] read directly into registers, no
 * SLM staging). Eliminates Lecture A's per-column loop (16 iters)
 * and 32 sub-group barriers per K_CHUNK. The outer kernel structure
 * (chunks_per_col loop, A SLM load, MMA, store) is unchanged.
 *
 * Phase 2a profiler bench/profile_v2_p2a.md identified §3.1 dequant
 * as top-1 bottleneck (~55% of t_full); Lecture B is the §4.2 fix.
 * Phase 2b §3.2 loadA vec-load is the §4.4 fix on top-2 (~37%).
 */

template <unsigned TILE_M,
          unsigned TILE_N,
          unsigned SG_SIZE,
          unsigned K_CHUNK>
static void kv2_launch_impl(sycl_queue_handle& q_handle,
                            std::size_t M,
                            std::size_t N,
                            std::size_t K,
                            const std::uint16_t* A_fp16,
                            const bitnet_arc_tq2_0_block* B_blocks,
                            std::uint16_t* C_fp16)
{
    /* Compile-time invariants (Phase 1 lock). */
    static_assert(TILE_M == 16u && TILE_N == 16u,
                  "kv2 Phase 1: TILE_M = TILE_N = 16 (joint_matrix frag)");
    static_assert(SG_SIZE == 16u,
                  "kv2 Phase 1: sg_size = 16 (joint_matrix on Xe2)");
    static_assert(K_CHUNK > 0u && K_CHUNK % 256u == 0u,
                  "K_CHUNK must be a positive multiple of TQ2_0 block (256)");

    /* SLM budget guard, ported from kernel_v1.cpp 81042d4. The Xe2 64
     * KB / WG hard limit must hold across all Phase 2 instantiations;
     * compile-time fail with a clear message protects the v2 sweep. */
    constexpr std::size_t KV2_A_SLAB_BYTES =
        static_cast<std::size_t>(TILE_M) * K_CHUNK * sizeof(std::uint16_t);
    constexpr std::size_t KV2_B_SLAB_BYTES =
        static_cast<std::size_t>(TILE_N) * K_CHUNK * sizeof(std::uint16_t);
    constexpr std::size_t KV2_SLM_BUDGET_BYTES = 64u * 1024u;
    static_assert(KV2_A_SLAB_BYTES + KV2_B_SLAB_BYTES
                      <= KV2_SLM_BUDGET_BYTES,
                  "kv2: A_slab + B_slab exceeds Xe2 64 KB SLM/WG hard "
                  "limit -- pick smaller TILE_M, TILE_N, or K_CHUNK");

    /* Host-side preconditions. */
    assert(M > 0 && "kv2: M must be > 0");
    assert(N > 0 && "kv2: N must be > 0");
    assert(K > 0 && "kv2: K must be > 0");
    assert(K % 256 == 0
           && "kv2: K must be a multiple of TQ2_0 block size (256)");
    assert(K % static_cast<std::size_t>(K_CHUNK) == 0
           && "kv2: K must be a multiple of K_CHUNK");
    assert(M % static_cast<std::size_t>(TILE_M) == 0
           && "kv2: M must be a multiple of TILE_M");
    assert(N % static_cast<std::size_t>(TILE_N) == 0
           && "kv2: N must be a multiple of TILE_N");

    constexpr unsigned BLOCKS_PER_CHUNK = K_CHUNK / 256u;
    constexpr unsigned FRAG_K           = 16u;
    constexpr unsigned FRAGS_PER_CHUNK  = K_CHUNK / FRAG_K;
    constexpr std::size_t A_SLAB_ELEMS  =
        static_cast<std::size_t>(TILE_M) * K_CHUNK;
    constexpr std::size_t B_SLAB_ELEMS  =
        static_cast<std::size_t>(TILE_N) * K_CHUNK;

    const std::size_t blocks_per_col = K / 256;
    const std::size_t chunks_per_col = K / K_CHUNK;
    const std::size_t tiles_M        = M / TILE_M;
    const std::size_t tiles_N        = N / TILE_N;
    const std::size_t total_tiles    = tiles_M * tiles_N;

    sycl::queue& q = q_handle.q;

    q.submit([&](sycl::handler& h) {
        /* Per-WG SLM. local_accessor in SYCL2020. */
        sycl::local_accessor<std::uint16_t, 1> A_slab(
            sycl::range<1>(A_SLAB_ELEMS), h);
        sycl::local_accessor<std::uint16_t, 1> B_slab(
            sycl::range<1>(B_SLAB_ELEMS), h);
        /* qs_local SLM scratch dropped in Phase 2b Lecture B switch:
         * each lane reads its own block's qs[] from global into
         * registers (compiler caches the 64-byte qs[] per block in
         * the lane's register file across the inner k loop). */

        /* Launch geometry: 1 WG per output tile, 1 WG = 1 SG of
         * SG_SIZE lanes (per Phase 1 brief §4). */
        const sycl::range<1> global_range(total_tiles * SG_SIZE);
        const sycl::range<1> local_range(SG_SIZE);

        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const unsigned lane = static_cast<unsigned>(
                    sg.get_local_linear_id());

                const std::size_t tile_id = it.get_group(0);
                const std::size_t tile_m  = tile_id / tiles_N;
                const std::size_t tile_n  = tile_id % tiles_N;
                const std::size_t m_group = tile_m * TILE_M;
                const std::size_t n_group = tile_n * TILE_N;

                /* Accumulator fragment lives in registers across all
                 * K_CHUNK outer iterations (per §3.3 brief). */
                mx::joint_matrix<sycl::sub_group, float,
                                 mx::use::accumulator,
                                 TILE_M, TILE_N> mC;
                mx::joint_matrix_fill(sg, mC, 0.0f);

                for (std::size_t c = 0; c < chunks_per_col; ++c) {
                    const std::size_t k0       = c * K_CHUNK;
                    const std::size_t k_chunk0 = c * BLOCKS_PER_CHUNK;

                    /* --- §3.1 cooperative TQ2_0 -> FP16 dequant ---
                     *
                     * Lecture B (Phase 2b): each lane owns one column
                     * n_local = lane and decodes all K_CHUNK positions
                     * of its column from its own block's qs[] read
                     * directly into registers (no SLM staging).
                     *
                     * MEMORY-PATTERN CAVEAT (per Phase 2b review #X
                     * sonnet-SYCL-semantics nit): the per-lane qs read
                     * pattern strides through B_blocks at
                     * `blocks_per_col * sizeof(block) = 4480 bytes`
                     * between lanes for K=14336. This is fully strided
                     * (16 cache-line misses per dequant-step) vs
                     * Lecture A's coalesced 64-byte cooperative load.
                     * The barrier savings (32 -> 0 per chunk) are
                     * traded against GDDR6 bandwidth pressure -- net
                     * 1.44x speedup observed but bounded by the
                     * memory regression. If a Phase 3 sweep ever
                     * needs more dequant speedup, consider hybrid
                     * (cooperative qs load + per-lane decode) or
                     * Path A (INT8 DPAS, no FP16 materialization).
                     *
                     * Inner outer loop over `n_local += SG_SIZE` makes
                     * this work for TILE_N > SG_SIZE (Phase 2+ sweep);
                     * for Phase 1's TILE_N = SG_SIZE = 16 it runs once.
                     * Inner outer loop over BLOCKS_PER_CHUNK handles
                     * K_CHUNK > 256 (multiple blocks per chunk per
                     * column); for K_CHUNK = 256 it runs once.
                     *
                     * Eliminates Lecture A's TILE_N inner iterations
                     * + 2 sub-group barriers per iteration = 32
                     * barriers per chunk. The SLM qs_local scratch is
                     * no longer needed; the compiler caches the
                     * 64-byte qs[] per block in the lane register
                     * file across the inner k loop.
                     */
                    for (unsigned n_local = lane;
                         n_local < TILE_N;
                         n_local += SG_SIZE)
                    {
                        for (unsigned blk_in_chunk = 0;
                             blk_in_chunk < BLOCKS_PER_CHUNK;
                             ++blk_in_chunk)
                        {
                            const std::size_t blk_idx =
                                (n_group + n_local) * blocks_per_col
                                + k_chunk0 + blk_in_chunk;
                            const bitnet_arc_tq2_0_block& blk =
                                B_blocks[blk_idx];
                            const float d_f = kv2_fp16_to_fp32(blk.d);
                            const unsigned k_base = blk_in_chunk * 256u;

                            for (unsigned k_in_blk = 0;
                                 k_in_blk < 256u;
                                 ++k_in_blk)
                            {
                                const std::size_t byte =
                                    static_cast<std::size_t>(
                                        (k_in_blk >> 7) * 32u
                                        + (k_in_blk & 31u));
                                const unsigned shift =
                                    static_cast<unsigned>(
                                        ((k_in_blk >> 5) & 3u) * 2u);
                                const std::uint8_t code =
                                    (blk.qs[byte] >> shift) & 0x3u;
                                const int s = static_cast<int>(code) - 1;
                                const float w =
                                    static_cast<float>(s) * d_f;
                                B_slab[static_cast<std::size_t>(
                                           k_base + k_in_blk)
                                       * TILE_N + n_local] =
                                    kv2_fp32_to_fp16(w);
                            }
                        }
                    }

                    /* --- §3.2 cooperative A SLM load (Phase 2b) --- *
                     *
                     * Vectorized lid-strided copy: each lane issues
                     * sycl::vec<half, VEC_W> wide loads from global
                     * to SLM, reducing LSU pipeline ops by VEC_W vs
                     * the scalar Phase 1 loop. Same coalescing
                     * pattern (16 lanes × VEC_W = 128 half = 256
                     * bytes per SG-wide step), within Arc B60 GDDR6
                     * cache-line size.
                     *
                     * Alignment guarantees: A_fp16 is sycl::malloc_
                     * device-aligned (>= 16-byte); offset
                     * (m_group+m_idx)*K + k0 + k_off is a multiple
                     * of VEC_W as long as K, K_CHUNK, and lane*VEC_W
                     * are all VEC_W-aligned. K is constrained by the
                     * outer assert (K % 256 == 0 and VEC_W=8 divides
                     * 256).
                     *
                     * SLM ALIGNMENT CAVEAT (per Phase 2b review #X
                     * sonnet-SYCL-semantics nit): SYCL2020 spec only
                     * guarantees `local_accessor<uint16_t>` is
                     * aligned to alignof(uint16_t) = 2 bytes.
                     * vec<half,8> requires 16-byte alignment.
                     * In practice icpx 2025.3 places SLM accessors
                     * on >=128-byte boundaries on Xe2 (verified by
                     * observed perf gain), but this is implementation
                     * behavior, not spec guarantee. If icpx changes
                     * the alignment policy, vec writes here may fall
                     * back to scalar silently. Path A pivot would
                     * eliminate this fragility.
                     *
                     * STRICT-ALIASING CAVEAT (same review): the
                     * reinterpret_cast<vec_half_t*> on uint16_t
                     * storage is type-punning and would be UB in
                     * standard C++. icpx kernel codegen is built
                     * with -fno-strict-aliasing equivalent for SPIR-V
                     * lowering, so this is safe under icpx. Other
                     * SYCL toolchains would need
                     * sycl::vec<half,8>::load() canonical pattern.
                     */
                    constexpr unsigned VEC_W = 8u;
                    static_assert(K_CHUNK % VEC_W == 0u,
                                  "Phase 2b vec loadA: K_CHUNK must be "
                                  "multiple of VEC_W=8");
                    static_assert((TILE_M * K_CHUNK) % (VEC_W * SG_SIZE) == 0u,
                                  "Phase 2b vec loadA: A_SLAB_ELEMS must "
                                  "be multiple of VEC_W * SG_SIZE");
                    using vec_half_t = sycl::vec<sycl::half, VEC_W>;
                    constexpr std::size_t A_SLAB_VEC_ELEMS =
                        A_SLAB_ELEMS / VEC_W;
                    for (std::size_t vidx = lane;
                         vidx < A_SLAB_VEC_ELEMS;
                         vidx += SG_SIZE)
                    {
                        const std::size_t idx = vidx * VEC_W;
                        const std::size_t m_idx = idx / K_CHUNK;
                        const std::size_t k_off = idx % K_CHUNK;
                        const auto* src =
                            reinterpret_cast<const vec_half_t*>(
                                &A_fp16[(m_group + m_idx) * K
                                        + k0 + k_off]);
                        auto* dst =
                            reinterpret_cast<vec_half_t*>(&A_slab[idx]);
                        *dst = *src;
                    }

                    /* sub-group barrier: A_slab + B_slab must be
                     * settled before the inner MMA loop reads from
                     * them. */
                    sycl::group_barrier(sg);

                    /* --- §3.3 inner MMA loop --- *
                     *
                     * 16 fragment-K steps per K_CHUNK. Each step
                     * loads a 16x16 tile of A and B from SLM and
                     * accumulates into mC. mC stays in registers
                     * across all FRAGS_PER_CHUNK steps and across all
                     * K_CHUNK outer iterations.
                     */
                    mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::a,
                                     TILE_M, FRAG_K,
                                     mx::layout::row_major> mA;
                    mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::b,
                                     FRAG_K, TILE_N,
                                     mx::layout::row_major> mB;

                    for (unsigned k_frag = 0;
                         k_frag < FRAGS_PER_CHUNK;
                         ++k_frag)
                    {
                        /* A_slab is row-major (TILE_M rows × K_CHUNK
                         * cols, element (m, k) at m*K_CHUNK+k).
                         * Fragment at (M=0..TILE_M, K=k_frag*FRAG_K..
                         * +FRAG_K-1): pointer to first row's first
                         * fragment column, stride = K_CHUNK between
                         * rows. */
                        /* reinterpret_cast<sycl::half*> : A_slab is
                         * std::uint16_t storage (FP16 bit-pattern); the
                         * joint_matrix FP16 use::a fragment load expects
                         * sycl::half*. Caught by @claude-opus build attempt
                         * post-#78 (review #77 nit pattern, deferred to
                         * build-time sanity). */
                        auto a_ptr = sycl::address_space_cast<
                            sycl::access::address_space::local_space,
                            sycl::access::decorated::no>(
                            reinterpret_cast<sycl::half*>(
                                &A_slab[k_frag * FRAG_K]));
                        mx::joint_matrix_load(sg, mA, a_ptr, K_CHUNK);

                        /* B_slab is K-major after the dequant fix:
                         * element (k, n) at offset k*TILE_N+n.
                         * Fragment at (K=k_frag*FRAG_K.., N=0..
                         * TILE_N): pointer to first K-row of the
                         * fragment = &B_slab[k_frag*FRAG_K*TILE_N],
                         * stride = TILE_N between K-rows. *Pre-fix
                         * (offset=k_frag*FRAG_K, stride=K_CHUNK) was
                         * a stride mismatch caught by @beta review
                         * #77 — would have read stride-16 K-positions
                         * instead of 16 contiguous K-rows.* */
                        /* reinterpret_cast<sycl::half*> : same rationale
                         * as a_ptr — B_slab is std::uint16_t storage,
                         * use::b fragment expects sycl::half*. */
                        auto b_ptr = sycl::address_space_cast<
                            sycl::access::address_space::local_space,
                            sycl::access::decorated::no>(
                            reinterpret_cast<sycl::half*>(
                                &B_slab[k_frag * FRAG_K * TILE_N]));
                        mx::joint_matrix_load(sg, mB, b_ptr, TILE_N);

                        mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    }

                    /* §3.4 sub-group barrier before next K_CHUNK
                     * overwrites SLM. With 1 WG = 1 SG geometry the
                     * full WG barrier collapses to a sub-group
                     * barrier; we use the explicit sub-group form
                     * (per @beta review #76 nit). */
                    sycl::group_barrier(sg);
                }

                /* --- §3.5 final store: mC (FP32) -> C_fp16 ---
                 *
                 * SYCL2020 joint_matrix_store on FP32 accumulator
                 * doesn't portably round to a different destination
                 * type. Per @beta review #77 minor (joint_matrix_apply
                 * 4-arg form was speculative): use the canonical
                 * pattern -- store FP32 fragment to SLM staging, then
                 * sub-group cooperative FP32 -> FP16 conversion +
                 * scattered write to global C_fp16.
                 */
                /* Reuse A_slab as the FP32 staging area for the
                 * accumulator: TILE_M*TILE_N = 256 floats = 1 KB,
                 * fits trivially in A_slab's 8 KB. The next K_CHUNK
                 * iteration won't run (we're at the end), so reuse
                 * is safe. */
                auto c_stage_fp32 = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(
                    reinterpret_cast<float*>(&A_slab[0]));
                mx::joint_matrix_store(sg, mC, c_stage_fp32, TILE_N,
                                       mx::layout::row_major);
                sycl::group_barrier(sg);

                /* Cooperative FP32 -> FP16 + global write. 16 lanes
                 * × (TILE_M*TILE_N / SG_SIZE = 16) elements per lane
                 * = 256 outputs total. Lane `lane` handles the row
                 * `lane` of the output tile. */
                {
                    const std::size_t row = m_group + lane;
                    for (unsigned col = 0; col < TILE_N; ++col) {
                        const float v = c_stage_fp32[lane * TILE_N + col];
                        C_fp16[row * N + n_group + col] =
                            kv2_fp32_to_fp16(v);
                    }
                }
            });
    });
}

/* -- variant registration -------------------------------------------- *
 *
 * Phase 1 ships exactly one entry. Phase 2 will extend this X-macro
 * to sweep tile / sg / K_CHUNK like kv1_variants[] does.
 */

#define BITNET_ARC_KV2_VARIANTS(X) \
    X(16, 16, 16, 256)

#define KV2_DEFINE_LAUNCHER(TM, TN, SG, KC)                              \
    static void kv2_launch_##TM##_##TN##_##SG##_##KC(                    \
        sycl_queue_handle& q,                                            \
        std::size_t M, std::size_t N, std::size_t K,                     \
        const std::uint16_t* A,                                          \
        const bitnet_arc_tq2_0_block* B,                                 \
        std::uint16_t* C)                                                \
    {                                                                    \
        kv2_launch_impl<TM, TN, SG, KC>(q, M, N, K, A, B, C);            \
    }

BITNET_ARC_KV2_VARIANTS(KV2_DEFINE_LAUNCHER)

#undef KV2_DEFINE_LAUNCHER

#define KV2_TABLE_ROW(TM, TN, SG, KC)                                    \
    { TM, TN, SG, KC,                                                    \
      "v2_" #TM "x" #TN "_sg" #SG "_k" #KC,                              \
      &kv2_launch_##TM##_##TN##_##SG##_##KC },

extern const kv2_variant_desc kv2_variants[] = {
    BITNET_ARC_KV2_VARIANTS(KV2_TABLE_ROW)
};

extern const std::size_t kv2_variants_count =
    sizeof(kv2_variants) / sizeof(kv2_variants[0]);

#undef KV2_TABLE_ROW
#undef BITNET_ARC_KV2_VARIANTS

/* -- runtime dispatcher (compat API) --------------------------------- */

void run_kernel_v2(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v2_config& cfg)
{
    /* Phase 1: one variant only. Look up by value, fall back with a
     * stderr warning if the caller passed an unsupported config. */
    for (std::size_t i = 0; i < kv2_variants_count; ++i) {
        const kv2_variant_desc& v = kv2_variants[i];
        if (v.tile_M  == cfg.tile_M
         && v.tile_N  == cfg.tile_N
         && v.sg_size == cfg.sg_size
         && v.k_chunk == cfg.k_chunk)
        {
            v.launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
            return;
        }
    }
    std::fprintf(stderr,
                 "kv2: unsupported cfg (TM=%u TN=%u SG=%u KC=%u), "
                 "falling back to default %s\n",
                 cfg.tile_M, cfg.tile_N, cfg.sg_size, cfg.k_chunk,
                 kv2_variants[0].name);
    kv2_variants[0].launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
}

} /* namespace bitnet_arc */
