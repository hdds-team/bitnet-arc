/*
 * bitnet-arc v1 SYCL kernel implementation -- SLM-tiled ternary x FP16
 * matmul.
 *
 * See kernel_v1.h for the public contract and docs/design-v1.md for
 * the full design rationale (review #66, approved 1/1).
 *
 * Diff vs v0 (kernel_v0.cpp):
 *   - Per-WG SLM slabs: A_slab (TILE_M x K_CHUNK FP16) and B_slab
 *     ((K_CHUNK/256) x TILE_N TQ2_0 blocks). Inner K-walk reads from
 *     SLM only; global loads happen once per K_CHUNK iteration via
 *     cooperative-load helpers.
 *   - Two SLM barriers per K_CHUNK iteration: one after the load
 *     (wait for all work-items to finish populating the slab) and one
 *     after the inner loop (wait for all readers to finish before the
 *     next iteration overwrites the slab).
 *   - BRANCHLESS-only inner loop (W1.5 finding 1; review #65 ratified
 *     BRANCHLESS as the empirical baseline).
 *   - K_CHUNK is a fourth template parameter, must be a multiple of
 *     256 (TQ2_0 block size). Phase 1 ships K_CHUNK=256 only;
 *     K_CHUNK=512/1024 are a follow-up commit.
 *
 * Build: requires DPC++ / icpx with SYCL2020. Linked into the same
 * libbitnet_arc_v0.a as kernel_v0.cpp (the lib name is historical;
 * src/Makefile aggregates v0 + v1 into one archive).
 */

#include "kernel_v0_sycl.hpp"
#include "kernel_v1.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bitnet_arc {

/* -- shared device helpers ------------------------------------------- *
 *
 * The TQ2_0 unpack and FP16 conversion helpers are duplicated from
 * kernel_v0.cpp (file-static there). v1 needs them inside SYCL device
 * code; rather than expose v0's static helpers we re-declare them
 * here with kv1_ prefix. Bit math is identical -- if oracle/tq2_0.c
 * or oracle/fp16.h change, both prefixes update in lockstep.
 */

static inline std::uint8_t kv1_unpack_code(const bitnet_arc_tq2_0_block& blk,
                                           std::size_t i)
{
    const std::size_t   byte  = (i >> 7) * 32u + (i & 31u);
    const unsigned      shift = static_cast<unsigned>(((i >> 5) & 3u) * 2u);
    return static_cast<std::uint8_t>((blk.qs[byte] >> shift) & 0x3u);
}

static inline float kv1_fp16_to_fp32(std::uint16_t h) {
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

static inline std::uint16_t kv1_fp32_to_fp16(float f) {
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
 * SLM layout (per design v1 section 5):
 *   A_slab : TILE_M rows of K_CHUNK FP16 elements each
 *   B_slab : (K_CHUNK / 256) rows of TILE_N TQ2_0 blocks each
 *
 * Cooperative-load distribution: we use a linear local id (lid =
 * m_local * TILE_N + n_local, range [0, TILE_M*TILE_N)) and stride by
 * the work-group size. This handles arbitrary TILE_M / TILE_N /
 * K_CHUNK combinations without per-shape special cases, and gives
 * stride-1 access within each step (-> coalesced loads).
 */

template <unsigned TILE_M,
          unsigned TILE_N,
          unsigned SG_SIZE,
          unsigned K_CHUNK>
static void kv1_launch_impl(sycl_queue_handle& q_handle,
                            std::size_t M,
                            std::size_t N,
                            std::size_t K,
                            const std::uint16_t* A_fp16,
                            const bitnet_arc_tq2_0_block* B_blocks,
                            std::uint16_t* C_fp16)
{
    /* Compile-time invariants. */
    static_assert(TILE_M > 0u && TILE_N > 0u, "tile dims must be > 0");
    static_assert(TILE_M * TILE_N <= 1024u,
                  "TILE_M * TILE_N must fit in Xe2 WG max (1024)");
    static_assert(SG_SIZE == 8u || SG_SIZE == 16u || SG_SIZE == 32u,
                  "SG_SIZE must be one of {8, 16, 32}");
    static_assert(K_CHUNK > 0u, "K_CHUNK must be > 0");
    static_assert(K_CHUNK % 256u == 0u,
                  "K_CHUNK must be a multiple of TQ2_0 block size (256)");

    /* SLM budget guard (per @theta + @sonnet review #68 catch). Arc Xe2
     * exposes ~64 KB of SLM per work-group as a hard limit -- past that
     * the runtime either refuses to launch or spills, both equally bad.
     * Any new (TILE_M, TILE_N, K_CHUNK) instantiation that would exceed
     * the budget must fail at compile time with a clear message, not
     * surprise us at hardware run. The 65536 bound is conservative; if
     * Xe3+ or a different target exposes more, this can be widened in
     * one place. */
    constexpr std::size_t KV1_A_SLAB_BYTES =
        static_cast<std::size_t>(TILE_M) *
        static_cast<std::size_t>(K_CHUNK) *
        sizeof(std::uint16_t);
    constexpr std::size_t KV1_B_SLAB_BYTES =
        static_cast<std::size_t>(K_CHUNK / 256u) *
        static_cast<std::size_t>(TILE_N) *
        sizeof(bitnet_arc_tq2_0_block);
    constexpr std::size_t KV1_SLM_BUDGET_BYTES = 64u * 1024u;
    static_assert(KV1_A_SLAB_BYTES + KV1_B_SLAB_BYTES
                      <= KV1_SLM_BUDGET_BYTES,
                  "kv1: A_slab + B_slab exceeds Xe2 64 KB SLM/WG hard "
                  "limit -- pick smaller TILE_M, TILE_N, or K_CHUNK, or "
                  "widen KV1_SLM_BUDGET_BYTES if targeting a device with "
                  "more SLM/WG");

    /* Host-side preconditions. */
    assert(M > 0 && "kv1: M must be > 0");
    assert(N > 0 && "kv1: N must be > 0");
    assert(K > 0 && "kv1: K must be > 0");
    assert(K % 256 == 0
           && "kv1: K must be a multiple of TQ2_0 block size (256)");
    assert(K % static_cast<std::size_t>(K_CHUNK) == 0
           && "kv1: K must be a multiple of K_CHUNK");
    assert(M % static_cast<std::size_t>(TILE_M) == 0
           && "kv1: M must be a multiple of TILE_M");
    assert(N % static_cast<std::size_t>(TILE_N) == 0
           && "kv1: N must be a multiple of TILE_N");

    constexpr unsigned BLOCKS_PER_CHUNK  = K_CHUNK / 256u;
    constexpr unsigned WG_SIZE           = TILE_M * TILE_N;
    constexpr std::size_t A_SLAB_ELEMS   =
        static_cast<std::size_t>(TILE_M) * static_cast<std::size_t>(K_CHUNK);
    constexpr std::size_t B_SLAB_BLOCKS  =
        static_cast<std::size_t>(BLOCKS_PER_CHUNK) *
        static_cast<std::size_t>(TILE_N);

    const std::size_t blocks_per_col = K / 256;
    const std::size_t chunks_per_col = K / K_CHUNK;

    sycl::queue& q = q_handle.q;

    q.submit([&](sycl::handler& h) {
        const sycl::range<2> global_range(M, N);
        const sycl::range<2> local_range(TILE_M, TILE_N);

        /* SLM allocations. local_accessor in SYCL2020. */
        sycl::local_accessor<std::uint16_t, 1> A_slab(
            sycl::range<1>(A_SLAB_ELEMS), h);
        sycl::local_accessor<bitnet_arc_tq2_0_block, 1> B_slab(
            sycl::range<1>(B_SLAB_BLOCKS), h);

        h.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const unsigned m_local = static_cast<unsigned>(
                    it.get_local_id(0));
                const unsigned n_local = static_cast<unsigned>(
                    it.get_local_id(1));
                const std::size_t m_group =
                    static_cast<std::size_t>(it.get_group(0)) * TILE_M;
                const std::size_t n_group =
                    static_cast<std::size_t>(it.get_group(1)) * TILE_N;
                const unsigned lid = m_local * TILE_N + n_local;

                float acc = 0.0f;

                for (std::size_t c = 0; c < chunks_per_col; ++c) {
                    const std::size_t k0       = c * K_CHUNK;
                    const std::size_t k_chunk0 = c * BLOCKS_PER_CHUNK;

                    /* Cooperative load A_slab.
                     *
                     * Total elements = TILE_M * K_CHUNK. We stripe by
                     * lid so consecutive lanes inside a subgroup hit
                     * consecutive K offsets -> coalesced. */
                    for (std::size_t idx = lid;
                         idx < A_SLAB_ELEMS;
                         idx += WG_SIZE)
                    {
                        const std::size_t m_idx = idx / K_CHUNK;
                        const std::size_t k_off = idx % K_CHUNK;
                        A_slab[idx] = A_fp16[(m_group + m_idx) * K
                                             + k0 + k_off];
                    }

                    /* Cooperative load B_slab.
                     *
                     * Total blocks = BLOCKS_PER_CHUNK * TILE_N. Same
                     * lid-stride pattern -- the per-block 66 B copy is
                     * a struct-by-value assignment; SYCL emits the
                     * appropriate load sequence. (A two-pass split
                     * (qs[64] coalesced + d gathered) is a v1.5
                     * follow-up if measurement shows the 66 B
                     * non-power-of-2 stride is hot.) */
                    for (std::size_t idx = lid;
                         idx < B_SLAB_BLOCKS;
                         idx += WG_SIZE)
                    {
                        const std::size_t k_blk = idx / TILE_N;
                        const std::size_t n_idx = idx % TILE_N;
                        B_slab[idx] = B_blocks[
                            (n_group + n_idx) * blocks_per_col
                            + k_chunk0 + k_blk];
                    }

                    /* Barrier 1/2: all WG members must see populated
                     * SLM before the inner loop reads from it. */
                    it.barrier(sycl::access::fence_space::local_space);

                    /* Inner K-walk over SLM. Per (m_local, n_local)
                     * work-item: walk BLOCKS_PER_CHUNK blocks of 256
                     * weights each, reading A_slab[m_local, *] and
                     * B_slab[*, n_local] from SLM only.
                     *
                     * Inner mode: BRANCHLESS only (design v1 sec 2.4).
                     * code in {0,1,2} -> s = code - 1 in {-1,0,+1}. */
                    for (unsigned k_blk = 0;
                         k_blk < BLOCKS_PER_CHUNK;
                         ++k_blk)
                    {
                        const bitnet_arc_tq2_0_block& blk =
                            B_slab[static_cast<std::size_t>(k_blk)
                                   * TILE_N + n_local];
                        const float scale = kv1_fp16_to_fp32(blk.d);
                        float partial = 0.0f;
                        const std::size_t k_base =
                            static_cast<std::size_t>(k_blk) * 256u;

                        for (unsigned i = 0; i < 256u; ++i) {
                            const std::uint16_t a_bits =
                                A_slab[static_cast<std::size_t>(m_local)
                                       * K_CHUNK + k_base + i];
                            const float a = kv1_fp16_to_fp32(a_bits);
                            const std::uint8_t code =
                                kv1_unpack_code(blk, i);
                            const int s = static_cast<int>(code) - 1;
                            partial = partial
                                    + static_cast<float>(s) * a;
                        }
                        acc = acc + scale * partial;
                    }

                    /* Barrier 2/2: all readers must finish before the
                     * next iteration overwrites SLM. The fence space
                     * is local_space only -- A/B/C in global memory
                     * are owned per-WG-item and don't need cross-WG
                     * sync inside the K-walk. */
                    it.barrier(sycl::access::fence_space::local_space);
                }

                /* Final FP32 -> FP16 round and store. */
                const std::size_t m = m_group + m_local;
                const std::size_t n = n_group + n_local;
                C_fp16[m * N + n] = kv1_fp32_to_fp16(acc);
            });
    });
}

/* -- variant registration -------------------------------------------- *
 *
 * X-macro listing the (TM, TN, SG, KCHUNK) tuples. Phase 1 (this drop)
 * ships K_CHUNK=256 only -- 5 tile/sg combos. Phase 2 follow-up adds
 * K_CHUNK=512 and K_CHUNK=1024 rows for the same 5 combos.
 *
 * Variant naming: "v1_<TM>x<TN>_sg<SG>_k<KCHUNK>".
 */

#define BITNET_ARC_KV1_VARIANTS(X)        \
    X(16, 16, 16, 256)                    \
    X(32, 16, 16, 256)                    \
    X(16, 32, 16, 256)                    \
    X(32, 32, 16, 256)                    \
    X(32, 32, 32, 256)                    \
    /* Phase 2a (task #153): K_CHUNK=512                              */ \
    /* tests H2 (barrier overhead) -- 28 barrier-pairs at K=14336    */ \
    /* instead of 56 with K_CHUNK=256.                                */ \
    X(16, 16, 16, 512)                    \
    X(32, 16, 16, 512)                    \
    X(16, 32, 16, 512)                    \
    X(32, 32, 16, 512)                    \
    X(32, 32, 32, 512)                    \
    /* Phase 2a (task #153): K_CHUNK=1024                              */ \
    /* tests H2 fully -- 14 barrier-pairs at K=14336 (4x fewer than   */ \
    /* the K_CHUNK=256 baseline). Variants are skipped at run time on */ \
    /* shapes where K % 1024 != 0 (e.g. (16,16,256)) via sweep_tile's */ \
    /* K%K_CHUNK guard.                                                */ \
    /*                                                                 */ \
    /* SLM budget at K_CHUNK=1024 (per @theta review #68 catch):       */ \
    /*   A_slab = TILE_M * 1024 * 2 bytes (FP16)                       */ \
    /*   B_slab = (1024/256) * TILE_N * 66 bytes                       */ \
    /*   Arc B60 / Xe2 hard limit per WG: ~64 KB.                      */ \
    /*                                                                 */ \
    /* TILE_M=32 means A_slab alone = 64 KB -- leaves zero room for    */ \
    /* B_slab and tanks occupancy to 1 WG/Xe-core. The TILE_M=32 rows  */ \
    /* at K_CHUNK=1024 are therefore dropped from this phase. They     */ \
    /* would be revisitable at v1.5 if SLM/WG exceeds 64 KB on Xe3+    */ \
    /* or via reduced-precision A_slab (FP8 staging, design v2).       */ \
    X(16, 16, 16, 1024)                   \
    X(16, 32, 16, 1024)

#define KV1_DEFINE_LAUNCHER(TM, TN, SG, KC)                              \
    static void kv1_launch_##TM##_##TN##_##SG##_##KC(                    \
        sycl_queue_handle& q,                                            \
        std::size_t M, std::size_t N, std::size_t K,                     \
        const std::uint16_t* A,                                          \
        const bitnet_arc_tq2_0_block* B,                                 \
        std::uint16_t* C)                                                \
    {                                                                    \
        kv1_launch_impl<TM, TN, SG, KC>(q, M, N, K, A, B, C);            \
    }

BITNET_ARC_KV1_VARIANTS(KV1_DEFINE_LAUNCHER)

#undef KV1_DEFINE_LAUNCHER

#define KV1_TABLE_ROW(TM, TN, SG, KC)                                    \
    { TM, TN, SG, KC,                                                    \
      "v1_" #TM "x" #TN "_sg" #SG "_k" #KC,                              \
      &kv1_launch_##TM##_##TN##_##SG##_##KC },

extern const kv1_variant_desc kv1_variants[] = {
    BITNET_ARC_KV1_VARIANTS(KV1_TABLE_ROW)
};

extern const std::size_t kv1_variants_count =
    sizeof(kv1_variants) / sizeof(kv1_variants[0]);

#undef KV1_TABLE_ROW
#undef BITNET_ARC_KV1_VARIANTS

/* -- runtime dispatcher (compat API) --------------------------------- */

void run_kernel_v1(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v1_config& cfg)
{
    /* Look up cfg in the variant table. If no match, fall back to the
     * default (32x32 / sg16 / K_CHUNK=256, which is at index 3 by the
     * X-macro order above -- but we look it up by value to stay
     * robust if the table is reordered). */
    for (std::size_t i = 0; i < kv1_variants_count; ++i) {
        const kv1_variant_desc& v = kv1_variants[i];
        if (v.tile_M  == cfg.tile_M
         && v.tile_N  == cfg.tile_N
         && v.sg_size == cfg.sg_size
         && v.k_chunk == cfg.k_chunk)
        {
            v.launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
            return;
        }
    }
    /* Fallback: search for the default (32x32/sg16/k256). */
    for (std::size_t i = 0; i < kv1_variants_count; ++i) {
        const kv1_variant_desc& v = kv1_variants[i];
        if (v.tile_M == 32u && v.tile_N == 32u
         && v.sg_size == 16u && v.k_chunk == 256u)
        {
            v.launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
            return;
        }
    }
    /* Last resort: row 0. */
    kv1_variants[0].launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
}

} /* namespace bitnet_arc */
