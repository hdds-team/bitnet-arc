/*
 * bitnet-arc v0 SYCL kernel implementation.
 *
 * See kernel_v0.h for the public contract and design anchors. The
 * inner-loop semantics here follow design v0 #2.2 "GPU path: ALU
 * vectorial" -- ternary {-1, 0, +1} maps to native sub / skip / add
 * with no multiplies in the hot path.
 *
 * #148 sweep: kernel body is a function template parameterized on
 * (TILE_M, TILE_N, SG_SIZE, MODE). Six explicit instantiations are
 * registered in kv0_variants[] below; the bench harness iterates
 * that table without knowing the SYCL types.
 *
 * Build: requires DPC++ / icpx with SYCL2020. See src/Makefile.
 */

#include "kernel_v0.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <sycl/sycl.hpp>

namespace bitnet_arc {

/* -- queue handle wrapper --------------------------------------------- */

class sycl_queue_handle {
public:
    sycl::queue q;

    sycl_queue_handle()
        : q(sycl::default_selector_v, sycl::property::queue::in_order{}) {}

    explicit sycl_queue_handle(const sycl::queue& src) : q(src) {}
};

sycl_queue_handle* make_default_queue_handle() {
    return new sycl_queue_handle();
}

void destroy_queue_handle(sycl_queue_handle* h) {
    delete h;
}

/* -- TQ2_0 unpack helper (device-callable) ---------------------------- */

/* Decode a 2-bit code at block-relative position i in [0, 256). Mirrors
 * oracle/tq2_0.c byte/shift formula bit-for-bit. */
static inline uint8_t kv0_unpack_code(const bitnet_arc_tq2_0_block& blk,
                                      std::size_t i)
{
    const std::size_t   byte  = (i >> 7) * 32u + (i & 31u);
    const unsigned      shift = static_cast<unsigned>(((i >> 5) & 3u) * 2u);
    return static_cast<uint8_t>((blk.qs[byte] >> shift) & 0x3u);
}

/* IEEE-754 binary16 -> FP32 (device variant). Same bit math as
 * oracle/fp16.h, duplicated here so the kernel stays self-contained
 * inside SYCL device code (avoids host-only headers leaking in). */
static inline float kv0_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)((h >> 15) & 0x1u);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31)
                 | ((uint32_t)(127 - 15 - e) << 23)
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

/* Inverse: FP32 -> binary16. Matches oracle/fp16.h fp32_to_fp16(). */
static inline uint16_t kv0_fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) {
        return (uint16_t)((sign << 15) | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    }
    if (exp > 15)  return (uint16_t)((sign << 15) | (0x1Fu << 10));
    if (exp < -14) {
        if (exp < -24) return (uint16_t)(sign << 15);
        uint32_t sub = (mant | 0x800000u) >> (-exp - 14 + 13);
        return (uint16_t)((sign << 15) | sub);
    }
    return (uint16_t)((sign << 15)
                    | ((uint32_t)(exp + 15) << 10)
                    | (mant >> 13));
}

/* -- templated kernel ------------------------------------------------- */

/* All four template parameters bake in compile-time. SG_SIZE feeds
 * [[sycl::reqd_sub_group_size(SG_SIZE)]] on the lambda; MODE drives
 * the inner-loop selection via constexpr if; TILE_M/TILE_N control
 * the work-group local range. */
template <unsigned TILE_M,
          unsigned TILE_N,
          unsigned SG_SIZE,
          kernel_v0_inner_mode MODE>
static void kv0_launch_impl(sycl_queue_handle& q_handle,
                            std::size_t M,
                            std::size_t N,
                            std::size_t K,
                            const std::uint16_t* A_fp16,
                            const bitnet_arc_tq2_0_block* B_blocks,
                            std::uint16_t* C_fp16)
{
    static_assert(TILE_M > 0u && TILE_N > 0u, "tile dims must be > 0");
    static_assert(SG_SIZE == 8u || SG_SIZE == 16u || SG_SIZE == 32u,
                  "SG_SIZE must be one of {8, 16, 32}");

    /* Host-side preconditions. v0 keeps these strict; padding is a
     * v0.5+ concern when we wire real BitNet 8B GGUF inputs. */
    assert(M > 0 && "kv0: M must be > 0");
    assert(N > 0 && "kv0: N must be > 0");
    assert(K > 0 && "kv0: K must be > 0");
    assert(K % 256 == 0
           && "kv0: K must be a multiple of TQ2_0 block size (256)");
    assert(M % static_cast<std::size_t>(TILE_M) == 0
           && "kv0: M must be a multiple of TILE_M");
    assert(N % static_cast<std::size_t>(TILE_N) == 0
           && "kv0: N must be a multiple of TILE_N");

    const std::size_t blocks_per_col = K / 256;

    sycl::queue& q = q_handle.q;

    q.submit([&](sycl::handler& h) {
        const sycl::range<2> global_range(M, N);
        const sycl::range<2> local_range(TILE_M, TILE_N);

        h.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const std::size_t m = it.get_global_id(0);
                const std::size_t n = it.get_global_id(1);

                /* Per-item accumulator in FP32. The output rounds to
                 * FP16 once at the end. Tile-local SLM accumulation is
                 * a #148 follow-up; v0 baseline keeps it simple. */
                float acc = 0.0f;

                for (std::size_t k_chunk = 0; k_chunk < blocks_per_col;
                     ++k_chunk)
                {
                    const bitnet_arc_tq2_0_block& blk =
                        B_blocks[n * blocks_per_col + k_chunk];

                    const float scale = kv0_fp16_to_fp32(blk.d);
                    float partial = 0.0f;
                    const std::size_t k_base = k_chunk * 256;

                    for (std::size_t i = 0; i < 256; ++i) {
                        const std::uint16_t a_bits = A_fp16[m * K + k_base + i];
                        const float a = kv0_fp16_to_fp32(a_bits);
                        const std::uint8_t code = kv0_unpack_code(blk, i);

                        if constexpr (MODE == kernel_v0_inner_mode::BRANCHFUL) {
                            /* Native ternary semantics:
                             *   code 0 (-1) -> partial -= a
                             *   code 1 ( 0) -> skip
                             *   code 2 (+1) -> partial += a
                             *   code 3      -> reserved, treated as skip
                             */
                            if (code == 0u) {
                                partial = partial - a;
                            } else if (code == 2u) {
                                partial = partial + a;
                            }
                        } else {
                            /* Branchless: (code - 1) cast to float. */
                            const int s = static_cast<int>(code) - 1;
                            partial = partial + static_cast<float>(s) * a;
                        }
                    }

                    acc = acc + scale * partial;
                }

                C_fp16[m * N + n] = kv0_fp32_to_fp16(acc);
            });
    });
}

/* -- variant registration -------------------------------------------- *
 *
 * X-macro listing the (TM, TN, SG, MODE) tuples that get explicit
 * launchers + table entries. Each row generates:
 *   - a non-templated wrapper kv0_launch_<TM>_<TN>_<SG>_<MODE>()
 *   - a row in kv0_variants[] pointing to that wrapper
 *
 * Smoke set for #148 (6 variants, ~5 min run on Arc B60):
 *   tile sweep   : (16x16) (32x16) (16x32) (32x32) at sg=16, branchful
 *   inner mode   : (16x16) sg=16, branchless
 *   subgroup     : (16x16) sg=32, branchful
 *
 * To extend the sweep, add rows here and rebuild. Compile time scales
 * roughly linearly with row count.
 */

#define BITNET_ARC_KV0_VARIANTS(X)            \
    X(16, 16, 16, BRANCHFUL)                  \
    X(32, 16, 16, BRANCHFUL)                  \
    X(16, 32, 16, BRANCHFUL)                  \
    X(32, 32, 16, BRANCHFUL)                  \
    X(16, 16, 16, BRANCHLESS)                 \
    X(16, 16, 32, BRANCHFUL)

#define KV0_DEFINE_LAUNCHER(TM, TN, SG, MODE)                            \
    static void kv0_launch_##TM##_##TN##_##SG##_##MODE(                  \
        sycl_queue_handle& q,                                            \
        std::size_t M, std::size_t N, std::size_t K,                     \
        const std::uint16_t* A,                                          \
        const bitnet_arc_tq2_0_block* B,                                 \
        std::uint16_t* C)                                                \
    {                                                                    \
        kv0_launch_impl<TM, TN, SG, kernel_v0_inner_mode::MODE>(         \
            q, M, N, K, A, B, C);                                        \
    }

BITNET_ARC_KV0_VARIANTS(KV0_DEFINE_LAUNCHER)

#undef KV0_DEFINE_LAUNCHER

#define KV0_TABLE_ROW(TM, TN, SG, MODE)                                  \
    { TM, TN, SG, kernel_v0_inner_mode::MODE,                            \
      #TM "x" #TN "_sg" #SG "_" #MODE,                                   \
      &kv0_launch_##TM##_##TN##_##SG##_##MODE },

extern const kernel_variant_desc kv0_variants[] = {
    BITNET_ARC_KV0_VARIANTS(KV0_TABLE_ROW)
};

extern const std::size_t kv0_variants_count =
    sizeof(kv0_variants) / sizeof(kv0_variants[0]);

#undef KV0_TABLE_ROW
#undef BITNET_ARC_KV0_VARIANTS

/* -- runtime dispatcher (compat API) ---------------------------------- */

void run_kernel_v0(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v0_config& cfg)
{
    /* Look up cfg in the variant table. If no match, fall back to the
     * baseline (always at index 0 by construction). */
    for (std::size_t i = 0; i < kv0_variants_count; ++i) {
        const kernel_variant_desc& v = kv0_variants[i];
        if (v.tile_M     == cfg.tile_M
         && v.tile_N     == cfg.tile_N
         && v.sg_size    == cfg.sg_size
         && v.inner_mode == cfg.inner_mode)
        {
            v.launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
            return;
        }
    }
    /* Fallback: baseline (16x16 / sg16 / branchful) is row 0. */
    kv0_variants[0].launch(q_handle, M, N, K, A_fp16, B_blocks, C_fp16);
}

} /* namespace bitnet_arc */
