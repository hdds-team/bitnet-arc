/*
 * bitnet-arc v0 SYCL kernel implementation.
 *
 * See kernel_v0.h for the public contract and design anchors. The
 * inner-loop semantics here follow design v0 §2.2 "GPU path: ALU
 * vectorial" -- ternary {-1, 0, +1} maps to native sub / skip / add
 * with no multiplies in the hot path.
 *
 * Build: this TU requires DPC++ / icpx / clang++ with SYCL2020. See
 * src/Makefile for the toolchain invocation.
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

/* -- TQ2_0 unpack helpers (device-callable) --------------------------- */

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

/* -- public API ------------------------------------------------------- */

void run_kernel_v0(sycl_queue_handle& q_handle,
                   std::size_t M,
                   std::size_t N,
                   std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16,
                   const kernel_v0_config& cfg)
{
    /* Host-side preconditions. v0 keeps these strict; padding is a
     * v0.5+ concern when we wire real BitNet 8B GGUF inputs.
     *
     * tile_M / tile_N are unsigned in the API (see kernel_v0.h note
     * on @codex review #60), so negative values are impossible. We
     * still need to forbid zero before the modulo. */
    assert(M > 0 && "kernel_v0: M must be > 0");
    assert(N > 0 && "kernel_v0: N must be > 0");
    assert(K > 0 && "kernel_v0: K must be > 0");
    assert(K % 256 == 0
           && "kernel_v0: K must be a multiple of TQ2_0 block size (256)");
    assert(cfg.tile_M > 0u && cfg.tile_N > 0u
           && "kernel_v0: tile dims must be > 0");
    assert(M % static_cast<std::size_t>(cfg.tile_M) == 0
           && "kernel_v0: M must be a multiple of tile_M");
    assert(N % static_cast<std::size_t>(cfg.tile_N) == 0
           && "kernel_v0: N must be a multiple of tile_N");

    const std::size_t blocks_per_col = K / 256;
    const std::size_t tile_M = static_cast<std::size_t>(cfg.tile_M);
    const std::size_t tile_N = static_cast<std::size_t>(cfg.tile_N);
    const auto inner_mode   = cfg.inner_mode;

    sycl::queue& q = q_handle.q;

    q.submit([&](sycl::handler& h) {
        /* nd_range:
         *   global = (M, N)            (one work-item per output element)
         *   local  = (tile_M, tile_N)  (work-group, default 16x16)
         */
        const sycl::range<2> global_range(M, N);
        const sycl::range<2> local_range(tile_M, tile_N);

        /* v0 baseline: subgroup size locked to 16 via the
         * compile-time attribute on the kernel lambda. Without this
         * attribute the runtime is free to pick any subgroup size,
         * which would make #148 tile sweep measurements meaningless
         * (varying tile_M/tile_N would not actually exercise the
         * subgroup dimension). #148 will introduce templated kernel
         * variants for sizes 8 / 16 / 32, each with its own
         * reqd_sub_group_size attribute (per @claude-opus + @sonnet
         * +@codex review #60 follow-up). */
        h.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                const std::size_t m = it.get_global_id(0);
                const std::size_t n = it.get_global_id(1);

                /* Per-item accumulator in FP32. The output rounds to
                 * FP16 once at the end. Tile-local SLM accumulation is
                 * a #148 sweep concern; v0 keeps it simple. */
                float acc = 0.0f;

                /* Walk K in TQ2_0 block chunks. Block (n, k_chunk) is
                 * laid out at index n * blocks_per_col + k_chunk. */
                for (std::size_t k_chunk = 0; k_chunk < blocks_per_col;
                     ++k_chunk)
                {
                    const bitnet_arc_tq2_0_block& blk =
                        B_blocks[n * blocks_per_col + k_chunk];

                    const float scale = kv0_fp16_to_fp32(blk.d);
                    float partial = 0.0f;
                    const std::size_t k_base = k_chunk * 256;

                    /* Inner block loop: 256 ternary weights against
                     * 256 contiguous activations. */
                    for (std::size_t i = 0; i < 256; ++i) {
                        const std::uint16_t a_bits = A_fp16[m * K + k_base + i];
                        const float a = kv0_fp16_to_fp32(a_bits);
                        const std::uint8_t code = kv0_unpack_code(blk, i);

                        if (inner_mode == kernel_v0_inner_mode::BRANCHFUL) {
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

                    /* Apply per-block scale once, after the 256
                     * unscaled ternary contractions. */
                    acc = acc + scale * partial;
                }

                /* Round to FP16 on output. */
                C_fp16[m * N + n] = kv0_fp32_to_fp16(acc);
            });
    });
}

} /* namespace bitnet_arc */
