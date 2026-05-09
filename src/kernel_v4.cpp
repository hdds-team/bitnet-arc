/*
 * bitnet-arc v4 ESIMD INT2 kernel implementation.
 *
 * See kernel_v4.h for the public contract and docs/design-v4.md for
 * the design (ratified 99c4237). VNNI INT2 packing helpers in
 * kernel_v4_packing.h are validated by 1131 host unit tests
 * (commit ff8f2d8).
 *
 * Phase 1 v4 strategy: HOST PRE-PACK (per @naskel review fold D1).
 *   - Helpers from kernel_v4_packing.h run on host to produce
 *     VNNI INT2 USM device buffers + per-row activation scales +
 *     per-N-column d_chunk values.
 *   - ESIMD kernel just block-loads pre-packed data + DPAS
 *     accumulates + folds d_chunk + scales + scatter store.
 *   - Phase 2 v4 optim will port the pre-pack into a device-side
 *     ESIMD prep kernel or fuse into the main kernel.
 *
 * Per-chunk register-only fold (per design v4 sec3.4 + @naskel fold D2):
 *   - 4 fragments per chunk all share the same d_chunk[n], so we
 *     accumulate INT32 across the 4 frags first, then fold once
 *     per chunk into the FP32 cross-chunk accumulator.
 */

#include "kernel_v4.h"
#include "kernel_v4_packing.h"
#include "kernel_v0_sycl.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace bitnet_arc {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

namespace {

/* ---------------------------------------------------------------- *
 * Host-side pre-pack:
 *   - For each (tile_m, frag_k) pair: extract M=8 x K=64 sub-block
 *     of A_fp16, compute per-row max-abs scales (or use existing),
 *     pack to VNNI INT2.
 *   - For each (tile_n, frag_k) pair: extract K=64 x N=16 sub-block
 *     of B (decoded ternary from TQ2_0), pack to VNNI INT2.
 *   - For each (tile_n, chunk) pair: extract 16 d_chunk[n] FP32 values
 *     for the per-chunk fold.
 *   - For each row m: per-row max-abs over full K (used for final scale).
 * ---------------------------------------------------------------- */

/* prepack_host_int2: INT2 act path. s_a = max_abs (no scale-back needed
 * because a_q ∈ {-1, 0, +1} = a/s_a rounded). */
void prepack_host_int2(
    std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A_fp16,
    const bitnet_arc_tq2_0_block* B_blocks,
    std::vector<std::uint32_t>& A_packed_h,
    std::vector<std::uint32_t>& B_packed_h,
    std::vector<float>&         d_per_n_h,
    std::vector<float>&         s_a_h)
{
    const std::size_t tiles_M       = M / KV4_TILE_M;
    const std::size_t tiles_N       = N / KV4_TILE_N;
    const std::size_t chunks_per_K  = K / KV4_K_CHUNK;
    const std::size_t frags_per_K   = K / KV4_TILE_K;
    const std::size_t blocks_per_col = K / 256;

    /* 1. Per-row activation max-abs over full K -> s_a_h[m] (M values). */
    s_a_h.assign(M, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        s_a_h[m] = kv4::compute_a_row_max_abs(A_fp16 + m * K, K);
    }

    /* 2. A pre-pack: per (tile_m, frag_k), 8 rows x 64 K -> 32 dwords. */
    A_packed_h.assign(tiles_M * frags_per_K * 32, 0u);
    for (std::size_t tm = 0; tm < tiles_M; ++tm) {
        const std::size_t m_off = tm * KV4_TILE_M;
        /* Get s_a for these 8 rows */
        float s_a_block[KV4_TILE_M];
        for (unsigned i = 0; i < KV4_TILE_M; ++i)
            s_a_block[i] = s_a_h[m_off + i];

        for (std::size_t f = 0; f < frags_per_K; ++f) {
            const std::size_t k_off = f * KV4_TILE_K;
            /* Extract 8 x 64 sub-block, contiguous K within each row. */
            std::uint16_t a_frag[KV4_TILE_M * KV4_TILE_K];
            for (unsigned i = 0; i < KV4_TILE_M; ++i) {
                std::memcpy(&a_frag[i * KV4_TILE_K],
                            A_fp16 + (m_off + i) * K + k_off,
                            KV4_TILE_K * sizeof(std::uint16_t));
            }
            kv4::pack_a_fragment_vnni_int2(
                a_frag, s_a_block,
                A_packed_h.data() + (tm * frags_per_K + f) * 32);
        }
    }

    /* 3. B pre-pack + d_per_n: per (tile_n, frag_k), 64 K x 16 N -> 64 dwords. */
    B_packed_h.assign(tiles_N * frags_per_K * 64, 0u);
    d_per_n_h.assign(tiles_N * chunks_per_K * KV4_TILE_N, 0.0f);

    for (std::size_t tn = 0; tn < tiles_N; ++tn) {
        const std::size_t n_off = tn * KV4_TILE_N;

        /* For each chunk, decode 16 N-cols x 256 K = 16 x 256 ternary,
         * then split into 4 frags of K=64 each. Also extract d_chunk[n]. */
        for (std::size_t c = 0; c < chunks_per_K; ++c) {
            std::int8_t ternary_chunk[KV4_K_CHUNK * KV4_TILE_N];
            for (unsigned n = 0; n < KV4_TILE_N; ++n) {
                const std::size_t blk_idx = (n_off + n) * blocks_per_col + c;
                const auto& blk = B_blocks[blk_idx];
                /* Per-chunk d_per_n: store FP32 of FP16 d. */
                d_per_n_h[(tn * chunks_per_K + c) * KV4_TILE_N + n] =
                    bitnet_arc_fp16_to_fp32(blk.d);

                std::int8_t ternary_col[256];
                kv4::unpack_tq2_0_block_to_ternary(blk, ternary_col);
                for (unsigned k = 0; k < KV4_K_CHUNK; ++k) {
                    ternary_chunk[k * KV4_TILE_N + n] = ternary_col[k];
                }
            }
            /* Split 256 K x 16 N -> 4 frags of 64 K x 16 N. */
            for (unsigned f = 0; f < KV4_FRAGS_PER_CHUNK; ++f) {
                std::int8_t ternary_frag[KV4_TILE_K * KV4_TILE_N];
                for (unsigned k = 0; k < KV4_TILE_K; ++k) {
                    std::memcpy(
                        &ternary_frag[k * KV4_TILE_N],
                        &ternary_chunk[(f * KV4_TILE_K + k) * KV4_TILE_N],
                        KV4_TILE_N * sizeof(std::int8_t));
                }
                const std::size_t frag_k = c * KV4_FRAGS_PER_CHUNK + f;
                kv4::pack_b_fragment_vnni_int2(
                    ternary_frag,
                    B_packed_h.data() + (tn * frags_per_K + frag_k) * 64);
            }
        }
    }
}

/* prepack_host_int4: INT4 act path. s_a stored as (max_abs / 7) so the
 * kernel's final `c = s_a_dev[m] * mC_fp32[m, n]` correctly reconstructs
 * c = sum_k(a*b) given that a_q = round(a/s_a*7), recovered as a = a_q * s_a / 7.
 * Output buffer sizes:
 *   A_packed_h: tiles_M * frags_per_K * 64 dwords (8 nibbles/dword × 8 rows)
 *   B_packed_h: tiles_N * frags_per_K * 128 dwords (8 nibbles/dword × 16 cols)
 */
void prepack_host_int4(
    std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A_fp16,
    const bitnet_arc_tq2_0_block* B_blocks,
    std::vector<std::uint32_t>& A_packed_h,
    std::vector<std::uint32_t>& B_packed_h,
    std::vector<float>&         d_per_n_h,
    std::vector<float>&         s_a_h)
{
    const std::size_t tiles_M       = M / KV4_TILE_M;
    const std::size_t tiles_N       = N / KV4_TILE_N;
    const std::size_t chunks_per_K  = K / KV4_K_CHUNK;
    const std::size_t frags_per_K   = K / KV4_TILE_K;
    const std::size_t blocks_per_col = K / 256;

    /* s_a = max_abs (raw, used to scale a/s_a for *7 quant). For final
     * reconstruction we store s_a / 7 in the kernel-visible buffer. */
    s_a_h.assign(M, 0.0f);
    std::vector<float> s_a_raw(M, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        s_a_raw[m] = kv4::compute_a_row_max_abs(A_fp16 + m * K, K);
        s_a_h[m]   = s_a_raw[m] / 7.0f;
    }

    /* A pre-pack: 64 dwords per fragment (vs 32 for INT2). */
    A_packed_h.assign(tiles_M * frags_per_K * 64, 0u);
    for (std::size_t tm = 0; tm < tiles_M; ++tm) {
        const std::size_t m_off = tm * KV4_TILE_M;
        float s_a_block[KV4_TILE_M];
        for (unsigned i = 0; i < KV4_TILE_M; ++i)
            s_a_block[i] = s_a_raw[m_off + i];

        for (std::size_t f = 0; f < frags_per_K; ++f) {
            const std::size_t k_off = f * KV4_TILE_K;
            std::uint16_t a_frag[KV4_TILE_M * KV4_TILE_K];
            for (unsigned i = 0; i < KV4_TILE_M; ++i) {
                std::memcpy(&a_frag[i * KV4_TILE_K],
                            A_fp16 + (m_off + i) * K + k_off,
                            KV4_TILE_K * sizeof(std::uint16_t));
            }
            kv4::pack_a_fragment_vnni_int4(
                a_frag, s_a_block,
                A_packed_h.data() + (tm * frags_per_K + f) * 64);
        }
    }

    /* B pre-pack + d_per_n: 128 dwords per fragment (vs 64 for INT2). */
    B_packed_h.assign(tiles_N * frags_per_K * 128, 0u);
    d_per_n_h.assign(tiles_N * chunks_per_K * KV4_TILE_N, 0.0f);

    for (std::size_t tn = 0; tn < tiles_N; ++tn) {
        const std::size_t n_off = tn * KV4_TILE_N;
        for (std::size_t c = 0; c < chunks_per_K; ++c) {
            std::int8_t ternary_chunk[KV4_K_CHUNK * KV4_TILE_N];
            for (unsigned n = 0; n < KV4_TILE_N; ++n) {
                const std::size_t blk_idx = (n_off + n) * blocks_per_col + c;
                const auto& blk = B_blocks[blk_idx];
                d_per_n_h[(tn * chunks_per_K + c) * KV4_TILE_N + n] =
                    bitnet_arc_fp16_to_fp32(blk.d);
                std::int8_t ternary_col[256];
                kv4::unpack_tq2_0_block_to_ternary(blk, ternary_col);
                for (unsigned k = 0; k < KV4_K_CHUNK; ++k) {
                    ternary_chunk[k * KV4_TILE_N + n] = ternary_col[k];
                }
            }
            for (unsigned f = 0; f < KV4_FRAGS_PER_CHUNK; ++f) {
                std::int8_t ternary_frag[KV4_TILE_K * KV4_TILE_N];
                for (unsigned k = 0; k < KV4_TILE_K; ++k) {
                    std::memcpy(
                        &ternary_frag[k * KV4_TILE_N],
                        &ternary_chunk[(f * KV4_TILE_K + k) * KV4_TILE_N],
                        KV4_TILE_N * sizeof(std::int8_t));
                }
                const std::size_t frag_k = c * KV4_FRAGS_PER_CHUNK + f;
                kv4::pack_b_fragment_vnni_int4(
                    ternary_frag,
                    B_packed_h.data() + (tn * frags_per_K + frag_k) * 128);
            }
        }
    }
}

/* FP32 -> FP16 inline (mirror kernel_v0_sycl helper for ESIMD-side
 * usage; ESIMD's simd<half> + simd<float> conversion is the path
 * but for scalar single-element conversion we keep the bit math). */
static inline std::uint16_t kv4_fp32_to_fp16_bits(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    std::uint32_t sign = (x >> 31) & 0x1u;
    std::int32_t  exp  = (std::int32_t)((x >> 23) & 0xFFu) - 127;
    std::uint32_t mant = x & 0x7FFFFFu;
    if (exp == 128) {
        return (std::uint16_t)((sign << 15) | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    }
    if (exp > 15)  return (std::uint16_t)((sign << 15) | (0x1Fu << 10));
    if (exp < -14) {
        if (exp < -24) return (std::uint16_t)(sign << 15);
        std::uint32_t sub = (mant | 0x800000u) >> (-exp - 14 + 13);
        return (std::uint16_t)((sign << 15) | sub);
    }
    return (std::uint16_t)((sign << 15) | ((std::uint32_t)(exp + 15) << 10) | (mant >> 13));
}

} /* anonymous namespace */

/* -------------------------------------------------------------------- */
/* run_kernel_v4                                                        */
/* -------------------------------------------------------------------- */

void run_kernel_v4(sycl_queue_handle& q_handle,
                   std::size_t M, std::size_t N, std::size_t K,
                   const std::uint16_t* A_fp16,
                   const bitnet_arc_tq2_0_block* B_blocks,
                   std::uint16_t* C_fp16)
{
    /* Risk #7 runtime contracts. */
    assert(K % KV4_K_CHUNK == 0);
    assert(M % KV4_TILE_M  == 0);
    assert(N % KV4_TILE_N  == 0);

    const std::size_t tiles_M       = M / KV4_TILE_M;
    const std::size_t tiles_N       = N / KV4_TILE_N;
    const std::size_t total_tiles   = tiles_M * tiles_N;
    const std::size_t chunks_per_K  = K / KV4_K_CHUNK;
    const std::size_t frags_per_K   = K / KV4_TILE_K;

    sycl::queue& q = q_handle.q;

    /* 0. The inputs A_fp16 / B_blocks are USM device pointers (per
     * bench/sweep_tile.cpp). Pull them to host-side staging vectors
     * so prepack_host can read them. Phase 1 v4 cost: 2 extra memcpys
     * device->host. Phase 2 v4 optim: port pre-pack to device-side
     * ESIMD prep kernel to eliminate this round-trip.
     */
    const std::size_t A_size = M * K;
    const std::size_t B_size = (K / 256) * N;
    std::vector<std::uint16_t> A_host(A_size);
    std::vector<bitnet_arc_tq2_0_block> B_host(B_size);
    q.memcpy(A_host.data(), A_fp16,    A_size * sizeof(std::uint16_t)).wait();
    q.memcpy(B_host.data(), B_blocks,  B_size * sizeof(bitnet_arc_tq2_0_block)).wait();

    /* 1. Host pre-pack into staging vectors (INT2 act path).
     * For INT4 act variant, see the dispatched variant launcher below. */
    std::vector<std::uint32_t> A_packed_h, B_packed_h;
    std::vector<float>         d_per_n_h, s_a_h;
    prepack_host_int2(M, N, K, A_host.data(), B_host.data(),
                      A_packed_h, B_packed_h, d_per_n_h, s_a_h);

    /* 2. Allocate USM device buffers + memcpy from staging. */
    auto A_packed = sycl::malloc_device<std::uint32_t>(A_packed_h.size(), q);
    auto B_packed = sycl::malloc_device<std::uint32_t>(B_packed_h.size(), q);
    auto d_per_n  = sycl::malloc_device<float>        (d_per_n_h.size(),  q);
    auto s_a_dev  = sycl::malloc_device<float>        (s_a_h.size(),      q);
    q.memcpy(A_packed, A_packed_h.data(), A_packed_h.size() * sizeof(std::uint32_t)).wait();
    q.memcpy(B_packed, B_packed_h.data(), B_packed_h.size() * sizeof(std::uint32_t)).wait();
    q.memcpy(d_per_n,  d_per_n_h.data(),  d_per_n_h.size()  * sizeof(float)).wait();
    q.memcpy(s_a_dev,  s_a_h.data(),      s_a_h.size()      * sizeof(float)).wait();

    /* 3. ESIMD kernel launch: 1 thread per output tile. */
    try {
    q.submit([&](sycl::handler& h) {
        /* Capture by value: pointers + dims (only those used in the kernel) */
        const auto t_N   = tiles_N;
        const auto cpK   = chunks_per_K;
        const auto fpK   = frags_per_K;
        const auto N_cap = N;

        h.parallel_for(
            sycl::nd_range<1>(total_tiles, 1),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                using esimd::simd;

                const std::size_t tid   = it.get_global_id(0);
                const std::size_t tm    = tid / t_N;
                const std::size_t tn    = tid % t_N;
                const std::size_t m_off = tm * KV4_TILE_M;
                const std::size_t n_off = tn * KV4_TILE_N;

                /* Cross-chunk FP32 accumulator (8 rows × 16 cols). */
                simd<float, 128> mC_fp32(0.0f);

                for (std::size_t c = 0; c < cpK; ++c) {
                    /* Per-chunk INT32 accumulator (reset per chunk). */
                    simd<int, 128> mC_chunk_i32(0);

                    for (unsigned f = 0; f < KV4_FRAGS_PER_CHUNK; ++f) {
                        const std::size_t frag_k = c * KV4_FRAGS_PER_CHUNK + f;

                        simd<int, 32> A_vnni;
                        A_vnni.copy_from(
                            reinterpret_cast<const int*>(
                                A_packed + (tm * fpK + frag_k) * 32),
                            esimd::element_aligned);

                        simd<int, 64> B_vnni;
                        B_vnni.copy_from(
                            reinterpret_cast<const int*>(
                                B_packed + (tn * fpK + frag_k) * 64),
                            esimd::element_aligned);

                        /* DPAS 3-arg form: accumulate into mC_chunk_i32. */
                        mC_chunk_i32 = xmx::dpas<
                            /*SystolicDepth=*/8, /*RepeatCount=*/8,
                            /*T=*/int, /*CT=*/int,
                            /*BT=*/int, /*AT=*/int,
                            /*BPrec=*/xmx::dpas_argument_type::s2,
                            /*APrec=*/xmx::dpas_argument_type::s2>(
                                mC_chunk_i32, B_vnni, A_vnni);
                    }

                    /* Per-chunk d fold: load 16 d_chunk[n] FP32 values
                     * for this (tn, c) and accumulate into mC_fp32. */
                    simd<float, 16> d_chunk_n;
                    d_chunk_n.copy_from(
                        d_per_n + (tn * cpK + c) * KV4_TILE_N,
                        esimd::element_aligned);

                    for (unsigned m = 0; m < KV4_TILE_M; ++m) {
                        auto row_i32 = mC_chunk_i32.template select<16, 1>(m * 16);
                        simd<float, 16> row_f32 = row_i32;
                        mC_fp32.template select<16, 1>(m * 16) +=
                            row_f32 * d_chunk_n;
                    }
                }

                /* Final: per-row scale + FP32 -> FP16 + store. */
                for (unsigned m = 0; m < KV4_TILE_M; ++m) {
                    const float sa = s_a_dev[m_off + m];
                    simd<float, 16> row_f32 =
                        mC_fp32.template select<16, 1>(m * 16) * sa;
                    simd<sycl::half, 16> row_h = row_f32;
                    /* C_fp16 is uint16_t* but ESIMD prefers half*. The
                     * bit pattern matches (FP16 storage = 2 bytes). */
                    auto* dst = reinterpret_cast<sycl::half*>(
                        C_fp16 + (m_off + m) * N_cap + n_off);
                    row_h.copy_to(dst, esimd::element_aligned);
                }
            });
    }).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "[kv4] sycl::exception during kernel: %s\n", e.what());
        sycl::free(A_packed, q);
        sycl::free(B_packed, q);
        sycl::free(d_per_n,  q);
        sycl::free(s_a_dev,  q);
        throw;
    }

    /* 4. Free staging USM. */
    sycl::free(A_packed, q);
    sycl::free(B_packed, q);
    sycl::free(d_per_n,  q);
    sycl::free(s_a_dev,  q);
    (void)kv4_fp32_to_fp16_bits;  /* unused for now; kept for fallback. */
}

/* -------------------------------------------------------------------- */
/* run_kernel_v4_aint4 -- INT4 activation path (Phase 1 v4 PRIMARY)     */
/* -------------------------------------------------------------------- *
 * Same outer structure as run_kernel_v4 (above) but uses INT4 helpers
 * and DPAS <s4, s4> for both operands. Dimensions per fragment:
 *   A: simd<int, 64>  (vs 32 for INT2)
 *   B: simd<int, 128> (vs 64 for INT2)
 *   C: simd<int, 128> (same -- M*N)
 */
void run_kernel_v4_aint4(sycl_queue_handle& q_handle,
                         std::size_t M, std::size_t N, std::size_t K,
                         const std::uint16_t* A_fp16,
                         const bitnet_arc_tq2_0_block* B_blocks,
                         std::uint16_t* C_fp16)
{
    assert(K % KV4_K_CHUNK == 0);
    assert(M % KV4_TILE_M  == 0);
    assert(N % KV4_TILE_N  == 0);

    const std::size_t tiles_M       = M / KV4_TILE_M;
    const std::size_t tiles_N       = N / KV4_TILE_N;
    const std::size_t total_tiles   = tiles_M * tiles_N;
    const std::size_t chunks_per_K  = K / KV4_K_CHUNK;
    const std::size_t frags_per_K   = K / KV4_TILE_K;

    sycl::queue& q = q_handle.q;

    /* 0. USM device -> host pull (A_fp16 / B_blocks are device pointers). */
    const std::size_t A_size = M * K;
    const std::size_t B_size = (K / 256) * N;
    std::vector<std::uint16_t> A_host(A_size);
    std::vector<bitnet_arc_tq2_0_block> B_host(B_size);
    q.memcpy(A_host.data(), A_fp16,    A_size * sizeof(std::uint16_t)).wait();
    q.memcpy(B_host.data(), B_blocks,  B_size * sizeof(bitnet_arc_tq2_0_block)).wait();

    /* 1. Host pre-pack INT4. */
    std::vector<std::uint32_t> A_packed_h, B_packed_h;
    std::vector<float>         d_per_n_h, s_a_h;
    prepack_host_int4(M, N, K, A_host.data(), B_host.data(),
                      A_packed_h, B_packed_h, d_per_n_h, s_a_h);

    auto A_packed = sycl::malloc_device<std::uint32_t>(A_packed_h.size(), q);
    auto B_packed = sycl::malloc_device<std::uint32_t>(B_packed_h.size(), q);
    auto d_per_n  = sycl::malloc_device<float>        (d_per_n_h.size(),  q);
    auto s_a_dev  = sycl::malloc_device<float>        (s_a_h.size(),      q);
    q.memcpy(A_packed, A_packed_h.data(), A_packed_h.size() * sizeof(std::uint32_t)).wait();
    q.memcpy(B_packed, B_packed_h.data(), B_packed_h.size() * sizeof(std::uint32_t)).wait();
    q.memcpy(d_per_n,  d_per_n_h.data(),  d_per_n_h.size()  * sizeof(float)).wait();
    q.memcpy(s_a_dev,  s_a_h.data(),      s_a_h.size()      * sizeof(float)).wait();

    try {
    q.submit([&](sycl::handler& h) {
        const auto t_N   = tiles_N;
        const auto cpK   = chunks_per_K;
        const auto fpK   = frags_per_K;
        const auto N_cap = N;

        h.parallel_for(
            sycl::nd_range<1>(total_tiles, 1),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                using esimd::simd;

                const std::size_t tid   = it.get_global_id(0);
                const std::size_t tm    = tid / t_N;
                const std::size_t tn    = tid % t_N;
                const std::size_t m_off = tm * KV4_TILE_M;
                const std::size_t n_off = tn * KV4_TILE_N;

                simd<float, 128> mC_fp32(0.0f);

                for (std::size_t c = 0; c < cpK; ++c) {
                    simd<int, 128> mC_chunk_i32(0);

                    for (unsigned f = 0; f < KV4_FRAGS_PER_CHUNK; ++f) {
                        const std::size_t frag_k = c * KV4_FRAGS_PER_CHUNK + f;

                        /* INT4: A = 64 dwords/frag, B = 128 dwords/frag */
                        simd<int, 64> A_vnni;
                        A_vnni.copy_from(
                            reinterpret_cast<const int*>(
                                A_packed + (tm * fpK + frag_k) * 64),
                            esimd::element_aligned);

                        simd<int, 128> B_vnni;
                        B_vnni.copy_from(
                            reinterpret_cast<const int*>(
                                B_packed + (tn * fpK + frag_k) * 128),
                            esimd::element_aligned);

                        mC_chunk_i32 = xmx::dpas<
                            /*SystolicDepth=*/8, /*RepeatCount=*/8,
                            /*T=*/int, /*CT=*/int,
                            /*BT=*/int, /*AT=*/int,
                            /*BPrec=*/xmx::dpas_argument_type::s4,
                            /*APrec=*/xmx::dpas_argument_type::s4>(
                                mC_chunk_i32, B_vnni, A_vnni);
                    }

                    simd<float, 16> d_chunk_n;
                    d_chunk_n.copy_from(
                        d_per_n + (tn * cpK + c) * KV4_TILE_N,
                        esimd::element_aligned);

                    for (unsigned m = 0; m < KV4_TILE_M; ++m) {
                        auto row_i32 = mC_chunk_i32.template select<16, 1>(m * 16);
                        simd<float, 16> row_f32 = row_i32;
                        mC_fp32.template select<16, 1>(m * 16) +=
                            row_f32 * d_chunk_n;
                    }
                }

                /* Final: per-row scale (s_a / 7 already pre-divided in
                 * prepack) + FP32 -> FP16 + store. */
                for (unsigned m = 0; m < KV4_TILE_M; ++m) {
                    const float sa = s_a_dev[m_off + m];
                    simd<float, 16> row_f32 =
                        mC_fp32.template select<16, 1>(m * 16) * sa;
                    simd<sycl::half, 16> row_h = row_f32;
                    auto* dst = reinterpret_cast<sycl::half*>(
                        C_fp16 + (m_off + m) * N_cap + n_off);
                    row_h.copy_to(dst, esimd::element_aligned);
                }
            });
    }).wait_and_throw();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "[kv4-aint4] sycl::exception: %s\n", e.what());
        sycl::free(A_packed, q);
        sycl::free(B_packed, q);
        sycl::free(d_per_n,  q);
        sycl::free(s_a_dev,  q);
        throw;
    }

    sycl::free(A_packed, q);
    sycl::free(B_packed, q);
    sycl::free(d_per_n,  q);
    sycl::free(s_a_dev,  q);
}

/* -------------------------------------------------------------------- */
/* Variant table                                                        */
/* -------------------------------------------------------------------- */

static void kv4_launch_aint2_8x16x64_k256(
    sycl_queue_handle& q,
    std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A,
    const bitnet_arc_tq2_0_block* B,
    std::uint16_t* C)
{
    run_kernel_v4(q, M, N, K, A, B, C);
}

static void kv4_launch_aint4_8x16x64_k256(
    sycl_queue_handle& q,
    std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A,
    const bitnet_arc_tq2_0_block* B,
    std::uint16_t* C)
{
    run_kernel_v4_aint4(q, M, N, K, A, B, C);
}

/* Phase 1 v4 variant table:
 *   [0] = INT4 act PRIMARY (less aggressive quant, W1 PASS expected)
 *   [1] = INT2 act EXPERIMENTAL (max throughput, W1 only on quant-aware oracle)
 */
extern const kv4_variant_desc kv4_variants[] = {
    { KV4_TILE_M, KV4_TILE_N, KV4_TILE_K, /*sg_size=*/16u, KV4_K_CHUNK,
      /*act_int2=*/false,
      "v4_8x16x64_aint4_k256",
      &kv4_launch_aint4_8x16x64_k256 },
    { KV4_TILE_M, KV4_TILE_N, KV4_TILE_K, /*sg_size=*/16u, KV4_K_CHUNK,
      /*act_int2=*/true,
      "v4_8x16x64_aint2_k256",
      &kv4_launch_aint2_8x16x64_k256 },
};

extern const std::size_t kv4_variants_count =
    sizeof(kv4_variants) / sizeof(kv4_variants[0]);

} /* namespace bitnet_arc */
