/*
 * bench/profile_v2.cpp -- Phase 2a profiler for kernel_v2 XMX path.
 *
 * Goal (per docs/design-v2-phase-2.md sec3, ratified 13f188b):
 *   Discriminate between the 4 candidate bottlenecks for the 2.3-2.6x
 *   v0_BL gap on Arc B60 (sub-saturation / barriers / MMA dispatch /
 *   SLM aliasing) per the priority-ranked decision tree at sec4.
 *
 * Methodology -- two-build pattern (per @beta review #79 fold,
 * @sonnet barrier_us nit folded too):
 *
 *   Build (a) FULL : single parallel_for with one SYCL event around
 *     the whole submit. Measures the truthful total time for the
 *     production kernel (no instrumentation bias). Used as the
 *     denominator for `% of total` ratios in the report.
 *
 *   Build (b) SECTION-SPLIT : the kernel is split into 5 sections
 *     (dequant / load_a / mma / store, plus the inferred barrier_us
 *     by subtraction). Each section runs as its own parallel_for;
 *     inter-section state crosses the launch boundary via USM device
 *     scratch buffers (substituting for the production kernel's SLM
 *     local_accessor). This introduces an *additional* global-memory
 *     bias on top of the cross-section-fusion-lost bias documented
 *     in the brief. We therefore report sections as RATIOS, not
 *     absolutes -- the (a) total is the absolute reference.
 *
 *   barrier_us : NOT measurable directly via SYCL events
 *     (sub_group_barrier is a GPU instruction, not a kernel boundary).
 *     Inferred by subtraction: total_(a) - sum(other 4 sections in (b)).
 *     If inferred barrier_us > 40% of total, follow-up VTune /
 *     Level Zero XPU timeline confirmation is recommended (out-of-
 *     scope for Phase 2a day-1 per brief sec3).
 *
 * Output:
 *   - stdout: CSV with per-shape per-section breakdown (build a total,
 *     build b per-section absolute, build b per-section ratio,
 *     inferred barrier_us, occupancy ratio, bandwidth GB/s).
 *   - stderr: per-shape summary + top-1/top-2 bottleneck identification
 *     per priority-ranked decision tree (brief sec4.5).
 *
 * Phase 2a NOT-IN-SCOPE (these belong to Phase 2b conditional on
 * @naskel decode-vs-prefill call):
 *   - Geometry sweep (TILE_M/N variants beyond 16x16). Phase 2b sec5.
 *   - Lecture B switch / SLM wrapper refactor / fragment coalesce.
 *     Phase 2b sec4.X fix application.
 *
 * NB on bias accounting (per @beta review #79 fold #1 + post-ship nit):
 *   The build (b) split kernel is ~10-30% slower than build (a) due to
 *   per-section launch overhead + lost cross-section fusion. The (b)
 *   ratios apply to (a)'s total, not (b)'s sum-of-sections, to keep
 *   the absolute estimates anchored to production-truth.
 *
 *   IMPORTANT: build (b) per-section absolute estimates are
 *   **UPPER BOUNDS**, not point estimates. Three stacked biases:
 *     1. Cross-section fusion lost  (compiler can't fuse adjacent
 *        sections that the production kernel fuses).
 *     2. Global-memory SLM substitute  (inter-section state crosses
 *        kernel boundary via USM device scratch, replacing
 *        local_accessor's role -- adds memory traffic absent in
 *        production).
 *     3. MMA section global-load bias  (per review #2 nit): the
 *        production kernel issues `joint_matrix_load` from SLM
 *        (~4-cycle L1 latency); the split MMA section issues from
 *        USM global, which hits L2 or DRAM. This INFLATES the MMA
 *        section's measured time vs production. Concretely, when
 *        the split says "MMA = 1-4% of t_full", the true production
 *        MMA is even smaller -- this only strengthens the conclusion
 *        that MMA is not the bottleneck, but consumers should not
 *        read the 1-4% as a calibrated production-truth number.
 *   Decision tree priority-rank in brief sec4.5 takes this into
 *   account: top-1 fix is applied to the production kernel, then we
 *   re-profile on build (a), not (b). (b) only identifies which
 *   section to attack first, never validates the fix.
 *
 *   Note on `barrier_inferred` (= t_full - sum(4 sections)): this is
 *   the residual after the 3 biases above eat into t_full. It is NOT
 *   the production sub_group_barrier overhead in isolation (that
 *   overhead is ~14 ns/barrier × 32 barriers/chunk × chunks_per_col =
 *   ~25 us = ~0.5% of t_full at K=14336). The reported `barrier_inf`
 *   ratios (1-10% observed) are an upper-bound residual that absorbs
 *   real barriers + the residue of the 3 biases.
 *
 * Build: bench/Makefile target profile_v2. Same toolchain as sweep_tile
 * and profile_v0_bl (icpx + Arc B60).
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../src/kernel_v2.h"
#include "../oracle/fp16.h"
#include "../oracle/tq2_0.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using bitnet_arc::kernel_v2_config;
using bitnet_arc::kernel_v2_config_default;
using bitnet_arc::run_kernel_v2;
using bitnet_arc::sycl_queue_handle;

/* --- fixture helpers (mirror profile_v0_bl.cpp) ---------------------- */

void gen_ternary_weights(std::vector<int8_t>& w, std::mt19937& rng) {
    constexpr double P_ZERO = 0.45, P_PLUS = 0.275;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& x : w) {
        const double r = u(rng);
        x = (r < P_ZERO) ? 0 : (r < P_ZERO + P_PLUS ? 1 : -1);
    }
}

void gen_fp16_activations(std::vector<uint16_t>& a, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& h : a) h = bitnet_arc_fp32_to_fp16(u(rng));
}

void pack_tq2_0_blocks(const std::vector<int8_t>& ternary_KxN,
                       std::size_t K, std::size_t N,
                       std::vector<bitnet_arc_tq2_0_block>& blocks)
{
    constexpr std::size_t BLK = BITNET_ARC_TQ2_0_BLOCK_SIZE;
    const std::size_t blocks_per_col = K / BLK;
    blocks.assign(blocks_per_col * N, bitnet_arc_tq2_0_block{});
    std::vector<int8_t> col(K);
    std::vector<float>  scales(blocks_per_col, 1.0f);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) col[k] = ternary_KxN[k * N + n];
        bitnet_arc_quantize_row_tq2_0(col.data(),
                                      &blocks[n * blocks_per_col],
                                      K, scales.data());
    }
}

/* RAII guard for USM device allocations. Scope-exit calls
 * sycl::free, so an exception in the kernel path (e.g.
 * q.wait_and_throw on a device error) cannot leak the 3 USM
 * buffers per shape iteration. Per @beta review #83 fold (~10 LOC,
 * no extra dep, no try/catch around the shape loop). */
template<typename T>
struct usm_device_uptr {
    T*           ptr{nullptr};
    sycl::queue* q  {nullptr};
    usm_device_uptr() = default;
    usm_device_uptr(std::size_t n, sycl::queue& queue)
        : ptr(sycl::malloc_device<T>(n, queue)), q(&queue) {}
    ~usm_device_uptr() { if (ptr && q) sycl::free(ptr, *q); }
    usm_device_uptr(const usm_device_uptr&)            = delete;
    usm_device_uptr& operator=(const usm_device_uptr&) = delete;
    usm_device_uptr(usm_device_uptr&& o) noexcept
        : ptr(o.ptr), q(o.q) { o.ptr = nullptr; o.q = nullptr; }
    usm_device_uptr& operator=(usm_device_uptr&& o) noexcept {
        if (this != &o) {
            if (ptr && q) sycl::free(ptr, *q);
            ptr = o.ptr; q = o.q;
            o.ptr = nullptr; o.q = nullptr;
        }
        return *this;
    }
    T* get() const noexcept { return ptr; }
};

struct stats_t { double t_min, t_med, t_mean, t_std, t_p99; };

stats_t summarize(std::vector<double> ts) {
    /* Defensive : empty input would UB on ts.front() and underflow on
     * (n - 1) -> SIZE_MAX -> OOB read. Caught by @beta + @theta on
     * review #83 (fold). The CLI validation (`--timed > 0`) is the
     * primary guard; this is the belt-and-braces fallback. */
    if (ts.empty()) return {};
    std::sort(ts.begin(), ts.end());
    const std::size_t n = ts.size();
    stats_t s{};
    s.t_min = ts.front();
    s.t_med = (n & 1u) ? ts[n / 2]
                       : 0.5 * (ts[n / 2 - 1] + ts[n / 2]);
    s.t_p99 = ts[std::min(n - 1, static_cast<std::size_t>(0.99 * n))];
    double sum = 0.0;
    for (double x : ts) sum += x;
    s.t_mean = sum / n;
    double sq = 0.0;
    for (double x : ts) sq += (x - s.t_mean) * (x - s.t_mean);
    s.t_std = std::sqrt(sq / n);
    return s;
}

/* --- build (a) FULL kernel timing ------------------------------------ */

stats_t time_full_v2(sycl::queue& q, sycl_queue_handle& qh,
                     std::size_t M, std::size_t N, std::size_t K,
                     std::uint16_t* A_d, bitnet_arc_tq2_0_block* B_d,
                     std::uint16_t* C_d,
                     unsigned warmup, unsigned timed)
{
    const auto cfg = kernel_v2_config_default();
    for (unsigned i = 0; i < warmup; ++i)
        run_kernel_v2(qh, M, N, K, A_d, B_d, C_d, cfg);
    q.wait_and_throw();

    std::vector<double> ts;
    ts.reserve(timed);
    for (unsigned i = 0; i < timed; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        run_kernel_v2(qh, M, N, K, A_d, B_d, C_d, cfg);
        q.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return summarize(std::move(ts));
}

/* --- build (b) SECTION-SPLIT kernels --------------------------------- *
 *
 * Each of the 4 sections (dequant / loadA / mma / store) becomes its
 * own parallel_for, timed via SYCL events with enable_profiling. The
 * cross-section state that the production kernel keeps in SLM
 * (A_slab, B_slab) is replaced by USM device scratch persisted across
 * launches.
 *
 * Hardcoded for Phase 1's single registered variant: TILE_M=TILE_N=16,
 * SG_SIZE=16, K_CHUNK=256, FRAG_K=16. Phase 2 sweep would need
 * templatization (out-of-scope for the profiler harness).
 *
 * Bias caveat (see file header §"NB on bias accounting"):
 *   - Cross-section fusion lost (compiler can't fuse adjacent sections).
 *   - Global-memory SLM substitute (extra memory traffic absent in
 *     production).
 *   Per-section absolute numbers are UPPER BOUNDS. Ratios identify
 *   which section to attack; absolute fix-validation runs on build (a).
 *
 * Layout of USM scratch buffers:
 *   a_persist[tiles_total][chunks_per_col][TILE_M][K_CHUNK]
 *      idx = t*chunks*TM*KC + c*TM*KC + m*KC + k
 *   b_persist[tiles_total][chunks_per_col][K_CHUNK][TILE_N]
 *      idx = t*chunks*KC*TN + c*KC*TN + k*TN + n
 *   mc_persist[tiles_total][TILE_M][TILE_N]  (FP32 accumulator output)
 *      idx = t*TM*TN + m*TN + n
 */

namespace mx = sycl::ext::oneapi::experimental::matrix;

/* FP16 <-> FP32 helpers (mirrored from kernel_v2.cpp; pure host/device
 * scalar code). */
static inline float kvp_fp16_to_fp32(std::uint16_t h) {
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
            bits = (sign << 31) | ((std::uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline std::uint16_t kvp_fp32_to_fp16(float f) {
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

/* Section 1: cooperative TQ2_0 -> FP16 dequant. Fills b_persist. */
sycl::event kv2_split_section_dequant(
    sycl::queue& q, std::size_t M, std::size_t N, std::size_t K,
    const bitnet_arc_tq2_0_block* B_blocks,
    std::uint16_t* b_persist)
{
    constexpr unsigned TILE_M = 16, TILE_N = 16, SG_SIZE = 16, K_CHUNK = 256;
    constexpr unsigned FRAG_K = 16;
    const std::size_t tiles_M = M / TILE_M;
    const std::size_t tiles_N = N / TILE_N;
    const std::size_t total_tiles = tiles_M * tiles_N;
    const std::size_t blocks_per_col = K / 256;
    const std::size_t chunks_per_col = K / K_CHUNK;

    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<std::uint8_t, 1> qs_local(sycl::range<1>(64), h);
        const sycl::range<1> global_range(total_tiles * SG_SIZE);
        const sycl::range<1> local_range(SG_SIZE);
        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const unsigned lane = static_cast<unsigned>(sg.get_local_linear_id());
                const std::size_t tile_id = it.get_group(0);
                const std::size_t tile_n = tile_id % tiles_N;
                const std::size_t n_group = tile_n * TILE_N;

                for (std::size_t c = 0; c < chunks_per_col; ++c) {
                    const std::size_t k_chunk0 = c * (K_CHUNK / 256);
                    const std::size_t b_base =
                        tile_id * chunks_per_col * K_CHUNK * TILE_N
                        + c * K_CHUNK * TILE_N;

                    for (unsigned n_local = 0; n_local < TILE_N; ++n_local) {
                        const std::size_t blk_idx =
                            (n_group + n_local) * blocks_per_col + k_chunk0;
                        const bitnet_arc_tq2_0_block& blk = B_blocks[blk_idx];

                        for (unsigned b = 0; b < 4u; ++b) {
                            qs_local[lane * 4u + b] = blk.qs[lane * 4u + b];
                        }
                        const float d_f = kvp_fp16_to_fp32(blk.d);
                        sycl::group_barrier(sg);

                        for (unsigned k_off = 0; k_off < FRAG_K; ++k_off) {
                            const unsigned k = lane * FRAG_K + k_off;
                            const std::size_t byte =
                                static_cast<std::size_t>((k >> 7) * 32u + (k & 31u));
                            const unsigned shift =
                                static_cast<unsigned>(((k >> 5) & 3u) * 2u);
                            const std::uint8_t code = (qs_local[byte] >> shift) & 0x3u;
                            const int s = static_cast<int>(code) - 1;
                            const float w = static_cast<float>(s) * d_f;
                            b_persist[b_base + (std::size_t)k * TILE_N + n_local]
                                = kvp_fp32_to_fp16(w);
                        }
                        sycl::group_barrier(sg);
                    }
                }
            });
    });
}

/* Section 2: cooperative A SLM load -> a_persist (USM scratch). */
sycl::event kv2_split_section_loada(
    sycl::queue& q, std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A_fp16,
    std::uint16_t* a_persist)
{
    constexpr unsigned TILE_M = 16, TILE_N = 16, SG_SIZE = 16, K_CHUNK = 256;
    const std::size_t tiles_M = M / TILE_M;
    const std::size_t tiles_N = N / TILE_N;
    const std::size_t total_tiles = tiles_M * tiles_N;
    const std::size_t chunks_per_col = K / K_CHUNK;
    constexpr std::size_t A_SLAB_ELEMS = TILE_M * K_CHUNK;

    return q.submit([&](sycl::handler& h) {
        const sycl::range<1> global_range(total_tiles * SG_SIZE);
        const sycl::range<1> local_range(SG_SIZE);
        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const unsigned lane = static_cast<unsigned>(sg.get_local_linear_id());
                const std::size_t tile_id = it.get_group(0);
                const std::size_t tile_m = tile_id / tiles_N;
                const std::size_t m_group = tile_m * TILE_M;

                for (std::size_t c = 0; c < chunks_per_col; ++c) {
                    const std::size_t k0 = c * K_CHUNK;
                    const std::size_t a_base =
                        tile_id * chunks_per_col * A_SLAB_ELEMS + c * A_SLAB_ELEMS;
                    for (std::size_t idx = lane; idx < A_SLAB_ELEMS; idx += SG_SIZE) {
                        const std::size_t m_idx = idx / K_CHUNK;
                        const std::size_t k_off = idx % K_CHUNK;
                        a_persist[a_base + idx] =
                            A_fp16[(m_group + m_idx) * K + k0 + k_off];
                    }
                }
            });
    });
}

/* Section 3: inner MMA loop. Reads a_persist + b_persist, accumulates
 * mC FP32 in registers across all chunks, then stores final mC to
 * mc_persist. The MMA section keeps mC in registers (matching the full
 * kernel pattern) but pays a one-shot mC->mc_persist write at end. */
sycl::event kv2_split_section_mma(
    sycl::queue& q, std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* a_persist,
    const std::uint16_t* b_persist,
    float* mc_persist)
{
    constexpr unsigned TILE_M = 16, TILE_N = 16, SG_SIZE = 16, K_CHUNK = 256;
    constexpr unsigned FRAG_K = 16;
    constexpr unsigned FRAGS_PER_CHUNK = K_CHUNK / FRAG_K;
    constexpr std::size_t A_SLAB_ELEMS = TILE_M * K_CHUNK;
    constexpr std::size_t B_SLAB_ELEMS = K_CHUNK * TILE_N;
    const std::size_t tiles_M = M / TILE_M;
    const std::size_t tiles_N = N / TILE_N;
    const std::size_t total_tiles = tiles_M * tiles_N;
    const std::size_t chunks_per_col = K / K_CHUNK;

    return q.submit([&](sycl::handler& h) {
        const sycl::range<1> global_range(total_tiles * SG_SIZE);
        const sycl::range<1> local_range(SG_SIZE);
        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const std::size_t tile_id = it.get_group(0);

                mx::joint_matrix<sycl::sub_group, float,
                                 mx::use::accumulator,
                                 TILE_M, TILE_N> mC;
                mx::joint_matrix_fill(sg, mC, 0.0f);

                mx::joint_matrix<sycl::sub_group, sycl::half,
                                 mx::use::a, TILE_M, FRAG_K,
                                 mx::layout::row_major> mA;
                mx::joint_matrix<sycl::sub_group, sycl::half,
                                 mx::use::b, FRAG_K, TILE_N,
                                 mx::layout::row_major> mB;

                for (std::size_t c = 0; c < chunks_per_col; ++c) {
                    const std::size_t a_base =
                        tile_id * chunks_per_col * A_SLAB_ELEMS + c * A_SLAB_ELEMS;
                    const std::size_t b_base =
                        tile_id * chunks_per_col * B_SLAB_ELEMS + c * B_SLAB_ELEMS;
                    for (unsigned k_frag = 0; k_frag < FRAGS_PER_CHUNK; ++k_frag) {
                        auto a_ptr = sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(
                            reinterpret_cast<const sycl::half*>(
                                &a_persist[a_base + k_frag * FRAG_K]));
                        mx::joint_matrix_load(sg, mA, a_ptr, K_CHUNK);

                        auto b_ptr = sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(
                            reinterpret_cast<const sycl::half*>(
                                &b_persist[b_base + k_frag * FRAG_K * TILE_N]));
                        mx::joint_matrix_load(sg, mB, b_ptr, TILE_N);

                        mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    }
                }

                /* Final: store FP32 mC -> mc_persist (one shot per tile,
                 * negligible vs the 16*chunks_per_col MMA ops above). */
                auto c_ptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(
                    &mc_persist[tile_id * TILE_M * TILE_N]);
                mx::joint_matrix_store(sg, mC, c_ptr, TILE_N,
                                       mx::layout::row_major);
            });
    });
}

/* Section 4: final FP32 -> FP16 store. Reads mc_persist, scatters FP16
 * to global C_fp16. Same lane-per-row pattern as the production kernel. */
sycl::event kv2_split_section_store(
    sycl::queue& q, std::size_t M, std::size_t N, std::size_t /*K*/,
    const float* mc_persist,
    std::uint16_t* C_fp16)
{
    constexpr unsigned TILE_M = 16, TILE_N = 16, SG_SIZE = 16;
    const std::size_t tiles_M = M / TILE_M;
    const std::size_t tiles_N = N / TILE_N;
    const std::size_t total_tiles = tiles_M * tiles_N;

    return q.submit([&](sycl::handler& h) {
        const sycl::range<1> global_range(total_tiles * SG_SIZE);
        const sycl::range<1> local_range(SG_SIZE);
        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const unsigned lane = static_cast<unsigned>(sg.get_local_linear_id());
                const std::size_t tile_id = it.get_group(0);
                const std::size_t tile_m = tile_id / tiles_N;
                const std::size_t tile_n = tile_id % tiles_N;
                const std::size_t m_group = tile_m * TILE_M;
                const std::size_t n_group = tile_n * TILE_N;

                const std::size_t row = m_group + lane;
                const std::size_t mc_base = tile_id * TILE_M * TILE_N;
                for (unsigned col = 0; col < TILE_N; ++col) {
                    const float v = mc_persist[mc_base + lane * TILE_N + col];
                    C_fp16[row * N + n_group + col] = kvp_fp32_to_fp16(v);
                }
            });
    });
}

/* Compute event execution time in ms via profiling info. Excludes
 * launch/dispatch latency by design (uses kernel command_start /
 * command_end timestamps from the device, not host wall-clock). */
double event_ms(sycl::event e) {
    e.wait();
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t1 - t0) / 1.0e6;  /* ns -> ms */
}

struct split_stats_t {
    stats_t dequant, loada, mma, store;
    stats_t total_split;       /* sum of 4 sections per timed iter */
    /* barrier_us inferred by subtraction is computed in the caller
     * (needs t_full from build (a)), not here. */
};

split_stats_t time_split_v2_sections(
    sycl::queue& q, std::size_t M, std::size_t N, std::size_t K,
    const std::uint16_t* A_d, const bitnet_arc_tq2_0_block* B_d,
    std::uint16_t* C_d,
    unsigned warmup, unsigned timed)
{
    constexpr unsigned TILE_M = 16, TILE_N = 16, K_CHUNK = 256;
    const std::size_t tiles_M = M / TILE_M;
    const std::size_t tiles_N = N / TILE_N;
    const std::size_t total_tiles = tiles_M * tiles_N;
    const std::size_t chunks_per_col = K / K_CHUNK;

    /* Allocate persist buffers (USM device scratch). Reused across all
     * warmup + timed iterations. */
    const std::size_t a_persist_n =
        total_tiles * chunks_per_col * TILE_M * K_CHUNK;
    const std::size_t b_persist_n =
        total_tiles * chunks_per_col * K_CHUNK * TILE_N;
    const std::size_t mc_persist_n =
        total_tiles * TILE_M * TILE_N;

    usm_device_uptr<std::uint16_t> a_persist(a_persist_n, q);
    usm_device_uptr<std::uint16_t> b_persist(b_persist_n, q);
    usm_device_uptr<float>         mc_persist(mc_persist_n, q);

    /* Warmup */
    for (unsigned i = 0; i < warmup; ++i) {
        kv2_split_section_dequant(q, M, N, K, B_d, b_persist.get());
        kv2_split_section_loada  (q, M, N, K, A_d, a_persist.get());
        kv2_split_section_mma    (q, M, N, K, a_persist.get(),
                                  b_persist.get(), mc_persist.get());
        kv2_split_section_store  (q, M, N, K, mc_persist.get(), C_d);
    }
    q.wait_and_throw();

    std::vector<double> td, tl, tm, ts, tt;
    td.reserve(timed); tl.reserve(timed);
    tm.reserve(timed); ts.reserve(timed); tt.reserve(timed);

    for (unsigned i = 0; i < timed; ++i) {
        auto e1 = kv2_split_section_dequant(q, M, N, K, B_d, b_persist.get());
        auto e2 = kv2_split_section_loada  (q, M, N, K, A_d, a_persist.get());
        auto e3 = kv2_split_section_mma    (q, M, N, K, a_persist.get(),
                                            b_persist.get(), mc_persist.get());
        auto e4 = kv2_split_section_store  (q, M, N, K, mc_persist.get(), C_d);
        q.wait_and_throw();
        const double dq = event_ms(e1);
        const double la = event_ms(e2);
        const double mm = event_ms(e3);
        const double st = event_ms(e4);
        td.push_back(dq); tl.push_back(la); tm.push_back(mm); ts.push_back(st);
        tt.push_back(dq + la + mm + st);
    }

    return split_stats_t{
        summarize(std::move(td)),
        summarize(std::move(tl)),
        summarize(std::move(tm)),
        summarize(std::move(ts)),
        summarize(std::move(tt))
    };
}

/* --- bytes / bandwidth helper (same as v0_BL profiler) --------------- */

double bytes_for(std::size_t M, std::size_t N, std::size_t K) {
    const double w = double(N) * (double(K) / 256.0)
                   * double(sizeof(bitnet_arc_tq2_0_block));
    const double a = double(M) * double(K) * sizeof(std::uint16_t);
    const double o = double(M) * double(N) * sizeof(std::uint16_t);
    return w + a + o;
}

/* --- occupancy probe (per brief sec3 add-meas #1) -------------------- */

struct occupancy_t {
    std::size_t wgs_launched;       /* (M / 16) * (N / 16) */
    std::size_t theoretical_max;    /* device max-CUs * threads-per-CU est */
    double      ratio;              /* launched / theoretical_max */
};

occupancy_t probe_occupancy(sycl::queue& q,
                            std::size_t M, std::size_t N, std::size_t /*K*/)
{
    occupancy_t o{};
    o.wgs_launched = (M / 16) * (N / 16);
    /* Arc B60 = 160 XMX engines per design v2 sec1. For occupancy
     * estimation we use compute_units * recommended-WGs-per-CU; the
     * SYCL portable info::device::max_compute_units gives CUs, and
     * recommend a coarse 8 WGs/CU (typical Xe2 occupancy ceiling for
     * SLM-heavy kernels with kv2's 16 KB SLM/WG). Refined estimates
     * via Level Zero ze_device_compute_properties_t are out-of-scope
     * for Phase 2a day-1. */
    const auto cu = q.get_device().get_info<sycl::info::device::max_compute_units>();
    o.theoretical_max = static_cast<std::size_t>(cu) * 8u;
    o.ratio = o.theoretical_max > 0
              ? double(o.wgs_launched) / double(o.theoretical_max)
              : 0.0;
    return o;
}

} /* anonymous namespace */

/* -------------------------------------------------------------------- */
/* main                                                                 */
/* -------------------------------------------------------------------- */

int main(int argc, char** argv) {
    /* Phase 2a target shapes per brief sec3 + sec7. The purpose
     * column is reported in the CSV row + stderr per-shape summary
     * so the bottleneck identification step can cross-reference
     * shape -> hypothesis directly. */
    struct shape_t { std::size_t M, N, K; const char* tag; const char* purpose; };
    std::vector<shape_t> shapes = {
        {  16,  16,   256, "smoke_16x16x256",    "smoke_single_block"          },
        {  16,  64,  4096, "attn_16x64x4096",    "attn_projection"             },
        {  16,  64, 14336, "ffn_16x64x14336",    "ffn_headline_W2_gate"        },
        {  64,  64, 14336, "ffn_64x64x14336",    "ffn_saturated_W2_gate"       },
        {  64, 256, 14336, "subsat_64x256x14336","sub_saturation_isolation"    },
    };

    unsigned warmup = 5, timed = 50;
    std::uint32_t seed = 1337;
    bool split_build = false; /* (b) section-split mode flag */

    /* Parse a positive-integer CLI value. Returns false (and prints
     * an error) on empty / non-numeric / non-positive input -- this
     * blocks the misfires caught by @beta + @theta on review #83 :
     * `--timed 0` would crash summarize(); `--warmup -1` would wrap
     * to UINT_MAX and burn the GPU for ~4e9 iterations. */
    auto parse_pos_uint = [](const char* s, const char* name,
                             unsigned& dst) -> bool {
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        if (end == s || end == nullptr || *end != '\0' || v <= 0) {
            std::fprintf(stderr,
                "error: %s requires a positive integer (got '%s')\n",
                name, s);
            return false;
        }
        dst = static_cast<unsigned>(v);
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--warmup" && i + 1 < argc) {
            if (!parse_pos_uint(argv[++i], "--warmup", warmup)) return 1;
        }
        else if (a == "--timed"  && i + 1 < argc) {
            if (!parse_pos_uint(argv[++i], "--timed",  timed))  return 1;
        }
        else if (a == "--seed"   && i + 1 < argc) {
            /* Seed accepts any uint32 incl. 0 ; non-numeric is the
             * only error path (no positivity constraint). */
            char* end = nullptr;
            const long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0) {
                std::fprintf(stderr,
                    "error: --seed requires a non-negative integer\n");
                return 1;
            }
            seed = static_cast<std::uint32_t>(v);
        }
        else if (a == "--split-build")            split_build = true;
        else if (a == "--help") {
            std::printf(
                "usage: %s [--warmup N] [--timed N] [--seed N]\n"
                "          [--split-build]\n"
                "  default: 5 warmup + 50 timed iters per shape, seed=1337,\n"
                "           build (a) FULL only.\n"
                "  --split-build : enable build (b) section-split, generate\n"
                "                  per-section ratios. ETA Phase 2a step 2\n"
                "                  (skeleton + global-mem SLM substitute).\n",
                argv[0]);
            return 0;
        }
    }

    sycl::queue q(sycl::default_selector_v,
                  sycl::property_list{
                      sycl::property::queue::in_order{},
                      sycl::property::queue::enable_profiling{}});
    sycl_queue_handle qh(q);
    std::fprintf(stderr, "device : %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(stderr, "shapes : %zu (W1 + 1 large)\n", shapes.size());
    std::fprintf(stderr, "build  : (a) FULL%s\n",
                 split_build ? " + (b) SECTION-SPLIT" : "");
    std::fprintf(stderr, "iters  : %u warmup + %u timed per shape\n\n",
                 warmup, timed);

    /* CSV header. Step 1 columns + step 2 split-build per-section
     * columns (NaN when --split-build is not enabled). The split-build
     * ratios `r_*` apply to t_full_med (build a) per brief sec3 -- the
     * production-truth absolute, not sum-of-(b)-sections. */
    std::printf("shape,purpose,M,N,K,t_full_min,t_full_med,t_full_mean,t_full_std,"
                "t_full_p99,bytes,bandwidth_gbs,wgs,occ_max,occ_ratio,"
                "t_dequant_med,t_loada_med,t_mma_med,t_store_med,"
                "t_split_total_med,t_barrier_inferred,"
                "r_dequant,r_loada,r_mma,r_store,r_barrier\n");

    for (const auto& s : shapes) {
        if (s.K % 256 != 0) {
            std::fprintf(stderr, "skip %s : K=%zu not %% 256\n",
                         s.tag, s.K);
            continue;
        }

        std::mt19937 rng(seed);
        std::vector<int8_t>   tern(s.K * s.N);
        std::vector<uint16_t> A(s.M * s.K);
        gen_ternary_weights(tern, rng);
        gen_fp16_activations(A, rng);
        std::vector<bitnet_arc_tq2_0_block> B;
        pack_tq2_0_blocks(tern, s.K, s.N, B);

        /* USM device allocations via RAII guard (review #83 fold) :
         * scope-exit frees automatically, so a SYCL exception in the
         * kernel path can no longer leak the buffers. */
        usm_device_uptr<uint16_t>                  A_d(s.M * s.K, q);
        usm_device_uptr<bitnet_arc_tq2_0_block>    B_d(B.size(),  q);
        usm_device_uptr<uint16_t>                  C_d(s.M * s.N, q);
        q.memcpy(A_d.get(), A.data(), s.M * s.K * sizeof(uint16_t)).wait();
        q.memcpy(B_d.get(), B.data(), B.size() * sizeof(B[0])).wait();

        /* --- build (a) FULL ------------------------------------------ */
        const stats_t full = time_full_v2(q, qh, s.M, s.N, s.K,
                                          A_d.get(), B_d.get(), C_d.get(),
                                          warmup, timed);
        const double bytes = bytes_for(s.M, s.N, s.K);
        const double gbs   = (bytes / 1e9) / (full.t_med / 1e3);
        const occupancy_t occ = probe_occupancy(q, s.M, s.N, s.K);

        /* --- build (b) SECTION-SPLIT (Phase 2a step 2) -------------- */
        double t_dq = std::nan(""), t_la = std::nan(""),
               t_mm = std::nan(""), t_st = std::nan("");
        double t_split_total = std::nan(""), t_barrier = std::nan("");
        double r_dq = std::nan(""), r_la = std::nan(""), r_mm = std::nan(""),
               r_st = std::nan(""), r_br = std::nan("");
        if (split_build) {
            const split_stats_t sp = time_split_v2_sections(
                q, s.M, s.N, s.K,
                A_d.get(), B_d.get(), C_d.get(),
                warmup, timed);
            t_dq = sp.dequant.t_med;
            t_la = sp.loada.t_med;
            t_mm = sp.mma.t_med;
            t_st = sp.store.t_med;
            t_split_total = sp.total_split.t_med;
            /* Barrier time = t_full - sum_of_4_sections. Negative or
             * tiny values mean the build (b) bias (fusion lost +
             * global-mem SLM substitute) exceeds the production
             * barrier overhead -- in that regime the inferred number
             * is degenerate and the ratios should be read directly
             * (not via barrier subtraction). */
            t_barrier = full.t_med - t_split_total;
            if (full.t_med > 0.0) {
                r_dq = t_dq / full.t_med;
                r_la = t_la / full.t_med;
                r_mm = t_mm / full.t_med;
                r_st = t_st / full.t_med;
                r_br = t_barrier / full.t_med;
            }
        }

        std::printf("%s,%s,%zu,%zu,%zu,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,"
                    "%.0f,%.2f,%zu,%zu,%.4f,"
                    "%.5f,%.5f,%.5f,%.5f,"
                    "%.5f,%.5f,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    s.tag, s.purpose, s.M, s.N, s.K,
                    full.t_min, full.t_med, full.t_mean, full.t_std, full.t_p99,
                    bytes, gbs,
                    occ.wgs_launched, occ.theoretical_max, occ.ratio,
                    t_dq, t_la, t_mm, t_st,
                    t_split_total, t_barrier,
                    r_dq, r_la, r_mm, r_st, r_br);
        std::fflush(stdout);
        std::fprintf(stderr,
            "  %-25s [%s] : t_med=%.3f ms, %.1f GB/s, "
            "WGs=%zu/%zu (occ=%.3f)\n",
            s.tag, s.purpose, full.t_med, gbs,
            occ.wgs_launched, occ.theoretical_max, occ.ratio);
        if (split_build) {
            std::fprintf(stderr,
                "    split: dequant=%.3f (%.1f%%), loadA=%.3f (%.1f%%), "
                "mma=%.3f (%.1f%%), store=%.3f (%.1f%%), barrier_inf=%.3f (%.1f%%)\n",
                t_dq, 100.0 * r_dq,
                t_la, 100.0 * r_la,
                t_mm, 100.0 * r_mm,
                t_st, 100.0 * r_st,
                t_barrier, 100.0 * r_br);
            /* Top-1 / top-2 bottleneck identification per brief sec4.5.
             * `barrier_inferred` is included in the ranking only when
             * positive (negative = degenerate, build (b) bias dominates). */
            struct sec_t { const char* name; double r; };
            sec_t secs[] = {
                {"dequant", r_dq}, {"loadA", r_la},
                {"mma", r_mm},     {"store", r_st},
                {"barrier", r_br > 0.0 ? r_br : -1.0},
            };
            int top1 = 0, top2 = 1;
            if (secs[1].r > secs[0].r) { top1 = 1; top2 = 0; }
            for (int i = 2; i < 5; ++i) {
                if (secs[i].r > secs[top1].r) { top2 = top1; top1 = i; }
                else if (secs[i].r > secs[top2].r) { top2 = i; }
            }
            std::fprintf(stderr,
                "    top-1: %s (%.1f%% of t_full)  top-2: %s (%.1f%%)\n",
                secs[top1].name, 100.0 * secs[top1].r,
                secs[top2].name, 100.0 * secs[top2].r);
            if (secs[top1].r < 0.50) {
                std::fprintf(stderr,
                    "    NOTE: top-1 < 50%%, likely composite bottleneck "
                    "(brief sec4.5 falsification path applies if Phase 2b fix fails)\n");
            }
        }

        /* A_d / B_d / C_d auto-freed at scope exit (RAII). */
    }

    /* --- summary + bottleneck call (per brief sec4.5) ---------------- */
    if (split_build) {
        std::fprintf(stderr,
            "\ndone. Phase 2a step 2 (build b SECTION-SPLIT) complete.\n"
            "  Per-shape top-1 / top-2 printed inline above.\n"
            "  Next: cross-tab against profile_v0_bl.csv (run profile_v0_bl\n"
            "  on same shapes for v0_BL comparison) and generate\n"
            "  bench/profile_v2_p2a.md report per brief sec4.5.\n");
    } else {
        std::fprintf(stderr,
            "\ndone. Phase 2a step 1 (build a FULL) complete. Next:\n"
            "  - run with --split-build for per-section ratios (step 2)\n"
            "  - cross-tab against profile_v0_bl.csv\n"
            "  - generate bench/profile_v2_p2a.md report\n");
    }
    std::fprintf(stderr,
        "\nNB methodology bias (per brief sec3 + this file's header):\n"
        "  build (b) ratios apply to build (a) total, not sum-of-(b)\n"
        "  sections. Cross-section fusion lost + global-memory SLM\n"
        "  substitute = upper-bound per-section estimates only.\n"
        "  Confirmation via VTune / Level Zero XPU timeline if any\n"
        "  inferred barrier_us > 40%% of total.\n");
    return 0;
}
