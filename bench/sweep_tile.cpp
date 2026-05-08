/*
 * bench/sweep_tile.cpp -- bitnet-arc #148 tile sweep harness.
 *
 * Iterates the variants registered in src/kernel_v0.cpp's kv0_variants[]
 * table, runs each on a fixed shape with profiling enabled, checks the
 * output against an FP32 reference (direct dequant + maison fp32_matmul,
 * matching voice 1 of bench/harness_3way.cpp), and emits a CSV row per
 * variant on stdout.
 *
 * Smoke shape: M=16, N=64, K=14336 (LLaMA-8B FFN intermediate dim).
 * Override via --M / --N / --K. The shape must satisfy:
 *   - K % 256 == 0
 *   - M % tile_M == 0  (per variant; incompatible variants are skipped)
 *   - N % tile_N == 0
 *
 * Build: bench/Makefile target sweep_tile (links libbitnet_arc_v0.a +
 * liboracle.a). Requires icpx (or clang++ with SYCL2020 plugin) at run
 * time on a host with a SYCL-visible device (Arc Pro B60 verified).
 *
 * Tolerance: SYCL output is compared against the FP32 reference using
 * BITNET_ARC_TOL_SYCL_VS_FP32REF from oracle/tolerance.h. The tighter
 * 1e-3 gate (FP32REF_VS_NUMPY_F64) is not enforced here -- the SYCL
 * kernel rounds to FP16 once at the end, so its accuracy budget is set
 * by the FP16 mantissa (~1e-3 relative on the layer output).
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../oracle/fp16.h"
#include "../oracle/tq2_0.h"
#include "../oracle/fp32_matmul.h"
#include "../oracle/tolerance.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using bitnet_arc::kernel_variant_desc;
using bitnet_arc::kernel_v0_inner_mode;
using bitnet_arc::kv0_variants;
using bitnet_arc::kv0_variants_count;
using bitnet_arc::sycl_queue_handle;

/* --- input generation ------------------------------------------------ */

void gen_ternary_weights(std::vector<int8_t>& w, std::mt19937& rng) {
    constexpr double P_ZERO = 0.45;
    constexpr double P_PLUS = 0.275;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& x : w) {
        const double r = u(rng);
        if (r < P_ZERO)               x =  0;
        else if (r < P_ZERO + P_PLUS) x =  1;
        else                          x = -1;
    }
}

void gen_fp16_activations(std::vector<uint16_t>& a, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& h : a) {
        h = bitnet_arc_fp32_to_fp16(u(rng));
    }
}

/* --- TQ2_0 layout helpers ------------------------------------------- */

/* Quantize ternary weights laid out row-major (K, N) into the kernel's
 * expected B_blocks layout: per column n, K/256 contiguous blocks. */
void pack_tq2_0_blocks(const std::vector<int8_t>&            ternary_KxN,
                       std::size_t                           K,
                       std::size_t                           N,
                       std::vector<bitnet_arc_tq2_0_block>&  blocks)
{
    constexpr std::size_t BLOCK = BITNET_ARC_TQ2_0_BLOCK_SIZE;
    assert(K % BLOCK == 0);
    const std::size_t blocks_per_col = K / BLOCK;
    blocks.assign(blocks_per_col * N, bitnet_arc_tq2_0_block{});

    std::vector<int8_t> col(K);
    std::vector<float>  scales(blocks_per_col, 1.0f);

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) {
            col[k] = ternary_KxN[k * N + n];
        }
        bitnet_arc_quantize_row_tq2_0(col.data(),
                                      &blocks[n * blocks_per_col],
                                      K,
                                      scales.data());
    }
}

/* --- FP32 reference (matches harness_3way voice 1) ------------------- */

void compute_fp32_ref(std::vector<float>&        C_ref,
                      const std::vector<uint16_t>& A_fp16,
                      const std::vector<int8_t>& ternary_KxN,
                      std::size_t                M,
                      std::size_t                N,
                      std::size_t                K)
{
    /* Direct-dequant: ternary -> float (scale=1, weights are exactly
     * {-1, 0, +1} and the FP16 round-trip in TQ2_0 is exact at amax=1). */
    std::vector<float> B_dequant(K * N);
    for (std::size_t i = 0; i < K * N; ++i) {
        B_dequant[i] = static_cast<float>(ternary_KxN[i]);
    }
    C_ref.assign(M * N, 0.0f);
    bitnet_arc_oracle_fp32_matmul(M, N, K,
                                  A_fp16.data(),
                                  B_dequant.data(),
                                  C_ref.data());
}

/* --- correctness gate ------------------------------------------------ */

struct check_result {
    double      max_rel_err;
    std::size_t over_threshold;
    bool        ok;
};

/* Compares the kernel output (FP16 -> FP32) against the FP32 reference
 * using a relative tolerance. Single-pass, bounded memory. */
check_result check_kernel_output(const std::vector<uint16_t>& C_kernel_fp16,
                                 const std::vector<float>&    C_ref_fp32,
                                 double tolerance)
{
    assert(C_kernel_fp16.size() == C_ref_fp32.size());
    constexpr double EPS = 1e-30;
    check_result r{0.0, 0, false};
    for (std::size_t i = 0; i < C_kernel_fp16.size(); ++i) {
        const double a = static_cast<double>(
            bitnet_arc_fp16_to_fp32(C_kernel_fp16[i]));
        const double b = static_cast<double>(C_ref_fp32[i]);
        const double denom = std::max({std::fabs(a), std::fabs(b), EPS});
        const double rel   = std::fabs(a - b) / denom;
        if (rel > r.max_rel_err) r.max_rel_err = rel;
        if (rel > tolerance)     ++r.over_threshold;
    }
    r.ok = (r.over_threshold == 0);
    return r;
}

/* --- timing helper --------------------------------------------------- */

/* Use the SYCL profiling clock if available, else wall-clock. The Arc
 * runtime supports profiling on Level Zero, so we prefer events. */
double median_ms(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();
    if (n == 0) return 0.0;
    if (n & 1u) return times[n / 2];
    return 0.5 * (times[n / 2 - 1] + times[n / 2]);
}

/* --- one variant run ------------------------------------------------- */

struct run_stats {
    double      time_ms_med;
    double      time_ms_min;
    double      bandwidth_gbs;
    bool        correct;
    double      max_rel_err;
    std::size_t over_threshold;
    bool        ran;            /* false if the variant was skipped */
    const char* skip_reason;    /* set when ran == false */
};

run_stats run_variant(sycl::queue&                          q,
                      sycl_queue_handle&                    q_handle,
                      const kernel_variant_desc&            v,
                      std::size_t                           M,
                      std::size_t                           N,
                      std::size_t                           K,
                      const std::uint16_t*                  A_fp16_dev,
                      const bitnet_arc_tq2_0_block*         B_blocks_dev,
                      std::uint16_t*                        C_fp16_dev,
                      const std::vector<float>&             C_ref_fp32,
                      std::vector<std::uint16_t>&           C_host_buf,
                      unsigned                              warmup,
                      unsigned                              timed)
{
    run_stats s{0.0, 0.0, 0.0, false, 0.0, 0, false, nullptr};

    if (M % v.tile_M != 0) { s.skip_reason = "M not multiple of tile_M"; return s; }
    if (N % v.tile_N != 0) { s.skip_reason = "N not multiple of tile_N"; return s; }

    /* Warm-up. */
    for (unsigned i = 0; i < warmup; ++i) {
        v.launch(q_handle, M, N, K,
                 A_fp16_dev, B_blocks_dev, C_fp16_dev);
    }
    q.wait_and_throw();

    /* Timed runs. We measure host-side wall clock around q.wait() per
     * launch -- on Level Zero this matches the device time within a
     * micros-level enqueue overhead, which is well below the kernel
     * times we expect at K=14336. */
    std::vector<double> times_ms;
    times_ms.reserve(timed);
    for (unsigned i = 0; i < timed; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        v.launch(q_handle, M, N, K,
                 A_fp16_dev, B_blocks_dev, C_fp16_dev);
        q.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    /* Read output back for correctness check. */
    q.memcpy(C_host_buf.data(), C_fp16_dev,
             M * N * sizeof(std::uint16_t)).wait();

    const check_result cr = check_kernel_output(
        C_host_buf, C_ref_fp32,
        static_cast<double>(BITNET_ARC_TOL_SYCL_VS_FP32REF));

    s.time_ms_med    = median_ms(times_ms);
    s.time_ms_min    = *std::min_element(times_ms.begin(), times_ms.end());
    /* Bandwidth model: each kernel invocation reads
     *   - weights : N * (K/256) * sizeof(tq2_0_block)  bytes (TQ2_0)
     *   - acts    : M * K * sizeof(uint16_t)           bytes (FP16)
     *   - writes  : M * N * sizeof(uint16_t)           bytes (FP16 out)
     * The (M, N) writes are negligible vs the weights; we include them
     * for completeness but the dominant term at K=14336 is weights. */
    const double w_bytes = static_cast<double>(N) *
                           (static_cast<double>(K) / 256.0) *
                           static_cast<double>(sizeof(bitnet_arc_tq2_0_block));
    const double a_bytes = static_cast<double>(M) * static_cast<double>(K) *
                           sizeof(std::uint16_t);
    const double o_bytes = static_cast<double>(M) * static_cast<double>(N) *
                           sizeof(std::uint16_t);
    const double total_bytes = w_bytes + a_bytes + o_bytes;
    s.bandwidth_gbs = (total_bytes / 1.0e9) / (s.time_ms_med / 1.0e3);

    s.correct        = cr.ok;
    s.max_rel_err    = cr.max_rel_err;
    s.over_threshold = cr.over_threshold;
    s.ran            = true;
    return s;
}

const char* mode_name(kernel_v0_inner_mode m) {
    return m == kernel_v0_inner_mode::BRANCHFUL ? "BRANCHFUL" : "BRANCHLESS";
}

} /* anonymous namespace */

int main(int argc, char** argv) {
    std::size_t  M = 16, N = 64, K = 14336;
    std::uint32_t seed = 1337;
    unsigned     warmup = 3, timed = 10;
    bool         emit_header = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](std::size_t& dst) {
            if (i + 1 < argc) dst = static_cast<std::size_t>(std::atoll(argv[++i]));
        };
        const auto nextu = [&](unsigned& dst) {
            if (i + 1 < argc) dst = static_cast<unsigned>(std::atoi(argv[++i]));
        };
        if      (a == "--M")        next(M);
        else if (a == "--N")        next(N);
        else if (a == "--K")        next(K);
        else if (a == "--seed")     { if (i + 1 < argc) seed = static_cast<std::uint32_t>(std::atoi(argv[++i])); }
        else if (a == "--warmup")   nextu(warmup);
        else if (a == "--timed")    nextu(timed);
        else if (a == "--no-header") emit_header = false;
        else if (a == "--help") {
            std::printf(
                "usage: %s [--M N] [--N N] [--K N] [--seed N] "
                "[--warmup N] [--timed N] [--no-header]\n",
                argv[0]);
            return 0;
        }
    }

    if (K % 256 != 0) {
        std::fprintf(stderr, "K=%zu must be a multiple of 256\n", K);
        return 2;
    }

    std::mt19937 rng(seed);

    /* --- host fixtures --- */
    std::vector<int8_t>   ternary(K * N);
    std::vector<uint16_t> A_fp16(M * K);
    gen_ternary_weights(ternary, rng);
    gen_fp16_activations(A_fp16, rng);

    std::vector<bitnet_arc_tq2_0_block> B_blocks;
    pack_tq2_0_blocks(ternary, K, N, B_blocks);

    std::vector<float> C_ref;
    compute_fp32_ref(C_ref, A_fp16, ternary, M, N, K);

    /* --- SYCL setup --- */
    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    sycl_queue_handle q_handle(q);
    {
        const sycl::device d = q.get_device();
        std::fprintf(stderr,
                     "device: %s (%s)\n",
                     d.get_info<sycl::info::device::name>().c_str(),
                     d.get_info<sycl::info::device::vendor>().c_str());
    }

    /* --- USM device allocations --- */
    const std::size_t a_bytes  = M * K * sizeof(uint16_t);
    const std::size_t b_bytes  = B_blocks.size() * sizeof(bitnet_arc_tq2_0_block);

    auto* A_dev = sycl::malloc_device<uint16_t>(M * K, q);
    auto* B_dev = sycl::malloc_device<bitnet_arc_tq2_0_block>(B_blocks.size(), q);
    auto* C_dev = sycl::malloc_device<uint16_t>(M * N, q);
    if (!A_dev || !B_dev || !C_dev) {
        std::fprintf(stderr, "USM allocation failed\n");
        return 3;
    }

    q.memcpy(A_dev, A_fp16.data(),   a_bytes).wait();
    q.memcpy(B_dev, B_blocks.data(), b_bytes).wait();
    /* C is written by the kernel, no upload needed. */

    std::vector<uint16_t> C_host(M * N, 0);

    /* --- sweep --- */
    if (emit_header) {
        std::printf(
            "variant,M,N,K,tile_M,tile_N,sg_size,mode,"
            "time_ms_med,time_ms_min,bytes,bandwidth_gbs,"
            "correct,max_rel_err,over_threshold\n");
    }

    for (std::size_t i = 0; i < kv0_variants_count; ++i) {
        const kernel_variant_desc& v = kv0_variants[i];

        const run_stats s = run_variant(
            q, q_handle, v, M, N, K,
            A_dev, B_dev, C_dev,
            C_ref, C_host,
            warmup, timed);

        if (!s.ran) {
            std::fprintf(stderr,
                         "skip %s: %s (M=%zu, N=%zu, tile=%ux%u)\n",
                         v.name, s.skip_reason, M, N, v.tile_M, v.tile_N);
            continue;
        }

        const double w_bytes = static_cast<double>(N) *
                               (static_cast<double>(K) / 256.0) *
                               static_cast<double>(sizeof(bitnet_arc_tq2_0_block));
        const double a_b = static_cast<double>(M) * static_cast<double>(K) *
                           sizeof(std::uint16_t);
        const double o_b = static_cast<double>(M) * static_cast<double>(N) *
                           sizeof(std::uint16_t);
        const double total_bytes = w_bytes + a_b + o_b;

        std::printf(
            "%s,%zu,%zu,%zu,%u,%u,%u,%s,%.4f,%.4f,%.0f,%.2f,%s,%.3g,%zu\n",
            v.name, M, N, K,
            v.tile_M, v.tile_N, v.sg_size, mode_name(v.inner_mode),
            s.time_ms_med, s.time_ms_min,
            total_bytes, s.bandwidth_gbs,
            s.correct ? "YES" : "NO",
            s.max_rel_err, s.over_threshold);
        std::fflush(stdout);
    }

    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);
    return 0;
}
