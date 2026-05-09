/*
 * bench/profile_v2_wg_isolation.cpp -- single-WG isolation test for
 * kernel_v2 (Task #160 Phase 2a step 1.5 -- @theta suggestion #1
 * preliminary before step 2 split-build).
 *
 * Why this exists
 * ---------------
 * @claude-opus's Phase 2a step 1 run on Arc B60 (commit fdc1747)
 * produced a counter-intuitive scaling result :
 *
 *      4 WGs  -> 5.47 ms (16x64x14336)
 *     16 WGs -> 5.41 ms (64x64x14336)
 *     64 WGs -> 5.38 ms (64x256x14336)
 *
 * Wall-clock is essentially flat across a 16x WG-count sweep. Two
 * interpretations remained on the table after that run :
 *
 *   (a) WGs run in true parallel, each WG saturates its own intra-WG
 *       resources (barriers / MMA dispatch / SLM aliasing -- the 3
 *       top-1 candidates left after sec4.1 sub-saturation was killed).
 *   (b) Some inter-WG effect (e.g. SLM bank contention shared across
 *       WGs, or a hidden serialization at the dispatcher level) caps
 *       wall-clock at ~5.4 ms regardless of WG count.
 *
 * The split-build step 2 (~3h dev) discriminates between the 3
 * intra-WG candidates assuming (a). If (b) is true instead, step 2
 * mis-attributes the bottleneck and we burn 3h building the wrong
 * tool.
 *
 * This 80-LOC preliminary forces the question : compare per-WG
 * wall-clock at WG_count = 1 vs WG_count = N.
 *
 *   - per-WG time at N = total_N / N
 *   - if per-WG_1 ~= per-WG_N within noise -> (a) confirmed,
 *     each WG carries ~5 ms of intra-WG work, true parallel exec
 *   - if per-WG_1 << per-WG_N -> (b), there's contention/serialization
 *     that grows with WG count
 *
 * Methodology (per @theta review #83 protocol notes)
 * --------------------------------------------------
 *   - 2 K shapes : 4096 (best, attn-projection K) + 14336 (worst,
 *     ffn W2/gate K). Keep K constant across single/multi to factor
 *     out K-dependent intra-WG work.
 *   - For each K : run M=16,N=16 (1 WG) + M=64,N=256 (64 WGs).
 *     The 16x16 output tile size is fixed by kernel_v2 (Phase 1 single
 *     registered variant 16x16_sg16_k256).
 *   - 5 timed runs after 5 warmup runs ; report min / median / max
 *     + measured variance (single-WG = more jitter expected, no
 *     masking by other WGs in flight -- per @theta caveat).
 *   - Caveat : single-WG runs hit launch-latency cost more visibly.
 *     The host steady_clock measures from submit to wait_and_throw,
 *     so launch overhead is included. We accept this as a known
 *     bias and flag it in the CSV `caveat` column.
 *
 * Output
 * ------
 *   stdout : CSV per (K, mode) with min/med/max/std + per_wg_med +
 *            ratio_vs_single (NaN for single rows).
 *   stderr : per-K summary + final intra-vs-inter verdict heuristic.
 *
 * Code reuse
 * ----------
 * The helper bodies (gen_ternary_weights, gen_fp16_activations,
 * pack_tq2_0_blocks, summarize, usm_device_uptr, parse_pos_uint) are
 * intentionally duplicated from bench/profile_v2.cpp rather than
 * extracted to a shared header. Reason : profile_v2.cpp passed review
 * #83 (audit @beta + @theta) at commit fdc1747 ; touching it now to
 * extract a header would force a re-audit of an already-locked file
 * just for the sake of a 80-LOC standalone bench. The duplication is
 * acknowledged debt, to be folded into a shared header if step 2
 * (split-build) lands and we end up with 3+ profile_v2_*.cpp files
 * sharing the same fixture infrastructure.
 *
 * Build : bench/Makefile target profile_v2_wg_isolation. Same icpx
 * + Arc B60 toolchain as profile_v2.
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../src/kernel_v2.h"
#include "../oracle/fp16.h"
#include "../oracle/tq2_0.h"

#include <sycl/sycl.hpp>

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

/* --- duplicated helpers (see file header re: code reuse) ----------- */

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

struct stats_t { double t_min, t_med, t_mean, t_std, t_max; };

stats_t summarize(std::vector<double> ts) {
    if (ts.empty()) return {};
    std::sort(ts.begin(), ts.end());
    const std::size_t n = ts.size();
    stats_t s{};
    s.t_min = ts.front();
    s.t_max = ts.back();
    s.t_med = (n & 1u) ? ts[n / 2]
                       : 0.5 * (ts[n / 2 - 1] + ts[n / 2]);
    double sum = 0.0;
    for (double x : ts) sum += x;
    s.t_mean = sum / n;
    double sq = 0.0;
    for (double x : ts) sq += (x - s.t_mean) * (x - s.t_mean);
    s.t_std = std::sqrt(sq / n);
    return s;
}

/* --- isolated kernel timer (mirror time_full_v2 in profile_v2.cpp) - */

stats_t time_kernel_v2(sycl::queue& q, sycl_queue_handle& qh,
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

/* --- one (K, mode) measurement run ---------------------------------- */

struct probe_result_t {
    std::size_t M, N, K;
    std::size_t wg_count;
    stats_t     stats;
    double      per_wg_med;
};

probe_result_t probe_one(sycl::queue& q, sycl_queue_handle& qh,
                         std::size_t M, std::size_t N, std::size_t K,
                         unsigned seed, unsigned warmup, unsigned timed)
{
    std::mt19937 rng(seed);
    std::vector<int8_t>   tern(K * N);
    std::vector<uint16_t> A(M * K);
    gen_ternary_weights(tern, rng);
    gen_fp16_activations(A, rng);
    std::vector<bitnet_arc_tq2_0_block> B;
    pack_tq2_0_blocks(tern, K, N, B);

    usm_device_uptr<uint16_t>               A_d(M * K, q);
    usm_device_uptr<bitnet_arc_tq2_0_block> B_d(B.size(), q);
    usm_device_uptr<uint16_t>               C_d(M * N, q);
    q.memcpy(A_d.get(), A.data(), M * K * sizeof(uint16_t)).wait();
    q.memcpy(B_d.get(), B.data(), B.size() * sizeof(B[0])).wait();

    const stats_t s = time_kernel_v2(q, qh, M, N, K,
                                     A_d.get(), B_d.get(), C_d.get(),
                                     warmup, timed);
    const std::size_t wgs = (M / 16) * (N / 16);
    return probe_result_t{M, N, K, wgs, s, s.t_med / double(wgs)};
}

} /* anonymous namespace */

/* -------------------------------------------------------------------- */
/* main                                                                 */
/* -------------------------------------------------------------------- */

int main(int argc, char** argv) {
    /* CLI : same parse_pos_uint helper as profile_v2.cpp (defensive
     * checks per review #83 fold). */
    unsigned warmup = 5, timed = 5;
    std::uint32_t seed = 1337;

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
            char* end = nullptr;
            const long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0) {
                std::fprintf(stderr,
                    "error: --seed requires a non-negative integer\n");
                return 1;
            }
            seed = static_cast<std::uint32_t>(v);
        }
        else if (a == "--help") {
            std::printf(
                "usage: %s [--warmup N] [--timed N] [--seed N]\n"
                "  default: 5 warmup + 5 timed iters per probe, seed=1337.\n"
                "  Runs kernel_v2 single-WG (M=16,N=16) vs multi-WG\n"
                "  (M=64,N=256, 64 WGs) for K in {4096, 14336}.\n",
                argv[0]);
            return 0;
        }
    }

    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    sycl_queue_handle qh(q);
    std::fprintf(stderr, "device : %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(stderr, "iters  : %u warmup + %u timed per probe\n",
                 warmup, timed);
    std::fprintf(stderr, "probes : 4 (2 K shapes x {single-WG, multi-WG})\n\n");

    /* CSV header. Columns chosen so the verdict (intra-vs-inter) is
     * computable per K from the ratio = single_per_wg / multi_per_wg.
     * Ratio close to 1.0 -> intra-WG bottleneck confirmed. */
    std::printf("K,mode,M,N,wg_count,t_min,t_med,t_mean,t_std,t_max,"
                "per_wg_med,ratio_vs_single,caveat\n");

    /* 2 shapes per @theta protocol : best (K=4096) + worst (K=14336).
     * Within each, single-WG (M=16,N=16) and multi-WG (M=64,N=256). */
    struct probe_pair_t { std::size_t K; const char* tag; };
    const probe_pair_t pairs[] = {
        { 4096,  "best_attn_K"   },
        { 14336, "worst_ffn_K"   },
    };

    for (const auto& p : pairs) {
        if (p.K % 256 != 0) {
            std::fprintf(stderr, "skip K=%zu : not %% 256\n", p.K);
            continue;
        }

        /* single WG : M=16, N=16 */
        const probe_result_t single = probe_one(q, qh, 16, 16, p.K,
                                                seed, warmup, timed);
        /* multi WG : M=64, N=256 -> 4 * 16 = 64 WGs */
        const probe_result_t multi  = probe_one(q, qh, 64, 256, p.K,
                                                seed, warmup, timed);

        const double ratio = (multi.per_wg_med > 0.0)
                             ? single.per_wg_med / multi.per_wg_med
                             : 0.0;

        /* CSV : single row (no ratio, NaN), then multi row (with ratio
         * vs single). The caveat column flags single-WG launch-latency
         * bias. */
        std::printf("%zu,single,%zu,%zu,%zu,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,"
                    "%.5f,nan,launch_latency_visible\n",
                    p.K, single.M, single.N, single.wg_count,
                    single.stats.t_min, single.stats.t_med,
                    single.stats.t_mean, single.stats.t_std,
                    single.stats.t_max,
                    single.per_wg_med);
        std::printf("%zu,multi,%zu,%zu,%zu,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,"
                    "%.5f,%.4f,launch_latency_amortized\n",
                    p.K, multi.M, multi.N, multi.wg_count,
                    multi.stats.t_min, multi.stats.t_med,
                    multi.stats.t_mean, multi.stats.t_std,
                    multi.stats.t_max,
                    multi.per_wg_med, ratio);
        std::fflush(stdout);

        /* stderr summary per K. */
        std::fprintf(stderr,
            "%-13s [K=%5zu] single (1 WG): t_med=%.3f ms (jitter %.3f-%.3f)\n",
            p.tag, p.K, single.stats.t_med,
            single.stats.t_min, single.stats.t_max);
        std::fprintf(stderr,
            "%-13s [K=%5zu] multi (64 WGs): t_med=%.3f ms total -> "
            "%.4f ms/WG (jitter %.3f-%.3f)\n",
            p.tag, p.K, multi.stats.t_med, multi.per_wg_med,
            multi.stats.t_min, multi.stats.t_max);
        std::fprintf(stderr,
            "%-13s [K=%5zu] RATIO single_per_wg / multi_per_wg = %.3f\n\n",
            p.tag, p.K, ratio);
    }

    /* Final heuristic. Conservative thresholds : ratio in [0.7, 1.5]
     * is treated as "intra-WG dominant" (true parallel WGs, single-WG
     * launch-latency bias accounts for the spread). Outside that band
     * suggests an inter-WG effect worth investigating before step 2. */
    std::fprintf(stderr,
        "interpretation guide :\n"
        "  ratio ~ 1.0 (range [0.7, 1.5])   -> intra-WG bottleneck CONFIRMED\n"
        "                                      (each WG carries ~constant\n"
        "                                      work, true parallel exec)\n"
        "                                      step 2 split-build is the\n"
        "                                      right next move.\n"
        "  ratio < 0.7                      -> single WG much faster than\n"
        "                                      per-WG share of multi.\n"
        "                                      Inter-WG contention likely\n"
        "                                      (SLM banks, dispatcher).\n"
        "                                      Re-think step 2 design.\n"
        "  ratio > 1.5                      -> single WG slower per WG\n"
        "                                      than the multi share.\n"
        "                                      Pure launch-latency bias\n"
        "                                      dominates ; report shows\n"
        "                                      noise floor only.\n"
        "  K-dependence (4096 vs 14336)     -> if the ratio differs\n"
        "                                      strongly between best and\n"
        "                                      worst shape, report it ;\n"
        "                                      may suggest K-scaled SLM\n"
        "                                      or barrier behavior.\n");
    return 0;
}
