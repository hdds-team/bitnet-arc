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
 *   **UPPER BOUNDS**, not point estimates. Two stacked biases:
 *     1. Cross-section fusion lost  (compiler can't fuse adjacent
 *        sections that the production kernel fuses).
 *     2. Global-memory SLM substitute  (inter-section state crosses
 *        kernel boundary via USM device scratch, replacing
 *        local_accessor's role -- adds memory traffic absent in
 *        production).
 *   Decision tree priority-rank in brief sec4.5 takes this into
 *   account: top-1 fix is applied to the production kernel, then we
 *   re-profile on build (a), not (b). (b) only identifies which
 *   section to attack first, never validates the fix.
 *
 * Build: bench/Makefile target profile_v2. Same toolchain as sweep_tile
 * and profile_v0_bl (icpx + Arc B60).
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

struct stats_t { double t_min, t_med, t_mean, t_std, t_p99; };

stats_t summarize(std::vector<double> ts) {
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

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--timed"  && i + 1 < argc) timed  = std::atoi(argv[++i]);
        else if (a == "--seed"   && i + 1 < argc) seed   = std::atoi(argv[++i]);
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
                  sycl::property::queue::in_order{});
    sycl_queue_handle qh(q);
    std::fprintf(stderr, "device : %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(stderr, "shapes : %zu (W1 + 1 large)\n", shapes.size());
    std::fprintf(stderr, "build  : (a) FULL%s\n",
                 split_build ? " + (b) SECTION-SPLIT (skeleton)" : "");
    std::fprintf(stderr, "iters  : %u warmup + %u timed per shape\n\n",
                 warmup, timed);

    /* CSV header. Phase 2a step 1 columns; step 2 will add per-section
     * absolute + ratio columns once (b) lands. `purpose` column makes
     * the bottleneck cross-reference explicit per @beta sanity nit. */
    std::printf("shape,purpose,M,N,K,t_full_min,t_full_med,t_full_mean,t_full_std,"
                "t_full_p99,bytes,bandwidth_gbs,wgs,occ_max,occ_ratio\n");

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

        auto* A_d = sycl::malloc_device<uint16_t>(s.M * s.K, q);
        auto* B_d = sycl::malloc_device<bitnet_arc_tq2_0_block>(B.size(), q);
        auto* C_d = sycl::malloc_device<uint16_t>(s.M * s.N, q);
        q.memcpy(A_d, A.data(), s.M * s.K * sizeof(uint16_t)).wait();
        q.memcpy(B_d, B.data(), B.size() * sizeof(B[0])).wait();

        /* --- build (a) FULL ------------------------------------------ */
        const stats_t full = time_full_v2(q, qh, s.M, s.N, s.K,
                                          A_d, B_d, C_d, warmup, timed);
        const double bytes = bytes_for(s.M, s.N, s.K);
        const double gbs   = (bytes / 1e9) / (full.t_med / 1e3);
        const occupancy_t occ = probe_occupancy(q, s.M, s.N, s.K);

        std::printf("%s,%s,%zu,%zu,%zu,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,"
                    "%.0f,%.2f,%zu,%zu,%.4f\n",
                    s.tag, s.purpose, s.M, s.N, s.K,
                    full.t_min, full.t_med, full.t_mean, full.t_std, full.t_p99,
                    bytes, gbs,
                    occ.wgs_launched, occ.theoretical_max, occ.ratio);
        std::fflush(stdout);
        std::fprintf(stderr,
            "  %-25s [%s] : t_med=%.3f ms, %.1f GB/s, "
            "WGs=%zu/%zu (occ=%.3f)\n",
            s.tag, s.purpose, full.t_med, gbs,
            occ.wgs_launched, occ.theoretical_max, occ.ratio);

        /* --- build (b) SECTION-SPLIT (skeleton) ---------------------- */
        if (split_build) {
            /* TODO Phase 2a step 2: split kernel_v2 into 4 sections
             * (dequant / load_a / mma / store), each as own
             * parallel_for. Inter-section state via USM device
             * scratch buffers (replaces local_accessor's role -- adds
             * global-memory bias documented in this file's header).
             * Time each section with sycl::event.
             *
             * Output schema (when ready):
             *   t_dequant_us, t_loada_us, t_mma_us, t_store_us,
             *   t_barrier_us_inferred (= t_full - sum(others)),
             *   r_dequant, r_loada, r_mma, r_store, r_barrier
             *   (ratios apply to t_full_med, not sum-of-(b)-sections).
             *
             * Per priority-ranked decision tree (brief sec4.5):
             *   top-1 = section with largest ratio
             *   if top-1 < 50%, signal composite -> falsify per sec7
             *
             * @claude-opus -- this is the iteration hook. The split
             * kernel's global-memory SLM-substitute is the bias that
             * matters most for Arc B60; we'll need at least one
             * iteration where ratios stabilize across runs.
             */
            std::fprintf(stderr,
                "  %s : split-build skeleton, see TODO in source\n",
                s.tag);
        }

        sycl::free(A_d, q);
        sycl::free(B_d, q);
        sycl::free(C_d, q);
    }

    /* --- summary + bottleneck call (per brief sec4.5) ---------------- */
    std::fprintf(stderr,
        "\ndone. Phase 2a step 1 (build a FULL) complete. Next:\n"
        "  - run with --split-build for per-section ratios (step 2,\n"
        "    requires landing the section-split kernel skeleton)\n"
        "  - cross-tab against profile_v0_bl.csv (run profile_v0_bl on\n"
        "    same shapes for direct comparison)\n"
        "  - generate bench/profile_v2_p2a.md with top-1 / top-2\n"
        "    bottleneck identification per brief sec4.5\n"
        "\nNB methodology bias (per brief sec3 + this file's header):\n"
        "  build (b) ratios apply to build (a) total, not sum-of-(b)\n"
        "  sections. Cross-section fusion lost + global-memory SLM\n"
        "  substitute = upper-bound per-section estimates only.\n"
        "  Confirmation via VTune / Level Zero XPU timeline if any\n"
        "  inferred barrier_us > 40%% of total.\n");
    return 0;
}
