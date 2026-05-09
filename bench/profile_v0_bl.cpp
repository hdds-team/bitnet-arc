/*
 * bench/profile_v0_bl.cpp -- task #155 v0_BL characterization profiler.
 *
 * Goal (per @claude-opus brief, design-v2 input):
 *   determine whether v0_BL on Arc B60 is compute-bound, memory-bound,
 *   or latency-bound on the headline shape (64, 64, 14336).
 *
 * Method: ablation timing across K = {256, 4096, 14336}. T(K) scaling
 * separates the regimes:
 *   - Linear T(K) starting near origin    -> memory- or compute-bound
 *     (bytes and FMAs both scale linearly with K, so the slope alone
 *      doesn't pick between them; combine with bandwidth_gbs to decide:
 *      if slope*1e3 saturates HBM peak -> memory-bound, else compute).
 *   - Plateau / heavy intercept at small K -> latency- / launch-bound.
 *   - Sub-linear at small K with cliff at large K -> cache spill.
 *
 * Output:
 *   - stdout: CSV (K, n, t_ms_min, t_ms_med, t_ms_mean, t_ms_stddev,
 *             t_ms_p99, bytes, gbs)
 *   - stderr: human-readable summary + slope analysis.
 *
 * NB: design v1 is failed, this profiler measures v0_BL only. If the
 * data points to compute-bound, design v2 should explore XMX. If
 * memory-bound, vec loads / B pre-shuffle. Let the data decide.
 *
 * Build: bench/Makefile target profile_v0_bl. Same toolchain as
 * sweep_tile (icpx + Arc B60 host).
 */

#include "../src/kernel_v0_sycl.hpp"
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

using bitnet_arc::kernel_variant_desc;
using bitnet_arc::kernel_v0_inner_mode;
using bitnet_arc::kv0_variants;
using bitnet_arc::kv0_variants_count;
using bitnet_arc::sycl_queue_handle;

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

const kernel_variant_desc* find_variant(const char* name) {
    for (std::size_t i = 0; i < kv0_variants_count; ++i) {
        if (std::strcmp(kv0_variants[i].name, name) == 0)
            return &kv0_variants[i];
    }
    return nullptr;
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

stats_t run_one_K(sycl::queue& q, sycl_queue_handle& qh,
                  const kernel_variant_desc& v,
                  std::size_t M, std::size_t N, std::size_t K,
                  std::uint16_t* A_dev,
                  bitnet_arc_tq2_0_block* B_dev,
                  std::uint16_t* C_dev,
                  unsigned warmup, unsigned timed)
{
    for (unsigned i = 0; i < warmup; ++i)
        v.launch(qh, M, N, K, A_dev, B_dev, C_dev);
    q.wait_and_throw();

    std::vector<double> ts;
    ts.reserve(timed);
    for (unsigned i = 0; i < timed; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        v.launch(qh, M, N, K, A_dev, B_dev, C_dev);
        q.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return summarize(std::move(ts));
}

double bytes_for(std::size_t M, std::size_t N, std::size_t K) {
    const double w = double(N) * (double(K) / 256.0)
                   * double(sizeof(bitnet_arc_tq2_0_block));
    const double a = double(M) * double(K) * sizeof(std::uint16_t);
    const double o = double(M) * double(N) * sizeof(std::uint16_t);
    return w + a + o;
}

} /* anonymous namespace */

int main(int argc, char** argv) {
    /* Defaults: headline shape + canonical v0_BL variant. */
    std::size_t M = 64, N = 64;
    std::vector<std::size_t> Ks = {256u, 4096u, 14336u};
    const char* variant_name = "32x32_sg16_BRANCHLESS";
    unsigned warmup = 5, timed = 50;
    std::uint32_t seed = 1337;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--variant" && i + 1 < argc) variant_name = argv[++i];
        else if (a == "--M"       && i + 1 < argc) M = std::atoll(argv[++i]);
        else if (a == "--N"       && i + 1 < argc) N = std::atoll(argv[++i]);
        else if (a == "--warmup"  && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--timed"   && i + 1 < argc) timed  = std::atoi(argv[++i]);
        else if (a == "--seed"    && i + 1 < argc) seed   = std::atoi(argv[++i]);
        else if (a == "--help") {
            std::printf("usage: %s [--variant NAME] [--M N --N N] "
                        "[--warmup N --timed N] [--seed N]\n"
                        "  default: variant=32x32_sg16_BRANCHLESS, "
                        "M=64 N=64, K sweep over {256,4096,14336}\n",
                        argv[0]);
            return 0;
        }
    }

    const kernel_variant_desc* v = find_variant(variant_name);
    if (!v) {
        std::fprintf(stderr, "unknown variant '%s' (try one of:\n", variant_name);
        for (std::size_t i = 0; i < kv0_variants_count; ++i)
            std::fprintf(stderr, "  %s\n", kv0_variants[i].name);
        return 2;
    }
    if (M % v->tile_M != 0 || N % v->tile_N != 0) {
        std::fprintf(stderr, "shape %zux%zu incompatible with variant tile %ux%u\n",
                     M, N, v->tile_M, v->tile_N);
        return 2;
    }

    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    sycl_queue_handle qh(q);
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(stderr, "variant: %s, shape M=%zu N=%zu, "
                         "K sweep: {", v->name, M, N);
    for (std::size_t k : Ks) std::fprintf(stderr, "%zu ", k);
    std::fprintf(stderr, "}\n");

    std::printf("variant,M,N,K,n,t_ms_min,t_ms_med,t_ms_mean,t_ms_stddev,"
                "t_ms_p99,bytes,bandwidth_gbs\n");

    for (const std::size_t K : Ks) {
        if (K % 256 != 0) continue;
        std::mt19937 rng(seed);
        std::vector<int8_t>   tern(K * N);
        std::vector<uint16_t> A(M * K);
        gen_ternary_weights(tern, rng);
        gen_fp16_activations(A, rng);
        std::vector<bitnet_arc_tq2_0_block> B;
        pack_tq2_0_blocks(tern, K, N, B);

        auto* A_d = sycl::malloc_device<uint16_t>(M * K, q);
        auto* B_d = sycl::malloc_device<bitnet_arc_tq2_0_block>(B.size(), q);
        auto* C_d = sycl::malloc_device<uint16_t>(M * N, q);
        q.memcpy(A_d, A.data(), M * K * sizeof(uint16_t)).wait();
        q.memcpy(B_d, B.data(), B.size() * sizeof(B[0])).wait();

        const stats_t s = run_one_K(q, qh, *v, M, N, K,
                                    A_d, B_d, C_d, warmup, timed);
        const double bytes = bytes_for(M, N, K);
        const double gbs   = (bytes / 1e9) / (s.t_med / 1e3);

        std::printf("%s,%zu,%zu,%zu,%u,%.5f,%.5f,%.5f,%.5f,%.5f,%.0f,%.2f\n",
                    v->name, M, N, K, timed,
                    s.t_min, s.t_med, s.t_mean, s.t_std, s.t_p99,
                    bytes, gbs);
        std::fflush(stdout);

        sycl::free(A_d, q);
        sycl::free(B_d, q);
        sycl::free(C_d, q);
    }

    std::fprintf(stderr, "\ndone. Slope analysis: compare t_med across K rows.\n"
                 "  if T(K)/K is roughly constant -> linear (mem or compute);\n"
                 "    cross-check bandwidth_gbs vs Arc B60 HBM peak (~456 GB/s):\n"
                 "    saturated -> memory-bound; far below -> compute-bound.\n"
                 "  if T(K) plateaus at small K -> launch-overhead-bound.\n");
    return 0;
}
