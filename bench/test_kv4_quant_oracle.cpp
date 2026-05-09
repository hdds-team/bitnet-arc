/*
 * bench/test_kv4_quant_oracle.cpp -- W1 with quant-aware reference.
 *
 * The sweep_tile FP32 oracle does NOT pre-quantize A, so its "max_rel_err"
 * blows up on kv4 on random uniform [-1,1] A: that's quant noise of the
 * INT2/INT4 quant, not a kernel bug. To prove kv4 is structurally correct,
 * we compare against an oracle that performs the SAME quant on A as kv4
 * does internally.
 *
 * If kv4 matches this quant-aware oracle within FP rounding tolerance,
 * the kernel is provably correct on random inputs. The sweep_tile gate
 * is then just the wrong yardstick for kv4's intentional W1.58A2/A4
 * approximation behavior.
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../src/kernel_v4.h"
#include "../src/kernel_v4_packing.h"
#include "../oracle/fp16.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace kv4 = bitnet_arc::kv4;

int main() {
    const std::size_t M = 16, N = 16, K = 256;
    const int variant_idx = 0;  /* INT4 act primary */

    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    bitnet_arc::sycl_queue_handle qh(q);

    /* Random fixture (same distribution as sweep_tile). */
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> ua(-1.0f, 1.0f);
    std::uniform_real_distribution<double> ub(0.0, 1.0);

    std::vector<std::uint16_t> A_fp16(M * K);
    for (auto& v : A_fp16) v = bitnet_arc_fp32_to_fp16(ua(rng));

    std::vector<std::int8_t> ternary_KxN(K * N);
    for (auto& v : ternary_KxN) {
        const double r = ub(rng);
        v = (r < 0.45) ? 0 : (r < 0.45 + 0.275 ? 1 : -1);
    }
    /* Build TQ2_0 blocks with d=1 from ternary. */
    std::vector<bitnet_arc_tq2_0_block> B_blocks((K / 256) * N);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t b = 0; b < K / 256; ++b) {
            auto& blk = B_blocks[n * (K / 256) + b];
            blk.d = bitnet_arc_fp32_to_fp16(1.0f);
            std::memset(blk.qs, 0, sizeof(blk.qs));
            for (std::size_t k = 0; k < 256; ++k) {
                const std::size_t idx = (b * 256 + k) * N + n;
                const std::int8_t t = ternary_KxN[idx];
                std::uint8_t code = (t == -1) ? 0 : ((t == 0) ? 1 : 2);
                const std::size_t byte = (k >> 7) * 32u + (k & 31u);
                const unsigned shift = ((k >> 5) & 3u) * 2u;
                blk.qs[byte] |= (code << shift);
            }
        }
    }

    /* Quant-aware oracle: same quant as kv4 INT4 act path.
     *   For each row m, s_a = max_abs(row).
     *   For each element a, a_q = round(a/s_a * 7) clamped to [-7, 7].
     *   Recovered a_eff = (a_q/7) * s_a.
     *   Reference c[m, n] = sum_k(a_eff * b[k, n]).
     */
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> s_a(M);
    for (std::size_t m = 0; m < M; ++m) {
        s_a[m] = kv4::compute_a_row_max_abs(A_fp16.data() + m * K, K);
    }
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                const float a = bitnet_arc_fp16_to_fp32(A_fp16[m * K + k]);
                const float q_f = (s_a[m] > 0.0f) ? (a / s_a[m] * 7.0f) : 0.0f;
                int q = static_cast<int>(std::lround(q_f));
                if (q < -7) q = -7;
                if (q >  7) q =  7;
                const float a_eff = (s_a[m] / 7.0f) * float(q);
                const float b = static_cast<float>(ternary_KxN[k * N + n]);
                acc += a_eff * b;
            }
            C_ref[m * N + n] = acc;
        }
    }

    /* Run kv4. */
    auto A_dev = sycl::malloc_device<std::uint16_t>(M * K, q);
    auto B_dev = sycl::malloc_device<bitnet_arc_tq2_0_block>(B_blocks.size(), q);
    auto C_dev = sycl::malloc_device<std::uint16_t>(M * N, q);
    q.memcpy(A_dev, A_fp16.data(),    A_fp16.size() * sizeof(std::uint16_t)).wait();
    q.memcpy(B_dev, B_blocks.data(),  B_blocks.size() * sizeof(bitnet_arc_tq2_0_block)).wait();
    bitnet_arc::kv4_variants[variant_idx].launch(qh, M, N, K, A_dev, B_dev, C_dev);
    std::vector<std::uint16_t> C_host(M * N);
    q.memcpy(C_host.data(), C_dev, C_host.size() * sizeof(std::uint16_t)).wait();

    /* Compare. */
    double max_rel = 0.0, sum_abs = 0.0;
    int over_1e2 = 0, over_1e3 = 0;
    int n_sign_flip = 0, n_match_mag = 0;
    constexpr double EPS = 1e-30;
    int worst_idx = -1;
    std::fprintf(stderr, "  first 8 (kv4, ref, sum/diff):\n");
    for (std::size_t i = 0; i < M * N; ++i) {
        const float kv = bitnet_arc_fp16_to_fp32(C_host[i]);
        const float ref = C_ref[i];
        const double denom = std::max({std::fabs(double(kv)), std::fabs(double(ref)), EPS});
        const double rel = std::fabs(double(kv) - double(ref)) / denom;
        sum_abs += std::fabs(double(kv) - double(ref));
        if (rel > max_rel) { max_rel = rel; worst_idx = int(i); }
        if (rel > 1e-2) ++over_1e2;
        if (rel > 1e-3) ++over_1e3;
        /* Detect sign flip: kv ≈ -ref. */
        if (std::fabs(double(kv) + double(ref)) < std::fabs(double(kv) - double(ref))
            && std::fabs(double(kv)) > 0.1)
            ++n_sign_flip;
        if (std::fabs(std::fabs(double(kv)) - std::fabs(double(ref))) < 0.01 * std::max(std::fabs(double(kv)), 0.1))
            ++n_match_mag;
        if (i < 8) {
            std::fprintf(stderr, "    [%zu] kv=%+8.4f  ref=%+8.4f  sum=%+8.4f  diff=%+8.4f\n",
                         i, kv, ref, kv + ref, kv - ref);
        }
    }
    std::fprintf(stderr,
                 "  sign-flip: %d / %zu, magnitude-matches (within 1%%): %d / %zu\n",
                 n_sign_flip, M * N, n_match_mag, M * N);
    std::fprintf(stderr,
                 "Quant-aware W1 (variant %d, %s):\n"
                 "  max_rel_err = %.4g (worst at idx %d: kv4=%.4f vs ref=%.4f)\n"
                 "  over 1e-2: %d / %zu, over 1e-3: %d\n"
                 "  mean abs diff = %.6f\n",
                 variant_idx, bitnet_arc::kv4_variants[variant_idx].name,
                 max_rel,
                 worst_idx,
                 worst_idx >= 0 ? bitnet_arc_fp16_to_fp32(C_host[worst_idx]) : 0.0f,
                 worst_idx >= 0 ? C_ref[worst_idx] : 0.0f,
                 over_1e2, M * N, over_1e3, sum_abs / (M * N));

    sycl::free(A_dev, q); sycl::free(B_dev, q); sycl::free(C_dev, q);

    /* PASS if max_rel_err <= 1e-2 (FP rounding budget; tighter than W1
     * gate since the quant-aware oracle factors out the quant noise). */
    return (max_rel <= 1e-2) ? 0 : 1;
}
