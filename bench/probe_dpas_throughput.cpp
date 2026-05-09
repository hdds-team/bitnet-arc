/*
 * bench/probe_dpas_throughput.cpp -- Path A Phase 0.5 hardstop gate.
 *
 * Per docs/design-v3.md sec6.0 (added in commit 1b542b6 per Opus
 * strategic fold #2): validate the 8x INT8 DPAS throughput claim
 * empirically BEFORE committing 400+ LOC of kernel implementation.
 *
 * Method: run many isolated joint_matrix_mad iterations of equivalent
 * INT8 (8x16x32) vs FP16 (16x16x16) ops, measure SYCL event kernel
 * times (excluding launch latency), compute ops/s ratio.
 *
 * Each MMA does 4096 ops in either case (8*16*32 = 16*16*16 = 4096).
 * The throughput ratio = time_fp16 / time_int8 (inverse since shorter
 * time = higher throughput for same op count).
 *
 * Acceptance gate (per design v3 sec6.0):
 *   PASS     : ratio >= 4x  (Path A throughput claim validated, Phase 1 GO)
 *   MARGINAL : 4x > ratio >= 2x  (proceed Phase 1 with reduced expectations)
 *   FAIL     : ratio < 2x  (Path A blocked, escalate v4 per sec6.3)
 *
 * Build: bench/Makefile target probe_dpas_throughput.
 * Standalone, no kernel lib dependency.
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

namespace {

double event_ms(sycl::event e) {
    e.wait();
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t1 - t0) / 1.0e6;
}

/* INT8 DPAS throughput micro-bench. Each lane runs N_REPS
 * joint_matrix_mad (8x16x32 INT8 -> int32 acc), with mC accumulating
 * to prevent DCE. Returns total kernel ms. */
double bench_int8(sycl::queue& q, unsigned N_REPS,
                  std::int8_t* dA, std::int8_t* dB, std::int32_t* dC,
                  unsigned warmups, unsigned timed_runs)
{
    constexpr unsigned MM = 8, NN = 16, KK = 32, SG = 16;
    auto submit_one = [&]() {
        return q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>({SG}, {SG}),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    sycl::sub_group sg = it.get_sub_group();
                    mx::joint_matrix<sycl::sub_group, std::int8_t,
                                     mx::use::a, MM, KK,
                                     mx::layout::row_major> mA;
                    mx::joint_matrix<sycl::sub_group, std::int8_t,
                                     mx::use::b, KK, NN,
                                     mx::layout::row_major> mB;
                    mx::joint_matrix<sycl::sub_group, std::int32_t,
                                     mx::use::accumulator, MM, NN> mC;
                    mx::joint_matrix_fill(sg, mC, std::int32_t{0});
                    mx::joint_matrix_load(sg, mA,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dA), KK);
                    mx::joint_matrix_load(sg, mB,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dB), NN);
                    /* Repeat the MAD N_REPS times. mC accumulates, so
                     * the compiler cannot DCE the loop. */
                    for (unsigned i = 0; i < N_REPS; ++i) {
                        mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    }
                    mx::joint_matrix_store(sg, mC,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dC),
                        NN, mx::layout::row_major);
                });
        });
    };
    for (unsigned w = 0; w < warmups; ++w) submit_one();
    q.wait_and_throw();
    std::vector<double> ts;
    ts.reserve(timed_runs);
    for (unsigned t = 0; t < timed_runs; ++t) {
        ts.push_back(event_ms(submit_one()));
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];  /* median */
}

/* FP16 joint_matrix throughput baseline. Same N_REPS, 16x16x16
 * fragment. */
double bench_fp16(sycl::queue& q, unsigned N_REPS,
                  sycl::half* dA, sycl::half* dB, float* dC,
                  unsigned warmups, unsigned timed_runs)
{
    constexpr unsigned MM = 16, NN = 16, KK = 16, SG = 16;
    auto submit_one = [&]() {
        return q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>({SG}, {SG}),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    sycl::sub_group sg = it.get_sub_group();
                    mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::a, MM, KK,
                                     mx::layout::row_major> mA;
                    mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::b, KK, NN,
                                     mx::layout::row_major> mB;
                    mx::joint_matrix<sycl::sub_group, float,
                                     mx::use::accumulator, MM, NN> mC;
                    mx::joint_matrix_fill(sg, mC, 0.0f);
                    mx::joint_matrix_load(sg, mA,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dA), KK);
                    mx::joint_matrix_load(sg, mB,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dB), NN);
                    for (unsigned i = 0; i < N_REPS; ++i) {
                        mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    }
                    mx::joint_matrix_store(sg, mC,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dC),
                        NN, mx::layout::row_major);
                });
        });
    };
    for (unsigned w = 0; w < warmups; ++w) submit_one();
    q.wait_and_throw();
    std::vector<double> ts;
    ts.reserve(timed_runs);
    for (unsigned t = 0; t < timed_runs; ++t) {
        ts.push_back(event_ms(submit_one()));
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

} /* anonymous namespace */

int main() {
    sycl::queue q(sycl::default_selector_v,
                  sycl::property_list{
                      sycl::property::queue::in_order{},
                      sycl::property::queue::enable_profiling{}});
    std::fprintf(stderr,
                 "device: %s (driver: %s)\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(),
                 q.get_device().get_info<sycl::info::device::driver_version>().c_str());

    /* INT8 buffers: 8*32 + 32*16 + 8*16 = 256+512+128 bytes */
    auto* iA = sycl::malloc_device<std::int8_t> (8 * 32, q);
    auto* iB = sycl::malloc_device<std::int8_t> (32 * 16, q);
    auto* iC = sycl::malloc_device<std::int32_t>(8 * 16, q);

    /* FP16 buffers: 16*16 + 16*16 + 16*16 elements (× sizeof) */
    auto* hA = sycl::malloc_device<sycl::half>(16 * 16, q);
    auto* hB = sycl::malloc_device<sycl::half>(16 * 16, q);
    auto* hC = sycl::malloc_device<float>     (16 * 16, q);

    if (!iA || !iB || !iC || !hA || !hB || !hC) {
        std::fprintf(stderr, "USM malloc_device failed\n");
        return 1;
    }

    /* Init with non-zero data so DCE can't kill the kernels. */
    std::vector<std::int8_t>  ia(8 * 32, 1), ib(32 * 16, 1);
    std::vector<sycl::half>   ha(16 * 16, sycl::half{1.0f}),
                              hb(16 * 16, sycl::half{1.0f});
    q.memcpy(iA, ia.data(), ia.size()).wait();
    q.memcpy(iB, ib.data(), ib.size()).wait();
    q.memcpy(hA, ha.data(), ha.size() * sizeof(sycl::half)).wait();
    q.memcpy(hB, hb.data(), hb.size() * sizeof(sycl::half)).wait();

    /* Run sweep over a few N_REPS values to verify scaling is linear
     * (catches situations where the compiler optimizes out part of
     * the inner loop). Each MMA does 4096 ops in either case, so
     * with N_REPS=1000, kernel does 4.096 M ops; should be measurable
     * in tens of microseconds. */
    constexpr unsigned warmups = 5, timed_runs = 25;
    const std::vector<unsigned> reps_list = {500u, 1000u, 2000u};

    std::printf("variant,N_REPS,t_ms_med,ops_per_iter,gops_per_s\n");

    double i_med_at_2k = 0, f_med_at_2k = 0;
    for (unsigned N_REPS : reps_list) {
        const double i_ms = bench_int8(q, N_REPS, iA, iB, iC, warmups, timed_runs);
        const double f_ms = bench_fp16(q, N_REPS, hA, hB, hC, warmups, timed_runs);
        constexpr double OPS_PER_MMA = 4096.0; /* 8*16*32 = 16*16*16 */
        const double i_gops = (N_REPS * OPS_PER_MMA) / (i_ms * 1e6);
        const double f_gops = (N_REPS * OPS_PER_MMA) / (f_ms * 1e6);
        std::printf("int8_dpas_8x16x32,%u,%.5f,%.0f,%.2f\n",
                    N_REPS, i_ms, OPS_PER_MMA, i_gops);
        std::printf("fp16_jm_16x16x16,%u,%.5f,%.0f,%.2f\n",
                    N_REPS, f_ms, OPS_PER_MMA, f_gops);
        std::fprintf(stderr,
                     "  N_REPS=%4u : INT8 t_med=%.4f ms (%.1f GOps/s) | "
                     "FP16 t_med=%.4f ms (%.1f GOps/s) | ratio=%.3fx\n",
                     N_REPS, i_ms, i_gops, f_ms, f_gops,
                     i_gops / f_gops);
        if (N_REPS == 2000) {
            i_med_at_2k = i_ms;
            f_med_at_2k = f_ms;
        }
    }

    sycl::free(iA, q); sycl::free(iB, q); sycl::free(iC, q);
    sycl::free(hA, q); sycl::free(hB, q); sycl::free(hC, q);

    /* Verdict gate per design v3 sec6.0 + sec6.3. */
    if (i_med_at_2k <= 0.0 || f_med_at_2k <= 0.0) {
        std::fprintf(stderr,
                     "\nphase 0.5 v3 INVALID: missing N_REPS=2000 measurement\n");
        return 1;
    }
    const double ratio = f_med_at_2k / i_med_at_2k;
    std::fprintf(stderr,
                 "\nthroughput ratio (INT8 / FP16) at N_REPS=2000: %.3fx\n",
                 ratio);
    if (ratio >= 4.0) {
        std::fprintf(stderr,
                     "phase 0.5 v3 PASS: INT8 DPAS >= 4x FP16 throughput "
                     "(claim validated). Phase 1 v3 GO.\n");
        return 0;
    } else if (ratio >= 2.0) {
        std::fprintf(stderr,
                     "phase 0.5 v3 MARGINAL: 4x > ratio >= 2x. Phase 1 v3 "
                     "may proceed but with reduced expectation on W2 gate. "
                     "Document in commit message.\n");
        return 0;
    } else {
        std::fprintf(stderr,
                     "phase 0.5 v3 FAIL: ratio < 2x = INT8 DPAS wrapper "
                     "is throttled. Path A blocked. Escalate v4 per "
                     "design v3 sec6.3 (Path C / hard pause / decode pivot).\n");
        return 1;
    }
}
