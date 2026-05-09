/*
 * bench/probe_esimd_int2.cpp -- Path 9 ESIMD INT2 probe.
 *
 * Per @naskel's lead pointing to ESIMD as the path for INT2 native
 * silicon access without inline asm. icpx 2025.3 headers expose
 * `sycl::ext::intel::esimd::xmx::dpas_argument_type::s2` (signed 2-bit)
 * which IS the silicon-native INT2 path that:
 *   - cl_intel_subgroup_matrix_multiply_accumulate (OpenCL ext) does NOT expose
 *   - sycl::ext::oneapi::experimental::matrix (joint_matrix) does NOT expose
 *
 * Reference: /opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/
 *           esimd/xmx/{common.hpp, dpas.hpp}
 *
 * Per dpas.hpp template (line 230):
 *   template <int SystolicDepth, int RepeatCount, T, CT, BT, AT,
 *             BPrecision, APrecision, N, BN, AN>
 *   simd<T, N> dpas(simd<CT, N> C, simd<BT, BN> B, simd<AT, AN> A);
 *
 * For INT2 s2 × s2 -> int32 with M=8, Xe2 ExecutionSize=16:
 *   AElemBitSize = 2, BElemBitSize = 2
 *   MaxElemsInDword = 32/2 = 16; OpsPerChannel = min(16, 8) = 8 (CAPPED)
 *   _K = SystolicDepth * OpsPerChannel = 8 * 8 = 64
 *   _M = RepeatCount = 8
 *   _N = ExecutionSize = 16 (Xe2 native)
 *   Result simd<int, 128>  (8*16 = 128 int32 = M*N)
 *   A = simd<int, 32> for M*K*2 bits / 32 bits per int = 512/32 = 16... wait let me redo
 *
 * Per dpas.hpp deduce_exec_size (lines 138-141):
 *   M*K*AElemBitSize == AN*sizeof(AT)*8
 *   K*N*BElemBitSize == BN*sizeof(BT)*8
 *
 *   AT = int (4 bytes), AElemBitSize = 2 bits
 *   AN = M*K*2 / (4*8) = 8*64*2 / 32 = 32   --> simd<int, 32>
 *   BN = K*N*2 / (4*8) = 64*16*2 / 32 = 64  --> simd<int, 64>
 *   N (result) = M*ExecutionSize = 8*16 = 128  --> simd<int, 128>
 *
 * All-element-1 init: signed INT2 value 1 = bit pattern '01'.
 * Packed into int32: 16 elements x 2 bits = 0x55555555 (alternating 01 01 01...).
 *
 * Validation: each result acc[m, n] = sum_k(1*1) = K = 64 per MMA.
 * After n_reps: each = 64 * n_reps. Sum of all 128 elements per
 * SIMD thread = 128 * 64 * n_reps = 8192 * n_reps.
 *
 * Acceptance gate:
 *   PASS     : INT2 ESIMD throughput >= 2.0× FP16 SYCL wrapper baseline
 *              (same level as INT4 K=64 OpenCL builtin we measured)
 *   MARGINAL : 1.5x - 2x (some win on memory but compute-cap holds)
 *   FAIL     : <1.5x (ESIMD path doesn't deliver expected silicon access)
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

int main() {
    sycl::queue q(sycl::default_selector_v,
                  sycl::property_list{
                      sycl::property::queue::in_order{},
                      sycl::property::queue::enable_profiling{}});
    std::fprintf(stderr, "device: %s (driver: %s)\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(),
                 q.get_device().get_info<sycl::info::device::driver_version>().c_str());

    constexpr int RESULT_N = 128;  // M*N = 8*16 result int32 elements
    int* dout = sycl::malloc_device<int>(RESULT_N, q);
    if (!dout) { std::fprintf(stderr, "malloc fail\n"); return 1; }

    auto submit_one = [&](int n_reps) {
        return q.submit([&](sycl::handler& h) {
            /* ESIMD kernel: single thread processing the whole MMA.
             * nd_range<1>(1, 1) = 1 thread = 1 SIMD16 unit on Xe2. */
            h.parallel_for(
                sycl::nd_range<1>(1, 1),
                [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                    /* All-element=1 INT2 init: 0x55555555 packs 16 x s2=1.
                     * 0x55 = 0101 0101 = four s2 values of 1 each. */
                    esimd::simd<int, 32>  A_data(0x55555555);
                    esimd::simd<int, 64>  B_data(0x55555555);
                    esimd::simd<int, RESULT_N> C_data(0);

                    for (int i = 0; i < n_reps; ++i) {
                        C_data = xmx::dpas<
                            /*SystolicDepth=*/8, /*RepeatCount=*/8,
                            /*T=*/int, /*CT=*/int, /*BT=*/int, /*AT=*/int,
                            /*BPrecision=*/xmx::dpas_argument_type::s2,
                            /*APrecision=*/xmx::dpas_argument_type::s2>(
                                C_data, B_data, A_data);
                    }

                    /* Reduce-sum to first int32, write to output. */
                    int s = 0;
                    for (int i = 0; i < RESULT_N; ++i) s += C_data[i];
                    esimd::simd<int, 1> out(s);
                    out.copy_to(dout);
                });
        });
    };

    /* Validation pass first */
    {
        constexpr int n_reps = 50;
        auto evt = submit_one(n_reps);
        evt.wait_and_throw();
        int host_out = -1;
        q.memcpy(&host_out, dout, sizeof(int)).wait();
        const int expected = 8192 * n_reps;
        if (host_out != expected) {
            std::fprintf(stderr,
                "VALIDATION FAIL n_reps=%d: expected %d, got %d\n",
                n_reps, expected, host_out);
            sycl::free(dout, q);
            return 5;
        }
        std::fprintf(stderr,
            "VALIDATION OK at n_reps=%d (expected=%d, got=%d)\n",
            n_reps, expected, host_out);
    }

    auto event_ms = [](sycl::event e) -> double {
        e.wait();
        const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        return double(t1 - t0) / 1.0e6;
    };

    constexpr unsigned warmups = 5, timed = 25;
    /* M=8, N=16, K=64, INT2 ops_per_MMA = 8*16*64*2 = 16384 */
    constexpr double OPS_PER_MMA = 8.0 * 16.0 * 64.0 * 2.0;
    std::printf("variant,N_REPS,t_ms_med,ops_per_mma,gops_per_s\n");

    double med_at_2k = 0;
    for (int n_reps : {500, 1000, 2000, 5000}) {
        for (unsigned w = 0; w < warmups; ++w) submit_one(n_reps);
        q.wait_and_throw();
        std::vector<double> ts;
        ts.reserve(timed);
        for (unsigned t = 0; t < timed; ++t) ts.push_back(event_ms(submit_one(n_reps)));
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size() / 2];
        const double gops = (n_reps * OPS_PER_MMA) / (med * 1e6);
        std::printf("esimd_dpas_int2_8x16x64,%d,%.5f,%.0f,%.2f\n",
                    n_reps, med, OPS_PER_MMA, gops);
        std::fprintf(stderr,
                     "  N_REPS=%5d : INT2 ESIMD t_med=%.4f ms (%.1f GOps/s)\n",
                     n_reps, med, gops);
        if (n_reps == 2000) med_at_2k = med;
    }

    sycl::free(dout, q);

    /* Verdict gate */
    constexpr double BASELINE_FP16_GOPS = 93.6;
    constexpr double BASELINE_INT4_GOPS = 244.2;
    if (med_at_2k <= 0) return 1;
    const double gops_at_2k = (2000.0 * OPS_PER_MMA) / (med_at_2k * 1e6);
    const double r_fp16 = gops_at_2k / BASELINE_FP16_GOPS;
    const double r_int4 = gops_at_2k / BASELINE_INT4_GOPS;
    std::fprintf(stderr,
                 "\nINT2 ESIMD at N_REPS=2000: %.1f GOps/s\n"
                 "  vs FP16 wrapper (93.6 GOps/s):    %.3fx\n"
                 "  vs INT4 K=64 OCL (244.2 GOps/s):  %.3fx\n",
                 gops_at_2k, r_fp16, r_int4);
    if (r_fp16 >= 2.0) {
        std::fprintf(stderr,
                     "PASS: INT2 ESIMD >= 2x FP16. Path 9 unlocked.\n");
        return 0;
    } else if (r_fp16 >= 1.5) {
        std::fprintf(stderr, "MARGINAL.\n");
        return 0;
    } else {
        std::fprintf(stderr, "FAIL.\n");
        return 1;
    }
}
