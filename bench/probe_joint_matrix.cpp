/*
 * bench/probe_joint_matrix.cpp -- task #156, design v2 Phase 0 probe.
 *
 * Goal : determine which (operand_type, accumulator_type, fragment_shape)
 * tuples of the SYCL2020 joint_matrix extension actually compile and
 * run correctly on Arc B60 (Xe2 / Battlemage) with icpx. The result
 * pins the operand+accumulator pair we lock in for Phase 1 of design v2.
 *
 * Acceptance criteria (per design v2 §3 fold):
 *   - At least one (operand_type, accumulator_type) pair must produce a
 *     correct 16x16 matmul within tol = 1e-2 vs an FP32 reference.
 *   - If neither FP16-acc nor FP32-acc passes, Phase 1 is BLOCKED and
 *     the escalation path in design v2 §3 opens.
 *
 * Probe matrix (4 combos, all on minimal 16x16x16 fragments):
 *
 *   operand   | accumulator | expected on Xe2
 *   ----------+-------------+------------------------------------------
 *   FP16      | FP32        | strongly preferred (Path B baseline)
 *   FP16      | FP16        | acceptable fallback if tolerance holds
 *   BF16      | FP32        | only if activation range needs > FP16
 *   BF16      | FP16        | unlikely useful; probed for completeness
 *
 * Output :
 *   - stdout : human-readable summary table + chosen pair
 *   - stderr : per-combo diagnostics (compile, run, tolerance)
 *   - exit code : 0 if any FP32-acc combo passes ; 1 if Phase 1 blocked
 *
 * Build : bench/Makefile target probe_joint_matrix. Same toolchain
 * as sweep_tile / profile_v0_bl (icpx + Arc B60 host).
 *
 * NB on portability : we use the oneAPI-experimental namespace
 * (sycl::ext::oneapi::experimental::matrix). icpx 2025.3 ships this as
 * the supported path on Xe2. Codeplay's Intel-specific extension under
 * sycl::ext::intel::experimental::matrix is *not* targeted here; it
 * lives behind a different header and is not portable per design v2
 * §2.2 portability rationale.
 */

#include "../oracle/fp16.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using bf16  = sycl::ext::oneapi::bfloat16;

namespace {

/* Tiny canonical fragment dim. Xe2 supports M=N=K=16 for FP16/BF16
 * operands per the Intel oneAPI SYCL2020 joint_matrix support matrix.
 * Other shapes (M=8, N=16, K=16) exist but the M=N=K=16 path is the
 * lowest-common-denominator probe; Phase 1 will sweep further. */
constexpr unsigned MX_DIM = 16u;

/* --- FP32 reference matmul ------------------------------------------ */

void fp32_ref_matmul(const float* A, const float* B, float* C,
                     std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

/* --- per-combo probe ------------------------------------------------- */

struct combo_result {
    const char* operand_name;
    const char* accumulator_name;
    bool        compiled;       /* always true if we reach here */
    bool        ran;            /* false on SYCL exception */
    bool        passed;         /* max_rel_err <= 1e-2 */
    double      max_rel_err;
    const char* notes;
};

template <typename TOp, typename TAcc>
combo_result probe_combo(sycl::queue& q,
                         const char* op_name, const char* acc_name,
                         const std::vector<float>& A_fp32,
                         const std::vector<float>& B_fp32,
                         const std::vector<float>& C_ref)
{
    combo_result r{op_name, acc_name, true, false, false, 0.0, ""};

    /* Convert A, B from FP32 to TOp on host, copy to USM device. */
    std::vector<TOp>  A_dev(MX_DIM * MX_DIM);
    std::vector<TOp>  B_dev(MX_DIM * MX_DIM);
    std::vector<TAcc> C_dev(MX_DIM * MX_DIM, TAcc{0});
    for (std::size_t i = 0; i < A_dev.size(); ++i)
        A_dev[i] = static_cast<TOp>(A_fp32[i]);
    for (std::size_t i = 0; i < B_dev.size(); ++i)
        B_dev[i] = static_cast<TOp>(B_fp32[i]);

    /* Allocs + memcpys inside try (per @beta review #74 nit): a thrown
     * exception on memcpy must not leak the device pointers. The catch
     * frees whichever pointers are non-null, then returns the failure
     * marker. */
    auto* dA = sycl::malloc_device<TOp>(A_dev.size(),  q);
    auto* dB = sycl::malloc_device<TOp>(B_dev.size(),  q);
    auto* dC = sycl::malloc_device<TAcc>(C_dev.size(), q);
    if (!dA || !dB || !dC) {
        r.notes = "USM malloc_device returned null (driver/budget issue)";
        if (dA) sycl::free(dA, q);
        if (dB) sycl::free(dB, q);
        if (dC) sycl::free(dC, q);
        return r;
    }

    try {
        q.memcpy(dA, A_dev.data(), A_dev.size() * sizeof(TOp)).wait();
        q.memcpy(dB, B_dev.data(), B_dev.size() * sizeof(TOp)).wait();
        q.memcpy(dC, C_dev.data(), C_dev.size() * sizeof(TAcc)).wait();
        q.submit([&](sycl::handler& h) {
            /* nd_range<1>({16},{16}) -> 1 WG = 1 SG of 16 lanes = exactly
             * what joint_matrix<16,16,16> expects. The earlier
             * nd_range<2>({16,16},{16,16}) launched 16 sub-groups all
             * writing the same dC -- correct values but UB per SYCL
             * spec (caught by @sonnet review #75). */
            h.parallel_for(
                sycl::nd_range<1>({MX_DIM}, {MX_DIM}),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    sycl::sub_group sg = it.get_sub_group();

                    mx::joint_matrix<sycl::sub_group, TOp,  mx::use::a,
                                     MX_DIM, MX_DIM, mx::layout::row_major> mA;
                    mx::joint_matrix<sycl::sub_group, TOp,  mx::use::b,
                                     MX_DIM, MX_DIM, mx::layout::row_major> mB;
                    mx::joint_matrix<sycl::sub_group, TAcc, mx::use::accumulator,
                                     MX_DIM, MX_DIM> mC;

                    mx::joint_matrix_fill(sg, mC, TAcc{0});
                    mx::joint_matrix_load(sg, mA, sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(dA), MX_DIM);
                    mx::joint_matrix_load(sg, mB, sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(dB), MX_DIM);
                    mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    mx::joint_matrix_store(sg, mC, sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(dC),
                        MX_DIM, mx::layout::row_major);
                });
        }).wait_and_throw();
        r.ran = true;
    } catch (const sycl::exception& e) {
        r.notes = "sycl::exception during launch (joint_matrix combo "
                  "likely unsupported)";
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return r;
    }

    /* Read back, compare against FP32 reference. */
    q.memcpy(C_dev.data(), dC, C_dev.size() * sizeof(TAcc)).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    constexpr double EPS = 1e-30;
    for (std::size_t i = 0; i < C_dev.size(); ++i) {
        const double a = static_cast<double>(static_cast<float>(C_dev[i]));
        const double b = static_cast<double>(C_ref[i]);
        const double denom = std::max({std::fabs(a), std::fabs(b), EPS});
        const double rel   = std::fabs(a - b) / denom;
        if (rel > r.max_rel_err) r.max_rel_err = rel;
    }
    r.passed = (r.max_rel_err <= 1e-2);
    if (r.passed) r.notes = "ok";
    else          r.notes = "tolerance exceeded";
    return r;
}

} /* anonymous namespace */

int main() {
    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    const sycl::device d = q.get_device();
    std::fprintf(stderr,
                 "device: %s (vendor: %s, driver: %s)\n",
                 d.get_info<sycl::info::device::name>().c_str(),
                 d.get_info<sycl::info::device::vendor>().c_str(),
                 d.get_info<sycl::info::device::driver_version>().c_str());

    /* Synthesize a small reproducible 16x16x16 problem in FP32. The
     * inputs are bounded so quantization to FP16 / BF16 round-trips
     * within the tolerance budget by construction (no random outliers
     * forcing FP16 to saturate). */
    std::vector<float> A(MX_DIM * MX_DIM), B(MX_DIM * MX_DIM);
    std::vector<float> C_ref(MX_DIM * MX_DIM, 0.0f);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>((i % 7) - 3) * 0.125f;       /* in [-0.375, +0.375] */
        B[i] = static_cast<float>((i % 11) - 5) * 0.0625f;     /* in [-0.3125, +0.3125] */
    }
    fp32_ref_matmul(A.data(), B.data(), C_ref.data(),
                    MX_DIM, MX_DIM, MX_DIM);

    /* Probe the 4 combos. Each combo does its own try/catch; a SYCL
     * exception on one does not abort the others. */
    std::vector<combo_result> results;
    results.push_back(probe_combo<sycl::half, float>(
        q, "fp16", "fp32", A, B, C_ref));
    results.push_back(probe_combo<sycl::half, sycl::half>(
        q, "fp16", "fp16", A, B, C_ref));
    results.push_back(probe_combo<bf16, float>(
        q, "bf16", "fp32", A, B, C_ref));
    results.push_back(probe_combo<bf16, bf16>(
        q, "bf16", "bf16", A, B, C_ref));

    /* Report. */
    std::printf("operand,accumulator,ran,passed,max_rel_err,notes\n");
    bool any_fp32_acc_passed = false;
    for (const auto& r : results) {
        std::printf("%s,%s,%s,%s,%.4g,%s\n",
                    r.operand_name, r.accumulator_name,
                    r.ran ? "yes" : "no",
                    r.passed ? "yes" : "no",
                    r.max_rel_err, r.notes);
        if (r.passed && std::strcmp(r.accumulator_name, "fp32") == 0)
            any_fp32_acc_passed = true;
    }
    std::fflush(stdout);

    /* Chosen pair for Phase 1: prefer (fp16, fp32) if it passes;
     * else first passing combo; else nothing (Phase 1 BLOCKED). */
    const combo_result* chosen = nullptr;
    for (const auto& r : results) {
        if (r.passed && std::strcmp(r.operand_name, "fp16") == 0
                     && std::strcmp(r.accumulator_name, "fp32") == 0)
            { chosen = &r; break; }
    }
    if (!chosen) {
        for (const auto& r : results) if (r.passed) { chosen = &r; break; }
    }

    /* Exit code (per @beta review #74 nit -- differentiate the 3 cases):
     *   0 = FP32 accumulator passes (preferred Phase 1 path)
     *   2 = only FP16-acc passes (fallback path; Phase 1 needs explicit
     *         decision on whether the relaxed tolerance is acceptable
     *         for the workload)
     *   1 = no combo passes (Phase 1 BLOCKED, design v2 §3 escalation) */
    if (chosen) {
        std::fprintf(stderr,
                     "\nphase 0 OK: chosen pair = %s operand x %s accumulator "
                     "(max_rel_err=%.4g)\n",
                     chosen->operand_name, chosen->accumulator_name,
                     chosen->max_rel_err);
        if (!any_fp32_acc_passed) {
            std::fprintf(stderr,
                         "warning: no FP32-accumulator combo passed -- "
                         "falling back to lower-precision accumulator. "
                         "Phase 1 should validate tolerance budget "
                         "explicitly before locking the path.\n");
            return 2;
        }
        return 0;
    }
    std::fprintf(stderr,
                 "\nphase 0 BLOCKED: no (operand, accumulator) combo passed "
                 "the 1e-2 tolerance gate. design v2 escalation path opens.\n");
    return 1;
}
