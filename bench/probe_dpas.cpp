/*
 * bench/probe_dpas.cpp -- Path A Phase 0 probe (post v2 falsification).
 *
 * Goal: determine whether icpx 2025.3 + Arc B60 (Xe2) supports
 * joint_matrix INT8 DPAS, and which (operand_type, accumulator_type,
 * fragment_shape) tuples actually compile + run correctly. Result
 * pins (or blocks) the Path A kernel design v3 brief.
 *
 * Acceptance criteria (per design v2 §3 ladder + Phase 2 falsification
 * report bench/v2_phase2_falsification.md):
 *   - At least one (int8, int8, int32) combo must compile, run, and
 *     produce a correct M=N=16 matmul within tol = 1e-2 vs an INT32
 *     reference.
 *   - If no combo passes, Path A is blocked and design v3 either
 *     pivots to Path C (custom packing) or hard-pauses.
 *
 * Probe matrix: 4 combos targeting Xe2 DPAS-supported shapes:
 *
 *   operand_a | operand_b | acc    | (M, N, K)   | rationale
 *   ----------+-----------+--------+-------------+----------------------
 *   int8      | int8      | int32  | 16, 16, 32  | typical Xe2 DPAS native
 *   int8      | int8      | int32  | 16, 16, 64  | 2x K-rep, fits ternary
 *                                                  K_CHUNK=256/4=64 frag
 *   int8      | int8      | int32  | 8, 16, 32   | minimal DPAS rep=1
 *   int8      | int8      | int32  | 16, 16, 16  | smallest probe, may
 *                                                  not be DPAS-supported
 *
 * The 4 combos test the 3 known Xe2 DPAS native shapes + the 16x16x16
 * shape from the FP16 probe for direct comparison. Different M means
 * different launch geometry; the 16x16x16 case can launch 1 WG = 1 SG
 * (matches FP16 probe Phase 0); the 8x16x* cases need adjusted launch.
 *
 * Build: bench/Makefile target probe_dpas. Same toolchain as
 * probe_joint_matrix (icpx + Arc B60 host).
 *
 * NB on portability: same as probe_joint_matrix -- we use the
 * sycl::ext::oneapi::experimental::matrix namespace (not the Intel-
 * specific extension); icpx 2025.3 ships this on Xe2.
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

namespace {

/* --- INT32 reference matmul ----------------------------------------- */

void int32_ref_matmul(const std::int8_t* A, const std::int8_t* B,
                      std::int32_t* C,
                      std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            std::int32_t acc = 0;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<std::int32_t>(A[m * K + k])
                     * static_cast<std::int32_t>(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
}

/* --- per-combo probe ------------------------------------------------- */

struct combo_result {
    const char*   tag;
    unsigned      M, N, K;
    bool          ran;
    bool          passed;
    double        max_rel_err;
    std::int32_t  worst_dev_val;
    std::int32_t  worst_ref_val;
    const char*   notes;
};

template <unsigned MM, unsigned NN, unsigned KK, unsigned SG_SIZE>
combo_result probe_combo(sycl::queue& q, const char* tag,
                         const std::vector<std::int8_t>& A,
                         const std::vector<std::int8_t>& B,
                         const std::vector<std::int32_t>& C_ref)
{
    combo_result r{tag, MM, NN, KK, false, false, 0.0, 0, 0, ""};

    auto* dA = sycl::malloc_device<std::int8_t> (A.size(), q);
    auto* dB = sycl::malloc_device<std::int8_t> (B.size(), q);
    auto* dC = sycl::malloc_device<std::int32_t>(MM * NN,  q);
    if (!dA || !dB || !dC) {
        r.notes = "USM malloc_device returned null";
        if (dA) sycl::free(dA, q);
        if (dB) sycl::free(dB, q);
        if (dC) sycl::free(dC, q);
        return r;
    }

    std::vector<std::int32_t> C_dev(MM * NN, 0);
    try {
        q.memcpy(dA, A.data(), A.size() * sizeof(std::int8_t)).wait();
        q.memcpy(dB, B.data(), B.size() * sizeof(std::int8_t)).wait();
        q.memcpy(dC, C_dev.data(), MM * NN * sizeof(std::int32_t)).wait();
        q.submit([&](sycl::handler& h) {
            /* Launch geometry: 1 WG = 1 SG of SG_SIZE lanes. The frag
             * MMxNNxKK is consumed by the SG cooperatively. The
             * accumulator MM x NN is written by the SG into dC at the
             * end. Same minimal launch as the FP16 probe. */
            h.parallel_for(
                sycl::nd_range<1>({SG_SIZE}, {SG_SIZE}),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
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
                    mx::joint_matrix_mad(sg, mC, mA, mB, mC);
                    mx::joint_matrix_store(sg, mC,
                        sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(dC),
                        NN, mx::layout::row_major);
                });
        }).wait_and_throw();
        r.ran = true;
    } catch (const sycl::exception& e) {
        r.notes = "sycl::exception during launch (joint_matrix INT8 "
                  "DPAS combo likely unsupported)";
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return r;
    }

    q.memcpy(C_dev.data(), dC, MM * NN * sizeof(std::int32_t)).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    /* For INT32 acc, exact match is the right tolerance criterion --
     * but we check max_rel_err for cross-comparison with the FP16
     * probe report. Also report worst (dev, ref) pair for diagnosis
     * if there's a mismatch. */
    constexpr double EPS = 1e-30;
    bool exact_match = true;
    for (std::size_t i = 0; i < C_dev.size(); ++i) {
        const std::int32_t a = C_dev[i];
        const std::int32_t b = C_ref[i];
        if (a != b) exact_match = false;
        const double da = static_cast<double>(a);
        const double db = static_cast<double>(b);
        const double denom = std::max({std::fabs(da), std::fabs(db), EPS});
        const double rel   = std::fabs(da - db) / denom;
        if (rel > r.max_rel_err) {
            r.max_rel_err = rel;
            r.worst_dev_val = a;
            r.worst_ref_val = b;
        }
    }
    /* INT32 path: bit-exact match expected (no rounding). Even an
     * 1e-2 tolerance would mask bugs; we require exact for PASS. */
    r.passed = exact_match;
    r.notes = exact_match ? "exact match" : "INT32 mismatch (bug)";
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

    /* Synthesize 4 fixtures, one per (M, K) pair we probe. B is shared
     * across fixtures with same N=16. Inputs are small int8 values to
     * keep the INT32 acc within representable range trivially. */
    constexpr unsigned MAX_M = 16, MAX_N = 16, MAX_K = 64;
    std::vector<std::int8_t> A_full(MAX_M * MAX_K), B_full(MAX_K * MAX_N);
    for (std::size_t i = 0; i < A_full.size(); ++i)
        A_full[i] = static_cast<std::int8_t>((i % 7) - 3); /* in [-3, +3] */
    for (std::size_t i = 0; i < B_full.size(); ++i)
        B_full[i] = static_cast<std::int8_t>((i % 5) - 2); /* in [-2, +2] */

    /* Helper to extract A[M x K], B[K x N] subviews from the maxed
     * fixture. Each combo's A/B/C_ref is an independent vector pair. */
    auto make_fixture = [&](unsigned M, unsigned N, unsigned K)
        -> std::tuple<std::vector<std::int8_t>,
                      std::vector<std::int8_t>,
                      std::vector<std::int32_t>>
    {
        std::vector<std::int8_t> A(M * K), B(K * N);
        for (unsigned m = 0; m < M; ++m)
            for (unsigned k = 0; k < K; ++k)
                A[m * K + k] = A_full[m * MAX_K + k];
        for (unsigned k = 0; k < K; ++k)
            for (unsigned n = 0; n < N; ++n)
                B[k * N + n] = B_full[k * MAX_N + n];
        std::vector<std::int32_t> C_ref(M * N, 0);
        int32_ref_matmul(A.data(), B.data(), C_ref.data(), M, N, K);
        return std::make_tuple(std::move(A), std::move(B), std::move(C_ref));
    };

    std::vector<combo_result> results;

    /* Probe 1: 16x16x32 -- typical Xe2 DPAS native shape. */
    {
        auto [A, B, C_ref] = make_fixture(16, 16, 32);
        results.push_back(probe_combo<16, 16, 32, 16>(
            q, "int8x16x16x32", A, B, C_ref));
    }

    /* Probe 2: 16x16x64 -- 2x K-rep, fits ternary K_CHUNK=256/4=64 fragment. */
    {
        auto [A, B, C_ref] = make_fixture(16, 16, 64);
        results.push_back(probe_combo<16, 16, 64, 16>(
            q, "int8x16x16x64", A, B, C_ref));
    }

    /* Probe 3: 8x16x32 -- minimal DPAS rep=1. Note: M=8 with SG=16
     * means each accumulator row maps to 2 lanes (cooperative split);
     * joint_matrix abstracts this. */
    {
        auto [A, B, C_ref] = make_fixture(8, 16, 32);
        results.push_back(probe_combo<8, 16, 32, 16>(
            q, "int8x8x16x32", A, B, C_ref));
    }

    /* Probe 4: 16x16x16 -- smallest probe, may not be DPAS-supported
     * (Xe2 INT8 DPAS native K is typically 32, not 16). */
    {
        auto [A, B, C_ref] = make_fixture(16, 16, 16);
        results.push_back(probe_combo<16, 16, 16, 16>(
            q, "int8x16x16x16", A, B, C_ref));
    }

    /* Report. */
    std::printf("tag,M,N,K,ran,passed,max_rel_err,worst_dev,worst_ref,notes\n");
    bool any_passed = false;
    for (const auto& r : results) {
        std::printf("%s,%u,%u,%u,%s,%s,%.4g,%d,%d,%s\n",
                    r.tag, r.M, r.N, r.K,
                    r.ran ? "yes" : "no",
                    r.passed ? "yes" : "no",
                    r.max_rel_err, r.worst_dev_val, r.worst_ref_val,
                    r.notes);
        if (r.passed) any_passed = true;
    }
    std::fflush(stdout);

    /* Verdict + chosen shape for design v3 brief. */
    const combo_result* chosen = nullptr;
    /* Prefer 16x16x32 (most general for our K_CHUNK=256 = 8 frags),
     * then 16x16x64, then any other passing shape. */
    for (const auto& r : results) {
        if (r.passed && r.M == 16 && r.N == 16 && r.K == 32)
            { chosen = &r; break; }
    }
    if (!chosen) {
        for (const auto& r : results) {
            if (r.passed && r.M == 16 && r.N == 16)
                { chosen = &r; break; }
        }
    }
    if (!chosen) {
        for (const auto& r : results) if (r.passed) { chosen = &r; break; }
    }

    /* Exit code semantics:
     *   0 = at least one INT8 DPAS combo passes -> Path A unblocked,
     *       design v3 brief can proceed
     *   1 = no combo passes -> Path A blocked, escalation to Path C
     *       or hard pause (D) per Phase 2 falsification report */
    if (chosen) {
        std::fprintf(stderr,
                     "\nphase 0 v3 OK: chosen DPAS shape = %s "
                     "(M=%u, N=%u, K=%u, exact match)\n",
                     chosen->tag, chosen->M, chosen->N, chosen->K);
        if (!any_passed) {
            std::fprintf(stderr, "(unreachable: chosen but !any_passed)\n");
        }
        return 0;
    }
    std::fprintf(stderr,
                 "\nphase 0 v3 BLOCKED: no INT8 DPAS combo passed. "
                 "Path A is blocked; design v3 must escalate to Path C "
                 "(custom packing) or D (hard pause / scope re-eval) per "
                 "v2_phase2_falsification.md.\n");
    return 1;
}
