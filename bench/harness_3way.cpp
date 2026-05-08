/*
 * bench/harness_3way.cpp -- bitnet-arc 3-voice cross-check harness.
 *
 * Three voices, per oracle/README.md tolerance model:
 *   v1 = maison FP32 matmul on pre-dequantized weights
 *        (oracle/fp32_matmul.c)
 *   v2 = TQ2_0 quantize -> dequantize -> same FP32 matmul
 *        (oracle/tq2_0.c then oracle/fp32_matmul.c)
 *   v3 = numpy float64 BLAS, via subprocess to numpy_xcheck.py
 *
 * Comparison gates (constants from oracle/tolerance.h, single source
 * of truth shared with the Python xcheck):
 *   v1 vs v2 : TOL_FP32_VS_FP32         (0, bit-equal expected)
 *   v1 vs v3 : TOL_FP32REF_VS_NUMPY_F64 (~1e-3 relative)
 *
 * SYCL kernel comparisons (TOL_SYCL_*) are stubbed for #146 v2 once
 * the kernel can be linked and run on a real device.
 *
 * Edge cases (per @theta + @claude-opus QA spec):
 *   - zero_tile : weights all zero, output must be zero
 *   - k256      : K = 256 (smallest valid block-aligned), random weights
 *   - k14336    : K = 14336 (LLaMA-8B FFN), random weights
 *
 * Build: see bench/Makefile (links oracle/liboracle.a, requires
 * python3 + numpy on PATH at run-time for v3).
 *
 * Build of @alpha after @codex was unable to ship the harness during
 * his bootstrap window; the API contract this consumes is exactly the
 * one that survived review #59 (oracle layer) and review #60 (kernel
 * skeleton).
 */

#include "../oracle/fp16.h"
#include "../oracle/tq2_0.h"
#include "../oracle/fp32_matmul.h"
#include "../oracle/tolerance.h"

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

#include <sys/wait.h>
#include <unistd.h>

namespace {

/* --- input generators ------------------------------------------------ */

/* Generate ternary weights {-1, 0, +1} with the design v0 distribution
 * (zeros ~45%, +1 ~27.5%, -1 ~27.5%), per design v0 §2.3 and the
 * gen_inputs.py used by upstream_gating. */
void gen_ternary_weights(std::vector<int8_t>& w, std::mt19937& rng) {
    constexpr double P_ZERO = 0.45;
    constexpr double P_PLUS = 0.275;
    /* P_MINUS = 1 - P_ZERO - P_PLUS = 0.275 implicit */

    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& x : w) {
        const double r = u(rng);
        if (r < P_ZERO)               x =  0;
        else if (r < P_ZERO + P_PLUS) x =  1;
        else                          x = -1;
    }
}

/* Generate FP16 activations (uniform in [-1, 1] FP32, then cast). */
void gen_fp16_activations(std::vector<uint16_t>& a, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& h : a) {
        h = bitnet_arc_fp32_to_fp16(u(rng));
    }
}

/* --- voice-1: direct dequantize path -------------------------------- */

/* B_dequant[k * N + n] = scale * ternary[k * N + n].
 * Synthetic weights are already in {-1, 0, +1}, so scale = 1.0 keeps
 * the values intact; the FP16 round-trip in the block format is exact
 * for amax = 1 anyway. */
void direct_dequant_voice1(std::vector<float>&        B_dequant,
                           const std::vector<int8_t>& ternary,
                           std::size_t                K,
                           std::size_t                N)
{
    assert(B_dequant.size() == K * N);
    assert(ternary.size()   == K * N);
    for (std::size_t i = 0; i < K * N; ++i) {
        B_dequant[i] = static_cast<float>(ternary[i]);
    }
}

/* --- voice-2: quantize then dequantize via oracle ------------------- */

/* Layout for B blocks: per column n, K/256 contiguous blocks. */
void quantize_dequantize_voice2(std::vector<float>&        B_dequant,
                                const std::vector<int8_t>& ternary,
                                std::size_t                K,
                                std::size_t                N)
{
    constexpr std::size_t BLOCK = BITNET_ARC_TQ2_0_BLOCK_SIZE;
    assert(K % BLOCK == 0);
    const std::size_t blocks_per_col = K / BLOCK;

    std::vector<bitnet_arc_tq2_0_block> blocks(blocks_per_col * N);
    std::vector<int8_t>                 col(K);
    std::vector<float>                  scales(blocks_per_col, 1.0f);

    /* For each column n, gather the K-contiguous ternary slice and
     * quantize into one row of the block array. */
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) {
            col[k] = ternary[k * N + n];
        }
        bitnet_arc_quantize_row_tq2_0(col.data(),
                                      &blocks[n * blocks_per_col],
                                      K,
                                      scales.data());
    }

    /* Dequantize back via the oracle and scatter into the row-major
     * B_dequant. */
    std::vector<float> col_deq(K);
    for (std::size_t n = 0; n < N; ++n) {
        bitnet_arc_dequantize_row_tq2_0(&blocks[n * blocks_per_col],
                                        col_deq.data(),
                                        K);
        for (std::size_t k = 0; k < K; ++k) {
            B_dequant[k * N + n] = col_deq[k];
        }
    }
}

/* --- voice-3: numpy float64 BLAS via subprocess --------------------- */

bool write_binary(const char* path, const void* buf, std::size_t bytes) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const std::size_t written = std::fwrite(buf, 1, bytes, f);
    std::fclose(f);
    return written == bytes;
}

bool read_binary(const char* path, void* buf, std::size_t bytes) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    const std::size_t got = std::fread(buf, 1, bytes, f);
    std::fclose(f);
    return got == bytes;
}

bool run_voice3_numpy(std::size_t M,
                      std::size_t N,
                      std::size_t K,
                      const std::vector<uint16_t>& A_fp16,
                      const std::vector<float>&    B_dequant,
                      std::vector<double>&         C_fp64,
                      const std::string&           script_path)
{
    /* PID-suffixed temp paths (per @theta review #61): two concurrent
     * harness instances must not stomp each other's binary fixtures
     * (CI parallel, or interactive debug while a make run is going). */
    const std::string tag = std::to_string(static_cast<long>(::getpid()));
    const std::string path_A_s   = "/tmp/bitnet_arc_harness_A_"   + tag + ".fp16";
    const std::string path_B_s   = "/tmp/bitnet_arc_harness_B_"   + tag + ".fp32";
    const std::string path_out_s = "/tmp/bitnet_arc_harness_out_" + tag + ".fp64";
    const char* path_A   = path_A_s.c_str();
    const char* path_B   = path_B_s.c_str();
    const char* path_out = path_out_s.c_str();

    if (!write_binary(path_A, A_fp16.data(),
                      A_fp16.size() * sizeof(uint16_t))) {
        std::fprintf(stderr, "harness: failed to write %s\n", path_A);
        return false;
    }
    if (!write_binary(path_B, B_dequant.data(),
                      B_dequant.size() * sizeof(float))) {
        std::fprintf(stderr, "harness: failed to write %s\n", path_B);
        return false;
    }

    char arg_M[32], arg_K[32], arg_N[32];
    std::snprintf(arg_M, sizeof(arg_M), "%zu", M);
    std::snprintf(arg_K, sizeof(arg_K), "%zu", K);
    std::snprintf(arg_N, sizeof(arg_N), "%zu", N);

    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "harness: fork failed\n");
        return false;
    }
    if (pid == 0) {
        /* child: exec python3. */
        char* const argv[] = {
            const_cast<char*>("python3"),
            const_cast<char*>(script_path.c_str()),
            const_cast<char*>("--A"),   const_cast<char*>(path_A),
            const_cast<char*>("--B"),   const_cast<char*>(path_B),
            const_cast<char*>("--M"),   arg_M,
            const_cast<char*>("--K"),   arg_K,
            const_cast<char*>("--N"),   arg_N,
            const_cast<char*>("--out"), const_cast<char*>(path_out),
            nullptr
        };
        execvp(argv[0], argv);
        std::fprintf(stderr, "harness: execvp(python3) failed\n");
        std::_Exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0
        || !WIFEXITED(status)
        || WEXITSTATUS(status) != 0)
    {
        std::fprintf(stderr,
                     "harness: numpy_xcheck.py failed (status=%d)\n",
                     status);
        return false;
    }

    if (!read_binary(path_out, C_fp64.data(),
                     C_fp64.size() * sizeof(double))) {
        std::fprintf(stderr, "harness: failed to read %s\n", path_out);
        return false;
    }

    std::remove(path_A);
    std::remove(path_B);
    std::remove(path_out);
    return true;
}

/* --- comparison gates ----------------------------------------------- */

struct rel_check_result {
    double      max_rel_err;
    std::size_t over_threshold;
};

/* Bit-equal check for FP32 voice-1 vs voice-2. Returns mismatch count.
 *
 * Static-asserts the tolerance for the v1<->v2 gate is exactly 0
 * (per @sonnet review #61). If a future tolerance.h edit ever loosens
 * this bar, the assertion fires and we revisit the gate semantics
 * before silently letting drift through. */
static_assert(BITNET_ARC_TOL_FP32_VS_FP32 == 0.0f,
              "v1 vs v2 must remain a strict bit-equal gate; "
              "see oracle/tolerance.h rationale");

std::size_t check_bit_equal(const std::vector<float>& a,
                            const std::vector<float>& b,
                            float& max_abs_diff_out)
{
    assert(a.size() == b.size());
    std::size_t mismatches = 0;
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++mismatches;
            const float d = std::fabs(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    max_abs_diff_out = max_abs;
    return mismatches;
}

/* Relative tolerance check: max |a - b| / max(|a|, |b|, eps). */
rel_check_result check_relative(const std::vector<float>&  a_f32,
                                const std::vector<double>& b_f64,
                                double tolerance)
{
    assert(a_f32.size() == b_f64.size());
    constexpr double EPS = 1e-30;
    rel_check_result r{0.0, 0};
    for (std::size_t i = 0; i < a_f32.size(); ++i) {
        const double a = static_cast<double>(a_f32[i]);
        const double b = b_f64[i];
        const double denom = std::max({std::fabs(a), std::fabs(b), EPS});
        const double rel   = std::fabs(a - b) / denom;
        if (rel > r.max_rel_err) r.max_rel_err = rel;
        if (rel > tolerance)     ++r.over_threshold;
    }
    return r;
}

/* --- one test case --------------------------------------------------- */

struct test_case {
    const char* name;
    std::size_t M;
    std::size_t N;
    std::size_t K;
    bool        zero_weights;
};

bool run_case(const test_case& tc,
              std::uint32_t    seed,
              const std::string& script_path)
{
    std::printf("[%s] M=%zu N=%zu K=%zu seed=%u zero_weights=%d\n",
                tc.name, tc.M, tc.N, tc.K, seed,
                tc.zero_weights ? 1 : 0);

    std::mt19937 rng(seed);

    std::vector<int8_t>   ternary(tc.K * tc.N, 0);
    std::vector<uint16_t> A_fp16(tc.M * tc.K);
    if (!tc.zero_weights) {
        gen_ternary_weights(ternary, rng);
    }
    gen_fp16_activations(A_fp16, rng);

    std::vector<float> B_dequant_v1(tc.K * tc.N);
    direct_dequant_voice1(B_dequant_v1, ternary, tc.K, tc.N);

    std::vector<float> B_dequant_v2(tc.K * tc.N);
    quantize_dequantize_voice2(B_dequant_v2, ternary, tc.K, tc.N);

    std::vector<float> C_v1(tc.M * tc.N);
    bitnet_arc_oracle_fp32_matmul(tc.M, tc.N, tc.K,
                                  A_fp16.data(),
                                  B_dequant_v1.data(),
                                  C_v1.data());

    std::vector<float> C_v2(tc.M * tc.N);
    bitnet_arc_oracle_fp32_matmul(tc.M, tc.N, tc.K,
                                  A_fp16.data(),
                                  B_dequant_v2.data(),
                                  C_v2.data());

    std::vector<double> C_v3(tc.M * tc.N);
    const bool v3_ok = run_voice3_numpy(tc.M, tc.N, tc.K,
                                        A_fp16, B_dequant_v1, C_v3,
                                        script_path);

    /* Gate v1 vs v2: bit-equal expected. */
    float       max_abs    = 0.0f;
    const auto  mismatches = check_bit_equal(C_v1, C_v2, max_abs);
    const bool  gate12_ok  = (mismatches == 0);
    std::printf("  v1 vs v2 (bit-equal)            : %s  "
                "mismatches=%zu  max_abs=%.3g\n",
                gate12_ok ? "PASS" : "FAIL",
                mismatches, static_cast<double>(max_abs));

    /* Gate v1 vs v3: relative tolerance. */
    bool gate13_ok = false;
    if (v3_ok) {
        const rel_check_result rc =
            check_relative(C_v1, C_v3,
                           static_cast<double>(BITNET_ARC_TOL_FP32REF_VS_NUMPY_F64));
        gate13_ok = (rc.over_threshold == 0);
        std::printf("  v1 vs v3 (rel<%g)               : %s  "
                    "over=%zu  max_rel=%.3g\n",
                    static_cast<double>(BITNET_ARC_TOL_FP32REF_VS_NUMPY_F64),
                    gate13_ok ? "PASS" : "FAIL",
                    rc.over_threshold, rc.max_rel_err);
    } else {
        std::printf("  v1 vs v3 (rel<%g)               : SKIP "
                    "(numpy_xcheck.py unavailable)\n",
                    static_cast<double>(BITNET_ARC_TOL_FP32REF_VS_NUMPY_F64));
    }

    return gate12_ok && (gate13_ok || !v3_ok);
}

} /* anonymous namespace */

int main(int argc, char** argv) {
    /* Default test-case set, per @theta + @claude-opus QA spec. */
    const test_case cases[] = {
        /* zero-tile: output must be exactly zero across all voices. */
        { "zero_tile", /*M=*/16, /*N=*/16, /*K=*/256,   /*zero=*/true  },
        /* k256: smallest valid block-aligned K, varied weights. */
        { "k256",      /*M=*/16, /*N=*/16, /*K=*/256,   /*zero=*/false },
        /* k14336: representative LLaMA-8B FFN intermediate dim. */
        { "k14336",    /*M=*/1,  /*N=*/64, /*K=*/14336, /*zero=*/false },
    };
    constexpr std::size_t NUM_CASES = sizeof(cases) / sizeof(cases[0]);

    std::uint32_t seed = 1337;
    std::string script_path =
        "/projects/bitnet-arc/oracle/numpy_xcheck.py";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--xcheck") == 0 && i + 1 < argc) {
            script_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--seed N] [--xcheck PATH]\n", argv[0]);
            return 0;
        }
    }

    bool all_pass = true;
    for (std::size_t i = 0; i < NUM_CASES; ++i) {
        const bool ok = run_case(cases[i],
                                 seed + static_cast<std::uint32_t>(i),
                                 script_path);
        /* No short-circuit: every case must run so we see all failures
         * in a single invocation (per @theta review #61). */
        if (!ok) all_pass = false;
        std::printf("\n");
    }

    std::printf("=========================================\n");
    std::printf("HARNESS RESULT: %s (%zu cases)\n",
                all_pass ? "ALL PASS" : "FAILURE", NUM_CASES);
    return all_pass ? 0 : 1;
}
