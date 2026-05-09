/*
 * bench/test_kv4_one.cpp -- all-+1 ternary, all-1.0 activation test.
 *
 * Manually-calculable expected output to discriminate which axis
 * is broken in kv4: VNNI layout, per-chunk d fold, or activation quant.
 *
 * Test setup (single chunk K=256, M=16 N=16):
 *   - All ternary B = +1 (TQ2_0 code = 2 in oracle convention)
 *   - All A_fp16 = 1.0
 *   - All d (FP16 scale) = 1.0
 *
 * Expected math:
 *   mC[m, n] = sum_k(A[m,k] * B[k,n]) for k in 0..K-1
 *           = K * 1 * 1 = 256
 *   s_a[m] = max_abs(row m of A_fp16) = 1.0
 *   With per-chunk d fold:
 *     mC_fp32 += d_chunk * float(mC_chunk_i32) = 1 * 256 = 256 per chunk
 *     But for K=256 = 1 chunk, mC_chunk_i32 = 256 per element after 4 frags
 *     -> mC_fp32 = 1 * 256 = 256
 *   Output: c[m, n] = s_a[m] * mC_fp32[m, n] = 1 * 256 = 256
 *
 *   Expected FP16 output: 256.0 (FP16 bit pattern 0x5C00).
 *
 * If output != 256 systematically -> layout/fold bug. The exact wrong
 * value gives diagnostic info:
 *   - 64 = only 1 fragment counted (per-chunk fold accumulating only
 *          last frag, not all 4)
 *   - 128 = only 2 frags
 *   - 256 / N (= 16) = layout permutation between A and B mismatched
 *   - 256 * 16 = same in other direction
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../src/kernel_v4.h"
#include "../src/kernel_v4_packing.h"
#include "../oracle/fp16.h"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const std::size_t M = 16, N = 16, K = 256;

    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    bitnet_arc::sycl_queue_handle qh(q);

    auto A_dev = sycl::malloc_device<std::uint16_t>(M * K, q);
    auto B_dev = sycl::malloc_device<bitnet_arc_tq2_0_block>((K / 256) * N, q);
    auto C_dev = sycl::malloc_device<std::uint16_t>(M * N, q);

    /* All-1.0 FP16 activations */
    const std::uint16_t one_h = bitnet_arc_fp32_to_fp16(1.0f);
    std::vector<std::uint16_t> A_one(M * K, one_h);
    q.memcpy(A_dev, A_one.data(), A_one.size() * sizeof(std::uint16_t)).wait();

    /* All-+1 ternary B with d=1.0:
     * In TQ2_0 convention (oracle/tq2_0.c): code 2 -> ternary +1.
     * 4 codes per byte (2 bits each), so all-+1 = each byte = 0xAA
     * (bits 10101010 = 4 codes of 2). */
    std::vector<bitnet_arc_tq2_0_block> B_one((K / 256) * N);
    for (auto& blk : B_one) {
        blk.d = bitnet_arc_fp32_to_fp16(1.0f);
        std::memset(blk.qs, 0xAA, sizeof(blk.qs));  /* every code = 2 = ternary +1 */
    }
    q.memcpy(B_dev, B_one.data(), B_one.size() * sizeof(bitnet_arc_tq2_0_block)).wait();

    /* Sanity: verify host-side helper produces all-+1 ternary from this. */
    std::int8_t ternary_check[256];
    bitnet_arc::kv4::unpack_tq2_0_block_to_ternary(B_one[0], ternary_check);
    int n_pos = 0, n_zero = 0, n_neg = 0;
    for (auto t : ternary_check) {
        if (t > 0) ++n_pos; else if (t == 0) ++n_zero; else ++n_neg;
    }
    std::fprintf(stderr,
                 "host ternary unpack: %d +1, %d 0, %d -1 (expected 256 +1)\n",
                 n_pos, n_zero, n_neg);

    /* Run kv4. */
    bitnet_arc::run_kernel_v4(qh, M, N, K, A_dev, B_dev, C_dev);

    std::vector<std::uint16_t> C_host(M * N);
    q.memcpy(C_host.data(), C_dev, C_host.size() * sizeof(std::uint16_t)).wait();

    /* Convert + analyze */
    const float expected = 256.0f;
    int n_match = 0, n_zero_out = 0;
    float min_v = 1e30f, max_v = -1e30f;
    float sum_v = 0.0f;
    for (auto v : C_host) {
        const float f = bitnet_arc_fp16_to_fp32(v);
        if (f == expected) ++n_match;
        if (v == 0) ++n_zero_out;
        if (f < min_v) min_v = f;
        if (f > max_v) max_v = f;
        sum_v += f;
    }
    const float mean = sum_v / float(C_host.size());
    std::fprintf(stderr,
                 "all-1 test M=%zu N=%zu K=%zu: expected 256.0\n"
                 "  matches: %d/%zu, zero outputs: %d\n"
                 "  min=%.2f max=%.2f mean=%.2f\n"
                 "  first 16 outputs (FP32):  ",
                 M, N, K, n_match, C_host.size(), n_zero_out, min_v, max_v, mean);
    for (unsigned i = 0; i < 16; ++i) {
        std::fprintf(stderr, "%.1f ", bitnet_arc_fp16_to_fp32(C_host[i]));
    }
    std::fprintf(stderr, "\n");

    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    return n_match == int(C_host.size()) ? 0 : 1;
}
