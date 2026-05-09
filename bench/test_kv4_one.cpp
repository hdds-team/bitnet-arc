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

    /* All-+1 ternary B EXCEPT k=0 of column n=0 set to -1.
     * TQ2_0: byte = (k>>7)*32 + (k&31), shift = ((k>>5)&3)*2.
     * For k=0: byte=0, shift=0 -> low 2 bits of qs[0].
     * To get code 0 (= ternary -1) at k=0: low 2 bits of qs[0] = 00.
     * Other codes in byte 0 (k=1,2,3 etc., shifts up) = 2 (= +1) -> 0xA8.
     */
    std::vector<bitnet_arc_tq2_0_block> B_one((K / 256) * N);
    for (std::size_t i = 0; i < B_one.size(); ++i) {
        auto& blk = B_one[i];
        blk.d = bitnet_arc_fp32_to_fp16(1.0f);
        std::memset(blk.qs, 0xAA, sizeof(blk.qs));  /* default all +1 */
        if (i == 0) {
            /* For column 0 only: set k=0 to -1. */
            blk.qs[0] = 0xA8;  /* low 2 bits = 00 = code 0 = ternary -1 */
        }
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

    /* Run kv4 INT4 act variant (variant index 0 = primary). */
    bitnet_arc::kv4_variants[0].launch(qh, M, N, K, A_dev, B_dev, C_dev);

    std::vector<std::uint16_t> C_host(M * N);
    q.memcpy(C_host.data(), C_dev, C_host.size() * sizeof(std::uint16_t)).wait();

    /* Expected pattern (single -1 at B[k=0, n=0], rest +1, A all +1):
     *   c[m, 0]   = sum_k(a*b)= -1 + 255*1 = 254 for ALL m
     *   c[m, n>0] = 256 for all m
     * So column 0 should be all 254, columns 1..15 all 256. */
    std::fprintf(stderr,
                 "single-elem test: expected col 0 = 254, cols 1..15 = 256\n");
    int wrong = 0;
    for (unsigned m = 0; m < M; ++m) {
        std::fprintf(stderr, "  row %u: ", m);
        for (unsigned n = 0; n < N; ++n) {
            const float f = bitnet_arc_fp16_to_fp32(C_host[m * N + n]);
            std::fprintf(stderr, "%6.1f ", f);
            const float expect_val = (n == 0) ? 254.0f : 256.0f;
            if (std::fabs(f - expect_val) > 0.5f) ++wrong;
        }
        std::fprintf(stderr, "\n");
        if (m == 1) {
            std::fprintf(stderr, "  ... (rows 2..%zu suppressed)\n", M - 1);
            break;
        }
    }
    std::fprintf(stderr, "  total wrong (out of %zu): %d\n", M * N, wrong);

    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    return wrong == 0 ? 0 : 1;
}
