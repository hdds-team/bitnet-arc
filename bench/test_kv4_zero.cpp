/*
 * bench/test_kv4_zero.cpp -- isolated all-zero correctness probe.
 *
 * If kv4 has a layout / packing bug, an all-zero input must still
 * produce all-zero output (any layout permutation of zeros = zero).
 * This eliminates the entire layout question and isolates the
 * "kernel runs at all + writes correctly" axis from the "data
 * interpretation" axis.
 */

#include "../src/kernel_v0_sycl.hpp"
#include "../src/kernel_v4.h"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const std::size_t M = 16, N = 16, K = 256;

    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    bitnet_arc::sycl_queue_handle qh(q);

    /* All-zero inputs */
    auto A_dev = sycl::malloc_device<std::uint16_t>(M * K, q);
    auto B_dev = sycl::malloc_device<bitnet_arc_tq2_0_block>((K / 256) * N, q);
    auto C_dev = sycl::malloc_device<std::uint16_t>(M * N, q);

    /* Zero-fill A. */
    std::vector<std::uint16_t> A_zero(M * K, 0);
    q.memcpy(A_dev, A_zero.data(), A_zero.size() * sizeof(std::uint16_t)).wait();

    /* Zero-fill B (TQ2_0 blocks: d=0, qs=all 0). With qs=0, every code is 0
     * which maps to ternary -1 in the TQ2_0 convention -- BUT d=0 means the
     * effective weight is 0 anyway. So per-element contribution = 0 * activation = 0.
     * Output = 0 expected. */
    std::vector<bitnet_arc_tq2_0_block> B_zero((K / 256) * N);
    std::memset(B_zero.data(), 0, B_zero.size() * sizeof(bitnet_arc_tq2_0_block));
    q.memcpy(B_dev, B_zero.data(), B_zero.size() * sizeof(bitnet_arc_tq2_0_block)).wait();

    /* Zero-fill C buffer too (so post-run we can detect kernel actually wrote). */
    std::vector<std::uint16_t> C_init(M * N, 0xCAFE);  /* sentinel */
    q.memcpy(C_dev, C_init.data(), C_init.size() * sizeof(std::uint16_t)).wait();

    /* Run kv4. */
    bitnet_arc::run_kernel_v4(qh, M, N, K, A_dev, B_dev, C_dev);

    /* Read C output. */
    std::vector<std::uint16_t> C_host(M * N, 0xCAFE);
    q.memcpy(C_host.data(), C_dev, C_host.size() * sizeof(std::uint16_t)).wait();

    /* Check output: all FP16 bits should be 0 (= +0.0). */
    int n_zero = 0, n_sentinel = 0, n_other = 0;
    for (auto v : C_host) {
        if (v == 0)         ++n_zero;
        else if (v == 0xCAFE) ++n_sentinel;
        else                ++n_other;
    }
    std::fprintf(stderr,
                 "all-zero test: out of %zu elements: %d zero, %d sentinel (kernel didn't write), %d other (kernel wrote non-zero!)\n",
                 C_host.size(), n_zero, n_sentinel, n_other);
    std::fprintf(stderr, "first 8 outputs: %04x %04x %04x %04x %04x %04x %04x %04x\n",
                 C_host[0], C_host[1], C_host[2], C_host[3],
                 C_host[4], C_host[5], C_host[6], C_host[7]);

    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    if (n_other > 0) {
        std::fprintf(stderr, "FAIL: kernel writes non-zero on all-zero input\n");
        return 1;
    }
    if (n_sentinel > 0) {
        std::fprintf(stderr, "FAIL: kernel didn't write some output positions\n");
        return 2;
    }
    std::fprintf(stderr, "PASS: kernel correctly outputs all-zero.\n");
    return 0;
}
