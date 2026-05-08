/*
 * Internal SYCL header for bitnet-arc v0.
 *
 * The public kernel_v0.h forward-declares sycl_queue_handle so that
 * non-SYCL translation units (oracle code, CPU-only harnesses) do not
 * have to include <sycl/sycl.hpp>. The full definition lives here and
 * is only consumed by translation units that already pull in SYCL --
 * src/kernel_v0.cpp itself, and bench/sweep_tile.cpp.
 *
 * Including this header in a non-SYCL TU is a build error by design.
 */

#ifndef BITNET_ARC_SRC_KERNEL_V0_SYCL_HPP
#define BITNET_ARC_SRC_KERNEL_V0_SYCL_HPP

#include <sycl/sycl.hpp>

#include "kernel_v0.h"

namespace bitnet_arc {

class sycl_queue_handle {
public:
    sycl::queue q;

    sycl_queue_handle()
        : q(sycl::default_selector_v, sycl::property::queue::in_order{}) {}

    explicit sycl_queue_handle(const sycl::queue& src) : q(src) {}
};

} /* namespace bitnet_arc */

#endif /* BITNET_ARC_SRC_KERNEL_V0_SYCL_HPP */
