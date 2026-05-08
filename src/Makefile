# bitnet-arc v0 SYCL kernel build.
#
# Requires oneAPI DPC++ (icpx) or LLVM clang++ with SYCL2020 plugin.
# Codeplay plugins extend this to AMD / NVIDIA targets without source
# changes (see oracle/Makefile rationale).
#
# Notes on flags:
#   -fsycl                : enable SYCL support (icpx default; clang
#                           with libsycl needs the explicit flag)
#   -fsycl-targets=...    : intel_gpu_pvc / intel_gpu_bmg / spir64 ;
#                           we leave default = spir64 (JIT) for now,
#                           AOT comes when we pin Arc B60 device id
#   -O2                   : conservative for the v0 baseline; #148
#                           sweep will compare with -O3 + loop unroll
#   -Wall -Wextra         : same diagnostic discipline as oracle
#   -std=c++17            : SYCL2020 minimum
#
# We intentionally do NOT pass -ffp-contract=off to the kernel TU.
# Voice 1 (FP32 maison) vs voice 2 (oracle dequant) bit-equality lives
# in the oracle, not here. The kernel is allowed to fuse / reorder for
# performance, with correctness checked against the FP64 BLAS voice
# (TOL_SYCL_VS_NUMPY_F64 = 1e-1) per oracle/tolerance.h.

CXX     ?= icpx
AR      ?= ar
CXXFLAGS:= -O2 -std=c++17 -Wall -Wextra -fsycl

INCLUDES := -I. -I..

OBJS    := kernel_v0.o kernel_v1.o
HEADERS := kernel_v0.h kernel_v1.h kernel_v0_sycl.hpp ../oracle/tq2_0.h

# Library name kept as v0 for backward compat with bench/Makefile (the
# archive aggregates both v0 and v1 kernel translation units; v1 is
# additive, no v0 changes per the design v1 contract).
libbitnet_arc_v0.a: $(OBJS)
	$(AR) rcs $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) libbitnet_arc_v0.a

.PHONY: clean
