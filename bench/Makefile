# bitnet-arc bench harness Makefile.
#
# Targets:
#   harness_3way -- 3-voice cross-check (CPU only, gcc/clang)
#   sweep_tile   -- #148 SYCL tile sweep (requires icpx + Arc device)
#
# harness_3way links oracle/liboracle.a; voice 3 uses numpy via subprocess
# to oracle/numpy_xcheck.py at run time, so python3 + numpy must be on
# PATH. No build-time numpy dependency.
#
# sweep_tile additionally links src/libbitnet_arc_v0.a (the SYCL
# kernel variants) and is built with -fsycl. Requires CXX=icpx (or a
# clang++ with the SYCL2020 plugin). Run on a host with a SYCL-visible
# device -- we target Arc Pro B60 via Level Zero.

CXX        ?= g++
CXXFLAGS   := -O2 -std=c++17 -Wall -Wextra -Wpedantic

# SYCL toolchain (overridable, but icpx is the only supported path right
# now). Build sweep_tile via `make CXX_SYCL=icpx sweep_tile`.
CXX_SYCL   ?= icpx
SYCLFLAGS  := -O2 -std=c++17 -Wall -Wextra -fsycl

INCLUDES := -I.. -I../oracle -I../src

ORACLE_LIB := ../oracle/liboracle.a
KERNEL_LIB := ../src/libbitnet_arc_v0.a
HEADERS    := ../oracle/fp16.h \
              ../oracle/tq2_0.h \
              ../oracle/fp32_matmul.h \
              ../oracle/tolerance.h
KERNEL_HDR := ../src/kernel_v0.h

harness_3way: harness_3way.cpp $(ORACLE_LIB) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) harness_3way.cpp $(ORACLE_LIB) -o $@

sweep_tile: sweep_tile.cpp $(KERNEL_LIB) $(ORACLE_LIB) $(HEADERS) $(KERNEL_HDR)
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) sweep_tile.cpp \
	    $(KERNEL_LIB) $(ORACLE_LIB) -o $@

# Build dependent libraries on demand.
$(ORACLE_LIB):
	$(MAKE) -C ../oracle

$(KERNEL_LIB):
	$(MAKE) -C ../src CXX=$(CXX_SYCL)

run: harness_3way
	./harness_3way

run-sweep: sweep_tile
	./sweep_tile

clean:
	rm -f harness_3way sweep_tile

.PHONY: clean run run-sweep
