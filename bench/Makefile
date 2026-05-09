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
KERNEL_HDR := ../src/kernel_v0.h ../src/kernel_v1.h

harness_3way: harness_3way.cpp $(ORACLE_LIB) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) harness_3way.cpp $(ORACLE_LIB) -o $@

sweep_tile: sweep_tile.cpp $(KERNEL_LIB) $(ORACLE_LIB) $(HEADERS) $(KERNEL_HDR)
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) sweep_tile.cpp \
	    $(KERNEL_LIB) $(ORACLE_LIB) -o $@

# Task #155 profiler -- v0_BL ablation timing for design v2 input.
# Same toolchain as sweep_tile (icpx + Arc B60 host).
profile_v0_bl: profile_v0_bl.cpp $(KERNEL_LIB) $(ORACLE_LIB) $(HEADERS) $(KERNEL_HDR)
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) profile_v0_bl.cpp \
	    $(KERNEL_LIB) $(ORACLE_LIB) -o $@

# Task #160 / design v2 Phase 2a -- kernel_v2 section profiler.
# Phase 2a step 1: build (a) FULL only. Step 2 = --split-build adds
# per-section ratios via section-split kernel skeleton. See
# bench/profile_v2.cpp header + docs/design-v2-phase-2.md sec3.
profile_v2: profile_v2.cpp $(KERNEL_LIB) $(ORACLE_LIB) $(HEADERS) ../src/kernel_v2.h
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) profile_v2.cpp \
	    $(KERNEL_LIB) $(ORACLE_LIB) -o $@

# Task #160 / Phase 2a step 1.5 -- single-WG isolation test.
# Per @theta review #83 suggestion #1 : preliminary to step 2 split-
# build. Validates the intra-WG bottleneck hypothesis (vs an inter-WG
# contention alternative) by comparing per-WG time at WG_count = 1 vs
# WG_count = 64 across 2 K shapes (best K=4096, worst K=14336).
# See bench/profile_v2_wg_isolation.cpp header for full methodology.
profile_v2_wg_isolation: profile_v2_wg_isolation.cpp $(KERNEL_LIB) $(ORACLE_LIB) $(HEADERS) ../src/kernel_v2.h
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) profile_v2_wg_isolation.cpp \
	    $(KERNEL_LIB) $(ORACLE_LIB) -o $@

# Task #156 / design v2 Phase 0 -- joint_matrix availability probe.
# Standalone (no kernel_v0 lib needed): only links libc++ + the oracle
# fp16 helpers via header. Same icpx toolchain.
probe_joint_matrix: probe_joint_matrix.cpp $(HEADERS)
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) probe_joint_matrix.cpp -o $@

# Path A Phase 0 probe (post-v2-falsification): probe INT8 DPAS support
# on Arc B60 / Xe2. Standalone, no kernel lib dependency. See
# bench/probe_dpas.cpp header for full rationale + acceptance criteria.
probe_dpas: probe_dpas.cpp
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) probe_dpas.cpp -o $@

# Path A Phase 0.5 hardstop gate (per docs/design-v3.md sec6.0):
# validate INT8 DPAS throughput vs FP16 joint_matrix baseline.
# >=4x = PASS, 2-4x = MARGINAL, <2x = FAIL (escalate v4).
probe_dpas_throughput: probe_dpas_throughput.cpp
	$(CXX_SYCL) $(SYCLFLAGS) $(INCLUDES) probe_dpas_throughput.cpp -o $@

# Path A Phase 0 v4 hardstop: standalone OpenCL probe testing if
# cl_intel_subgroup_matrix_multiply_accumulate (DPAS direct, bypass
# SYCL joint_matrix wrapper) delivers >2x FP16 wrapper baseline.
# Standalone, no SYCL toolchain involvement; just libOpenCL.so + IGC.
probe_dpas_opencl: probe_dpas_opencl.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) probe_dpas_opencl.cpp -lOpenCL -o $@

# Build dependent libraries on demand.
$(ORACLE_LIB):
	$(MAKE) -C ../oracle

$(KERNEL_LIB):
	$(MAKE) -C ../src CXX=$(CXX_SYCL)

run: harness_3way
	./harness_3way

run-sweep: sweep_tile
	./sweep_tile

# W1.5 multi-shape sweep over the LLM-relevant shapes (k256, k4096-attn,
# k14336-FFN-M16, k14336-M64). Output goes to sweep_w15_llm_baseline.csv
# (overwrites). Run on a SYCL-visible device (Arc Pro B60).
run-sweep-llm: sweep_tile
	./sweep_tile --shapes-preset llm > sweep_w15_llm_baseline.csv 2>&1
	@echo "wrote sweep_w15_llm_baseline.csv"

# #147 stop-gate: parses a sweep CSV and emits the W1/W2 report.
# Default GATE_CSV picks up the smoke baseline; override on the
# command line: `make gate GATE_CSV=path/to/run.csv`.
GATE_CSV ?= sweep_w1_smoke_baseline.csv
gate:
	python3 gate_w1w2.py $(GATE_CSV)

gate-strict:
	python3 gate_w1w2.py --strict-gates $(GATE_CSV)

clean:
	rm -f harness_3way sweep_tile profile_v0_bl profile_v2 \
	      profile_v2_wg_isolation probe_joint_matrix probe_dpas

.PHONY: clean run run-sweep run-sweep-llm gate gate-strict
