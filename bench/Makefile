# bitnet-arc bench harness Makefile.
#
# Builds harness_3way and links against oracle/liboracle.a.
#
# Voice 3 (numpy float64 BLAS) is invoked at runtime via subprocess to
# oracle/numpy_xcheck.py, so python3 + numpy must be on PATH. No
# build-time dependency on numpy.

CXX     ?= g++
CXXFLAGS:= -O2 -std=c++17 -Wall -Wextra -Wpedantic

INCLUDES := -I.. -I../oracle

ORACLE_LIB := ../oracle/liboracle.a
HEADERS    := ../oracle/fp16.h \
              ../oracle/tq2_0.h \
              ../oracle/fp32_matmul.h \
              ../oracle/tolerance.h

harness_3way: harness_3way.cpp $(ORACLE_LIB) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) harness_3way.cpp $(ORACLE_LIB) -o $@

# Build the oracle library on demand.
$(ORACLE_LIB):
	$(MAKE) -C ../oracle

run: harness_3way
	./harness_3way

clean:
	rm -f harness_3way

.PHONY: clean run
