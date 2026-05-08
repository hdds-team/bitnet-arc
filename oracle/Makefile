# bitnet-arc oracle reference library.
#
# Builds a small static lib (liboracle.a) consumed by:
#   - bench/harness_3way.cpp (W1 critical path)
#   - upstream_gating/gate.c (when llama.cpp pin lands)
#
# Critical build flag: -ffp-contract=off
# ---------------------------------------
# We rely on voice 1 vs voice 2 being bit-equal at FP32 (same canonical
# m-n-k loop, only weight source differs). gcc/clang are allowed by the
# C standard to fuse `(a * w) + acc` into a single rounded FMA at
# compile time, which would silently round differently than the
# unfused two-step we wrote. -ffp-contract=off forbids that.
#
# Other notable flags:
#   -fno-strict-aliasing : we round-trip floats via memcpy in fp16.h
#                          and never want the optimizer to assume the
#                          aliased uint32_t / float overlap is invalid.
#   -Wpedantic           : we use only ISO C11 features in the oracle
#                          (no zero-length arrays, no VLA, no GNU
#                          extensions in source).

CC      ?= gcc
AR      ?= ar

CFLAGS  := -O2 \
           -Wall -Wextra -Wpedantic -std=c11 \
           -ffp-contract=off \
           -fno-strict-aliasing

OBJS    := tq2_0.o fp32_matmul.o
HEADERS := fp16.h tq2_0.h fp32_matmul.h tolerance.h

liboracle.a: $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# upstream_gating/gate -- one-shot bit-identical validator.
# Vendored upstream code lives in upstream_gating/upstream_ref.h.
upstream_gating/gate: upstream_gating/gate.c upstream_gating/upstream_ref.h $(HEADERS) tq2_0.o
	$(CC) $(CFLAGS) -I. upstream_gating/gate.c tq2_0.o -lm -o $@

gate: upstream_gating/gate

clean:
	rm -f $(OBJS) liboracle.a upstream_gating/gate

.PHONY: clean gate
