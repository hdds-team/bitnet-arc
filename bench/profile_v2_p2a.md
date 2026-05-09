# Phase 2a Profiler Report — kernel_v2 Per-Section Breakdown

**Date:** 2026-05-09
**Hardware:** Intel Arc Pro B60 Graphics (Xe2 / Battlemage)
**Profiler:** `bench/profile_v2.cpp` build (a) FULL + (b) SECTION-SPLIT
**Method:** SYCL events with `enable_profiling`, kernel
`command_start`/`command_end` timestamps (excludes launch/dispatch
latency). 5 warmup + 50 timed iterations per shape per section.
**Inputs:** brief `docs/design-v2-phase-2.md` §3-§4 (ratified `13f188b`)
+ Phase 2a step 1 result `fdc1747` (sub-saturation excluded) + step
1.5 single-WG bench `923929d` (intra-WG bottleneck confirmed at K=14336).

---

## Headline verdict

**Top-1 bottleneck: DEQUANT** (§3.1 cooperative TQ2_0 → FP16) at
**~55% of t_full** across all 4 K≥4096 shapes.

**Top-2 bottleneck: LOADA** (§3.2 cooperative SLM A load) at
**~37% of t_full**.

**MMA dispatch ELIMINATED** as a candidate: the joint_matrix MMA
section accounts for only **1.1-3.9% of t_full**. The XMX compute
path itself is fast; the bottleneck is entirely in the SLM-fill
phases that feed it.

---

## Per-shape breakdown (5 warmup + 50 timed)

| Shape | t_full (ms) | dequant | loadA | mma | store | barrier |
|-------|-------------|---------|-------|-----|-------|---------|
| smoke 16x16x256 | 0.20 | 50.4% | 31.0% | 1.7% | 3.2% | 13.5% |
| attn 16x64x4096 | 1.49 | 54.5% | 34.2% | 1.1% | 0.2% | 10.1% |
| ffn 16x64x14336 | 5.14 | 55.1% | 34.4% | 1.1% | 0.1% | 9.3% |
| ffn 64x64x14336 | 5.15 | 55.5% | 36.6% | 2.1% | 0.1% | 5.6% |
| subsat 64x256x14336 | 5.16 | 55.9% | 39.3% | 3.9% | 0.1% | 0.9% |

Numbers are stable across re-runs (variance < 1% on grand shapes).
Smoke shape (16×16×256, single WG, single chunk) is included for
sanity but has higher relative noise.

### Bias accounting

`barrier` column = `t_full - sum(4 sections)`, inferred by
subtraction. Per the brief §3 + review #2 nit, build (b) carries
three stacked biases:
1. **Cross-section fusion lost** — the compiler cannot fuse adjacent
   sections that the production kernel fuses inside one parallel_for.
2. **Global-memory SLM substitute** — inter-section state crosses the
   kernel boundary via USM device scratch, replacing `local_accessor`'s
   role. Adds memory traffic absent in production. Mostly affects
   dequant + loadA sections.
3. **MMA section global-load bias** — the production kernel issues
   `joint_matrix_load` from SLM (~4-cycle L1 latency); the split MMA
   section issues from USM global, which hits L2 or DRAM. **Inflates**
   the MMA section's measured time vs production. The reported "MMA =
   1-4% of t_full" is therefore an *upper bound*; true production MMA
   is even smaller. This only strengthens the §4.3 elimination — MMA
   is not the bottleneck.

The `barrier` column is NOT the production sub_group_barrier overhead
in isolation. Production barrier overhead estimate: ~14 ns/barrier ×
32 barriers/chunk × chunks_per_col = ~25 µs = ~0.5% of t_full at
K=14336. The reported barrier_inf ratios (1-10%) are an upper-bound
residual that absorbs real barriers + the residue of the 3 biases.

The headline fact remains valid: **dequant + loadA together account
for ~90% of t_full** under any reasonable bias allocation. MMA cannot
be the bottleneck at ≤1-4% (and is likely lower in production).

---

## Decision-tree mapping (brief §4)

| Hypothesis | Brief § | Status post-Phase-2a |
|-----------|---------|----------------------|
| Sub-saturation | §4.1 | **EXCLUDED** by step 1 + step 1.5 |
| Barriers (Lecture A overhead) | §4.2 | **CANDIDATE** — dequant carries 2 barriers per N column × 16 columns/chunk = 32 barriers/chunk in Lecture A |
| MMA dispatch overhead | §4.3 | **EXCLUDED** — MMA = 1-4% of t_full |
| SLM aliasing | §4.4 | **CANDIDATE** — dequant + loadA both write to SLM heavily; bank conflicts plausible |

The two surviving candidates (§4.2 and §4.4) both manifest in the
SLM-fill phases. They are not mutually exclusive: a Lecture B switch
(§4.2 fix) would reduce barriers AND change the SLM access pattern
(§4.4 surface), addressing both hypotheses with one change.

### Composite or single?

`top-1 < 50%` only on the smoke shape (which is dominated by launch
overhead anyway). On all 4 K≥4096 shapes top-1 is 54-56%, between
the "structurally balanced" threshold (50% in the brief §4.5 text)
and a clear "single dominant" regime. The kernel has one section
slightly above half the time and a strong second; this is an
**asymmetric composite**, not a structural balance.

Phase 2b strategy implication: **fixing top-1 alone may yield
~1.5-2x improvement** (eliminating most of dequant's 55%), but
catching the W2 ≥1.5× v0_BL gate cleanly likely requires
addressing top-2 (loadA) too.

---

## Phase 2b recommendations

### Priority 1: switch to Lecture B for dequant (§3.1)

In Lecture A the 16 lanes co-decode ONE block per inner iteration,
looping over `TILE_N=16` columns (= 16 cooperative passes per
K_CHUNK, with 32 sub-group barriers each). In Lecture B each lane
owns one column; all 16 columns are decoded in parallel in a single
pass. Eliminates the 32 internal barriers per chunk and changes the
SLM access pattern to lane-private writes (no bank conflicts on
write-out).

Expected gain: **2-3× on the dequant section** (the barrier
overhead + serialized N-column loop both go away). Translates to
~30-40% reduction on t_full, well above the W2 1.5× gate.

LOC estimate: ~40 LOC kernel rewrite in `src/kernel_v2.cpp`
§3.1 (the surrounding kernel structure is unchanged). Plus a SLM
budget recheck (Lecture B may need a per-lane qs scratch instead
of a shared 64-byte one — verify against the 64 KB / WG hard limit).

### Priority 2: vectorized loadA (§3.2)

The current cooperative A load is lid-strided scalar copies of
`std::uint16_t` (FP16 bit-pattern) from global to SLM. On Arc B60
each lane issues 16 scalar loads per chunk (`A_SLAB_ELEMS / SG_SIZE
= 4096 / 16 = 256` elements / lane / chunk total).

Switching to `sycl::vec<sycl::half, 8>` or `sub_group::load<8>`
would issue 1/8 the number of memory transactions and let the GPU
coalesce them into wider loads. Even a 2-3× speedup on loadA shaves
another ~15-20% off t_full.

LOC estimate: ~10-15 LOC change in `src/kernel_v2.cpp` §3.2.

### Priority 3 (deferred): geometry sweep §5

The brief's geometry sweep would have been the right move if §4.1
(sub-saturation) were the bottleneck. With the actual top-1 in
SLM-fill cooperative work, larger TILE_M/TILE_N changes the
per-tile work but does not address the dominant bottleneck. Defer
to Phase 3 perf optim once Lecture B + vec loadA land.

---

## What this rules out / confirms for design v2

**Confirms:** the v2 XMX path is the right architectural choice. MMA
itself is fast (1-4% of t_full); the kernel is bottlenecked by the
data-staging *into* MMA, not by MMA itself.

**Rules out:** Path A (INT8 DPAS) and Path C (custom packing) being
necessary as v2 fallback. The compute path is not the issue. Both
remain valid v3+ options for further compute-side speedups but
should not be promoted to "v2 falsified, pivot to A/C" — Phase 2a
shows a clear v2-internal optimization path with realistic gains.

**Phase 2 W2 gate** (1.5× v0_BL on `(16, 64, 14336)`):
- Current v2: 5.14 ms vs v0_BL 2.19 ms = **0.43×** (i.e. 2.34× *slower*)
- Target W2: ≥3.28 ms (=2.19/1.5 → wait, ≥1.5× faster means
  v2 t_med ≤ v0_BL/1.5 = 1.46 ms)
- Gap to close: 5.14 → 1.46 ms = **3.5× speedup needed**
- Realistic from Lecture B + vec loadA: ~30-40% of dequant + ~15-20%
  of loadA reduction → ~50% of t_full removed → 5.14 → ~2.6 ms
  (1.97× faster than current v2; 0.84× v0_BL = still ~1.2× *slower*)

This single-iteration estimate suggests **Phase 2b will close most
but not all of the W2 gap**. The remaining ~80% would need a
second-order optim (vec loads on B-write inside dequant, or
algorithmic change in the inner MMA pipeline). Phase 2b is not
guaranteed to pass W2 gate at first try; the brief §7 hardstop
applies.

---

## Caveats

1. **Smoke shape (16×16×256)** is dominated by launch overhead and
   has higher relative noise. Use only as a sanity check, not for
   ratio quotation in design discussions.

2. **`barrier_inferred` is a residual, not a measurement.** Negative
   values would signal that build (b) bias exceeds production
   barrier cost. All values here are positive but small (<10% on
   grand shapes), suggesting the build (b) bias is modest in
   absolute terms but eats the entire production barrier budget.

3. **Build (b) section absolutes are upper bounds.** Ratios are the
   reportable quantity. The narrative uses ratios throughout and
   only invokes absolutes for context.

4. **Phase 2b fixes must be re-validated on build (a)**, not (b).
   Build (b) is for bottleneck identification; production-truth
   absolutes only come from build (a) (the un-instrumented kernel).

---

## Raw data

`bench/profile_v2.csv` and `bench/profile_v2.log` contain the full
50-iter run data. Both are `.gitignore`d per repo policy. Reproduce
with:

```bash
cd bench && make profile_v2 && ./profile_v2 --split-build \
    > profile_v2.csv 2> profile_v2.log
```

---

## Phase 2b unblock condition

Phase 2a delivers the data the brief §3-§4 asked for. Phase 2b
proceeds with:
- Top-1 fix: Lecture B switch on §3.1 dequant (~40 LOC)
- Top-2 fix: vectorized loadA on §3.2 (~15 LOC)
- Re-bench on build (a) only after each fix
- W2 gate evaluation per brief §7

Decode-vs-prefill scope question (§6.1) remains open for @naskel,
but is **not gating Phase 2b** — Lecture B + vec loadA are
prefill-and-decode-agnostic optimizations on the SLM-fill pipeline.
