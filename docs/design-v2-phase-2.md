# Phase 2 Implementation Brief — Profile, Diagnose, Tune (or Falsify)

**Parent design:** `docs/design-v2.md` §3 (ratified `e3fb4f2`, math fix
`806be5e`, body fixes `a5d4283`).
**Phase 1 record:** `8e83745` (kernel_v2 W1 correctness 4/4 pass, perf
2.3-2.6x SLOWER than v0_BL scalar — see §2 below).
**Task ID:** #158.
**Reviewers:** @sonnet, @beta, @haiku (same triplet as Phase 1, drill
known; @theta offline this session).

---

## 1. Scope

Phase 1 confirmed correctness (joint_matrix on Xe2 works, FP16/FP32 acc
combo OK, SLM staging + lane mapping race-free) but the naive XMX
kernel is **2.3-2.6x SLOWER than v0_BL scalar**. Inversion totale vs
design v2 §2.1 estimate (5-15x faster cible).

Phase 2 is **profile-first, then a single tuning pass**, time-boxed at
2 days. Either the bottleneck is identifiable + fixable inside the
XMX path, or we falsify Phase 1 honestly and pivot per design v2 §3
escalation ladder.

- **In scope:**
  - `bench/profile_v2.cpp` (~50 LOC) — SYCL events instrumentation
    around kernel_v2 inner sections (dequant / inner MMA / barrier
    wait / staging / final store).
  - `bench/profile_v0_bl.cpp` (extension) — same per-section
    instrumentation on v0_BL for direct comparison (the existing
    `profile_v0_bl.md` was peak-gap deduction, never measured).
  - Geometry sweep — register additional `kv2_variants[]` for
    TILE_M/N pairs in {32x32, 32x16, 16x64, 16x32}. K_CHUNK stays at
    256 (TQ2_0 block size, fixed).
  - Decision tree execution: identify dominant bottleneck from
    profiler output, apply the matching fix, re-bench.

- **Out of scope (gated behind Phase 2 falsification):**
  - Path A (INT8 DPAS) — disqualified at design v2 §2.2 unless
    Phase 2 falsifies the FP16 path entirely.
  - Path C (custom packing) — same gate as Path A.
  - BF16 path — Phase 0 confirmed it works, deferred to Phase 3+.
  - Multi-WG cooperative output tile (split-K reduce) — only if
    geometry sweep alone proves insufficient.
  - Decode regime (M=1) — see §6, prefill-only by design.

---

## 2. Phase 1 results (the inputs)

From `8e83745` bench output, run by @claude-opus on Arc B60:

| Shape           | v0_BL t_med | v2 t_med  | ratio       | max_rel_err |
|-----------------|-------------|-----------|-------------|-------------|
| (16, 16, 256)   | 0.076 ms    | 0.199 ms  | 2.6x slow   | 0.00084     |
| (16, 64, 4096)  | 0.630 ms    | 1.477 ms  | 2.3x slow   | 0.00097     |
| (16, 64, 14336) | 2.185 ms    | 5.134 ms  | 2.35x slow  | 0.00095     |
| (64, 64, 14336) | 2.172 ms    | 5.141 ms  | 2.37x slow  | 0.00096     |

**Correctness:** 4/4 PASS (all shapes ≤1e-3, well under the 1e-2 W1
gate; smoke shape passes the strict 1e-3 sub-gate).

**Perf signal:** 2.3-2.6x slower across all shapes. Two analytical
observations from @sonnet + @alpha reads (commit thread):

1. **Ratio constant ≈ 2.35x across K = {256, 4096, 14336}.** A
   fixed-cost launch overhead would yield a *decreasing* ratio with
   larger K. Ratio constant means the overhead scales linearly with
   work — bottleneck is **inside the inner loop**, not setup.

2. **Sub-saturation Arc B60.** TILE_M=TILE_N=16 yields 1-16 WGs total
   on the bench shapes (160 XMX engines on Arc B60). v0_BL scalar
   saturates via per-output-element parallel threads (M*N work items,
   not WG-bound). v2 cannot expose enough parallelism with TILE_M/N=16
   on M ≤ 64 inputs.

   | Shape           | TILE_M=16 row tiles | TILE_N=16 col tiles | WGs total |
   |-----------------|---------------------|---------------------|-----------|
   | (16, 16, 256)   | 1                   | 1                   | 1         |
   | (16, 64, 4096)  | 1                   | 4                   | 4         |
   | (16, 64, 14336) | 1                   | 4                   | 4         |
   | (64, 64, 14336) | 4                   | 4                   | 16        |

These two observations are **hypotheses**, not yet measurements.
Phase 2 begins by discriminating them.

---

## 3. Profiler harness (the first deliverable)

`bench/profile_v2.cpp` instruments the kernel with SYCL events to
break down per-K_CHUNK iter into:

| Section                   | Source lines (kernel_v2.cpp) | Counter         |
|---------------------------|------------------------------|-----------------|
| Cooperative dequant       | §3.1 (qs load + barrier + decode + write) | `dequant_us` |
| A SLM cooperative load    | §3.2                         | `load_a_us`     |
| Inner MMA loop (16 frags) | §3.3                         | `mma_us`        |
| Sub-group barrier per K_CHUNK | §3.4                     | `barrier_us`    |
| Final store (FP32 staging + scatter) | §3.5              | `store_us`      |

**Implementation pattern** (~50 LOC, mirrors `bench/profile_v0_bl.cpp`
existing harness):

- **Methodology pinned pre-implementation** (per @beta review #79 nit
  on §3 instrumentation bias + @sonnet barrier_us observability nit):
  - **Method = (c) split build with documented bias + reference
    baseline.** The production kernel does cross-section fusion that
    a split build loses; we acknowledge this rather than hide it.
    The split build's per-section *ratios* are reported with an
    explicit "cross-section fusion lost" disclaimer; absolute timing
    comes from the full-kernel reference run (a) below.
  - **Two profiler builds:**
    - (a) **Full kernel** with one SYCL event marker around the
      whole `parallel_for` invocation. Measures the truthful total
      time for the production code (no instrumentation bias). Used
      as the denominator for `% of total`.
    - (b) **Section-split kernel** — same kernel logic split into
      per-section `parallel_for` launches with shared SLM region
      preserved across launches by re-binding the same
      `local_accessor`. Each launch has its own SYCL event; we
      report ratios `t_section / Σt_sections` from this build, then
      apply those ratios against (a)'s total to estimate per-section
      absolute time. **Bias caveat:** (b) is ~10-30% slower than
      (a) due to extra launches + lost cross-section fusion; we
      report ratios not absolute (b) times, and document the bias.
  - **`barrier_us` specifically** (per @sonnet nit): SYCL events
    alone CANNOT isolate `sub_group_barrier` since it's a GPU
    instruction inside a kernel, not a kernel boundary. Two paths:
    - **Inferred by subtraction** (default): `barrier_us = total −
      Σ(other 4 sections)`. Works as long as the other 4 sections
      cover the kernel cleanly with no overlap.
    - **Level Zero XPU timeline** or **VTune GPU perf counters**:
      direct measurement of barrier stall cycles. Out-of-scope for
      Phase 2 day-1 profiler harness (extra tooling install +
      parsing budget); used as confirmation if inferred `barrier_us`
      lands suspiciously high (>40% of total).
  - **Rejected alternatives** (for the record):
    - "Comment-out sections + subtraction" — invalid for interlocked
      sections (commenting dequant means MMA reads garbage).
    - "Dummy kernels" — measure dispatch overhead, not section time.
      Terminology deprecated.
- Run on the 4 W1 shapes (16x16x256, 16x64x4096, 16x64x14336,
  64x64x14336) + 1 large shape (64x256x14336) for sub-saturation
  isolation.
- Output: per-shape per-section CSV + a derived `% of total` column.

**Additional measurements** (within the same harness):

1. **Occupancy probe** — `sycl::queue::get_device().get_info<...max
   work group...>` + actual launched WG count vs theoretical max
   concurrent WGs on Arc B60. Output: `actual_wg / theoretical_max`
   ratio.
2. **Memory bandwidth** — `B_blocks` bytes read total / kernel time.
   Compare to Arc B60 spec peak (456 GB/s GDDR6, per design v2 §1).
3. **v0_BL per-section breakdown** (extension of
   `bench/profile_v0_bl.cpp`) — same shape coverage. The existing
   `profile_v0_bl.md` deduced compute-bound from the peak gap; we
   never measured *what* dominates inside v0_BL.

---

## 4. Decision tree (post-profiler)

The profiler output discriminates the 4 main hypotheses. Each has a
distinct fix path within Phase 2 scope:

### 4.1 Bottleneck = sub-saturation (low actual_wg / theoretical_max)

**Signal:** Occupancy ratio < 0.3, ratio constant across K, MMA time
per fragment is fast in absolute terms.

**Fix:** Geometry sweep (§5). Larger TILE_M/N exposes more output
tiles or amortizes setup.

**Phase 2 path:** registered variants {32x32, 32x16, 16x64, 16x32};
re-bench all shapes; expect 1.5-3x improvement on M >= 32 shapes.

### 4.2 Bottleneck = barriers (Lecture A overhead dominant)

**Signal:** `barrier_us` >= 30% of total per-K_CHUNK time;
`dequant_us` is also high (cooperative dequant requires the barriers).

**Fix:** Re-evaluate Lecture A (cooperative dequant in SLM) vs
Lecture B (lane = output column, no SG cooperation, dequant inline).
The Lecture A choice was made in design v2 §3 ratification; if it's
the bottleneck, switching to Lecture B is a fold.

**Phase 2 path:** ~40 LOC kernel rewrite to Lecture B; re-bench;
discriminate from sub-saturation by holding TILE_M/N=16 and only
flipping the dequant strategy.

### 4.3 Bottleneck = MMA dispatch overhead (tiny fragment cost)

**Signal:** `mma_us` per fragment-K is high relative to FLOPs done
(per-fragment 16x16x16 MMA does 8192 FLOPs; if each takes >50 ns,
dispatch is dominant).

**Fix:** Coalesce fragment-K calls (joint_matrix supports
larger fragment shapes on Xe2 if exposed; or manually unroll the
inner 16-fragment loop). May also be addressed by §5 geometry change.

### 4.4 Bottleneck = SLM aliasing inhibits compiler optim

**Signal:** `dequant_us` and `store_us` both high; comparison run
with separate `local_accessor<sycl::half>` (no reinterpret_cast)
shows >20% improvement.

**Fix:** Drop the SLM scratch reuse pattern, use separate typed
accessors. +1 KB SLM (still in 64 KB budget per Phase 1 §5).

### 4.5 Composite bottlenecks (priority-ranked sequential)

Per @beta review #79 nit on §4 exclusivity: bottlenecks in real GPU
kernels are often co-dominant (e.g. occupancy=0.5 simultaneous with
barriers=25%). The decision tree is **priority-ranked sequential**,
not exclusive:

1. **Identify top-1 bottleneck** from profiler output (largest %).
2. **Apply the matching §4.X fix** for it.
3. **Re-profile + re-bench.** If gap is now >= 1.5x v0_BL on
   headline shape (16, 64, 14336), Phase 2 PASSES.
4. **If still <1.5x:** identify top-2 bottleneck from re-profile,
   apply its §4.X fix, re-profile + re-bench.
5. **End of day-2 hardstop** (§7): if still <1.5x after fixing top-1
   and top-2, falsify honestly. No third fix attempt — that's
   composite-bottleneck regime, geometry+algorithm sweep alone
   cannot close the gap.

**Threshold for "single dominant section"** (informal): one section
> 50% of total. If the top-1 is <50%, the kernel is structurally
balanced and the gap is composite. In this case, fix top-1 anyway
(largest bang/buck) and re-profile to confirm; if re-profile shows
the structural gap remains, **falsify per §7** without a top-2 attempt.

---

## 5. Geometry sweep candidates (Phase 2 work, conditional on §4.1)

Registered as additional `kv2_variants[]` entries, each compatible
with `parallel_for` shape filter (skip if M % TILE_M != 0 or
N % TILE_N != 0):

| Variant name           | TILE_M | TILE_N | SG_SIZE | K_CHUNK |
|------------------------|--------|--------|---------|---------|
| v2_16x16_sg16_k256 (P1)| 16     | 16     | 16      | 256     |
| v2_32x16_sg16_k256     | 32     | 16     | 16      | 256     |
| v2_16x32_sg16_k256     | 16     | 32     | 16      | 256     |
| v2_32x32_sg16_k256     | 32     | 32     | 16      | 256     |
| v2_16x64_sg16_k256     | 16     | 64     | 16      | 256     |

K_CHUNK stays at 256 (TQ2_0 block size, fixed by quant format).

**Geometry implications:**
- TILE_M=32: needs 2 row tiles per WG, doubled accumulator fragments
  (2 mC instead of 1). Doubles the register pressure but halves the
  MMA dispatch overhead per output column. **A_slab cooperative load
  pattern also changes** (per @sonnet review #79 nit): with 16 lanes
  and 2 row tiles, each lane loads 2 row contributions instead of 1
  (or the load loops twice over the lid stride). The load remains
  coalesced as long as the per-lane K-stride matches; verify at
  implementation that the per-lane K offset wraps correctly across
  the 2 row tiles.
- TILE_N=32: needs 2 col tiles per WG, doubled mB loads per fragment-K.
- TILE_N=64: needs 4 col tiles, may overflow SLM budget for B_slab
  (4 × 8KB = 32KB + A_slab 8KB = 40KB, still <64KB).

**SG_SIZE constraint:** joint_matrix on Xe2 expects sg_size=16. We do
NOT explore sg_size=32 in Phase 2 (would require multi-SG WG +
revisiting §3.4 barriers — Phase 3+ scope).

**Sweep pruning:** the bench harness skips variants where the input
shape is incompatible (M % tile_M != 0, etc.). Same pattern as kv0/kv1.

---

## 6. Prefill-only scope + decode fallback + Phase 2a/2b split

**Strategic clarification** (per @beta + @alpha read post-Phase 1):

Kernel v2 with TILE_M=16 minimum **cannot run M=1 decode regime**
(single-token autoregressive inference). For LLM workloads:

| Regime    | M typical    | Kernel choice              |
|-----------|--------------|----------------------------|
| Prefill   | M = seq_len, often >= 64 | **kernel_v2** (this design) |
| Decode    | M = 1        | **kernel_v0_BL** (scalar)  |

Phase 2 commits to **prefill-only scope** for kernel_v2. The dispatch
layer (out of scope here, Phase 4+) will route by M:
- M >= 16: kernel_v2 with the largest applicable TILE_M variant
- M < 16: kernel_v0_BL fallback

### 6.1 Phase 2a/2b split (per @beta review #79 nit)

To avoid wasting day-1 effort if @naskel pivots to decode-priority,
Phase 2 splits into two sub-phases with distinct gating:

| Sub-phase | Scope                                    | Decode-gating? | Day budget |
|-----------|------------------------------------------|----------------|------------|
| **2a**    | Profiler harness ship + v2 + v0_BL breakdown measured | **NO — decode-agnostic** (data useful in either scenario) | 1 day |
| **2b**    | Geometry sweep + §4.X fix + W2 perf gate | **YES — gated on @naskel = prefill-relevant** | 1 day |

**Phase 2a unblocks immediately** post-brief ratification + parser-
hardening fix (§10 backlog item 2). Output:
- Per-section breakdown for v2 on 4 W1 shapes + 1 large shape
- Per-section breakdown for v0_BL (informs decode kernel design too)
- `bench/profile_v2_p2a.md` report

**Phase 2b is conditional:**
- **Naskel confirms prefill-relevant** -> Phase 2b proceeds as
  scoped (§4 decision tree, §5 geometry sweep, §7 hardstop gate).
- **Naskel pivots to decode-priority** -> Phase 2b skipped.
  Phase 2a profiler data feeds a separate `kernel_v3_decode` brief
  (M=1 fast-path, possibly INT8 DPAS scalar instead of XMX matrix).
  Same 2-day total budget, just allocated differently.

**Same hardstop applies** in either branch: end-of-day-2,
Phase 2 wraps with either a W2-pass commit or a falsification report.

---

## 7. Time-box hardstop & falsification gate

Per design v2 §3 escalation ladder:

| Phase | Result    | Action                                          |
|-------|-----------|-------------------------------------------------|
| v1    | 1.07x v0_BL max | Falsified, pivot to v2 (XMX FP16)         |
| v2 P1 | 0.4x v0_BL (2.5x slower) | Diagnose first, then decide      |
| v2 P2 | TBD (this brief) | Hardstop 2 days post-profiler tuning    |

**2-day hardstop semantics** (per §6.1 Phase 2a/2b split):

Day 1 (Phase 2a, decode-agnostic): profiler harness ships, v2 + v0_BL
per-section breakdown measured on all 4 W1 shapes + 1 large shape.
Phase 2a output is independent of @naskel decode-vs-prefill call.

Day 2 (Phase 2b, prefill-conditional): apply top-1 §4.X fix, re-
profile, if still <1.5x apply top-2 §4.X fix per §4.5 priority-ranked
sequential, re-bench. **Skipped entirely** if @naskel pivots decode-
priority (Phase 2a data instead seeds `kernel_v3_decode` brief).

**End of day 2 gate:**
- **>=1.5x v0_BL on (16, 64, 14336)** (headline shape, prefill FFN
  width): Phase 2 PASSES, proceed to Phase 3 (multi-tile sweep
  perf optim, BF16 path, dispatch layer).
- **<1.5x v0_BL after best fix:** Phase 2 falsified honestly.
  Pivot per design v2 §3 ladder:
  - Path A (INT8 DPAS) — re-evaluate now that we have profiler data
  - Path C (custom packing) — alternative to A
  - Hard pause + design v2.1 brief (multi-month scope, naskel call)

**No endless tuning.** Discipline lesson from v1: do not pursue
speculative optim past the hardstop without explicit naskel call.

---

## 8. Acceptance criteria (W2 gate)

| Check                                  | Threshold                  | Action on fail        |
|----------------------------------------|----------------------------|-----------------------|
| Profiler harness runs clean            | All 4 shapes + 1 large     | Block Phase 2 step 2  |
| v2 best variant on (16, 64, 14336)     | >= 1.5x v0_BL              | Falsify per §7        |
| v2 best variant on (64, 64, 14336)     | >= 1.5x v0_BL              | Falsify per §7        |
| Correctness (all variants, all shapes) | max_rel_err <= 1e-2        | Block Phase 2 step 2  |
| Smoke (16, 16, 256)                    | max_rel_err <= 1e-3        | Investigate dequant   |

**Stretch gates** (informational, not blocking):
- 5-15x v0_BL on (16, 64, 14336) — the original design v2 §2.1 cible
- 10x+ v0_BL on (64, 64, 14336) — saturated regime ideal

---

## 9. Out of scope (avoid re-litigation)

- **Path A (INT8 DPAS)** — disqualified at design v2 §2.2 unless
  Phase 2 falsifies the FP16 path entirely. Re-evaluation deferred
  to falsification post-§7.
- **Path C (custom packing)** — same gate as Path A.
- **BF16 path** — Phase 0 confirmed it works, deferred Phase 3+.
- **Decode regime (M=1)** — see §6, prefill-only by design.
- **Multi-WG cooperative output tile (split-K reduce)** — only if
  geometry sweep §5 alone proves insufficient AND profiler points
  to occupancy as primary bottleneck.
- **sg_size != 16** — joint_matrix on Xe2 expects sg=16, exploring
  multi-SG WG is Phase 3+ scope.
- **Dispatch layer (M-based routing)** — Phase 4+, requires Phase 2
  to ship a viable v2 first.

---

## 10. Phase 2 backlog folded

Items deferred from Phase 1 review thread, integrated into Phase 2
scope where applicable:

1. **SLM wrapper helper** (per @beta nit, post-Phase 1 #78):
   `reinterpret_cast<T*>(local_accessor<storage_t>)` pattern appears
   3x in kernel_v2 (mA load, mB load, final-store FP32 staging).
   Refactor into a single inline helper (`slm_typed_ptr<T>(accessor)`
   or similar) reduces aliasing surface and improves readability.
   Apply during Phase 2 step 2 if profiler points to §4.4 (SLM
   aliasing) as bottleneck; otherwise deferred to Phase 3 cleanup.

2. **Parser-hardening backlog** (6 misfires this session per @beta
   tally: #68, #69, #74, #76, #78, plus the chat-side close on #78):
   `_infer_activity` daemon-side fix to stop auto-detecting MCP
   votes from chat phrases. **Critical priority** — blocks clean
   review record on Phase 2 reviews. Tracked in aircp project, NOT
   bitnet-arc; flagged here so reviewers are aware.

3. **MAC vs FMA convention** doc gap (per @sonnet 60x->30x->253x
   thread): all peak-FLOPS cites must use 2 FLOPs/FMA. Already
   applied in `bench/profile_v0_bl.md`; verify Phase 2 profiler
   output uses the same convention.

---

## 11. LOC estimate & sequencing

- `bench/profile_v2.cpp` ~50 LOC (per-section SYCL events + CSV out)
- `bench/profile_v0_bl.cpp` ~30 LOC extension (per-section breakdown)
- `src/kernel_v2.cpp` +0 to +40 LOC depending on §4 fix path
- `src/kernel_v2.h` +5 LOC (additional `kv2_variant_desc` entries)
- `bench/sweep_tile.cpp` +10 LOC (extended variant table iteration)
- **Total:** ~100-150 LOC across 5 files (depends on which §4 fix wins).

**Sequencing inside Phase 2** (per §6.1 split):

**Day 1 — Phase 2a (profiler, decode-agnostic):**
1. `bench/profile_v2.cpp` skeleton + SYCL events wiring (~2 h)
2. Two-build pattern (full + section-split) per §3 methodology (~3 h)
3. Run on 4 W1 + 1 large shape, generate `bench/profile_v2_p2a.md`
   (~1 h)
4. Extend `bench/profile_v0_bl.cpp` with same per-section (~2 h)
5. Comparison table + top-1/top-2 bottleneck identification
   (~1 h, in `bench/profile_p2a_comparison.md`)

**Day 2 — Phase 2b (geometry+fix, prefill-conditional, skipped if
@naskel pivots decode-priority):**
6. Apply top-1 §4.X matching fix per §4.5 priority-ranked (~3 h
   typical, may slip to ~5 h if it's geometry sweep with TILE_N=64
   debug surface — see @beta minor on Day-2 timing tightness)
7. Re-bench W2 shapes; if >= 1.5x v0_BL on (16, 64, 14336),
   Phase 2 PASSES.
8. **If still <1.5x:** apply top-2 §4.X fix (~2 h budget, trim
   geometry sweep to 3 variants instead of 4 if needed per @beta
   day-2 timing nit), re-bench. End of day-2 hardstop applies.
9. Open review/request -> 3-voice for Phase 2b results.

**Day 1 fallback (decode-pivot scenario):** If @naskel pivots before
Day 2 starts, skip §6-§9 above. Day 2 instead drafts `docs/design-
v3-decode.md` brief seeded by Phase 2a profiler data.

**Hardstop:** end of day 2 in either branch. If Phase 2b W2 gate
fails, falsify report (`bench/v2_phase2_falsification.md`) + pivot
brief draft starts. If decode-pivot branch, day-2 output is the
v3-decode brief draft (not a falsification, a redirection).

---

## 12. Reviewer focus

- **@sonnet** — algorithm correctness on §4.X fix variants;
  geometry sweep math sanity (TILE_M/N expanded fragment scheme,
  accumulator register pressure); decision tree completeness.
- **@beta** — QA on profiler harness (no instrumentation-induced
  perf bias; SYCL events not inhibiting compiler optim); §6
  prefill-only scope verbiage; §7 hardstop gate semantics; backlog
  fold completeness.
- **@haiku** — phasing alignment with design v2 §3 ladder;
  acceptance criteria vs. risk register; W2 gate is achievable in
  2 days given §4.X options; falsification path well-defined.

@theta offline this session — voice welcome on Phase 3 brief if back.

---

## 13. Inherited lessons (forward-port from Phase 1)

| Lesson                    | Source           | Phase 2 application              |
|---------------------------|------------------|----------------------------------|
| `nd_range` geometry race  | @sonnet #75      | Larger TILE_M/N → multi-SG WG, re-evaluate barriers per §3.4 |
| SLM budget static_assert  | @theta+@sonnet #68| Re-check guard for TILE_N=64 (40KB total, still <64KB) |
| MAC vs FMA convention     | @sonnet 60x->30x->253x | Profiler output uses 2 FLOPs/FMA explicitly |
| ZERO BULLSHIT denominators| @alpha 12.28 TFLOPS | Phase 2 cites Arc B60 spec values explicitly |
| Discipline post-v1        | v1 falsification | 2-day hardstop, no endless tuning |
| Self-catch on premature claims | @claude-opus task #157 catch (post-#78) | Status reports must reflect actual working tree state |

---

**Status on commit of this brief:** Phase 2 development can start
once 3-voice approval lands (or 2/3 with explicit delegate). The
profiler harness (§3) is the unblocked first deliverable; the
geometry sweep + §4 fix path is gated on profiler output.

Phase 2 result determines whether design v2 ships or gets falsified
honestly per §7. No third option without explicit @naskel call.
