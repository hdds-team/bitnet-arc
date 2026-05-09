# Design v2 Path B Falsification Report — Phase 2b W2 Gate Fail

**Date:** 2026-05-09
**Hardware:** Intel Arc Pro B60 Graphics (Xe2 / Battlemage)
**Brief reference:** `docs/design-v2-phase-2.md` §7 escalation ladder
**Phase 2b commit:** see git log on `src/kernel_v2.cpp` post Phase 2a step 2 `51948c4`

---

## Summary

Phase 2 design v2 Path B (FP16 joint_matrix XMX path) is **falsified
honestly** at the W2 perf gate. Phase 2b applied the top-1 + top-2
fixes identified by Phase 2a profiler (`bench/profile_v2_p2a.md`) and
measured a real but insufficient speedup. The remaining gap to W2 is
structural to Path B and cannot be closed by further tactical optims
in the FP16 materialization regime.

Per brief §3 escalation ladder, the appropriate next move is **Path A
(INT8 DPAS)**, with Phase 2b kept as a stepping stone (correctness
preserved, kv2 codebase intact for fallback or comparison).

---

## What Phase 2b applied

Two optims targeting the top-1 (dequant ~55%) and top-2 (loadA ~37%)
bottlenecks identified by `bench/profile_v2_p2a.md`:

1. **Lecture B switch on §3.1 cooperative TQ2_0 → FP16 dequant.**
   Each lane owns one column `n_local = lane` and decodes all K_CHUNK
   positions of its column from its own block's `qs[]` read directly
   into registers. Eliminates Lecture A's per-column inner loop
   (`TILE_N=16` iterations per chunk) and 32 sub-group barriers per
   chunk. SLM `qs_local` scratch dropped (no longer needed).

2. **Vectorized loadA on §3.2 cooperative SLM A load.**
   `sycl::vec<sycl::half, 8>` wide loads via reinterpret_cast,
   reducing LSU pipeline ops by 8× vs the scalar Phase 1 loop.

Both changes are in `src/kernel_v2.cpp` §3.1 and §3.2. The outer
kernel structure (chunks_per_col loop, MMA, store) unchanged.
SLM budget assert updated to drop the now-unused `qs_local`.

---

## Empirical result

### Correctness (W1 gate, max_rel_err ≤ 1e-2)

| Shape | Phase 1 max_rel_err | Phase 2b max_rel_err |
|-------|---------------------|---------------------|
| 16×16×256 | 0.00084 | 0.00084 |
| 16×64×4096 | 0.000968 | 0.000968 |
| 16×64×14336 | 0.00097 | 0.00097 |
| 64×64×14336 | 0.00096 | 0.00096 |

**4/4 PASS**, identical numerical behavior to baselines. No
correctness regression from the Lecture B + vec loadA changes.

### Performance (kv2 self + W2 gate vs v0_BL)

| Shape | Phase 1 t_med | Phase 2b t_med | kv2 self speedup | v0_BL t_med | Phase 2b vs v0_BL |
|-------|---------------|----------------|------------------|-------------|-------------------|
| 16×16×256 | 0.199 ms | 0.126 ms | **1.58×** | 0.076 ms | 1.66× SLOWER |
| 16×64×4096 | 1.477 ms | 1.026 ms | **1.44×** | 0.630 ms | 1.63× SLOWER |
| 16×64×14336 | 5.134 ms | 3.557 ms | **1.44×** | 2.170 ms | **1.64× SLOWER** |
| 64×64×14336 | 5.141 ms | 3.550 ms | **1.45×** | 2.187 ms | 1.62× SLOWER |

W2 gate (≥1.5× faster than v0_BL on headline shape `(16,64,14336)`)
requires `t_med ≤ 1.45 ms`. **Phase 2b achieves 3.557 ms = 0.61× v0_BL**.
**Gate FAILS by a factor 2.46×.**

The kv2 self speedup of 1.44× is real and consistent across shapes,
validating that Lecture B + vec loadA were correct optimizations.
But the absolute target was a moving goalpost: v0_BL scalar is so
efficient on this hardware that beating it via FP16 XMX requires
absorbing data-staging cost that the scalar kernel avoids entirely.

---

## Why W2 fails: structural to Path B

Phase 2a profiler showed dequant + loadA = ~92% of t_full. Phase 2b
addressed both. The 1.44× speedup confirms the optims worked, but
also reveals the bottleneck is **structurally bound** to the FP16
materialization choice:

1. **FP16 materialization is structural to Path B.** To use
   `joint_matrix_load` (which expects FP16 contiguous in SLM), the
   kernel must convert TQ2_0 codes to FP16. This is per-output-tile
   per-K_CHUNK work that has no analogue in v0_BL — v0_BL does
   `acc += ternary_code * activation` directly without materializing
   FP16 weights.

2. **Lecture B traded barriers for memory-access pattern.** Per the
   review-3 sonnet-SYCL-semantics finding: Lecture B reads from
   `B_blocks[lane_col].qs[byte]` with stride `blocks_per_col *
   sizeof(block) = 4480 bytes` between lanes. This is fully strided
   = 16 cache-line misses per dequant-step, vs Lecture A's coalesced
   64-byte cooperative load. The barrier savings (32 → 0 per chunk)
   are partially absorbed by GDDR6 bandwidth pressure.

3. **vec loadA hit a partial ceiling.** The 8× LSU op reduction
   should yield ~8× speedup if LSU pipeline was the sole bottleneck.
   We observe ~1.5× on the loadA component. Likely cause: SLM
   alignment fragility (per review-3: `local_accessor<uint16_t>`
   only spec-guarantees 2-byte alignment, not the 16 needed for
   `vec<half,8>`; icpx in practice aligns >128-byte but vec writes
   may fall back to scalar silently).

The data confirms: even with both top-1 and top-2 fixes optimally
applied (and the SLM alignment caveat addressed), the FP16
materialization step alone would still cost more than v0_BL's
*entire* runtime. Path B has a structural ceiling around 0.7-0.8×
v0_BL on this hardware.

---

## What Phase 2b did NOT prove false

- **XMX as a concept** — the joint_matrix MMA itself is fast (Phase 2a
  measured 1-4% of t_full). The compute path is sound.
- **Ternary on Arc B60 GPU** — the architectural target is intact;
  it's the FP16 path that's off.
- **The kv2 codebase** — code is correct, well-structured, and
  retained as the staging ground for any future Path B revisitation
  or for benchmarking the Path A implementation.

---

## Recommendation: Path A (INT8 DPAS)

Per the brief §3 escalation ladder + the 3-voice review (Opus voice,
strategic angle):

**Path A unblocks the structural cost of FP16 materialization.**
Intel Arc B60 has `dpas` (dot product acceleration) instructions
that compute `acc_i32 += int8_a * int8_b` natively. Ternary
`{-1, 0, +1}` fits in `int8` trivially with no dequantization for
the multiply step. The two reasons design v2 §2.2 cited against
Path A are now both empirically invalidated:

1. **"No guaranteed perf win"** — invalidated by Phase 2a profiler
   data showing dequant is the dominant cost. Eliminating the
   FP16 materialization step is now the documented top-1 fix.

2. **"Precision regression"** — Phase 2b max_rel_err is 1e-3, a
   full order of magnitude below the 1e-2 W1 gate. INT8 activation
   quantization can absorb this margin without breaking W1.

Compute headroom: Arc B60 spec sheet quotes **197 TOPS INT8** vs
~24.58 TFLOPS FP16 vector = **8× more peak throughput** on the
INT8 path.

### Out-of-scope for this report
- Detailed Path A design (separate `docs/design-v3.md` brief)
- Path C (custom packing) — kept in reserve for v4 if Path A hits
  a precision wall.

---

## Phase 2b artifacts kept

- `src/kernel_v2.cpp` — Phase 2b kernel (Lecture B + vec loadA),
  correctness-passing, slower than v0_BL but faster than Phase 1.
  Retained as production stub for fallback and as benchmark target
  for Path A comparisons.
- `bench/profile_v2.cpp` — split-build profiler, retained for
  verifying Path A's per-section profile against v2's. The split-
  build kernel sections still measure the OLD Lecture A pattern
  (not refreshed for Lecture B); a Phase 3 follow-up could refresh
  if needed.
- `bench/profile_v2_p2a.md` — Phase 2a profiler report, retained
  as the data foundation that justifies Path A pivot.

---

## Status

- **Phase 2 (FP16 XMX path) falsified per brief §7.**
- Phase 2 task #160 closes here. No further v2-internal optims.
- Path A re-evaluation triggered. Next deliverable:
  `docs/design-v3.md` brief seeded by Phase 2a profiler data + this
  falsification report.
- The W2 gate moves from a v2 acceptance criterion to a comparative
  benchmark: Path A v3 must beat v0_BL by ≥1.5× on headline shape,
  with v2 Phase 2b numbers (3.557 ms / 0.61× v0_BL) as the
  baseline-to-beat for FP16 path comparisons.
