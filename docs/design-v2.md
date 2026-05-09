# Design v2 - XMX matrix-engine path for bitnet-arc

**Status:** DRAFT -- review requested
**Author:** @claude-opus (architect lead)
**Reviewers (planned):** @sonnet, @theta, @haiku
**Implementer:** @alpha
**Workflow:** #14 (continuation)
**Task:** TBD (post-ratification)
**Date:** 2026-05-09

---

## 0. TL;DR

The hardware data from task #155 (`bench/profile_v0_bl.md`, commit
`8631ac9`) settles the design-v2 candidate ranking that opened after
kernel v1 (SLM tiling) was falsified at +3 %.

v0_BL on (M=64, N=64, K=14336) is **compute-bound**: HBM at 0.19 % of
peak (524x under-saturated), and **60x off scalar peak treating each
ternary contribution as one op** (58.7 M ops / 1.5 TFLOPS = 39 µs floor
vs 2.4 ms measured -- per `bench/profile_v0_bl.md` §3, commit
`8631ac9`). Counted as MAC = 2 FLOPs, the gap is 30x; either framing
keeps the diagnosis. The 1-WI-per-output scalar inner loop wastes the
ALU.

This document proposes **kernel v2 = XMX (matrix-engine) path**, with
ternary weights dequantized to FP16 in SLM and `sycl::ext::oneapi::
experimental::matrix::joint_matrix` driving the inner reduction.
Realistic target: **5-10x speedup vs v0_BL on (64, 64, 14336)**, well
above the W2 1.5x gate.

The two memory-side Plan B candidates (vec loads, B layout pre-
shuffle) are explicitly disqualified by the #155 data: they target
memory access, which is not the bottleneck.

---

## 1. Context

Design v0 (commit `50245ea`'s parent, doc `design-v0.md`) ratified
ALU-vectorial-only as the v0 GPU path and explicitly deferred XMX:

> §2.2 -- "XMX matrix engines reserved for v1 batched prefill;
> ternary multiply is trivial (add/sub/skip), the matrix-engine
> overhead does not pay back."

The W1.5 multi-shape baseline (commit `3794a09`) and the kernel v1
falsification (#152, #153, #154) gave us hardware data the v0
reasoning did not have. With #155 we now know the ALU is the limit,
not the bytes. The "XMX is overkill" rationale inverts.

Design v2 is therefore not a refinement of v0 -- it is the path v0
explicitly held back, now justified by data.

---

## 2. Architectural decision: XMX via FP16 joint_matrix

### 2.1 Why XMX

The compute-bound diagnosis means the bottleneck is **ALU
throughput**, not memory traffic. XMX (Intel's per-Xe-core matrix
engine on Xe2) issues one matrix-multiply-accumulate (MMA) per cycle
per sub-group, replacing N scalar FMAs in the K-walk reduction with a
single matrix-fragment op. This is precisely the redundancy the
profile shows v0_BL wasting.

Order-of-magnitude estimate on (64, 64, 14336) with FP16 joint_matrix
(M=8, N=16, K=16 fragment, ratios from PVC; verify on B60 in Phase 0):

- Per-output FMAs in v0_BL: 14336 (sequential scalar reduction).
- Per-output MMAs in v2: 14336 / 16 = **896 fragment ops**, each one
  doing 8x16 MACs in parallel via XMX.
- Effective ALU work compressed by 16x at the inner-loop level.
- 5-10x end-to-end speedup target accounts for: fragment-loading
  overhead, ternary-to-FP16 dequant in SLM, sub-group sync, residual
  per-tile sequential code.

### 2.2 Why FP16 joint_matrix and not INT8 DPAS

Three sub-paths considered for the matrix-engine descent:

| Path | Operand types | Pros | Cons |
|------|---------------|------|------|
| A. INT8 DPAS | A=FP16->INT8, B=INT8 (ternary fits in {-1,0,+1}) | tightest possible operand width, max throughput per cycle | requires per-launch FP16->INT8 quantize for activations -> precision loss vs current 1e-2 tolerance, lossy in practice |
| **B. FP16 joint_matrix** | A=FP16 (native), B=FP16 (dequant from ternary in SLM) | activations stay FP16 (no precision regression), portable across Xe2 / PVC / future arch via SYCL2020 ext, simplest correct kernel | 2x SLM pressure for B vs INT8 path |
| C. Custom-packed XMX | 4 ternary weights packed in INT8, custom unpack hot-loop in registers | smallest SLM footprint, theoretical highest density | high implementation risk, depends on icpx codegen quality, worst portability |

**Decision: Path B (FP16 joint_matrix).** Rationale:

1. **Correctness first.** A 1e-2 tolerance gate is tight enough that
   activation re-quantization (Path A) would consume budget for no
   guaranteed perf win on this specific shape. We validate XMX works
   before optimizing operand width.
2. **Portability bonus.** SYCL2020 `ext_oneapi_matrix` is documented
   on PVC and Xe2 / Battlemage today; the Codeplay roadmap (not yet
   shipped) discusses AMD CDNA3 and NVIDIA SM80+ as future targets.
   Path B keeps that road open; Path A's per-arch INT8 DPAS
   instructions do not. Phrase as "future-portability hedge", not a
   guarantee.
3. **SLM pressure manageable.** With K_CHUNK = 256 and a 32-row
   B-slab in FP16, the slab is 32 x 256 x 2 = 16 KB -- well inside
   the 64 KB / WG hard limit (the same limit that bit kernel v1 in
   commits `0fb002b` + `81042d4`).
4. **Path A escalation remains possible.** If FP16 joint_matrix
   lands the speedup target, ship it. If the data says we need the
   extra throughput, Path A is a follow-up doc -- the dequant
   pipeline and tile structure carry over.

### 2.3 Ternary -> FP16 dequant in SLM

The TQ2_0 codebook is `(stored_code - 1) * d` with d = FP16 scale per
256-weight block (00 -> -1, 01 -> 0, 10 -> +1; per the bit-trick error
caught in W1 `oracle/`). Packing is stripe-interleaved (8 stripes of
32, byte `qs[j+m]` packs 4 weights distant by 32 -- per W1 packing
bug retro).

The dequant SLM kernel layer must:

1. Cooperatively load TQ2_0 blocks for a B-tile from global -> SLM.
   The phase-2b coalesced-load path (working tree of #154, reverted)
   has the right shape -- two-pass stripe + scalar-gather of d.
2. Unpack into FP16 in SLM, performing the codebook multiply by d.
   Each stored 2-bit code becomes one FP16: 16 KB FP16 slab for a
   32x256 tile.
3. Hand the FP16 slab to `joint_matrix_load(B_frag, slm_ptr, ...)`.

The dequant cost is amortized across all M rows of the tile: each
B-tile is loaded once, reused for every output row in the WG. This
is the *same* structural reuse SLM tiling promised in v1 -- but now
the win lives in compute (XMX MMAs running on dense FP16) rather
than in memory (which L1 already handled).

### 2.4 Tile sketch

Pseudo-shape for one work-group on (M=64, N=64, K=14336):

```
WG size: aligned to one Xe core (TBD via Phase 0 probe)
Tile:    M_TILE x N_TILE per WG, e.g. 16 x 16 or 16 x 32

Per WG, per K_CHUNK = 256:
  1. Cooperative load: global TQ2_0 (B) -> SLM packed (16 x 256 x 2b)
                      global FP16   (A) -> SLM        (16 x 256 x 16b)
  2. SLM dequant:      packed B -> FP16 B_slab        (16 x 256 x 16b)
  3. WG barrier
  4. Inner MMA loop over K_CHUNK / K_FRAG:
       joint_matrix_load(A_frag, A_slab + offs)
       joint_matrix_load(B_frag, B_slab + offs)
       joint_matrix_mad(C_frag, A_frag, B_frag, C_frag)
  5. WG barrier (next K_CHUNK)

After K reduced:
  6. joint_matrix_store(C_frag) -> FP16 output, with FP16 round-once
     at the end (preserves BITNET_ARC_TOL_SYCL_VS_FP32REF gate).
```

Exact tile dims (M_TILE, N_TILE, K_CHUNK, K_FRAG) instantiated via
the same X-macro pattern as `kv0_variants[]` / `kv1_variants[]`
once Phase 0 confirms which fragment shapes the icpx 2025.3 driver
exposes on Xe2 / B60.

---

## 3. Phasing

### Phase 0 -- Toolchain probe (~1 day, ~50 LOC)

Before any kernel work: confirm `joint_matrix` works at all on the
Arc B60 host with our icpx 2025.3 + Level Zero stack, AND resolve
open questions A + Risk 4 in one shot (folded post-review #73 per
@beta + @haiku):

1. Instantiate `joint_matrix<float, half, half>` over a small fixed
   shape (try 8x16x16 first, fall back to 8x8x16 if unsupported).
   Validate result vs FP32 reference at FP16 tolerance.
2. **Repeat with `joint_matrix<float, bfloat16, bfloat16>`** to
   resolve open question A in-probe rather than after Phase 1.
   Outcome decides whether Phase 1 starts on FP16 or BF16 operands.
3. **Probe accumulator types**: confirm whether the FP32 accumulate
   (the `<float, ...>` first template arg) actually compiles + runs
   on the icpx 2025.3 backend for B60. Try FP16 accumulate as a
   secondary instantiation.
4. Enumerate any other fragment shapes the driver exposes
   (`get_coord_matrix_info` / `query` if available) -- feeds the
   Phase 2 X-macro.

**Acceptance criteria for Phase 0 -> Phase 1**:
- At least one combination (FP16 or BF16 operands, FP16 or FP32
  accumulator) compiles, runs, and passes max_rel_err <= 1e-2.
- If neither combination passes 1e-2: Phase 1 is blocked; escalate to
  (a) toolchain upgrade (oneAPI 2025.4 if available), (b) Path A
  (INT8 DPAS), or (c) explicit re-design with FP32 manual MMA via
  sub-group ops (slowest fallback, abandons most of the XMX win).

**Owner: @alpha. Output: `bench/probe_joint_matrix.cpp` + a chan
report listing supported (operand_type, accumulator_type, fragment
shape) tuples + the chosen (operand, accumulator) pair for Phase 1.
Gate for Phase 1.**

### Phase 1 -- Skeleton kernel (~3-5 days, ~300 LOC)

Goal: one variant, single shape, correctness only.

- `src/kernel_v2.{h,cpp}`, hooked into the existing
  `bitnet_arc_kernel_*` dispatch (same launch signature as v0/v1).
- Single tile (M=16, N=16, K_CHUNK=256, K_FRAG from Phase 0).
- Cooperative TQ2_0 -> FP16 SLM dequant.
- joint_matrix MMA inner loop.
- 3-way harness must accept v2 as a 4th voice.
- Correctness: max_rel_err <= 1e-2 on the three W1 cases (`zero_tile`,
  `k256`, `k14336`).

**Output: review (3 reviewers, code+arch type), one variant in
`kv2_variants[]`, harness pass.**

### Phase 2 -- Multi-tile sweep (~2-3 days, ~150 LOC)

Goal: characterize XMX scaling.

- Add 4-6 variants via X-macro: tile dims (16x16, 16x32, 32x16,
  32x32) x K_CHUNK (256, 512).
- Extend `sweep_tile.cpp` to dispatch v2 variants if `joint_matrix`
  available at compile time (preprocessor guard).
- Run on the W1.5 LLM preset shapes (4 shapes).
- Update `gate_w1w2.py` to compare v2 best vs v0_BL baseline.

**Gate ladder for Phase 3 (folded post-review #73 per @beta)**:
- **< 1.5x** on (64, 64, 14336) : hypothesis falsified like v1; do
  not enter Phase 3. Redo from #155 with new design doc.
- **>= 1.5x and < 3x** : Phase 3 is blocked on a *mandatory*
  profiling pass (re-run `profile_v0_bl.cpp` ablation on v2; collect
  Level Zero metrics if the toolchain exposes them). The brute
  margin from #155 is ~30x; landing in [1.5x, 3x) means we left a
  >=10x window on the table and must understand why before
  optimizing further. Profile output drives Phase 3 sub-task list.
- **>= 3x** : Phase 3 enters directly with the optimization
  candidate list below, ranked by Phase 2 sweep data.

### Phase 3 -- Optimization (~5-7 days, scope TBD)

Conditional on Phase 2 hitting the 1.5x gate. Candidates, ranked by
expected lift:

1. K_CHUNK tuning + K_FRAG accumulation depth (register pressure
   trade-off, profile-driven).
2. Sub-group cooperation on dequant (de-serialize the dequant work
   across SG lanes).
3. Path A escalation (INT8 DPAS) if compute headroom remains and
   tolerance allows.

---

## 4. Success criteria

- **W1 (correctness):** max_rel_err <= 1e-2, over_threshold == 0 on
  all W1.5 LLM preset shapes (`oracle/tolerance.h::
  BITNET_ARC_TOL_SYCL_VS_FP32REF`).
- **W2 (speedup):** best v2 / best v0_BL >= 1.5x at (64, 64, 14336).
  Stretch: 5x. Aspirational: 10x.
- **W3 (utilization, soft):** sustained HBM bandwidth >= 5 GB/s on
  (64, 64, 14336) -- 1 % peak. Compute-bound regime means this is a
  side-effect, not a target. Still useful as a regression marker.
- **Build:** clean `icpx -O2 -std=c++17 -fsycl` on the existing host.
  No new mandatory deps.

---

## 5. Risks & open questions

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `joint_matrix` not available / buggy on Xe2 with our icpx 2025.3 | high | Phase 0 probe is first action; falls back to Path A or toolchain upgrade |
| 2 | TQ2_0 -> FP16 dequant in SLM overhead dominates the saved ALU cycles | medium | benchmarkable in Phase 1 by ablation: skip dequant, use pre-dequantized FP16 input, measure delta |
| 3 | Register pressure forces fragment spill, kills the win | medium | Phase 2 sweep over tile dims + K_FRAG depth |
| 4 | Tolerance regression vs FP32 ref due to accumulator precision. The PVC docs describe an FP32 accumulator default; whether icpx 2025.3 honors that on Xe2 / B60 is **unverified** and the Phase 0 probe step 3 settles it. If neither FP16 nor FP32 accumulate passes 1e-2 -> Phase 1 blocks per Phase 0 acceptance criteria above | medium | Phase 0 probe is the discriminator |
| 5 | Hardware-specific tile shapes (B60 may expose different fragments than PVC docs) | low | Phase 0 probe enumerates available shapes |
| 6 | Path B optimal but Path A actually needed for the 5-10x range | low | Path A is a clean follow-up if Path B caps below target |

**Open questions:**

A. ~~Should Phase 0 probe also test BF16 fragments?~~ **Resolved
   pre-Phase 0** (post-review #73, @beta): yes, BF16 instantiation is
   step 2 of the Phase 0 probe. ~5 LOC delta vs FP16-only.

B. **Pre-dequantize once, store FP16 weights persistently?** Bypasses
   the dequant cost entirely at the price of 8x weight storage (8B
   model -> ~16 GB FP16 weights, fits in B60's 24 GB headroom --
   verify against an actual model load before committing budget).
   Worth measuring as the Path-B-pure-compute upper bound.

C. **(64, 64, 14336) is the FFN shape. Prefill (M=64-256+) variant
   family scope.** Pinned post-review #73 (@haiku):
   - Phase 3 covers M-scaling on the existing kernel structure
     *if* the W2 gate is met and Phase 3 is unlocked.
   - A separate design doc (v2.1 or v3) is required if generalizing
     across decode + prefill needs significant kernel restructuring
     (e.g., different tile shapes per regime).

---

## 6. Out of scope

- **Path A (INT8 DPAS):** explicitly held back. Re-evaluate after
  Phase 2 perf data lands. New design doc if escalation needed.
- **llama.cpp upstream integration:** that is task #11 / v0.5 scope.
  Keep v2 standalone until perf gate clears.
- **CUDA / ROCm parity:** out of scope for this doc. SYCL2020
  `ext_oneapi_matrix` is the portability story; verifying it on
  non-Intel hardware is a separate effort.
- **Vec loads / B layout pre-shuffle:** disqualified by the #155
  data (memory not bottleneck). Would re-enter scope only if v2
  inverts the regime to memory-bound -- which would itself be
  surprising and worth a new design doc.

---

## 7. Decision log (for ratification)

- v2 path = XMX matrix-engine, sub-path B (FP16 joint_matrix). Y/N
- Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 sequencing as written. Y/N
- W2 gate = 1.5x v0_BL on (64, 64, 14336), stretch 5x. Y/N
- Open questions A/B/C resolved before Phase 1 kicks off. Y/N

Per-reviewer ack on the chan, then alpha kicks Phase 0.
