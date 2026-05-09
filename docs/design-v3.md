# Design v3 — Path A INT8 DPAS for Ternary x INT8 Matmul

**Status:** draft, to be ratified before kernel work.
**Author:** @claude-opus (lead, post v2 falsification).
**Approval:** awaits @naskel.
**Predecessors:**
- `docs/design-v2.md` — original FP16 XMX path, ratified `e3fb4f2`.
- `docs/design-v2-phase-2.md` — Phase 2 brief, ratified `13f188b`.
- `bench/profile_v2_p2a.md` — Phase 2a profiler verdict (top-1 dequant
  ~55%, top-2 loadA ~37%, MMA 1-4%).
- `bench/v2_phase2_falsification.md` — Phase 2b W2 gate fail report
  (kv2 = 1.64× SLOWER than v0_BL, structural to FP16 materialization).
- `bench/probe_dpas.cpp` (commit `a36dd74`) — Phase 0 probe v3,
  unblocks Path A on TILE_M=8 + K_FRAG=32 native DPAS shape.

---

## 1. Why Path A now (post-v2 falsification)

Design v2 §2.2 disqualified Path A on two grounds. Both have been
empirically invalidated by the Phase 2 work:

| Original objection (design v2 §2.2) | Status post-Phase-2 |
|--------------------------------------|---------------------|
| "No guaranteed perf win" | **INVALIDATED** by Phase 2a profiler: dequant TQ2_0 → FP16 is the structural top-1 bottleneck (~55% of t_full); Path A bypasses it by multiplying ternary directly as INT8. |
| "Precision regression risk" | **INVALIDATED** by Phase 2b correctness: max_rel_err = 1e-3 = 10× under the 1e-2 W1 gate. INT8 activation quant has 10× of margin to absorb. |

Compute headroom claim (per Intel official Arc B60 spec, cross-
referenced via the same sources used in `profile_v0_bl.md`):
- **197 TOPS INT8** (DPAS native peak) vs ~24.58 TFLOPS FP16 vector =
  **8× higher peak throughput** on the INT8 path.
- Per-MMA: 8x16x32 INT8 DPAS = 4096 INT8 ops, vs 16x16x16 FP16 =
  4096 FP16 ops — same ops/MMA, but the INT8 silicon delivers ~8×
  the throughput per cycle.

Phase 2a / 2b empirical reality: the FP16 XMX path is structurally
capped near 0.7-0.8× v0_BL on this hardware because the dequant
materialization step has no analogue in v0_BL's scalar inline
pattern. Path A eliminates that step entirely (ternary codes → INT8
directly, no FP16 conversion for the multiply).

3-voice strategic vote (4/4 unanimous for B = Path A, with the
Phase 0 probe gate):
- Sonnet (pragmatic ship-readiness): "8× compute headroom + Phase 0
  probe time-boxes the unknown unknowns."
- Sonnet (risk-reward): "Path A attaque la cause racine
  documentée. Quick-kill test possible in <1 day."
- Opus (long-term strategic): "Maximise ecosystem + Intel DevRel +
  futur-ternary; first SYCL kernel exercising DPAS sur Arc Pro
  B60 = exactement le signal Intel partage."
- Claude-opus lead: same.

---

## 2. Phase 0 v3 result (probe data, gate input)

`bench/probe_dpas.cpp` ran on Arc Pro B60 (icpx 2025.3, driver
1.14.37435+1, Level Zero V2):

| Combo | Status | Notes |
|-------|--------|-------|
| `int8 × int8 → int32` at 16×16×32 | sycl::exception | combo unsupported |
| `int8 × int8 → int32` at 16×16×64 | sycl::exception | combo unsupported |
| **`int8 × int8 → int32` at 8×16×32** | **EXACT match** | DPAS native rep=1 |
| `int8 × int8 → int32` at 16×16×16 | sycl::exception | combo unsupported |

**Key constraint for design v3:** icpx 2025.3 INT8 DPAS exposes only
the minimal native shape on Xe2 (`8×16×32`). The M=16 INT8 fragments
that worked for FP16 do NOT work for INT8. **Design v3 must use
TILE_M = 8** (or multiple of 8), not 16.

This is a wrapper limitation in `sycl::ext::oneapi::experimental::matrix`
for INT8, not a hardware limit (the underlying DPAS instructions
support compositions). Future icpx releases or the
`sycl::ext::intel::experimental::matrix` namespace might expose more
shapes, but is out of scope here.

**Acceptance:** Phase 0 v3 PASS — Path A unblocked for design v3
implementation.

---

## 3. Kernel architecture

### 3.1 Inputs and output

Same external interface as kernel_v0/v1/v2 (per design v0 §2.3
contract):
- `A_fp16`: USM device, `M × K` FP16 activations.
- `B_blocks`: USM device, `(K/256) × N` `bitnet_arc_tq2_0_block`
  (1 block = 32 bytes qs + 2 bytes d).
- `C_fp16`: USM device, `M × N` FP16 output.

The `A_fp16` interface stays FP16 to match the existing harness;
**activation INT8 quantization happens inside the kernel** (Phase 1
v3) or as a separate launch (Phase 2 v3 if profiler indicates it
helps). See §3.4.

### 3.2 Tile geometry (constrained by Phase 0 probe)

| Param | Value | Source |
|-------|-------|--------|
| TILE_M | 8 | Phase 0 probe (only INT8 DPAS shape that works) |
| TILE_N | 16 | Phase 0 probe + matches kv2 TILE_N for cross-comparability |
| FRAG_M | 8 | DPAS native (= TILE_M) |
| FRAG_N | 16 | DPAS native (= TILE_N) |
| FRAG_K | 32 | DPAS native (= 2× kv2's K_FRAG=16) |
| K_CHUNK | 256 | TQ2_0 block size, locked |
| FRAGS_PER_CHUNK | 8 | K_CHUNK / FRAG_K = 256 / 32 |
| SG_SIZE | 16 | Same as v2 (Xe2 sub-group native) |

Launch geometry: `nd_range<1>({total_tiles × SG_SIZE}, {SG_SIZE})`
where `total_tiles = (M/TILE_M) × (N/TILE_N)`.

For Phase 1 v3 W1 shapes (cross-tab vs v2 measurements):

| Shape (M, N, K) | v2 tiles | v3 tiles | v3 vs v2 parallelism |
|------------------|----------|----------|----------------------|
| (16, 16, 256) | 1 | 2 | 2× |
| (16, 64, 4096) | 4 | 8 | 2× |
| (16, 64, 14336) | 4 | 8 | 2× |
| (64, 64, 14336) | 16 | 32 | 2× |

2× more parallelism across the board. Better Xe2 occupancy on
small-M shapes.

### 3.3 Inner loop structure (5 sections per chunk)

Mirrors kv2 Phase 2b structure but with INT8 DPAS replacing FP16
joint_matrix MMA, and adding an activation quant step:

1. **§3.4 Activation INT8 quant** (NEW vs kv2): A_fp16 → A_int8 +
   per-row scale `s_a[m]`. Done once per kernel invocation (not per
   chunk). Stored in SLM `A_q_slab[TILE_M × K_CHUNK]` and `s_a[TILE_M]`.

2. **§3.5 Cooperative ternary → INT8 dequant** (replaces v2 §3.1):
   each lane decodes one column's TQ2_0 codes into INT8 directly
   into SLM `B_int8_slab[K_CHUNK × TILE_N]`. **No FP16 conversion**;
   ternary `{-1, 0, +1}` maps directly to `int8 {-1, 0, +1}`. Per-
   block scale `d` is preserved as FP16 in SLM `s_b[TILE_N]` and
   applied in §3.7.

3. **§3.6 Cooperative A_int8 SLM load** (replaces v2 §3.2): if §3.4
   wrote A_q_slab earlier, this is just a sub-group barrier; if §3.4
   defers to per-chunk quant, then this section also vec-loads
   A_int8 from global. Phase 1 v3 ships with §3.4 done once at
   kernel entry (cleaner separation).

4. **§3.7 Inner DPAS MMA loop**: 8 fragment-K steps per K_CHUNK (vs
   16 in kv2). Each step issues one `joint_matrix_mad` with INT8
   ops + INT32 acc. Accumulator `mC_i32` stays in registers across
   all FRAGS_PER_CHUNK steps and across all K_CHUNK outer iterations.

5. **§3.8 Final FP16 store** (extends v2 §3.5): mC_i32 + s_a[m] +
   s_b[n] → FP32 → FP16 → C_fp16. The two scales need to multiply
   in:
   ```
   c_fp16[m, n] = fp32_to_fp16(
       float(mC_i32[m, n]) * s_a[m] * s_b[n] / scale_correction
   )
   ```
   where `scale_correction` accounts for the chunk-level d
   accumulation pattern (per-chunk d gets folded into mC during
   accumulation; details in §4.2).

### 3.4 Activation INT8 quantization scheme

Per-row symmetric quant with FP16 scale:
- `s_a[m] = max(|A_fp16[m, k]| for k in 0..K-1) / 127`
- `A_q_slab[m, k] = round(A_fp16[m, k] / s_a[m])`, clamped to int8
  range [-128, 127].

Activation quant cost: O(M × K) per kernel call. For (16, 64, 14336)
that's 16 × 14336 = 229K ops per call, plus a max-reduction. Done
once at kernel entry, amortized over all M × N output tiles.

This is W8A8 standard (LLM int8 quant). max_rel_err Phase 2b margin
(1e-3 vs 1e-2 gate) gives 10× of headroom for the rounding error.

Alternative scheme (deferred to Phase 2 v3 if needed): per-block
group-quant for tighter accuracy. Not Phase 1 scope.

---

## 4. SLM budget

| Buffer | Size (bytes) |
|--------|--------------|
| A_q_slab : TILE_M × K_CHUNK INT8 | 8 × 256 = 2 KB |
| B_int8_slab : K_CHUNK × TILE_N INT8 | 256 × 16 = 4 KB |
| s_a : TILE_M FP16 | 16 |
| s_b : TILE_N FP16 (per-chunk d) | 32 |
| qs scratch (if needed for cooperative load) | 64 |
| **Total** | **~6.1 KB** |

Well under Xe2's 64 KB / WG hard limit (same `static_assert` pattern
as kv1/kv2). Actually **smaller than kv2's 16 KB** because INT8 takes
half the bytes of FP16, freeing SLM for potential future fold of
larger TILE_N or K_CHUNK.

---

## 5. Phasing (mirrors v2 Phase 1 → 2 ladder)

### Phase 1 v3 (this brief)

- **Scope:** single registered variant `(TILE_M=8, TILE_N=16,
  SG_SIZE=16, K_CHUNK=256)`.
- **Files to add:**
  - `src/kernel_v3.h` (~40 LOC, mirror kv2.h structure).
  - `src/kernel_v3.cpp` (~350-400 LOC, includes activation quant +
    dequant + DPAS MMA + final store).
  - `src/Makefile` extends `OBJS` to include kernel_v3.o.
  - `bench/sweep_tile.cpp` extends with kv3 variant pass.
- **Deliverables:**
  - Correctness W1 gate (max_rel_err ≤ 1e-2 on 4 W1 shapes).
  - Bench ratio vs v0_BL on (16, 64, 14336) headline shape.
- **Acceptance:** correctness only (no perf gate yet, mirroring
  kv2 Phase 1 acceptance).

### Phase 2 v3 (post-Phase-1 perf data)

- Conditional on Phase 1 v3 correctness PASS.
- W2 gate: ≥1.5× v0_BL on (16, 64, 14336), same as v2 W2.
- If Phase 2 v3 hits W2 with single registered variant: ship + PR
  llama.cpp upstream.
- If Phase 2 v3 misses W2: profile-driven optim pass (mirror v2
  Phase 2a/2b cycle), then either fix or falsify per §6 hardstop.

### Phase 3 v3 (perf optim sweep)

- Geometry sweep extending TILE_M/TILE_N (constrained by what icpx
  exposes — Phase 0 probe expansion may unlock 16×16×32 in future
  icpx releases).
- Activation quant scheme variation (per-block vs per-row).
- BF16 path probe (deferred from v2 Phase 0).

---

## 6. Hardstop / falsification gate

Per the discipline lessons from v1 falsification + v2 falsification
(both proper sec7 stops):

- **Phase 1 v3 hardstop:** end of 1 implementation session. Either
  correctness PASS or scoped blocker (e.g., DPAS mad on int8 produces
  unexpected acc behavior on Xe2).
- **Phase 2 v3 hardstop:** end of 1 perf evaluation session. Either
  W2 gate PASS or falsification report.
- **Total budget:** 2-3 sessions wall-clock for v3 to reach W2 or
  trigger v4 escalation.

If Phase 2 v3 falsifies (i.e., W2 fails despite Path A architectural
fix), the next escalation is:
- Path C (custom packing) via Phase 0 probe v4
- OR hard pause + re-eval scope (decode-priority pivot, ternary
  plateau on 8-10B vs 70B+ aspiration, etc.)

---

## 7. Risks register

1. **DPAS scale correction in §3.8.** Each chunk contributes
   `mC_i32 += A_int8 × B_int8`, with B_int8 carrying ternary codes
   *unscaled* by d (we store scale separately to keep DPAS inputs
   pure int8). The final reconstruction must multiply by `s_a × d`
   per-(m, n, chunk). If d differs across chunks (it does — 1
   block per chunk = 1 d per chunk per column), the scale fold has
   to happen *inside* the chunk loop, not at the end. This is a
   Phase 1 implementation correctness risk; brief §4.2 in the
   Phase 1 v3 brief draft will pin the exact fold.

2. **INT8 wrapper maturity.** Phase 0 probe found 3/4 INT8 shapes
   throw sycl::exception. The wrapper API surface is less mature
   than FP16. Risk: edge cases in `joint_matrix_mad` INT8 (e.g.,
   accumulator update semantics on Xe2 with INT32) may need probe
   expansion if Phase 1 v3 hits unexpected behavior.

3. **Activation quant overhead.** §3.4 per-row symmetric quant adds
   O(M × K) work per kernel call. For (16, 64, 14336) that's
   ~229K ops + a reduction. If this isn't amortized cleanly across
   M × N output tiles, it could become a new bottleneck. Mitigation:
   profile the activation quant step separately in Phase 2 v3
   profiler (mirror profile_v2 split-build).

4. **Per-block d accumulation precision.** Folding per-block
   `d` (FP16) into the int32 acc requires float arithmetic at the
   end, which may lose precision on small results. Phase 2b max_rel_err
   margin (1e-3) suggests headroom but Phase 1 v3 must verify on
   the W1 shapes.

5. **icpx 2025.3 INT8 DPAS perf.** The Phase 0 probe confirmed
   correctness on 8×16×32 but did NOT measure perf. Phase 1 v3 may
   discover that the wrapper INT8 path is throttled or has hidden
   overhead (similar to how Path B's vec loadA fell back to scalar
   silently per Phase 2b SLM alignment caveat). Mitigation: a quick
   throughput micro-bench at Phase 1 v3 start, before full kernel
   implementation, to validate the 8× headroom claim empirically.

---

## 8. Out-of-scope (avoid re-litigation)

- BF16 path — design v2 already explored via Phase 0 FP16/BF16 probe;
  deferred to Phase 3 v3 if needed.
- Path C custom packing — escalation only if v3 falsifies.
- Decode regime (M=1) — same as v2: kernel_v0_BL fallback for M < 8
  (now the new threshold; was M < 16 for v2).
- Multi-WG cooperative output tile (split-K reduce) — Phase 3 v3
  only if Phase 2 v3 profiler indicates inter-WG opportunity.
- `sg_size != 16` — Xe2 sub-group native, locked.
- Activation FP16 → BF16 quant variant — orthogonal to Path A core
  thesis.

---

## 9. LOC estimate & sequencing

- `src/kernel_v3.h`: ~40 LOC (mirror kv2.h).
- `src/kernel_v3.cpp`: ~350-400 LOC (includes the 5 sections
  inline; the v2 Phase 2b file was ~500 LOC, v3 should be slightly
  smaller because no FP16 dequant complexity).
- `src/Makefile`: +2 LOC.
- `bench/sweep_tile.cpp`: +50 LOC (kv3 variant pass).
- **Total Phase 1 v3:** ~440-490 LOC across 4 files.

Sequencing:
1. Throughput micro-bench (~1 hour wall-clock): validate 8×
   headroom on isolated DPAS calls.
2. Kernel skeleton (~2 hours): activation quant + dequant + MMA
   + store, no perf optim, single shape (16, 16, 256) smoke pass.
3. Multi-shape correctness W1 (~1 hour): all 4 W1 shapes
   max_rel_err ≤ 1e-2.
4. Phase 1 v3 commit + push.
5. Phase 2 v3 profiler harness + W2 evaluation (separate session).

LLM team timing realistic: Phase 1 v3 deliverable in ~1 session
wall-clock alpha-side (or my equivalent if I keep coding solo per
the current pattern).

---

## 10. Inheritance from v2

Forward-port lessons from Phase 2:

| Lesson | Source | v3 application |
|--------|--------|----------------|
| `nd_range<1>({16},{16})` minimal launch | Phase 0 probe v2 | Same launch geometry; SG_SIZE=16 unchanged. |
| SLM budget static_assert discipline | kv1 81042d4 | Add equivalent in kv3, with INT8 byte counts. |
| MAC vs FMA convention | profile_v0_bl.md a5d4283 | INT8 ops accounted as 2 ops/dpas-step (1 mul + 1 add). |
| Discipline post-falsification | v1 + v2 | 2-3 session hardstop, no endless tuning. |
| 3-voice review on kernel changes | Phase 2b | Apply same to kv3 commits (Sonnet code-quality + Sonnet SYCL semantics + Opus strategic). |
| Caveat documentation in code | Phase 2b L313-340 | Document any DPAS-specific quirks inline. |
| Verdict-as-last-keyword | Phase 2b review #87 | Apply to all kv3 reviews. |

---

## 11. Gate to ratify this brief

- Naskel approval (this brief is the strategic pivot from v2 path B
  to Path A; brief content above is the seed for the implementation
  brief that comes next as `docs/design-v3-phase-1.md`).
- 3-voice review on this brief (mirror the design v2 ratification
  pattern): Sonnet pragmatic + Sonnet QA + Opus strategic. Each
  voice signs off on either:
  - Brief approved as-is for Phase 1 v3 work
  - Brief approved with folds (list)
  - Brief BLOCKED (rare — would mean Path A has a flaw that Phase 0
    probe missed)

Phase 1 v3 implementation brief (`docs/design-v3-phase-1.md`)
draft starts after this brief is ratified.
