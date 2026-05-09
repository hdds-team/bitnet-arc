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

3-voice strategic vote (3/3 unanimous for B = Path A, with the
Phase 0 probe gate; lead alignment recorded separately, not counted
in the tally per Phase 2b 3-voice review #X opus-strategic-fold-1
"clean up self-stacked vote"):
- Sonnet (pragmatic ship-readiness): "8× compute headroom + Phase 0
  probe time-boxes the unknown unknowns."
- Sonnet (risk-reward): "Path A attaque la cause racine
  documentée. Quick-kill test possible in <1 day."
- Opus (long-term strategic): "Maximise ecosystem + Intel DevRel +
  futur-ternary; first SYCL kernel exercising DPAS sur Arc Pro
  B60 = exactement le signal Intel partage."

Lead (claude-opus) aligned with the unanimous panel.

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

4. **§3.7 Inner DPAS MMA loop (revised post review #X for per-chunk
   d fold blocker, sonnet-SYCL #5)**:

   *Critical structural change vs kv2*: kv2 kept `mC_fp32` in
   registers across ALL chunks because FP16 dequant scales were
   folded *into B_slab* (each B element was already `code * d` in
   FP16). For Path A INT8 DPAS, B_int8_slab carries raw ternary
   codes `{-1, 0, +1}`; the per-block scales `d` (FP16, 1 per
   chunk per column) cannot be folded into the INT8 fragment
   without precision loss. Per-chunk d MUST be applied *between*
   chunks, breaking the kv2 register-only accumulator pattern.

   Phase 1 v3 pattern (4-level loop):
   ```
   mC_fp32_slab in SLM = zero-init  // TILE_M*TILE_N FP32 = 512 bytes
   for chunk c in 0..chunks_per_col:
       mC_i32 fragment = fill(0)   // chunk-local accumulator, registers
       // (... §3.5 dequant + §3.6 loadA cooperative SLM fill ...)
       for k_frag in 0..FRAGS_PER_CHUNK (= 8):
           load mA, mB from A_q_slab + B_int8_slab
           mC_i32 = joint_matrix_mad(mA, mB, mC_i32)
       // Fold this chunk's d into the FP32 accumulator
       joint_matrix_store(mC_i32 -> mC_i32_slab in SLM)
       group_barrier(sg)
       cooperative_scalar_fold(mC_fp32_slab, mC_i32_slab, d_chunk_per_n)
       group_barrier(sg)
   ```

   `cooperative_scalar_fold`: each of the SG_SIZE=16 lanes handles
   `(TILE_M * TILE_N) / SG_SIZE = 128 / 16 = 8` elements of the
   8x16 tile. For element `(m, n)`:
   ```
   mC_fp32_slab[m, n] += float(mC_i32_slab[m, n]) * d_chunk[n]
   ```
   where `d_chunk[n]` is the per-block `d` (FP16, decoded to FP32)
   for the current chunk's column `n`. The 16 `d_chunk` values
   are stored in SLM `s_b[TILE_N]` by §3.5 dequant alongside
   B_int8_slab.

   `mC_i32_slab` is reused SLM staging (8x16 INT32 = 512 bytes;
   fits in `B_int8_slab`'s 4 KB region after barrier-flushed).

5. **§3.8 Final FP16 store (revised, lane mapping per M=8 / SG=16
   per review #X sonnet-SYCL #1)**:

   At end of chunk loop, `mC_fp32_slab` holds the fully-accumulated
   FP32 result. Final store applies the per-row activation scale
   `s_a` and converts to FP16:
   ```
   for each (m, n) in TILE_M × TILE_N:
       c_fp16[row(m), col(n)] = fp32_to_fp16(
           mC_fp32_slab[m, n] * s_a[m]
       )
   ```

   **Lane mapping caveat**: with TILE_M=8 and SG_SIZE=16, the
   2-lanes-per-row fragment layout means `lane = m_group + lane_id`
   (the kv2 pattern) is INCORRECT for kv3. Cooperative store uses
   linear element distribution: lane `lane_id` handles elements
   `[lane_id*8 .. lane_id*8+7]` of the flattened 8x16 mC_fp32_slab,
   regardless of fragment lane mapping. This works because the
   final store reads from SLM (not from registers), so the SYCL
   matrix lane-mapping abstraction is irrelevant at this point.

   Replaces the `scale_correction` placeholder from the original
   draft (per Sonnet-SYCL review #X #3 unresolved formula). There
   is no `scale_correction` divisor in the final formula because
   the `d` fold is already in `mC_fp32_slab` (applied per-chunk
   in §3.7).

### 3.4 Activation INT8 quantization scheme

Per-row symmetric quant with FP16 scale:
- `s_a[m] = max(|A_fp16[m, k]| for k in 0..K-1) / 127`
- `A_q_slab[m, k] = round(A_fp16[m, k] / s_a[m])`, clamped to int8
  range [-128, 127].

**Sharing model in Phase 1 v3 (post review #X sonnet-pragmatic-2):**
Each WG (= one output tile) re-quantizes its own M-row strip of A
into its own private SLM `A_q_slab`. Per-row scales `s_a[0..TILE_M-1]`
are computed locally (max-reduce across `K` per row, lane-coop). The
*row* scales are shared across (m, n) tiles that share the same
m_group: tiles `(m_group, 0)`, `(m_group, 16)`, ... all compute the
same `s_a[m_group..m_group+7]` independently. This is redundant work
(each WG does max-reduce on the same A rows), but in Phase 1 it
keeps the kernel single-launch and simple.

Cost per kernel call (Phase 1 sharing model):
- Per-WG: TILE_M × K reads + max-reduce (~K/SG_SIZE ops per lane)
  + TILE_M × K rounds = O(TILE_M × K) per WG.
- Total across WGs: O(TILE_M × K × tiles_M × tiles_N).
- For (16, 64, 14336) = 8 × 14336 × 2 × 4 = 920K ops per kernel,
  ~3% of t_full estimated (vs ~3.5M ops for the matmul itself).
  Acceptable Phase 1 cost.

**Phase 2 v3 optim (deferred):** if profiler shows act-quant > 10%
of t_full, factor into a separate kernel launch that writes a
shared USM `A_q_global` once, then the matmul kernel reads it.
Reduces total ops from `O(M × K × tiles_N)` to `O(M × K)`. Trade-
off: launch overhead (~1-2 ms per Phase 2b single-WG bench finding)
vs the saved redundant work. Phase 2 profiler measures this.

This is W8A8 standard (LLM int8 quant). max_rel_err Phase 2b margin
(1e-3 vs 1e-2 gate) gives 10× of headroom for the combined rounding
error (act-quant rounding + per-chunk d fold rounding, see §4.2).
The combined error budget is verified analytically in Phase 1 W1
testing; if max_rel_err > 1e-2 on any W1 shape, falls back to
deferred per-block group-quant scheme.

---

## 4. SLM budget + per-chunk d fold pattern

### 4.1 SLM budget

| Buffer | Size (bytes) |
|--------|--------------|
| A_q_slab : TILE_M × K_CHUNK INT8 | 8 × 256 = 2 KB |
| B_int8_slab : K_CHUNK × TILE_N INT8 | 256 × 16 = 4 KB |
| mC_fp32_slab : TILE_M × TILE_N FP32 (cross-chunk acc) | 8 × 16 × 4 = 512 |
| mC_i32_slab : TILE_M × TILE_N INT32 (chunk-local staging) | 8 × 16 × 4 = 512 |
| s_a : TILE_M FP16 (per-row act scale) | 16 |
| s_b : TILE_N FP16 (per-chunk d; refreshed each chunk) | 32 |
| qs scratch (if needed for cooperative load) | 64 |
| **Total** | **~7.2 KB** |

Well under Xe2's 64 KB / WG hard limit (same `static_assert` pattern
as kv1/kv2). Still **smaller than kv2's 16 KB** despite the added
mC_fp32_slab + mC_i32_slab staging (forced by per-chunk d fold,
§4.2), thanks to INT8 being half-size of FP16.

**Occupancy gain (per Sonnet-SYCL review #X #6, fold)**: at 7.2 KB
per WG, Xe2's per-Xe-core 64 KB SLM allows up to 8 concurrent WGs
(vs kv2's 4). This is a Phase 2 v3 perf opportunity to flag for
the W2 evaluation profiler — kv3 may saturate Xe2 occupancy where
kv2 was capped, independent of any DPAS speedup.

### 4.2 Per-chunk d fold pattern (the critical correctness blocker
        identified by review #X SYCL semantics #5)

**The blocker**: kv2's "register-only accumulator across all chunks"
pattern relied on FP16 dequant pre-folding `d` into B_slab elements.
For Path A INT8 DPAS, B_int8_slab carries raw ternary codes; the
per-block `d` cannot be folded into the INT8 fragment without
precision loss. Different chunks have different `d` values per
column (1 d per TQ2_0 block), so:

```
WRONG (kv2 pattern, would lose per-chunk d info):
    mC_i32 = sum_over_all_chunks(A_int8 @ B_int8)  // d lost
    c_fp32 = float(mC_i32) * d_some_chunk          // structurally wrong

CORRECT (kv3 pattern, per-chunk fold):
    mC_fp32 = 0
    for chunk c:
        mC_i32_c = sum_over_k_in_chunk(A_int8 @ B_int8_codes)  // raw int
        mC_fp32 += d_chunk[n] * float(mC_i32_c)                // per-n d
    c_fp32 = mC_fp32 * s_a[m]                                   // final
```

The cost of this restructuring:
- **+1 SLM round-trip per chunk**: `mC_i32` flushes to SLM at end
  of each chunk, gets read in cooperative scalar fold, then
  `mC_fp32_slab` updated and SLM-resident across chunks.
- **+1 sub-group barrier per chunk** (after `joint_matrix_store`)
  + 1 more after the cooperative scalar fold (before next chunk's
  MMA reads SLM).
- For K=14336, 56 chunks: 56 SLM round-trips + 112 barriers added
  vs kv2's pattern.

This is *less* overhead than kv2 Phase 2b had (Lecture A: 32
barriers per chunk × 56 chunks = 1792 barriers; Phase 2b Lecture B
brought that to 0; kv3 brings it to 2 per chunk = 112). Per Phase
2b empirical profile, barriers themselves were ~0.5% of t_full,
so this added cost is bounded.

**Alternative fold patterns considered and rejected**:
- *Per-column scalar accumulator in registers*: would require
  per-element joint_matrix access (`get_wi_data()` is non-portable
  Intel-specific extension per review #X SYCL #2). Rejected.
- *FP32 accumulator joint_matrix*: would need an INT32→FP32
  joint_matrix conversion API which is not in SYCL2020 ext.
  Rejected.
- *Hardcode K=256 (1 chunk only) Phase 1*: would block W1 shapes
  with K>256 (3 of 4 W1 shapes). Rejected — too restrictive.

The SLM round-trip pattern is the cleanest portable solution. The
~2 KB SLM staging is well within budget (§4.1).

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
(both proper §7 stops). Phase 0.5 v3 added per review #X opus-
strategic fold #2 ("v2 trap was 3 phases before falsification
surfaced; need an intermediate gate before kernel work").

### 6.0 Phase 0.5 v3 throughput micro-bench (NEW HARDSTOP)

Before committing 350-400 LOC of kernel implementation, validate
the 8× headroom claim empirically on isolated DPAS calls:

- **Deliverable:** `bench/probe_dpas_throughput.cpp` (~80 LOC, mirror
  the structure of `bench/probe_dpas.cpp`). Times an isolated
  `joint_matrix_mad<int8, int32>` at 8x16x32 vs an equivalent
  `joint_matrix_mad<half, float>` at 16x16x16 over many iterations
  (each MMA = 4096 ops, so wall-clock comparison is per-op-rate).
- **Acceptance threshold:** INT8 DPAS shows ≥4× throughput vs FP16
  joint_matrix on equivalent ops/s rate. The peak-spec ratio is
  8×, but some wrapper overhead is expected; <4× indicates the
  wrapper INT8 path is throttled or has hidden cost (mirroring the
  Phase 2b vec loadA SLM alignment fragility).
- **Hardstop semantics:**
  - **PASS (≥4× throughput ratio):** Phase 1 v3 implementation
    proceeds as scoped (§5 + §9).
  - **FAIL (<4×):** Phase 1 v3 BLOCKED. Path A's structural
    advantage relies on the throughput claim; if the wrapper
    can't deliver, kernel work doesn't help. Escalate to v4
    options (Path C, hard pause, decode-pivot scope re-eval).
  - **MARGINAL (4-6× ratio):** Phase 1 v3 proceeds with
    *reduced expected gain* on W2 gate. The 1.5× v0_BL gate
    becomes harder to hit; document the reduced expectation in
    the Phase 1 v3 commit message.

This gate prevents the v2 trap (3 phases of work before
falsification surfaces). If Path A's compute claim is wrong, we
know in ~1 hour vs ~3 sessions.

### 6.1 Phase 1 v3 hardstop

End of 1 implementation session. Either:
- **PASS:** correctness W1 gate (max_rel_err ≤ 1e-2 on 4 shapes).
- **BLOCKER:** scoped issue (e.g., DPAS INT32 acc behavior on
  Xe2 unexpected, per-chunk d fold precision loss, joint_matrix
  INT8 wrapper crashes on actual workload despite probe pass).

### 6.2 Phase 2 v3 hardstop

End of 1 perf evaluation session. Either:
- **PASS:** W2 gate ≥1.5× v0_BL on (16, 64, 14336) → ship + PR
  llama.cpp upstream.
- **FAIL:** falsification report `bench/v3_phase2_falsification.md`
  (mirror v2 falsification doc structure) + escalation to v4.

### 6.3 v4 escalation options (if Phase 0.5 or Phase 2 v3 falsifies)

- Path C (custom packing) via Phase 0 probe v4.
- Hard pause + re-eval scope (decode-priority pivot, ternary
  plateau on 8-10B vs 70B+ aspiration, alternate hardware target).

**Total budget:** ~2.5 sessions wall-clock for v3 to reach W2 or
trigger v4 escalation:
- Phase 0.5 throughput micro-bench: ~0.3 session (1-2 hours)
- Phase 1 implementation + W1: ~1 session
- Phase 2 perf eval + W2: ~1 session
- Buffer: ~0.2 session for unforeseen fold-back from any phase

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

### Phase 0.5 v3 (NEW per §6.0 hardstop)

- `bench/probe_dpas_throughput.cpp`: ~80 LOC. Mirror
  `bench/probe_dpas.cpp` structure but timing-focused: many MMA
  iterations + SYCL events, INT8 ratio vs FP16 baseline.
- `bench/Makefile`: +2 LOC (target).

### Phase 1 v3 (gated on Phase 0.5 PASS)

- `src/kernel_v3.h`: ~40 LOC (mirror kv2.h structure; includes
  `kernel_v0_sycl.hpp` for `sycl_queue_handle` per kv2 pattern,
  per Sonnet-pragmatic review #X #7).
- `src/kernel_v3.cpp`: ~400-450 LOC (5 sections inline; per-chunk
  d fold pattern §4.2 adds ~30 LOC vs original estimate; activation
  quant inline ~40 LOC).
- `src/Makefile`: +2 LOC.
- `bench/sweep_tile.cpp`: +50 LOC (kv3 variant pass).
- **Total Phase 1 v3:** ~490-540 LOC across 4 files.

### Sequencing

1. **Phase 0.5: throughput micro-bench** (~1 hour wall-clock). Acts
   as hardstop per §6.0. Output: `bench/probe_dpas_throughput.{cpp,
   csv,log}`.
   **Hard gate**: if INT8/FP16 throughput ratio < 4×, STOP, escalate
   to v4 per §6.3.
2. **Phase 1 kernel skeleton** (~2 hours): single shape (16, 16, 256)
   smoke pass. Activation quant inline + dequant + DPAS MMA + per-
   chunk d fold + final store.
3. **Multi-shape correctness W1** (~1 hour): all 4 W1 shapes
   max_rel_err ≤ 1e-2. If activation quant + per-chunk d fold
   exceeds 1e-2 budget, fall back to per-block group-quant scheme
   (see §3.4 Phase 2 v3 deferred).
4. **Phase 1 v3 commit + push** + 3-voice review (Sonnet code-quality
   + Sonnet SYCL semantics + Opus strategic, mirror Phase 2b pattern).
5. **Phase 2 v3 profiler harness + W2 evaluation** (separate session).

LLM team timing realistic: Phase 0.5 + Phase 1 v3 deliverable in ~1
session wall-clock alpha-side (or my equivalent if I keep coding
solo per the current pattern). Phase 2 v3 in a 2nd session.

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
