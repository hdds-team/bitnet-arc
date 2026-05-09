# Design v4 — Path 9 ESIMD INT2 for Ternary Matmul on Arc B60

**Status:** draft, awaits 2-voice review + @naskel ratification.
**Author:** @claude-opus (lead, post v3 falsification + R&D session 2026-05-09).
**Predecessors:**
- `docs/design-v0.md` / `design-v1.md` / `design-v2.md` / `design-v3.md`
- `bench/v3_phase0_5_falsification.md` (v3 wrapper-path FAIL)
- `bench/probe_dpas_int4_throughput.cpp` (Path 6, 2.61× FP16)
- `bench/probe_esimd_int2.cpp` (Path 9, **3.76× FP16, CHAMPION**)
- `docs/research-notes-paths.md` (12 paths surveyed)
- Memory `feedback_esimd_silicon_native.md`

---

## 1. Why Path 9 ESIMD INT2 (post-R&D)

The R&D session 2026-05-09 surveyed 12 candidate paths after v3 Path A
(INT8 via SYCL `joint_matrix` wrapper) was falsified at 1.0× FP16
throughput. Empirical findings, validated probes (output buffer +
dmesg pre/post + reproducibility):

| Path | API | Compute vs FP16 | Status |
|------|-----|-----------------|--------|
| 4 INT8 wrapper | SYCL `joint_matrix` | 1.00× | wrapper plafonné |
| 5 INT8 builtin | OpenCL ext | 1.30× | marginal |
| 6 INT4 K=64 | OpenCL ext | 2.61× | PASS |
| 7 INT2 builtin | OpenCL ext | not exposed | ❌ |
| 8 2D block IO | OpenCL ext | 0.25× scalar (8x32) | small tile lose |
| 11 dpasw split | OpenCL ext | not exposed | ❌ |
| **9 ESIMD INT2** | **`sycl::ext::intel::esimd::xmx::dpas`** | **3.76×** ⭐ | **CHAMPION** |

Per-MMA wall-clock (Arc B60, validated 2026-05-09):
- ESIMD INT2 8x16x64: **47 ns/MMA**, 16384 ops/MMA, 351.6 GOps/s
- INT4 OpenCL 8x16x64: 67 ns/MMA, 16384 ops/MMA, 244.2 GOps/s
- INT8 OpenCL 8x16x32: 67 ns/MMA, 8192 ops/MMA, 122.1 GOps/s
- FP16 wrapper 16x16x16: 43 ns/MMA, 4096 ops/MMA, 93.6 GOps/s

ESIMD INT2 is empirically the fastest path on this silicon/toolchain.

**Ternary fit in INT2 signed**: `{00=0, 01=+1, 11=-1, 10=-2 (unused)}`
— zero storage waste, no precision loss for the multiply step.
Storage rate = 2 bpw = exactly the TQ2_0 packing rate (no waste vs
INT4 path which stores ternary in 4 bits = 50% waste).

---

## 2. Kernel architecture

### 2.1 ESIMD programming model (vs kv0/v1/v2/v3)

ESIMD is a fundamentally different SYCL programming model than
kv0/v1/v2/v3 used:

| Aspect | kv0..v3 (cooperative SG) | kv4 (ESIMD) |
|--------|--------------------------|-------------|
| Thread model | nd_range with SG of 16 lanes cooperating | Single SIMD16 thread per WG |
| Lane semantics | Implicit per-lane data, `joint_matrix` aggregates | Explicit `simd<T, N>` per thread |
| Synchronization | `sycl::group_barrier(sg)` | None at thread level (DPAS is single-instruction) |
| MMA call | `joint_matrix_mad(sg, mC, mA, mB, mC)` | `xmx::dpas<8, 8, T, ..., s2, s2>(C, B, A)` |
| Header | `<sycl/ext/oneapi/matrix/matrix.hpp>` | `<sycl/ext/intel/esimd.hpp>` |
| Kernel marker | none / `sycl::reqd_sub_group_size` | `SYCL_ESIMD_KERNEL` |
| Launch | `nd_range<1>(N*16, 16)` | `nd_range<1>(N, 1)` (1 thread per tile) |

This shift is intentional and well-tested: probe_esimd_int2.cpp
validates the model, output buffer matches expected ternary math,
zero Xe driver dmesg events, reproducible <0.1% drift.

### 2.2 Tile geometry

Hardcoded to the Path 9 probed shape (Phase 1 v4):

| Param | Value | Source |
|-------|-------|--------|
| TILE_M | 8 | `RepeatCount` (M dimension) per dpas template |
| TILE_N | 16 | `ExecutionSize` Xe2 native |
| TILE_K (per DPAS) | 64 | `SystolicDepth × OpsPerChannel = 8 × 8` for INT2 |
| K_CHUNK | 256 | TQ2_0 block size (locked) |
| FRAGS_PER_CHUNK | 4 | K_CHUNK / TILE_K = 256 / 64 |

For Phase 1 v4 W1 shapes (cross-tab vs kv2/v3):
| Shape (M, N, K) | tiles_M | tiles_N | total tiles |
|-----------------|---------|---------|-------------|
| (16, 16, 256) | 2 | 1 | 2 |
| (16, 64, 4096) | 2 | 4 | 8 |
| (16, 64, 14336) | 2 | 4 | 8 |
| (64, 64, 14336) | 8 | 4 | 32 |

(Same tile counts as kv3 = same parallelism class.)

Launch geometry: `nd_range<1>(total_tiles, 1)` — one ESIMD thread
per output tile. Each thread internally is a SIMD16 unit doing the
8x16x64 DPAS sequence over chunks_per_col.

### 2.3 Inner loop (4 sections)

```
ESIMD thread per tile:
    simd<int, 128> mC = 0;                    // M*N FP32-ish int32 acc

    for chunk c in 0..chunks_per_col:
        # §A. Ternary B unpack from TQ2_0 -> INT2 simd
        simd<int, 64> B_int2_packed = unpack_ternary_chunk(b_blocks, ...);

        # §B. Activation A quant FP16 -> INT2
        simd<int, 32> A_int2_packed = quant_a_chunk(a_fp16, ...);

        # §C. DPAS MMA (single inline call, K=64 in one go)
        mC = xmx::dpas<8, 8, int, int, int, int, s2, s2>(mC, B_int2_packed, A_int2_packed);

    # §D. Final scale (s_a * s_b) + FP32->FP16 + scatter to global C_fp16
    write_C_fp16(c_fp16_global, mC, s_a, s_b, ...);
```

K_CHUNK = 256 = 4 fragments × K=64 per chunk. **Per-chunk d fold
problem (the v3 design blocker)** is naturally handled here because
INT2 path can fuse `d_chunk` into the per-chunk MMA via:
- `mC_chunk` (chunk-local FP32 accumulator)
- After DPAS: `mC_global += d_chunk × float(mC_chunk)`
- Reset `mC_chunk = 0` for next chunk

But thanks to ESIMD's `simd<>` flexibility, we can also do this
inline as register operations (no SLM round-trip needed like kv3
required). Concrete pseudocode in §3.4 below.

---

## 3. Ternary packing & quantization

### 3.1 Ternary B (weights) unpack: TQ2_0 → INT2

TQ2_0 storage: `{ d (FP16), qs[64] }` per 256-element block.
- `qs[64]` packs 256 codes via stripe-interleaved 2-bit pattern
  (per oracle/tq2_0.h formula `byte=(k>>7)*32+(k&31)`,
  `shift=((k>>5)&3)*2`).
- Per code: `00→-1, 01→0, 10→+1` (per oracle's reference decode).

INT2 signed target encoding: `{00=0, 01=+1, 11=-1, 10=-2 unused}`.

**Decode formula** (per-code remapping):
- TQ2_0 `00` → ternary `-1` → INT2 signed `11`
- TQ2_0 `01` → ternary `0` → INT2 signed `00`
- TQ2_0 `10` → ternary `+1` → INT2 signed `01`

This is a 2-bit lookup, easily computed via:
```c
// tq2_0_code (2 bits) -> int2 signed (2 bits)
int2_signed = (tq2_0_code == 0) ? 0b11 :
              (tq2_0_code == 1) ? 0b00 :
                                  0b01;
```
Or as a 6-bit LUT if we prefer (3 codes × 2 bits = 6 bits in a const).

For the chunk dequant, per-chunk we read 256 codes (= 64 bytes of qs)
and produce 256 INT2 codes packed into 16 ints (`simd<int, 16>` for
B per chunk... wait, K=64 per DPAS frag means we feed 64 INT2 per K
to DPAS, with N=16 cols, so per-frag B = 64×16 = 1024 INT2 = 256
bytes = `simd<int, 64>` per ESIMD thread).

Detailed packing layout TBD in Phase 1 v4 implementation brief
(`design-v4-phase-1.md`).

### 3.2 Activation A (FP16 → INT2 quant)

This is the most aggressive design choice. Standard BitNet uses
W1.58A8 (8-bit activations); we propose **W1.58A2 effectively**.

**Per-row symmetric quant** (formula corrected per review fold B3):
- `s_a[m] = max(|A_fp16[m, k]| for k in 0..K-1)` (the row's max abs)
- `A_q_int2[m, k] = round(A_fp16[m, k] / s_a[m])` clamped to {-1, 0, +1}

After this quant, INT2 values are in `{-1, 0, +1}` exactly, mapping
to INT2 signed bit patterns `{11, 00, 01}` respectively.

**Precision risk**: BitNet b1.58 was trained with INT8 activations;
INT2 quant of activations at inference time is more aggressive.
Phase 2b max_rel_err margin (1e-3 vs 1e-2 gate) gives ~10× headroom,
but INT2 act quant could blow that budget.

**Phase 1 v4 W1 gate strategy**:
- Test correctness on synthetic inputs (Phase 1 fixture).
- If max_rel_err > 1e-2 on real ternary weights + non-trivial
  activations, fall back to: keep B as INT2 + use ESIMD INT4 path
  with A quantized to INT4 (less aggressive). The 2.61× FP16 from
  INT4 path is still W2-passing.
- Document the activation quant policy as a kv4 variant flag:
  `KV4_ACT_INT2` vs `KV4_ACT_INT4`.

### 3.3 Output reconstruction (revised, post review B3)

After the inner loop (§3.4 below) processes all chunks with per-chunk
d folding, `mC` holds an FP32 simd vector that already incorporates
all `d_chunk[c, n]` scales. The final per-output reconstruction
applies ONLY the per-row activation scale:

```c
c_fp16[m, n] = fp32_to_fp16(s_a[m] × mC_fp32[m, n])
```

There is **no** `s_b[n]` term in this final formula — the per-N-column
b-side scaling has already been applied inside the chunk loop in §3.4.
This resolves the inconsistency the review caught between the §2.3
inner loop pattern and the previous draft of this formula.

### 3.4 Per-chunk d fold pattern (ESIMD register-only, no SLM)

This section was missing in the initial draft; per review B1 (caught
independently by both reviewers), it is the critical correctness
spec. Resolved here.

The `xmx::dpas<8, 8, int, int, int, int, s2, s2>(C, B, A)` call
returns `simd<int, 128>` = the M*N=8*16 INT32 accumulator. We
**cannot** reuse the same `mC` accumulator across chunks like kv2
did, because each chunk has its own per-N-column `d_chunk[n]` (FP16)
that must be folded in BEFORE the next chunk's INT32 result mixes in.

**ESIMD register-only fold pattern** (no SLM round-trip needed —
this is the win vs kv3's SLM-staged d fold):

```c
// Globals throughout the chunk loop, all in ESIMD registers
simd<float, 128> mC_fp32 = 0;     // FP32 accumulator across all chunks

for chunk c in 0..chunks_per_col:
    // (1) Load + remap ternary B for this chunk into VNNI INT2 layout
    simd<int, 64>  B_int2_packed = load_b_chunk_vnni_int2(b_blocks, ..., c);
    // (2) Quant A for this chunk into VNNI INT2 layout
    simd<int, 32>  A_int2_packed = quant_a_chunk_vnni_int2(a_fp16, s_a, ..., c);
    // (3) Load per-N-column d_chunk into FP32 simd (broadcast pattern)
    simd<float, 16> d_chunk_n = load_d_chunk_fp32(b_blocks, ..., c);

    // (4) DPAS: chunk-local INT32 acc (NOT mixed across chunks)
    simd<int, 128> mC_chunk_i32 = xmx::dpas<8, 8, int, int, int, int,
                                            xmx::dpas_argument_type::s2,
                                            xmx::dpas_argument_type::s2>(
        simd<int, 128>(0),  // chunk-local C-zero (reset per chunk)
        B_int2_packed, A_int2_packed);

    // (5) Convert chunk-local INT32 to FP32, apply per-N-column d_chunk,
    //     accumulate into mC_fp32 (cross-chunk).
    //
    //     Layout: mC_chunk_i32 holds 8 row × 16 col = 128 elements
    //     in row-major order (M=8 outer, N=16 inner). For each
    //     (m, n) element at index m*16+n, the d_chunk_n[n] applies.
    //     ESIMD provides simd::convert_to<>() and broadcast/replicate
    //     ops to make this a register-only operation.
    for m in 0..8:
        // Slice 16-element row, multiply by d_chunk_n (broadcast)
        simd<int, 16> row_i32 = mC_chunk_i32.select<16, 1>(m * 16);
        simd<float, 16> row_f32 = row_i32.convert_to<float>();
        mC_fp32.select<16, 1>(m * 16) += row_f32 * d_chunk_n;

// After loop: mC_fp32 holds the fully folded result for this tile
// (with all per-chunk d already mixed in).
```

This pattern keeps everything in ESIMD registers — no SLM staging,
no barriers (single SIMD16 thread per tile, no inter-thread coop).
The cost vs kv2's "register-only across all chunks" is a per-chunk
INT32→FP32 convert + a per-row × d_chunk multiply — both cheap
register operations. Total overhead estimated at ≤5% of t_full.

**Failure mode to test in Phase 1 v4 W1**: per-chunk fold precision.
INT32 accumulator + FP16→FP32 convert × per-chunk float-multiply
can accumulate small rounding errors over 56 chunks (K=14336 case).
The 1e-3 max_rel_err margin from Phase 2b should hold but verify.

### 3.5 VNNI INT2 layout for B and A (per review B2)

Reviewer ESIMD-semantics caught that the `xmx::dpas` API requires B
in **VNNI-encoded layout** (per `dpas.hpp` line 226). The Phase 0.5
v4 probe used uniform `0x55555555` constants for both A and B, so
it validated the **throughput** correctly but did NOT validate the
**layout** (a constant matrix is invariant to any element permutation).

VNNI for INT2 on Xe2 (per Intel docs + dpas.hpp `OpsPerChannel = 8`
at `K = SystolicDepth × 8 = 64`):

- **B matrix shape K × N = 64 × 16** INT2 elements = 128 bytes total
  per chunk per ESIMD thread.
- **VNNI encoding**: per N column n, the K=64 elements are packed
  into 4 `int32` dwords. Each dword packs 16 K-positions × 2 bits
  in K-ascending order at bit positions [0..1, 2..3, ..., 30..31].
  - Dword[0] holds K positions {0, 1, ..., 15} of column n.
  - Dword[1] holds K positions {16, 17, ..., 31}.
  - Dword[2] holds K positions {32, 33, ..., 47}.
  - Dword[3] holds K positions {48, 49, ..., 63}.
- **VNNI grouping across N**: the 16 N columns are interleaved in
  the `simd<int, 64>` storage as `{N_dword0, N_dword1, ..., N_dword15,
  N+1_dword0, ...}` — i.e., 4 contiguous dwords per N column,
  consecutive in the simd vector.
- Total: 16 N columns × 4 dwords/col = 64 ints = simd<int, 64>. ✓

For A matrix shape M × K = 8 × 64 INT2 elements = 128 bytes:
- VNNI: per M row, K=64 elements packed into 4 dwords (same K
  position ordering as B per-N-column).
- 8 M rows × 4 dwords/row = 32 ints = simd<int, 32>. ✓

**Mapping from TQ2_0 storage to VNNI INT2 layout**:
- TQ2_0 byte/shift formula: `byte = (k>>7)*32 + (k&31)`, 
  `shift = ((k>>5)&3)*2`.
- For each (k, n_local) position in our chunk:
  1. Look up TQ2_0 code via the byte/shift formula.
  2. Remap to INT2 signed: `{0→0b11, 1→0b00, 2→0b01}`.
  3. Place at VNNI dword `dword_idx = k / 16`, bit position
     `(k % 16) * 2` within that dword.
  4. The N-column striping in simd<int, 64> means the dword goes
     to position `n_local * 4 + dword_idx`.

For activation A FP16 → INT2 quant + VNNI:
- Per-row max-abs reduce → s_a[m].
- Per-element divide + round + clamp to {-1, 0, +1} → INT2 signed.
- Pack into VNNI same way (M-row striping × 4 dwords per row).

**Phase 1 v4 W1 must include hetero data fixtures**: random
non-uniform A and B values to validate the VNNI layout actually
produces correct output (vs the oracle FP32 reference). The
probe's uniform-1 fixture is insufficient.

---

## 4. Phasing & hardstops (mirror v3 pattern)

### Phase 0.5 v4 (DONE)

`bench/probe_esimd_int2.cpp`: ESIMD INT2 throughput = **3.76× FP16**,
validated, reproducible. Path 9 unblocked.

### Phase 1 v4 (this brief targets)

- **Files:**
  - `src/kernel_v4.h` (~50 LOC, mirror kv2/v3 .h structure with
    ESIMD-aware config struct)
  - `src/kernel_v4.cpp` (~400-500 LOC, ESIMD kernel + 4 sections
    per §2.3 + ternary packing helpers)
  - `bench/sweep_tile.cpp` extension (~30 LOC, kv4 variant pass)
- **Acceptance**: W1 correctness max_rel_err ≤ 1e-2 on 4 W1 shapes.
  Phase 1 v4 fall-back to INT4 act if INT2 act fails W1 (per §3.2).
- **Hardstop**: 1 implementation session.

### Phase 2 v4

- **W2 gate**: ≥1.5× v0_BL on (16, 64, 14336) headline shape.
  Estimated kv4 = 4-6× v0_BL (very comfortable).
- **Hardstop**: 1 perf evaluation session. Pass → ship + PR upstream.

### Phase 3 v4 (perf optim, conditional)

- Variant sweep: TILE_M ∈ {4, 8} (kv4 INT2 supports M=1, 2, 4, 8 per
  dpas template line 78), TILE_N fixed = ExecutionSize=16.
- BF16 path probe (deferred from v2 Phase 0).
- 2D block IO retest on 16x32 / 32x32 tiles (Path 8 follow-up).
- Multi-WG cooperative output tile if profiler shows occupancy issue.

### Total budget (per @naskel's 3-4 session estimate)

- Phase 0.5: DONE (~2 hours wall-clock)
- Phase 1: ~1 session
- Phase 2: ~1 session
- Phase 3 (optional): ~1 session
- **Total: 3-4 sessions if Phase 3 needed, 2 sessions if Phase 2
  passes W2 directly.**

---

## 5. Risk register

1. **INT2 activation quant precision**. Most likely correctness blocker.
   Mitigation: variant flag `KV4_ACT_INT4` fallback (per §3.2).
   2.61× FP16 from INT4 path is still W2-passing.

2. **ESIMD register pressure**. The `simd<int, 128>` accumulator
   plus per-chunk B+A simd vectors may exceed register budget on
   Xe2. Spill = perf collapse. Mitigation: reduce TILE_M to 4 or 2
   if spill measured; the dpas template supports it (line 78
   `RepeatCount ∈ [1..8]`).

3. **ESIMD driver stability**. ESIMD path is less battle-tested than
   joint_matrix path on Battlemage. We have driver 1.14.37435+1
   working; future driver upgrades may regress. Mitigation: pin
   driver in CI / docs; @naskel's kernel custom + X11/Mate setup
   already off-limits for upgrades during this project.

4. **~~Compilation flags~~ — RESOLVED** (per review fold F1).
   Empirically the probe `bench/probe_dpas_int2.cpp` builds and
   runs correctly with the standard Makefile flags (`-fsycl
   -O2 -std=c++17`), no `-fsycl-device-code-split-esimd` needed.
   The flag is required only when ESIMD and SYCL classic kernels
   coexist in the same compilation unit; our `kernel_v4.cpp` is
   purely ESIMD (no `joint_matrix` etc.) so the flag is not needed.

5. **~~Mixing ESIMD and SYCL classic in same TU~~ — RESOLVED**
   (per review fold F2). The architecture naturally separates:
   `kernel_v4.cpp` is purely ESIMD (no `joint_matrix`); `kernel_v0/
   v1/v2/v3.cpp` are SYCL classic (no ESIMD). The host dispatcher
   in `sweep_tile.cpp` calls into both via `q.submit(...)` — this
   is fine because the kernels are in separate device TUs. The
   risk only materializes if someone tries to put `joint_matrix`
   inside `kernel_v4.cpp`, which the brief explicitly forbids.

6. **Per-chunk d fold precision**. INT32 accumulator + per-chunk d
   FP16 → FP32 conversion. Float arithmetic accumulates rounding
   over 56 chunks (K=14336 case). 1e-3 margin should hold but
   verify in W1 testing.

7. **Kernel input contracts**. Per fold F3 (review pragmatic):
   `static_assert(K_CHUNK == 256, "TQ2_0 block size locked")` and
   runtime assert `K % K_CHUNK == 0` must be in `kernel_v4.h`
   public API (mirroring `kernel_v2.h` line 80-86 pattern).
   `M % TILE_M == 0` and `N % TILE_N == 0` similarly enforced.

---

## 6. Out of scope (deferred to v5+)

- **Inline asm Xe2** (true low-level). ESIMD already gives the
  silicon-native types; inline asm gain estimate +10-30% only
  per intel/sycl-tla pattern, not worth the complexity vs
  3.76× already achieved.
- **Vulkan compute shader path** (Path 10). Different toolchain,
  3-4 sessions effort, deferred unless ESIMD path hits
  unrecoverable blocker.
- **Decode regime (M=1)**. ESIMD INT2 dpas template requires
  RepeatCount ≥ 1. Theoretically M=1 works but TILE_M=1 means
  underutilized DPAS unit. Decode falls back to v0_BL scalar
  (same dispatch policy as v2/v3 designs).
- **Multi-tile output reduction (split-K)**. Not needed for typical
  prefill shapes; Phase 3 v4 only if profiler shows occupancy gap.

---

## 7. LOC estimate & sequencing

### Phase 1 v4

- `src/kernel_v4.h`: ~50 LOC
- `src/kernel_v4.cpp`: ~400-500 LOC (ESIMD kernel + ternary unpack
  + activation quant inline + DPAS sequence + scale fold + store)
- `src/Makefile`: +2 LOC (kv4.o + ESIMD flags if needed)
- `bench/sweep_tile.cpp`: ~30 LOC (kv4 variant in the iteration)
- **Total: ~480-580 LOC across 4 files.**

### Sequencing

1. Phase 1 v4 implementation (1 session)
   - Hour 1: kernel_v4 skeleton + ternary unpack helpers
   - Hour 2: integration with sweep_tile + first build
   - Hour 3: W1 correctness on 4 shapes
   - Hour 4: 2-voice spec-vs-code review (ESIMD dpas headers vs kernel)
   - Commit + push
2. Phase 2 v4 perf evaluation (1 session)
   - Run sweep_tile, measure t_med
   - W2 gate eval; if pass, ship. If <1.5× v0_BL, profile & decide.
3. Phase 3 v4 optim (conditional, 1 session)

---

## 8. Inheritance from prior phases

| Lesson | Source | v4 application |
|--------|--------|----------------|
| Spec-vs-code 2-agent review | v3 commit a3293d7 | Apply on kv4 implementation pre-W1 |
| Output buffer validation in probes | v3 (caught int8 a SG=8 UB) | Built into all kv4 fixture tests |
| dmesg pre/post diff | R&D 2026-05-09 | Apply during Phase 1 v4 W1 runs |
| Hardstop discipline | v1 + v2 + v3 | 1 session per phase, no endless tuning |
| Per-chunk d fold pattern | v3 §4.2 | Apply in §2.3 inner loop (cleaner with ESIMD simd<>) |
| Verdict-as-last-keyword | Phase 2b reviews | Apply to all kv4 review chat |
| ESIMD = silicon-native path | This R&D session | New pattern, memory'd globally |

---

## 9. Ratification gate

This brief targets 2-voice review (Sonnet pragmatic ship-readiness +
Sonnet ESIMD/SYCL semantics) + @naskel approval before Phase 1 v4
implementation starts.

**Open questions for review/@naskel:**
- (a) Activation INT2 quant aggressive vs INT4 fallback variant —
  ship both as runtime variants or only INT2 in Phase 1 then add
  INT4 if W1 fails?
- (b) Phase 3 v4 BF16 path probe priority — defer to v5 entirely
  given INT2 success, or keep as Phase 3 follow-up?
- (c) PR upstream llama.cpp timing — after Phase 2 v4 W2 PASS,
  before or after Phase 3 sweep optims?

Phase 1 v4 implementation brief (`docs/design-v4-phase-1.md`) draft
starts after this brief is ratified.
