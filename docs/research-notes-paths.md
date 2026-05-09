# Research Notes — Paths Explored for INT8/INT4 DPAS on Arc B60

**Date:** 2026-05-09 (live, append-only)
**Scope:** All paths considered for getting ternary GPU compute faster
than v0_BL scalar on Intel Arc Pro B60 (Xe2 / Battlemage), with
results / verdicts for each.

---

## Tested paths (with empirical data)

### Path 1 — kernel_v0_BL (scalar SYCL baseline) ✅ BASELINE

- File: `src/kernel_v0.cpp`
- Result: 2.17 ms on (16, 64, 14336). Correctness PASS.
- Status: SHIPPED, BASELINE for all comparisons.

### Path 2 — kernel_v1 (SLM tiling) ❌ FALSIFIED

- File: `src/kernel_v1.cpp`
- Result: 1.07× v0_BL (no-op). SLM tiling does not help; L1 already
  absorbs apparent redundancy.
- Status: shipped, kept as record.

### Path 3 — kernel_v2 (FP16 XMX via SYCL `joint_matrix`) ❌ FALSIFIED

- Files: `src/kernel_v2.cpp` (Phase 1 + Phase 2b Lecture B + vec
  loadA), `bench/v2_phase2_falsification.md`
- Result: 0.61× v0_BL (1.64× SLOWER) on (16, 64, 14336).
- Reason: FP16 dequant is structural cost, ~55% of t_full. Cannot
  beat scalar v0_BL via FP16 path on this hardware.
- Status: shipped, kept as production stub + benchmark target.

### Path 4 — INT8 DPAS via SYCL `joint_matrix` wrapper ❌ FALSIFIED

- File: `bench/probe_dpas_throughput.cpp`
- Result: 1.00× FP16 wrapper baseline (no compute speedup).
- Reason: SYCL ext::oneapi::experimental::matrix wraps to KHR
  CooperativeMatrixMulAddKHR, not the Intel-specific DPAS
  intrinsic. Wrapper level overhead caps at FP16 rate.
- Status: shipped (`bench/v3_phase0_5_falsification.md`).

### Path 5 — INT8 DPAS direct (OpenCL builtin, M=8) ⚠️ MARGINAL

- File: `bench/probe_dpas_opencl.cpp`
- Result (post spec-vs-code review fix):
  - INT8 8x16x32 M=8: **122 GOps/s = 1.30× FP16 wrapper baseline**
  - Per-MMA: 67 ns INT8 (8192 ops) vs 43 ns FP16 (4096 ops)
  - Silicon ops/ns: 122 INT8 vs 95 FP16 = **1.28× compute physiologique**
- Reason failed gate (>=1.5×): the SYCL wrapper was NOT the
  bottleneck. Real silicon DPAS gives ~1.3× compute, not 8× peak.
- Status: shipped, lessons learned.

---

## Untested paths (open opportunities)

### Path 6 — INT4 K=64 DPAS direct (`intel_sub_group_i4_i4_matrix_mad_k64`) ✅ PASS (2.609× FP16)

**Result on Arc B60 (driver 26.09.37435.1) — validated 2026-05-09:**
- Builtin found, kernel compiles with `__attribute__((intel_reqd_sub_group_size(16)))`.
- VALIDATION OK: kernel writes expected `512*n_reps` in all 16 lanes (= MMA actually computed correctly).
- **244.2 GOps/s sustained** at N_REPS = 2000 / 5000.
- Ratio vs FP16 SYCL wrapper baseline: **2.609×**.
- Ratio vs INT8 DPAS direct: **2.000×** (exactly 2× more ops/MMA at same wall-clock).
- Per-MMA wall-clock: 67 ns INT4 K=64 (16384 ops) — **identical to INT8 K=32** (67 ns / 8192 ops). Silicon packs 2× more ops/cycle in INT4 mode.

**Empirical signal: the 8× INT8 peak claim was theoretical;** the realistic exposed compute speedup on Xe2 / icpx 2025.3 / driver 26.09 is:
- ~1.3× INT8 vs FP16 (compute physiologique)
- ~2.6× INT4 vs FP16 (compute + 2× K-dim packing)
- Both well below the 8× peak headroom claim.

**The big opportunity.** Why this could be 4× compute on our workload:

- Builtin signature for SG=16, M=8 (spec line 162):
  ```c
  int8 intel_sub_group_i4_i4_matrix_mad_k64(short8 a, int8 b, int8 acc);
  ```
- Ops per MMA: M*N*K*2 = 8*16*64*2 = **16384 ops/MMA** (= 4× FP16
  16x16x16 4096 ops/MMA)
- If silicon time per MMA stays similar to INT8 K=32 (~67 ns),
  **INT4 K=64 throughput estimate = 245 GOps/s = 2.6× FP16 baseline**.
- **Ternary {-1, 0, +1} fits PERFECTLY in INT4 signed** (4-bit
  signed = -8..+7, plenty of headroom — wastes 1 bit per weight
  but no precision loss).

Per-lane storage analysis (SG=16, M=8, K=64):
- `short8 a`: 8 shorts × 2 bytes = 16 bytes/lane = **32 INT4/lane**
  (2 nibbles per byte). SG=16 × 32 = 512 INT4 = M*K = 8*64 ✓
- `int8 b`: 8 ints × 4 bytes = 32 bytes/lane = **64 INT4/lane**.
  SG=16 × 64 = 1024 INT4 = K*N = 64*16 ✓
- `int8 acc`: 8 int32/lane × 16 lanes = 128 int32 = M*N ✓

All-nibble=1 init for validation:
- `short = 0x1111` = 4 nibbles all = 1 → `(short8)(0x1111, ...)`
- `int = 0x11111111` = 8 nibbles all = 1 → `(int8)(0x11111111, ...)`
- Expected: each acc[m, n] = K*n_reps = 64 * n_reps
- Per-lane sum (acc.s0..s7) = 8 * 64 * n_reps = **512 * n_reps**

### Path 7 — INT2 path ❌ FAIL (not exposed via OpenCL builtin)

**Tested via compile-attempt probe 2026-05-09** (driver 26.09.37435.1):
```c
acc = intel_sub_group_i2_i2_matrix_mad_k128(a, b, acc);
```
IGC error: `use of undeclared identifier 'intel_sub_group_i2_i2_matrix_mad_k128'; did you mean 'intel_sub_group_i8_i8_matrix_mad_k32'?`

Conclusion: only INT8 + INT4 + BF16 + FP16 + TF32 builtins are exposed
in the OpenCL extension as of icpx 2025.3 / driver 26.09. The XMX
silicon supports INT2 per Intel hardware docs, but it's not accessible
via this API surface. SPIR-V intrinsic level or inline asm Xe2 would
be the only paths to access it (= Path 9 territory).

For ternary specifically, INT4 K=64 (Path 6, 2.6× FP16, validated)
is the practical near-native option since INT2 builtin doesn't exist.

### Path 8 — `cl_intel_subgroup_2d_block_io` ❌ MODEST (on 8x32 tile)

**Tested via `bench/probe_2d_block_io.cpp` 2026-05-09** with 8r32x1c
(8 rows × 32 cols × 1 byte = 256 bytes per call).

Result: **block IO is ~4× SLOWER than scalar coalesced reads** on
this shape (2.7 GB/s vs 10.1 GB/s, ratio 0.247×).

Likely explanation: 8x32 = 256 bytes is too small to amortize the
block read setup overhead (coord computation, pitch handling). The
scalar pattern (16 lanes × uint reads = 64-byte coalesced load,
compiler-optimized) wins on small tiles.

**Verdict for kv4 ternary on this shape**: not worth integrating.
Worth retesting on larger tiles (16r or 32r variants) which may
amortize better. Defer to Phase 2/3 v4 optim if ever needed.

The 2D block IO extension itself works (compiles, runs) — just not
beneficial for our small-tile size workload at first measurement.

### Path 9 — ESIMD `xmx::dpas` (Intel SYCL extension, NOT inline asm) ✅ PASS (3.76× FP16) ⭐ CHAMPION

**@naskel pointed to ESIMD as the path. It paid off enormously.**

`sycl::ext::intel::esimd::xmx::dpas<>` is a templated function in icpx
2025.3 that exposes the Xe2 DPAS instruction with **all the precision
types** including the silicon-native ones. Located at
`/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/`.

The `dpas_argument_type` enum (`common.hpp`) lists:
```c
u2 = 3,    // unsigned 2 bits
s2 = 4,    // signed 2 bits  ← TERNARY {-1, 0, +1} fit PARFAIT
u4, s4, s8, bf16, tf32, ...
```

**INT2 (s2) IS exposed via ESIMD**, even though it's NOT in the
OpenCL `cl_intel_subgroup_matrix_multiply_accumulate` extension or
SYCL `joint_matrix` / `precision::*` types. ESIMD is the documented
Intel-specific path to this silicon feature.

**Empirical result (validated, `bench/probe_esimd_int2.cpp`)**:
- INT2 (s2) × INT2 (s2) → INT32 acc, M=8, K=64, N=16 (Xe2 ExecSize=16)
- N_REPS=2000: **351.6 GOps/s** (= 46.6 ns/MMA, 16384 ops/MMA)
- N_REPS=5000: **367.6 GOps/s** (asymptotic)
- VALIDATED: output = 8192 × 50 = 409600 exact

**Ratios:**
- vs FP16 SYCL wrapper baseline (93.6 GOps/s): **3.76×** STRONG PASS
- vs INT4 K=64 OpenCL builtin (244.2 GOps/s): **1.44×** (ESIMD path
  is empirically faster than OpenCL builtin path even at same ops/MMA;
  likely due to single-thread SIMD16 vs SG-cooperative overhead)
- vs theoretical Intel "8× INT8 peak" claim: **3.76× achieved** (~47%
  of theoretical peak — best we've seen on this hardware/toolchain)

**Per-MMA wall-clock by precision (Arc B60, validated)**:
| Path | ns/MMA | ops/MMA | GOps/s |
|------|-------:|--------:|-------:|
| FP16 SYCL wrapper (16x16x16) | 43 | 4096 | 93.6 |
| INT8 OpenCL builtin (8x16x32) | 67 | 8192 | 122.1 |
| INT4 OpenCL builtin (8x16x64) | 67 | 16384 | 244.2 |
| **ESIMD INT2 (8x16x64)** | **47** | **16384** | **351.6** |

**Trade-off vs INT4 path**: ESIMD requires a different programming
model than SYCL classic (single-thread SIMD instead of SG-cooperative
nd_range). Phase 1 v4 implementation effort is +0.5-1 session vs
INT4 path. The 1.44× extra speedup vs INT4 OpenCL is worth the
complexity for the headline shape kv4 perf.

**Storage match for ternary**: INT2 signed `{00=0, 01=1, 11=-1}` —
ternary fits exactly, no storage waste. INT4 wastes 50% of B-matrix
bits per ternary code.

- intel/sycl-tla `include/cute/arch/mma_xe.hpp` uses inline asm
  Xe2 instructions: `"dpas. #TB . #TA .8.%3 (M1, 16) DST.0 ..."`
- Bypasses the OpenCL builtin layer entirely; direct vISA assembly.
- llama.cpp maintainers quote: "We don't think the joint_matrix
  extension will be a good fit ... we're planning to use DPAS
  instructions more 'directly'".
- Estimated +10-30% over OpenCL builtin (per Intel community).
- Trade-off: less portable (Intel-specific, possibly Xe2-only),
  more complex to maintain.

### Path 10 — Vulkan compute + KHR_cooperative_matrix

- `KHR_cooperative_matrix | int dot: 1` exposed on Arc Pro B-series
  via Mesa.
- llama.cpp ggml-vulkan uses `mul_mmq.comp` with
  `GL_EXT_integer_dot_product` for INT8 high-throughput.
- Different toolchain (no SYCL); rewrite kernel in GLSL compute.
- llama.cpp issue #21517 confirms Vulkan int8 path on Battlemage
  works at acceptable throughput post Q8_0 reorder (PR #21527
  lifted Q8_0 from 4.88 → 15.24 t/s = 3.1× via data layout fix).
- Effort: ~3-4 sessions to learn Vulkan compute + write a ternary
  kernel.

### Path 11 — `cl_intel_subgroup_split_matrix_multiply_accumulate` (`dpasw`)

- Cross-subgroup cooperation reduces register pressure
- Could help if our kernel is register-pressure bound
- Less documented, untested on Xe2

### Path 12 — oneDNN reference patterns

- oneDNN v3.8 explicitly optimized for Battlemage INT8 matmul
- benchdnn = canonical perf measurement tool
- Not a path per se, but a reference we should use to validate
  our kernel's perf relative to vendor-optimized

---

## Critical insight: kernel data layout matters massively

llama.cpp issue #21517 (Q8_0 bandwidth on Arc Pro B70):
- Q8_0 token gen achieves **21-24% of bandwidth** vs Q4_K_M **53-64%**
- Q8_0 is **4-5× slower than Q4_K_M** despite both being quantized
- On NVIDIA RTX 40/30: **Q8_0 < 5% slower than Q4** (INT8 tensor cores)
- Conclusion: **Battlemage has a software gap, not hardware gap**, in
  INT8 kernel data layouts (DMMV path vs MMVQ reorder path).

PR #21527 (merged) lifted Q8_0 tg from 4.88 → 15.24 t/s = **3.1×
speedup** via Q8_0 reorder + MMVQ path — pure data layout change,
no compute change.

Implication for our kernel: even if compute is the bottleneck (Path 6
INT4 K=64), **if our memory access pattern is wrong we can leave 3-5×
on the table.** Path 8 (2D block IO) + DPAS-friendly layout (= same
shape that DPAS expects, no transpose during compute) is critical.

---

## Decision tree (current state)

```
                 v0_BL = baseline (2.17 ms on headline)
                  |
                  v
  Try INT8 path? -> Path 4 (wrapper) FAIL, Path 5 (direct) MARGINAL 1.3x
                  |
                  v
  ===> NEXT: Path 6 INT4 K=64 direct ===
                  |
        +---------+---------+
        |                   |
   PASS (>= 2x)         FAIL/MARGINAL
        |                   |
        v                   v
   Phase 1 v4 kv4    Try Path 7 INT2?
   ternary INT4              |
                    +--------+--------+
                    |                 |
                  PASS              FAIL
                    |                 |
                    v                 v
               Phase 1 v4    Try Path 8 (2D blockIO)
               kv4 ternary    + Path 9 (inline asm)
               INT2          combined push
                                     |
                              if STILL FAIL:
                              Ship as-is (Path 10 Vulkan deferred,
                              follow-up via PR upstream).
```

---

## Method discipline (carved-in-stone since 2026-05-09)

1. **Save spec text in `docs/specs/`** for every API touched.
2. **2-agent cross-check spec vs code** before trusting any empirical
   probe result.
3. **Output buffer validation** in every probe (read back actual GPU
   output, verify against expected math) — never trust timings alone.
4. **dmesg pre/post diff** to catch GPU resets / coredumps.
5. **Apply fixes as a single batch** post-review, not incremental
   edits without QA cycle.

These rules saved us a multi-session false-implementation effort
when the Path 5 probe initially reported 1.991× (artefact, output=0
silently due to spec violation).

---

## File index (this project's paths-related artifacts)

| File | Purpose |
|------|---------|
| `docs/specs/cl_intel_subgroup_matrix_multiply_accumulate.asciidoc` | Khronos spec for the OpenCL DPAS extension |
| `docs/specs/probe_dpas_opencl_BROKEN_int8a.cpp.txt` | Forensic snapshot of broken probe state |
| `bench/probe_joint_matrix.cpp` | Phase 0 v2 — FP16/BF16 joint_matrix availability probe |
| `bench/probe_dpas.cpp` | Phase 0 v3 — INT8 DPAS shape availability probe |
| `bench/probe_dpas_throughput.cpp` | Phase 0.5 v3 — SYCL wrapper INT8 vs FP16 (= falsified) |
| `bench/probe_dpas_opencl.cpp` | Phase 0 v4 — OpenCL DPAS direct INT8 (= 1.3×) |
| `bench/v2_phase2_falsification.md` | v2 FP16 XMX falsification report |
| `bench/v3_phase0_5_falsification.md` | v3 wrapper-path falsification report |
| `bench/profile_v0_bl.md` | v0_BL profiler verdict (compute-bound) |
| `bench/profile_v2_p2a.md` | v2 Phase 2a profiler (top-1 dequant 55%) |
| `docs/design-v2.md` + `design-v2-phase-1.md` + `design-v2-phase-2.md` | v2 design + phasing |
| `docs/design-v3.md` | v3 Path A INT8 DPAS brief (revised expectations needed post-Path-5) |
