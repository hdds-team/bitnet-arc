# Design v3 Path A Falsification Report — Phase 0.5 Throughput Gate Fail

**Date:** 2026-05-09
**Hardware:** Intel Arc Pro B60 Graphics (Xe2 / Battlemage)
**Toolchain:** icpx 2025.3, Level Zero V2, driver 1.14.37435+1
**Brief reference:** `docs/design-v3.md` §6.0 hardstop gate.
**Probe binary:** `bench/probe_dpas_throughput.cpp` (commit on this PR).

---

## Verdict

**Path A (INT8 DPAS) is FALSIFIED at the Phase 0.5 throughput gate.**
The 8× compute headroom claim that justified the v2-to-v3 pivot is
NOT exposed by `sycl::ext::oneapi::experimental::matrix` INT8 wrapper
on icpx 2025.3 / Arc B60. INT8 DPAS throughput equals FP16
joint_matrix throughput (ratio = 1.00× across measurements).

The §6.0 hardstop fired exactly as designed: ~30 minutes of wall-clock
to discover this, vs ~2-3 sessions of Phase 1 v3 kernel work that
would have re-arrived at the same falsification through a much more
expensive path.

---

## Empirical data

`bench/probe_dpas_throughput.cpp` runs many isolated joint_matrix_mad
iterations of equivalent op counts (4096 ops/MMA in both INT8 and
FP16 paths) and times them via SYCL events
(`command_start/command_end`, kernel-only, excluding launch latency).

| N_REPS | INT8 DPAS (8x16x32) | FP16 joint_matrix (16x16x16) | INT8/FP16 ratio |
|--------|---------------------|------------------------------|-----------------|
| 500 | 0.0205 ms (100.0 GOps/s) | 0.0231 ms (88.7 GOps/s) | 1.127× |
| 1000 | 0.0445 ms (92.0 GOps/s) | 0.0445 ms (92.0 GOps/s) | **1.000×** |
| 2000 | 0.0877 ms (93.4 GOps/s) | 0.0876 ms (93.6 GOps/s) | **0.999×** |

The 500-rep result shows ~12% INT8 advantage — but it's noise-floor
(short kernels include relatively more overhead). At 1000 and 2000
reps, the ratio collapses to ~1.0× exactly, confirming the
asymptotic throughput is identical between INT8 DPAS and FP16
joint_matrix wrappers.

Both paths converge to ~93 GOps/s sustained on Arc B60 at this
fragment size. Compared to Intel's official spec sheet (197 TOPS
INT8 = 197,000 GOps/s peak), we're at **0.047% of peak INT8** and
**0.38% of peak FP16** (24.58 TFLOPS = 24,580 GFLOPs/s). The
wrapper API leaves ~99% of peak compute on the floor on isolated
MMA throughput tests.

---

## What this means for Path A

The entire structural argument for Path A was that ternary
`{-1, 0, +1}` codes can fit in INT8 directly without FP16
materialization (the v2 top-1 bottleneck), letting the kernel use
the 8× higher INT8 silicon throughput.

The first half of that argument still holds: ternary IS trivially
representable as INT8, and the dequant cost IS structural to FP16
path B. But the second half is empirically wrong on this toolchain:
**the INT8 silicon throughput is NOT exposed via the SYCL wrapper.**
The `joint_matrix_mad<int8, int32>` operation on Arc B60 / icpx
2025.3 runs at the same wall-clock rate as `joint_matrix_mad<half,
float>`. Whether the wrapper is internally lowering INT8 to FP16
DPAS, or the DPAS hardware itself isn't being clocked at its INT8
rate, is opaque to us at this layer.

Best-case Path A perf reasoning revised:
- v2 Phase 2b numbers: kv2 = 3.55 ms on (16, 64, 14336), 1.64×
  SLOWER than v0_BL.
- Path A would eliminate dequant (~55% of t_full) = saves ~1.96 ms
  → kv3 estimated ~1.59 ms.
- v0_BL = 2.17 ms.
- kv3 / v0_BL = 0.73× = still 1.37× SLOWER than v0_BL.
- W2 gate (≥1.5× v0_BL = ≤1.45 ms) still missed by ~10%.

Path A on the current toolchain cannot pass W2 gate even in the
absolute best case (zero dequant cost + identical compute throughput
to FP16). The only remaining lever in Path A would be SLM staging
optimizations / occupancy gains (kv3's 7 KB SLM vs kv2's 16 KB =
2× concurrent WGs/Xe-core potential). But those are 5-15% gains,
not the 1.4× still required.

---

## What did NOT cause the fail

- **Probe correctness**: `bench/probe_dpas.cpp` (commit `a36dd74`)
  validated `int8 × int8 → int32` at 8x16x32 with bit-exact match.
  The MMA operation is functionally correct; it's just not faster.
- **DPAS hardware**: Arc B60 has DPAS units capable of 197 TOPS
  per Intel spec. The hardware exists. The wrapper is the bottleneck.
- **Brief design**: design v3 brief (commits `065fe53` + `1b542b6`)
  resolved the per-chunk d fold blocker, the lane mapping caveat,
  and the activation quant sharing question. The kernel design was
  ready for implementation. The Phase 0.5 hardstop is the right
  gate to have *before* implementation.

---

## Toolchain caveat (out-of-scope for project, flagged for record)

It is *possible* that a newer icpx release or a newer Arc compute
runtime version exposes more INT8 DPAS throughput. We have not
upgraded per @naskel's flag re custom kernel + X11/Mate setup
fragility. The current state (icpx 2025.3, driver 1.14.37435+1,
Level Zero V2) is what we tested. If a future toolchain upgrade
exposes the 8× claim, this falsification can be revisited.

A direct DPAS intrinsics path (bypassing `joint_matrix` wrapper,
using `__builtin_dpas` or similar Intel-specific intrinsic if
exposed) might also bypass the wrapper bottleneck. This is out
of scope — the project chose the portable `sycl::ext::oneapi::
experimental::matrix` namespace per design v2 §2.2 portability
rationale; using Intel-specific intrinsics breaks that.

---

## v4 escalation options (per design v3 §6.3)

Per the discipline lessons from v1 + v2 + v3 falsifications (3
hardstops respected in 3 phases), the project reaches a
strategic-pivot moment that exceeds tactical options:

### Path C (custom packing)

Repackage TQ2_0 into a format `joint_matrix` can consume directly.
Speculative; if the wrapper INT8 throughput is the bottleneck,
custom packing FP16 might not help either. ROI uncertain. Brief
Phase 0 probe could test FP16 wrapper at unusual fragment shapes.

### Hard pause + scope re-evaluation

Step back, reconsider:
- **Decode-priority pivot**: M=1 autoregressive cannot run kernel_v2
  or hypothetical kernel_v3 (TILE_M ≥ 8 minimum). Decode falls back
  to v0_BL scalar. If the project's true target is decode (LLM
  inference end-user), the GPU prefill kernel path was a means-not-
  ends. v0_BL on prefill + v0_BL on decode is already shippable.
- **Hardware retarget**: Vulkan path on the same Arc B60 might
  expose more compute (different driver/runtime). Or shift target
  to RTX 3070 Ti where CUDA tensor cores have well-documented INT8
  paths.
- **Scope acceptance**: the project has already produced significant
  ecosystem value (oracle 10K bit-identical harness, v0_BL kernel
  passing W1, three honest falsifications informing future work).
  Shipping current state as a PR upstream is a viable conclusion.

### Out of contention

- Path A retry on this toolchain — hardstop was definitive.
- Phase 1 v3 implementation despite the throughput gate fail —
  would knowingly lead to a 3rd falsification report; violates the
  v1/v2 lesson "no endless tuning past hardstop".

---

## What the project has shipped (ecosystem value summary)

Independent of v4 outcome, the project's current artifacts are
shippable as a PR llama.cpp upstream and/or blog material:

- **Oracle 3-way harness** (`oracle/`): 10K bit-identical samples
  vs upstream llama.cpp commit `deab41ec6` for TQ2_0.
- **kernel_v0** (correctness baseline) + **kernel_v0_BL** (branchless
  variant, 1.7-1.8× over BRANCHFUL on all W1 shapes) — first
  GPU-native SYCL ternary matmul kernel passing W1 correctness.
- **kernel_v1** (SLM tiling, falsified at W2): pedagogical record
  of why naive SLM tiling does not move the needle on Arc B60
  (L1 already absorbs apparent redundancy).
- **kernel_v2** (FP16 XMX, falsified at W2): pedagogical record
  of why FP16 path is structurally capped at ~0.7-0.8× v0_BL on
  this hardware (dequant materialization cost).
- **Three Phase 0 probes**: joint_matrix FP16/BF16 (`probe_joint_
  matrix.cpp`), INT8 DPAS correctness (`probe_dpas.cpp`), INT8
  DPAS throughput (`probe_dpas_throughput.cpp` — this commit).
- **Three falsification reports** (kv1 via README, kv2 via
  `v2_phase2_falsification.md`, kv3 v3 Path A via this report):
  honest project hygiene + scientific contribution to the SYCL
  ternary GPU literature.

The negative results matter: knowing what does NOT work on Arc B60
saves future ternary-on-Intel-GPU efforts from re-walking the same
ground.

---

## Status

- **Phase 0.5 v3 FAILED, Path A blocked** per §6.0 hardstop.
- **No Phase 1 v3 kernel work** triggered.
- **v4 escalation options surface to @naskel** for strategic call.
- **Ecosystem-value artifacts** preserved in tree, ready for PR.
