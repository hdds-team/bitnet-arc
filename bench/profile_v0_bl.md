# v0_BL Hardware Profile (Arc B60) — Task #155

**Date:** 2026-05-09
**Hardware:** Intel Arc Pro B60 Graphics (Xe2 / Battlemage)
**Variant profiled:** v0 `32x32_sg16_BRANCHLESS`
**Shape:** M = 64, N = 64, K ∈ {256, 4096, 14336}
**Method:** ablation timing — T(K) scaling + bandwidth saturation analysis
**Tool:** `bench/profile_v0_bl.cpp`, 50 timed runs after 5 warmup, host
wall-clock per launch (`q.wait_and_throw()` synced)
**Run by:** @claude-opus on Arc B60 host

---

## Verdict: COMPUTE-BOUND (ALU-throughput-bound)

| K     | t_med (ms) | bandwidth (GB/s) | % HBM peak |
|-------|------------|------------------|------------|
| 256   | 0.092      | 0.49             | 0.11%      |
| 4096  | 0.798      | 0.75             | 0.16%      |
| 14336 | 2.402      | 0.87             | 0.19%      |

HBM peak on Arc B60 = 456 GB/s. The kernel is **524× under-saturated**
on memory at K = 14336.

---

## Analysis

### Slope T(K)/K — linear, no launch-overhead floor

T(K) / K ≈ 0.16–0.18 µs per K-unit across the three points:

- (256 → 14336): 56× K, 26× t_med → near-linear after subtracting fixed
  launch overhead.
- The intercept implied by t(256) = 92 µs minus the per-K linear
  contribution is ~50–80 µs of fixed launch + setup, **negligible**
  vs the 2.4 ms steady time at K = 14336.

Linear scaling with negligible intercept rules out **launch-bound**.

### Bandwidth saturation — far below HBM peak

At K = 14336: 0.87 GB/s actual / 456 GB/s peak = **0.19%**. The kernel
reads ~2.1 MB per invocation (weights + activations + output) and
takes 2.4 ms — that is 524× more time than HBM alone would require
to deliver those bytes. There is **no contention on memory**.

This rules out **memory-bound** conclusively.

### Compute sanity check — 60× off scalar peak

At K = 14336:

- Total MAC count (output-element view): M × N × K = 64 × 64 × 14336 ≈
  58.7 M ternary FMAs (treating each ternary contribution as one op).
- Arc B60 theoretical FP32 scalar peak: ~1.5 TFLOPS.
- Compute-only floor: 58.7M / 1.5T ≈ **39 µs**.
- Measured: 2.402 ms = **~60× off the scalar peak**.

The v0 kernel runs 1 work-item per output element with a serial inner
K-walk. Each work-item issues scalar FMAs only. SIMD lanes inside each
sub-group execute the *same* op on adjacent outputs — useful — but the
inner K-loop is a sequential scalar reduction with no vectorization
across K. That is where the 60× efficiency gap lives: the ALU is
idle most of the time waiting on its own previous output.

---

## Implications for design v2

### Strong candidate: **XMX path**

Design v0 §2.2 disqualified XMX with the rationale "ternary is trivial,
no need for matrix accel". The hardware data **inverts that
reasoning**:

- We are 60× off the scalar peak.
- The kernel is *not* trivial in practice — it is a sequential
  reduction that wastes ALU cycles.
- XMX (Intel's matrix engine on Xe2) issues one matrix-multiply-and-
  accumulate per cycle per sub-group, which directly addresses this
  inefficiency by parallelizing the K-walk reduction.

Realistic v2 target on this shape: **~5–10× speedup via XMX**, which
would land us comfortably above the 1.5× gate (and possibly above
2× depending on how cleanly the ternary codes can be unpacked into the
XMX operand registers).

### Disqualified: vec loads, B layout pre-shuffle

Both proposed Plan B candidates address *memory access patterns*:

- `sycl::vec<half, N>` / `sub_group::load` — vectorized activation
  reads.
- B layout pre-shuffle — eliminate the 66 B stride at encode time.

With HBM at 0.19% saturation, **memory access is not the bottleneck**.
These optimizations would not move the needle on this regime. They
remain valid as second-order optimizations *after* XMX brings compute
into the same order of magnitude as memory traffic — but they cannot
unlock the 60× gap on their own.

---

## Caveats

1. **Variant choice**. Profile was run on `32x32_sg16_BRANCHLESS`,
   chosen because (M, N) = (64, 64) is exactly divisible by the
   32×32 tile and gives the natural full-occupancy WG. The canonical
   W1.5 baseline is `16x16_sg16_BRANCHLESS`. The compute-bound
   verdict is structural (1-WI-per-output scalar reduction) and holds
   across all v0 variants — the bottleneck lives in the inner K-walk,
   not in the tile dim.

2. **B-null ablation skipped**. @beta's review #72 flagged the
   B-null variant as a potential Phase 2 add if results were ambiguous
   mid-range. With bandwidth at 0.19% peak (524× under-saturation),
   the discriminator is unambiguous and Phase 2 is not needed.

3. **Host wall-clock timing**, not Level Zero device metrics. The
   µs-scale launch overhead and ms-scale kernel time give a
   30–100× signal/noise ratio — more than sufficient for the regime
   determination. Level Zero metrics would only be required for
   sub-µs accuracy, which we do not need.

---

## Raw measurement data

The headline numbers above are the actionable summary committed to
git history.

The full per-K record (columns: `variant, M, N, K, n, t_ms_min,
t_ms_med, t_ms_mean, t_ms_stddev, t_ms_p99, bytes, bandwidth_gbs`) is
captured in `bench/profile_v0_bl.csv` after running:

```bash
cd bench && make profile_v0_bl && ./profile_v0_bl > profile_v0_bl.csv 2> profile_v0_bl.log
```

CSV and log files are `.gitignore`d per repo policy (`*.csv`, `*.log`).

---

## Next steps

1. **@claude-opus** drafts a design v2 brief (XMX-first), pending
   @naskel ratification.
2. **No new kernel code** until the design v2 brief is reviewed and
   approved.
3. **Task #155** is closed on commit of this report.

The kernel v1 effort (#152, #153, #154) stays in git history as
contract data: SLM tiling does not move the needle on Arc B60
because L1 already absorbs the apparent redundancy. Design v2 will
target compute throughput directly via XMX.
