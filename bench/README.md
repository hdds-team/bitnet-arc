# bench/ -- bitnet-arc cross-check harness

Three-voice correctness harness for the bitnet-arc oracle, per
`oracle/README.md` tolerance model.

## Files

| File              | Role                                            |
|-------------------|-------------------------------------------------|
| `harness_3way.cpp`| Main harness (v1 / v2 / v3 cross-check)         |
| `sweep_tile.cpp`  | #148 tile sweep over `kv0_variants[]` (CSV out) |
| `gate_w1w2.py`    | #147 stop-gate over a sweep CSV (dry-run)       |
| `Makefile`        | Builds harness + sweep, runs gate via Python    |

## Voices and gates

```
  v1 = maison FP32 matmul on direct-dequantized weights
  v2 = TQ2_0 quantize -> dequantize -> same FP32 matmul
  v3 = numpy float64 BLAS (via subprocess to numpy_xcheck.py)
```

Comparison gates (constants from `oracle/tolerance.h`):

| Pair      | Tolerance                          | Catches                |
|-----------|-------------------------------------|------------------------|
| v1 vs v2  | `TOL_FP32_VS_FP32` (0)              | quantize / dequant bugs|
| v1 vs v3  | `TOL_FP32REF_VS_NUMPY_F64` (~1e-3) | bugs in our FP32 ref   |

The TOL_SYCL_* gates are stubbed; they activate when the harness
plugs the production kernel from `src/kernel_v0.{h,cpp}`.

## Test cases

| Case        | Shape             | Weights      | Why                              |
|-------------|-------------------|--------------|----------------------------------|
| `zero_tile` | M=16 N=16 K=256   | all zero     | output must be exactly zero      |
| `k256`      | M=16 N=16 K=256   | random ternary | smallest block-aligned K       |
| `k14336`    | M=1  N=64 K=14336 | random ternary | LLaMA-8B FFN representative dim|

Synthetic ternary weights follow the design v0 distribution:
zeros ~45%, +1 ~27.5%, -1 ~27.5%.

## Build & run

```bash
# 1. Build the oracle library if not done.
make -C ../oracle

# 2. Build and run the harness.
make -C .          # produces ./harness_3way
make run           # runs the 3 cases with seed 1337

# Override seed / script path:
./harness_3way --seed 42
./harness_3way --xcheck /custom/path/numpy_xcheck.py
```

Exit 0 = all cases pass, exit 1 = at least one gate failed.

## Status (3-way harness)

- [x] Voice 1 (FP32 maison) wiring
- [x] Voice 2 (TQ2_0 quantize/dequantize round-trip) wiring
- [x] Voice 3 (numpy float64 BLAS subprocess) wiring
- [x] Tolerance gates from `oracle/tolerance.h`
- [x] Three QA cases (zero_tile, k256, k14336)
- [x] First run on hardware: ALL PASS (margins ~3 orders below tol)

---

# `sweep_tile.cpp` -- #148 SYCL tile sweep

Iterates the variants registered in `src/kernel_v0.cpp::kv0_variants[]`
on a single shape (default `M=16 N=64 K=14336`, LLaMA-8B FFN), runs
warmup + timed launches, checks correctness vs an FP32 reference, and
emits a CSV row per variant on stdout.

```bash
make CXX_SYCL=icpx sweep_tile     # build (requires icpx + Arc device)
./sweep_tile                       # smoke shape, default seed
./sweep_tile --M 16 --N 64 --K 14336 --warmup 3 --timed 10
```

Tolerance: SYCL output rounds to FP16 once at the end, so the gate is
`BITNET_ARC_TOL_SYCL_VS_FP32REF` (1e-2). The tighter FP32-vs-FP32
gate from the 3-way harness does not apply here.

## Smoke baseline (Arc Pro B60, 2026-05-08)

First sweep on the registered 6 variants at `M=16 N=64 K=14336`:

| Variant                 | time_ms_med | bw_gbs | correct | max_rel_err |
|-------------------------|-------------|--------|---------|-------------|
| `16x16_sg16_BRANCHFUL`  | 3.71        | 0.19   | YES     | 9.47e-4     |
| `16x32_sg16_BRANCHFUL`  | 3.73        | 0.19   | YES     | 9.47e-4     |
| `16x16_sg16_BRANCHLESS` | **2.17**    | **0.32** | YES   | 9.47e-4     |
| `16x16_sg32_BRANCHFUL`  | 4.24        | 0.16   | YES     | 9.47e-4     |

(`32x16` and `32x32` skipped: `tile_M=32` incompatible with `M=16`.)

Observations:

- **Correctness 4/4** at this shape -- templated dispatch works, kernel
  output is FP16-stable vs the FP32 reference (10x under tolerance).
- **BRANCHLESS wins ~1.7x** over BRANCHFUL on identical config. GPU
  warps prefer SIMD-aligned multiplies-by-zero over divergent control
  flow, even when 45% of the multiplies are by zero.
- **`sg=32` is slower** than `sg=16` for a 16x16 tile (14% slower).
  A subgroup of 32 has idle items when the work-group is only 256 wide;
  this suggests the tile sweep should pair `sg` with the matching tile
  dim once we land 32x32 / 64x64 variants.
- **Bandwidth 0.16-0.32 GB/s** vs Arc B60 HBM peak 456 GB/s -- we are
  at 0.04-0.07% of peak. Expected for the v0 baseline (one work-item
  per output, no SLM, no coalesced loads); the value of #148 is
  precisely to make this gap visible. Stop-gates calibration (#147)
  must therefore use *relative* targets (each variant >=1.5x faster
  than baseline), not absolute fractions of HBM peak.

---

# `gate_w1w2.py` -- #147 stop-gate (dry-run by default)

Consumes a sweep CSV and emits a per-shape summary, a per-variant
verdict table, and a violation list. The gate model is **relative
to a designated baseline variant**, not absolute fractions of HBM
peak -- the v0 baseline lives at <0.1% of peak, so absolute
thresholds are not meaningful at this stage (per claude-opus's
brief in chat after the first Arc B60 smoke).

Gate semantics:

- **W1 (correctness)** : per variant, `correct == YES` AND
  `over_threshold == 0`. Hard gate: a `NO` always raises an error.
- **W2 (speedup)**     : per variant, `bandwidth_gbs / baseline_bw
  >= --min-speedup` (default 1.5x). Warn by default, upgraded to
  error in `--strict-gates`. The designated baseline is
  `16x16_sg16_BRANCHFUL` (matches design v0 narrative); if missing
  from a shape, falls back to the slowest runnable variant in that
  shape and emits a `baseline_missing` warning.
- **Outlier**          : per variant, `bw < --outlier-frac (=0.7)
  x shape_mean`. Always a warning -- catches bad tile/sg combos.
- **Soft peak**        : per shape, best `bw / peak <
  --soft-peak-pct (=0.5%)`. Pure diagnostic, never an error.

```bash
make gate                              # default: smoke baseline CSV
make gate GATE_CSV=path/to/run.csv     # arbitrary CSV
make gate-strict                       # exit 1 on error-severity hits
python3 gate_w1w2.py --csv-out v.csv sweep_w1_smoke_baseline.csv
python3 gate_w1w2.py --help            # full flag list
```

`--csv-out` writes a stable per-variant verdict CSV (columns:
`variant,M,N,K,bandwidth_gbs,speedup_vs_baseline,is_baseline,
gate_correctness_ok,gate_speedup_ok,is_outlier,pct_peak,verdict,
reason`) suitable for downstream calibration notebooks (review
#62) and cross-run regression diffs.

Lenient parser: `device:` headers and `skip <variant>: <reason>`
diagnostic lines from `sweep_tile.cpp` (when stderr is merged via
`2>&1`) are recognized and reported separately, not flagged as
errors. Header row is keyed by column name; reordered columns and
extra columns are tolerated as long as the required set
(`variant, M, N, K, bandwidth_gbs, correct`) is present.

## Severity model

| Finding                                       | Default | --strict-gates |
|-----------------------------------------------|---------|----------------|
| `correct == NO`                               | error   | exit 1         |
| `over_threshold > 0` on a correct row         | warning | unchanged      |
| `over_threshold > 0` on an incorrect row      | error   | exit 1         |
| `bw < 0.7 x shape_mean` (outlier)             | warning | unchanged      |
| `speedup < --min-speedup` (W2)                | warning | upgraded error |
| Best variant `< --soft-peak-pct` of peak      | warning | unchanged      |
| Canonical baseline missing in a shape         | warning | unchanged      |
| Parse errors                                  | logged  | not gated      |

The smoke baseline (`sweep_w1_smoke_baseline.csv`) currently passes
W1 4/4 and exposes one W2 pass (BRANCHLESS at 1.68x baseline) plus
two W2 regressions (16x32 and sg=32, both <=1.0x). That distribution
is what review #62 will use to calibrate the final `--min-speedup`
value.
