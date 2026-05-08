# bench/ -- bitnet-arc cross-check harness

Three-voice correctness harness for the bitnet-arc oracle, per
`oracle/README.md` tolerance model.

## Files

| File              | Role                                            |
|-------------------|-------------------------------------------------|
| `harness_3way.cpp`| Main harness (v1 / v2 / v3 cross-check)         |
| `Makefile`        | Builds against `oracle/liboracle.a`             |

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

## Status

- [x] Voice 1 (FP32 maison) wiring
- [x] Voice 2 (TQ2_0 quantize/dequantize round-trip) wiring
- [x] Voice 3 (numpy float64 BLAS subprocess) wiring
- [x] Tolerance gates from `oracle/tolerance.h`
- [x] Three QA cases (zero_tile, k256, k14336)
- [ ] First run on hardware (requires `python3 + numpy` available)
- [ ] SYCL kernel v0 plugged in (#146 v2 / W1 close)
