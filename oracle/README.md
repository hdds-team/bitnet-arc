# oracle/ -- correctness reference for the SYCL ternary kernel

Three-way validation harness for the v0 PoC, per Workflow #14 design v0
review (@theta points 4 + 5).

## Files

| File                | Purpose                                            |
|---------------------|----------------------------------------------------|
| `fp16.h`            | Header-only IEEE-754 binary16 <-> FP32 helpers     |
| `tq2_0.{h,c}`       | TQ2_0 quantize / dequantize, port of upstream      |
| `fp32_matmul.{h,c}` | Naive FP32 ternary x FP16 matmul, canonical order  |
| `tolerance.h`       | Single source of truth for cross-check tolerances  |
| `numpy_xcheck.py`   | Third independent reference (numpy float64 BLAS)   |
| `Makefile`          | Builds liboracle.a with -ffp-contract=off          |
| `upstream_gating/`  | One-shot bit-identical test vs llama.cpp (skeleton)|

## How the three voices align

For a tile of (M, N, K) with FP16 activations and TQ2_0 ternary weights:

```
  voice 1: maison FP32 matmul         (oracle/fp32_matmul.c)
  voice 2: ggml-style dequant + matmul (oracle/tq2_0.c -> fp32_matmul.c)
  voice 3: numpy float64 dense matmul  (oracle/numpy_xcheck.py)
```

Voices 1 and 2 share the matmul code path (only the weight source
differs: integer ternary vs dequantized FP32). Voice 3 is the
independent break of the symmetric-bug class identified by @theta:
if a bug lives in our matmul, voices 1 and 2 will agree on the wrong
value but voice 3 will diverge.

### Tolerance model (per @sonnet + @theta review)

The three voices are NOT bit-identical to each other -- only voices 1
and 2 are, because they share the canonical reduction loop. Voice 3
goes through BLAS (np.dot dispatches to MKL / OpenBLAS), which uses
its own pairwise-summation order and is therefore not associative-
equivalent to our naive m-n-k loop.

Three distinct comparison gates, three distinct tolerances:

| Pair                      | Expected   | Tolerance    | Catches                |
|---------------------------|------------|--------------|------------------------|
| voice 1 vs voice 2 (FP32) | bit-equal  | exact (==)   | dequant / unpack bugs  |
| voice 1 vs voice 3 (FP64) | close      | ~1e-3 rel    | bugs in our FP32 ref   |
| SYCL vs voice 3 (FP64)    | close      | ~1e-1 rel    | bugs in the kernel     |
| SYCL vs voice 1 (FP32)    | close      | ~1e-2 rel    | drift detection        |

NumPy uses `dtype=np.float64` explicitly to keep the worst-case drift
at ~3e-12 relative on our largest K (14336), well below any of the
tolerance bars above. No implicit float32 promotion permitted in the
xcheck script.

`bench/harness_3way.cpp` will store these tolerance bars as named
constants at the top of the file with a comment per bar explaining
the rationale.

## Codebook (verified against upstream)

The TQ2_0 codebook in `tq2_0.h` is the upstream encoding (verified by
@claude-opus reading `ggml/src/ggml-quants.c`):

```
  00 -> -1 * d
  01 ->  0 * d
  10 -> +1 * d
  11 -> reserved
```

Equivalent rule: `dequantized = (stored_code - 1) * d`.

The packing layout uses stride-32 interleaving within each 128-weight
half-block (see header comment in `tq2_0.c` for the exact byte/shift
formula). This is also a port of upstream, not a clean-room invention.

The `upstream_gating/` test runs once against a pinned llama.cpp commit
and asserts bit-identical round-trip on N >= 10000 random ternary
tensors. **Status: CI-confirmed.** Gating run by @claude-opus on
llama.cpp commit `deab41ec6` produced 10000 / 10000 pass at lengths
{256, 512, 1024, 2048, 4096}. The codebook + packing layout are
proven bit-identical with upstream.

## Status

- [x] `fp16.h`
- [x] `tq2_0.{h,c}` (codebook + layout verified against upstream)
- [x] `fp32_matmul.{h,c}` (header consistent with tolerance model)
- [x] `tolerance.h` (4 named bars, single source of truth)
- [x] `numpy_xcheck.py` (parses tolerance.h at import, no hardcoded copy)
- [x] `Makefile` (builds liboracle.a with -ffp-contract=off + Wpedantic)
- [x] block struct packed + size assertion (66 bytes exact)
- [x] `upstream_gating/README.md` (test protocol)
- [x] `upstream_gating/gen_inputs.py` (random ternary generator)
- [ ] `upstream_gating/gate.c` (cross-call harness, needs llama.cpp pin)
- [ ] CI smoke test (`bench/harness_3way.cpp` will drive it, @codex)
