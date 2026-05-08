# upstream_gating/ -- one-shot TQ2_0 round-trip vs llama.cpp

Validates that our clean-room TQ2_0 codebook (`oracle/tq2_0.c`) is
bit-identical to the upstream llama.cpp implementation.

## Status: PASS

Last gating run (by @claude-opus): **10000 / 10000 pass, 0 fail**
against llama.cpp commit `deab41ec6`, on tensor lengths
{256, 512, 1024, 2048, 4096}. Quantized bytes byte-identical and
dequantized FP32 byte-identical across all variants.

The codebook + packing layout in `oracle/tq2_0.{h,c}` are therefore
proven bit-identical with upstream. The PLACEHOLDER caveat in
`oracle/tq2_0.h` was lifted in the same commit cycle that landed the
gating pass.

## Test protocol

1. Generate N tensors of valid ternary input (each entry in
   `{-1, 0, +1}`), block-aligned (sizes 256, 512, 1024, 2048, 4096).
2. Quantize each via our `bitnet_arc_quantize_row_tq2_0()` ->
   sequence of `bitnet_arc_tq2_0_block` records.
3. Pass the same raw input to upstream `quantize_row_tq2_0()`.
4. Compare the two byte streams: must be byte-equal.
5. Dequantize both via the respective implementations.
6. Compare the two FP32 outputs: must be bit-equal.

Subtlety on auto-scaling: upstream derives the FP16 scale `d` from
`amax` of the block, while our `bitnet_arc_quantize_row_tq2_0()`
takes the scale as a parameter. `gate.c` precomputes amax-style
scales in a pre-pass and passes them to our function. This validates
the **format** (codebook + packing + FP16 scale encoding) without
coupling to the auto-scaling **policy** (which is not part of the
on-disk format spec).

## What divergence patterns would tell us

If a future re-run regresses, failure mode -> likely fix in
`oracle/tq2_0.c`:

- All values zero in our dequant but non-zero upstream
  -> `k_code_to_ternary[]` LUT is permuted; map upstream codes to ours.
- Signs flipped uniformly
  -> `+1` and `-1` codebook entries swapped (1 <-> 2 in our LUT).
- Magnitudes off by a power of two
  -> FP16 scale endianness or position in the block layout is wrong.
- Off-by-one between blocks (drift increases linearly with block index)
  -> block stride or `qs[64] + d` ordering swapped.

## Why one-shot

The TQ2_0 on-disk format is part of the upstream llama.cpp ABI. A
breaking change there would show up in their commit log and trigger
their own version bump. So we run the gate exactly once per pinned
llama.cpp tag, store the pass/fail result in CI, and trust the
codebook afterward. The gate is not a hot-loop check.

## Files

| File              | Purpose                                          |
|-------------------|--------------------------------------------------|
| `gen_inputs.py`   | Random ternary tensor generator (N tensors)      |
| `gate.c`          | Cross-call harness (our + upstream functions)    |
| `upstream_ref.h`  | Vendored upstream code (MIT, commit deab41ec6)   |

Build is wired in `oracle/Makefile` via the `gate` target.

## Status checklist

- [x] Test protocol documented
- [x] `gen_inputs.py` (random ternary generator)
- [x] `gate.c` (cross-call harness, by @claude-opus)
- [x] `upstream_ref.h` (vendored, MIT, with #error guard)
- [x] `oracle/Makefile` `gate` target
- [x] **Gating pass 10000/10000 on `deab41ec6`**
- [ ] CI integration (re-run on llama.cpp pin update)
