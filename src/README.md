# src/ -- bitnet-arc v0 SYCL kernel

Production ternary x FP16 -> FP16 matmul kernel for Intel Arc B60
(Xe2 / Battlemage). Matches the design v0 contract (see
`/projects/bitnet-arc/docs/design-v0.md`).

## Files

| File              | Role                                            |
|-------------------|-------------------------------------------------|
| `kernel_v0.h`     | Public API + config struct + queue handle decl  |
| `kernel_v0.cpp`   | SYCL implementation                             |
| `Makefile`        | Builds `libbitnet_arc_v0.a` via DPC++           |

## v0 design choices

- **Dispatch**: 2D `nd_range` with `global = (M, N)`, `local = (16, 16)`.
  One work-item per output element. No SLM tiling in v0 -- that lands
  in #148 (tile sweep) and gets us closer to the bandwidth ceiling.
- **Subgroup size**: v0 uses the SYCL implementation default. The
  config does NOT carry a runtime `subgroup_size` field, since SYCL
  enforces this via a compile-time `[[reqd_sub_group_size(N)]]`
  attribute (per @codex review #60). #148 will introduce templated
  kernel variants for 8 / 16 / 32 instead of pretending the runtime
  value is honored.
- **Inner loop**: ternary {-1, 0, +1} expressed natively as
  sub / skip / add (no multiplies). Branchful form by default,
  branchless form available behind `kernel_v0_config::inner_mode`
  for #148 to compare.
- **Accumulator**: FP32 per work-item. Output rounded to FP16 once at
  the end. Per-block scale applied at block boundary, not per weight.
- **Weights layout**: `(N, K/256)` blocks of `bitnet_arc_tq2_0_block`,
  K-contiguous within a block, N-contiguous block-by-block. Lets the
  hot loop walk K in stride-256 chunks with one FP16 scale per chunk.
- **No XMX in v0**: design decision per workflow #14 (memory-bound,
  multiplies are not the bottleneck). XMX returns in v1 for batched
  prefill.

## What v0 is NOT yet

- Not tile-tiled (one work-item per output element, not one per tile)
- Not subgroup-coalesced (each item reads its own activation slice)
- Not autotuned (single 16x16 work-group baseline)
- Not AOT-compiled to Arc B60 (JIT via spir64 default)

All four are explicit follow-ups, gated on review #59 oracle merge
(done) and harness wiring (#146, pending @codex).

## Build

```bash
# In src/, after oneAPI / DPC++ is on PATH:
make                        # icpx default (oneAPI)
CXX=clang++ make           # LLVM clang with SYCL plugin
```

The kernel TU does NOT enforce `-ffp-contract=off` (unlike the oracle).
The kernel is allowed to fuse / reorder for perf; correctness lives in
the oracle harness via `TOL_SYCL_VS_NUMPY_F64` (1e-1 relative).

## Status

- [x] `kernel_v0.{h,cpp}` skeleton (compile + correct shape)
- [x] `Makefile` (DPC++ wiring, override `CXX` for clang)
- [ ] First compile clean on icpx (needs DPC++ on PATH, manual)
- [ ] Wire into `bench/harness_3way.cpp` (@codex)
- [ ] Tile sweep 8x8 -> 64x64 (#148)
- [ ] Stop-gates instrumentation (#147)
