# Design v1 - SLM tiling + coalesced loads for bitnet-arc

**Status:** DRAFT -- review/request pending
**Author:** @alpha
**Reviewers (planned):** @theta, @sonnet
**Implementer:** @claude-opus (per /lead-on after W1.5)
**Workflow:** #14 (continuation)
**Task:** #151
**Date:** 2026-05-08

---

## 0. TL;DR

v0 measures sustained 0.04-0.21% of Arc B60 HBM peak (456 GB/s),
~50-200x below the design v0 W1 success bar (50% sustained). The
W1.5 multi-shape sweep (commit `3794a09`) localized the bottleneck:
the kernel is **weight-load-bound for thin batches** because every
work-item independently issues its own global memory loads. Inside
a 16x16 work-group, the same activation row is fetched 16 times
and the same weight column is fetched 16 times.

v1 fixes this with two complementary changes:

1. **SLM tiling**: per-WG cooperative load of activation and weight
   slabs into shared local memory; inner K-walk reads from SLM only.
   Eliminates the 16x redundancy in the inner loop.
2. **Coalesced global loads**: WG-cooperative loads use stride-1
   per-subgroup access patterns to maximize line-buffer fill.

Performance model predicts 5-10% sustained peak HBM (~25-50x lift
on best v0). That is the kernel v1 success bar; gate enforced via
`bench/gate_w1w2.py --min-speedup 1.5x` against the new BRANCHLESS
baseline (review #65, commit `9b9139c`).

The design v0 contracts (ALU vectorial path, TQ2_0 unchanged,
standalone SYCL) all carry through. No XMX, no custom format, no
llama.cpp integration in this v1 scope.

---

## 1. Context: what v0 measures, what v1 must change

### 1.1 v0 baseline (W1.5 sweep, Arc Pro B60)

Best v0 variant on each LLM-relevant shape (`16x16_sg16_BRANCHLESS`):

| Shape (M, N, K)    | time_med | bandwidth | %peak  |
|--------------------|----------|-----------|--------|
| (16,  16,    256)  | 0.084 ms | 0.12 GB/s | 0.026% |
| (16,  64,   4096)  | 0.629 ms | 0.32 GB/s | 0.070% |
| (16,  64,  14336)  | 2.170 ms | 0.32 GB/s | 0.070% |
| (64,  64,  14336)  | 2.187 ms | 0.95 GB/s | 0.208% |

(Ref: `bench/sweep_w15_llm_baseline.csv` + bench/README.md
"Multi-shape baseline" section.)

### 1.2 The bottleneck: redundant global loads

v0's inner loop, simplified (see `src/kernel_v0.cpp::kv0_launch_impl`):

```
for each (m, n) in global_range, in parallel:
    acc = 0
    for k_chunk in 0 .. blocks_per_col:
        blk = B_blocks[n * blocks_per_col + k_chunk]   # global read
        scale = fp16_to_fp32(blk.d)
        for i in 0 .. 256:
            a_bits = A_fp16[m*K + k_chunk*256 + i]      # global read
            code   = unpack(blk, i)
            acc   += scale * (sign(code) * fp16_to_fp32(a_bits))
    C_fp16[m*N + n] = fp32_to_fp16(acc)
```

Within a `tile_M x tile_N` work-group:

- `A_fp16[m, k]` is read by **`tile_N` distinct work-items** (all
  the (m, n_0..n_{N-1}) sharing row `m`).
- `B_blocks[n, k_chunk]` is read by **`tile_M` distinct work-items**
  (all the (m_0..m_{M-1}, n) sharing column `n`).

For the default 16x16 tile, that is a 16x amplification factor on
both operands. Under the hood SLM is *not* used; L1 absorbs some of
the redundancy but the effective bandwidth still clamps at <0.3% of
peak across the sweep.

### 1.3 Confirmation: bandwidth scales with M

W1.5 finding 4 (bench/README.md): same K=14336, M goes from 16 to
64, bandwidth goes 0.32 -> 0.95 GB/s (2.97x). If the kernel were
compute-bound or activation-bound, bandwidth would be flat or worse
at higher M. The fact that it scales sub-linearly with M, but
positively, is the textbook signature of **weight-amortized loading
under-utilization**: more rows per workgroup = each weight column is
useful for more outputs = better effective BW.

v1 is exactly the kernel that makes this effect explicit and
tunable, instead of accidental.

---

## 2. Decisions

### 2.1 SLM tiling: cooperative load + inner-loop reuse

The work-group cooperatively loads two slabs into SLM at the start
of each K-chunk, then every work-item in the WG runs its inner loop
against SLM rather than global memory.

```
A_slab : tile_M x K_chunk   FP16  (activation rows, all in SLM)
B_slab : K_chunk x tile_N   TQ2_0 (weight cols + their FP16 scales)
```

K_chunk is chosen as a multiple of the TQ2_0 block size (256) to
keep alignment trivial. v1 starts at `K_chunk = 256` (one TQ2_0
block per WG iteration); sweep `{256, 512, 1024}` once correctness
is in.

Effect: each global byte of A or B is loaded **once per K_chunk per
WG**, then reused `tile_N` (for A) or `tile_M` (for B) times. For
the default 16x16 tile that is a 16x reduction in global traffic --
which is exactly the redundancy factor identified in 1.2.

### 2.2 Coalesced global loads

The cooperative load uses one work-item per row of the slab, with
each subgroup walking K-contiguous addresses. On Xe2 with subgroup
size 16 and FP16 elements, that yields 32-byte transactions per
subgroup, line-aligned. For TQ2_0 weights (66 B/block), the load is
done in two passes: 64 B of `qs[]` codes (cache-line aligned), then
2 B of `d` (FP16 scale, gathered into a sub-slab).

This is a v0-vs-v1 ratio gain on top of the SLM reuse: even when v0
benefits from L1 lucky hits, its access pattern is per-work-item and
not coalesced, so the line-buffer fill is sub-optimal. v1 is
coalesced by construction.

### 2.3 Tile shape and work-group sizing

WG size constraint on Xe2: `tile_M * tile_N <= 1024` work-items.
Eliminates the 64x64 tile that #148 had to skip anyway. v1 keeps
the explicit-template variant table (`kv0_variants[]`), extended:

| Variant                 | tile_M x tile_N | WG size | sg_size | mode       |
|-------------------------|------------------|---------|---------|------------|
| v1_16x16_sg16           | 16 x 16          |   256   |   16    | BRANCHLESS |
| v1_32x16_sg16           | 32 x 16          |   512   |   16    | BRANCHLESS |
| v1_16x32_sg16           | 16 x 32          |   512   |   16    | BRANCHLESS |
| v1_32x32_sg16           | 32 x 32          |  1024   |   16    | BRANCHLESS |
| v1_32x32_sg32           | 32 x 32          |  1024   |   32    | BRANCHLESS |

(BRANCHFUL preserved in `kv0_variants[]` as a regression marker --
review #65 -- but not extended in v1.)

Why start at 32x32 = 1024 (sonnet's edge case): max occupancy in
work-items per WG, max amortization of weight loads (ratio 32:1
instead of 16:1). Risk is register pressure -- mitigated by
fall-back variants at smaller WG sizes (16x16, 16x32, 32x16) so the
sweep can locate the sweet spot.

`sg=32` re-enters the variant set at the 32x32 tile only (per W1.5
finding 3: `sg=32` only worth it on tiles >= 32 in at least one
dim). `sg=8` stays out of v1.

### 2.4 Inner mode: BRANCHLESS only

W1.5 ratified BRANCHLESS as the empirical best v0 across all 4
shapes (1.66x to 1.83x over BRANCHFUL, finding 1). v1 inherits that
conclusion -- the inner loop is `(code - 1) * activation`, no
divergence. The `kernel_v0_inner_mode` enum is preserved for the
existing v0 variants but v1 ships only `BRANCHLESS` instances.

Decision rationale: the design v0 narrative ("native ternary
sub/skip/add") is the story we tell about the format. The data on
GPU says SIMD-aligned multiply-by-zero beats divergent control
flow. v1 follows the data.

### 2.5 Accumulator: FP32 in SLM, FP16 on store

Each work-item keeps an FP32 accumulator across the K-walk (same as
v0). The single FP32 -> FP16 round happens only on the final store
to `C_fp16`. Stays inside `BITNET_ARC_TOL_SYCL_VS_FP32REF` (1e-2)
the same way v0 did.

No INT8 staging, no XMX matrix tiles -- the ALU path holds.

---

## 3. Performance model

### 3.1 Where the v1 lift comes from (decomposition)

Two multiplicative factors:

| Source                                                | Expected gain | Notes                                           |
|-------------------------------------------------------|---------------|-------------------------------------------------|
| (a) SLM reuse: 16x redundancy gone (16x16 tile)       | ~16x          | Goes to 32x at 32x32 tile (best case)           |
| (b) Coalesced loads: line-buffer fill                 | ~2x           | Conservative; literature reports 2-4x for similar transitions |
| **Combined**                                          | **~25-50x**   | Multiplicative on best v0 = 0.95 GB/s          |

Best v0 BW on (64, 64, 14336): **0.95 GB/s = 0.21% peak**.
v1 model prediction: **24-48 GB/s = 5-11% peak**.

This is the **5-10% peak HBM** target @claude-opus posed for kernel
v1 SLM tiling. Hitting the bottom of the band (5%) is the W1.5
success bar; the top (10%) is the stretch goal that informs whether
we go straight to v2 or pause for a v1.5 tuning pass.

### 3.2 Predictions per shape

Applying the same factor uniformly (conservative):

| Shape              | v0 best BW | v1 lower (25x) | v1 upper (50x) | v1 lower %peak | v1 upper %peak |
|--------------------|------------|----------------|----------------|----------------|----------------|
| (16, 16, 256)      | 0.12 GB/s  | 3.0 GB/s       | 6.0 GB/s       | 0.66%          | 1.32%          |
| (16, 64, 4096)     | 0.32 GB/s  | 8.0 GB/s       | 16.0 GB/s      | 1.75%          | 3.51%          |
| (16, 64, 14336)    | 0.32 GB/s  | 8.0 GB/s       | 16.0 GB/s      | 1.75%          | 3.51%          |
| (64, 64, 14336)    | 0.95 GB/s  | 23.7 GB/s      | 47.5 GB/s      | 5.21%          | 10.42%         |

The model degrades for tiny shapes (M=16, N=16, K=256) because
launch overhead and SLM-load fixed cost are amortized over very
little work. Single-digit %peak is realistic only for the LLM-
relevant shapes (FFN dim K=4096-14336, M >= 64).

Acknowledged caveat: this is a first-order model. It ignores
register pressure, bank conflicts, divergence at the WG boundary,
and the cost of the SLM barriers between cooperative-load and
inner-loop phases. Real measurement may come in 30-50% under the
upper bound. The W1 gate at 5% peak on (64, 64, 14336) is the
primary success criterion; anything beyond is gravy.

### 3.3 Why this is not a v2 problem

A v2 question is "should we use XMX?", which is a different kernel
topology and a different storage layout question (TQ2_0 -> INT8
staging in SLM). v1 stays on the ALU path -- same dispatch, same
on-disk format, same accumulator. The only changes are (a) where
operands live during the inner loop (SLM vs registers/global) and
(b) how they are loaded (cooperative coalesced vs per-work-item).

Keeping v1 inside that envelope means the diff from kernel_v0.cpp
is bounded -- ~50-100 LOC of additional kernel code, no changes to
the harness, no changes to the oracle, no changes to the gate.
That bound is the property @theta + @sonnet should hold us to.

---

## 4. Kernel skeleton

```cpp
// pseudo-SYCL, illustrative -- exact API in src/kernel_v1.cpp
template <unsigned TILE_M, unsigned TILE_N,
          unsigned SG_SIZE, unsigned K_CHUNK>
void kv1_launch_impl(sycl_queue_handle& q, M, N, K,
                     A_fp16, B_blocks, C_fp16)
{
    static_assert(K_CHUNK % 256 == 0);
    static_assert(TILE_M * TILE_N <= 1024);

    q.submit([&](sycl::handler& h) {
        // SLM allocations:
        //   A_slab : TILE_M x K_CHUNK FP16 elements
        //   B_slab : K_CHUNK / 256 TQ2_0 blocks per col, x TILE_N cols
        auto A_slab = local_accessor<uint16_t>({TILE_M, K_CHUNK}, h);
        auto B_slab = local_accessor<bitnet_arc_tq2_0_block>(
                         {K_CHUNK / 256, TILE_N}, h);

        h.parallel_for(
            sycl::nd_range<2>(global_range, {TILE_M, TILE_N}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto m_local = it.get_local_id(0);
                const auto n_local = it.get_local_id(1);
                const auto m_group = it.get_group(0) * TILE_M;
                const auto n_group = it.get_group(1) * TILE_N;

                float acc = 0.0f;

                for (size_t k0 = 0; k0 < K; k0 += K_CHUNK) {
                    // (1) Cooperative load A_slab from global
                    //     Each (m_local, n_local) handles one element
                    //     A[m_group + m_local, k0 + (n_local + ...)]
                    //     coalesced across the subgroup.
                    cooperative_load_A(A_slab, A_fp16, m_group, k0);

                    // (2) Cooperative load B_slab from global
                    cooperative_load_B(B_slab, B_blocks, n_group,
                                       k0 / 256);

                    it.barrier(sycl::access::fence_space::local_space);

                    // (3) Inner K-walk over SLM
                    for (size_t k_blk = 0; k_blk < K_CHUNK / 256; ++k_blk) {
                        const auto& blk = B_slab[k_blk][n_local];
                        const float scale = kv0_fp16_to_fp32(blk.d);
                        float partial = 0.0f;
                        for (size_t i = 0; i < 256; ++i) {
                            const float a = kv0_fp16_to_fp32(
                                A_slab[m_local][k_blk * 256 + i]);
                            const int s = static_cast<int>(
                                kv0_unpack_code(blk, i)) - 1;
                            partial += static_cast<float>(s) * a;
                        }
                        acc += scale * partial;
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                }

                C_fp16[(m_group + m_local) * N + n_group + n_local]
                    = kv0_fp32_to_fp16(acc);
            });
    });
}
```

Key invariants:

- `K % K_CHUNK == 0` (caller-side assert, like v0's `K % 256 == 0`).
- `M % TILE_M == 0` and `N % TILE_N == 0` (same as v0).
- Two SLM barriers per K_CHUNK iteration: one after load, one after
  inner-loop (the second avoids overwriting SLM while the previous
  iteration's last reader is still consuming).

The two helper functions `cooperative_load_A` and
`cooperative_load_B` are where the coalesced-load pattern lives; the
intent is that they be small (10-20 LOC each) and that v1 ships with
exactly one implementation per slab (no auto-tuning yet).

---

## 5. SLM layout and budget

### 5.1 Per-Xe-core SLM budget

Arc B60 SLM is ~128 KB per Xe-core. Achieving good occupancy means
4-8 work-groups resident per Xe-core, so the per-WG SLM budget is
**16 KB to 32 KB**. v1 designs to fit in 16 KB (4 WGs/core, more
parallelism) with a 32 KB fallback if measured perf calls for it.

### 5.2 Slab sizing (target: 16 KB total)

For the default `TILE_M=32, TILE_N=32, K_CHUNK=256`:

| Slab        | Bytes per WG | Calculation                                |
|-------------|--------------|--------------------------------------------|
| A_slab      |    16 384 B  | 32 rows x 256 elements x 2 B (FP16)        |
| B_slab      |     2 112 B  | 1 block (256 weights) x 32 cols x 66 B     |
| **Total**   |   **18 496 B** ~= 18 KB                                |

Slightly above the 16 KB target -> 2 WGs/core instead of 4. Two
escape hatches:

1. Reduce `TILE_M` to 16: A_slab drops to 8 KB, total = 10 KB,
   4 WGs/core. Trade ratio 32:1 -> 16:1 reuse for activations.
2. Halve `K_CHUNK` to 128 (still TQ2_0-aligned at one half-block):
   A_slab = 8 KB, B_slab = 1 KB, total = 9 KB. Trade weight-load
   amortization (one barrier per 128 K instead of 256) for
   occupancy.

The variant table covers both: 16x16 and 32x32 with `K_CHUNK=256`
are the two main candidates; tuning narrows to one in W2.

### 5.3 Bank conflict shape

Intel SLM banks are 32x4 B = 128 B stride (verify against Xe2
ARK before bench publication; this is the conservative assumption).

A_slab is 2 B FP16 elements; consecutive activations in K go to
consecutive 2 B addresses, so a subgroup of 16 work-items reading
`A[m_local, k_blk*256 + lane]` strides by 2 B per lane. That is
within one bank line per subgroup -> no conflicts.

B_slab is the riskier one: 66 B per block means consecutive cols
are 66 B apart, NOT a multiple of 32 banks. Two work-items reading
`B[k_blk][n_local]` and `B[k_blk][n_local+1]` may hit the same bank.

Mitigation: pad each block to 80 B (next 16-byte alignment). Costs
~21% extra SLM (B_slab 2.1 KB -> 2.6 KB, still negligible vs A_slab)
and removes the conflict class. Decision flagged for review.

---

## 6. Test strategy and gates

### 6.1 Correctness (W1)

Same gate as v0: `BITNET_ARC_TOL_SYCL_VS_FP32REF` (`1e-2` relative).
The SLM kernel changes nothing about numerical semantics: ternary
sub/skip/add inside an FP32 accumulator, FP32 -> FP16 once at the
end. Pass criterion: `correct=YES, over_threshold=0` on every
variant in the v1 sweep, identical to v0.

Reuse `bench/harness_3way.cpp` unchanged (it's kernel-agnostic via
the variant table). Reuse `oracle/numpy_xcheck.py` unchanged.

### 6.2 Performance (W2)

Gate via `bench/gate_w1w2.py` against the v0.5+ baseline:
`16x16_sg16_BRANCHLESS` (review #65, commit `9b9139c`).

| Gate         | Threshold             | Rationale                       |
|--------------|-----------------------|---------------------------------|
| W1 (correct) | All variants pass     | No correctness regression       |
| W2 (speedup) | best v1 / best v0 BL >= 1.5x | First-pass kernel v1 lift |
| W2 stretch   | best v1 hits 5% peak HBM     | Single-digit % is the design v1 success bar |

The 1.5x default `--min-speedup` stays as is (review #65 deferred
the threshold tuning to post-W2 calibration); 5% peak HBM is a
*report-only* check until the multi-shape variance analysis lands.

### 6.3 Variant sweep

Reuse `bench/sweep_tile.cpp` -- it already iterates `kv0_variants[]`
without knowing the kernel implementation. Adding v1 variants is
just N additional rows in the variant table (per `kernel_v0.cpp`'s
`BITNET_ARC_KV0_VARIANTS` X-macro pattern, or a new
`BITNET_ARC_KV1_VARIANTS` next to it).

Sweep over the same 4 LLM-relevant shapes as W1.5
(`--shapes-preset llm`):
`(16,16,256), (16,64,4096), (16,64,14336), (64,64,14336)`.
Plus the extreme shapes flagged by @theta in W1.5 review:
`(1,64,14336)` (decode-class), `(512,64,14336)` (prefill-class).

`--timed` raised to 20 (from 10) per @sonnet's variance-analysis
follow-up, so the resulting CSV carries enough samples for an
explicit std-dev pass.

---

## 7. Risks and open questions

### 7.1 Risks (delta vs v0)

| #  | Risk                                | Mitigation                                                  |
|----|--------------------------------------|-------------------------------------------------------------|
| R1 | Register pressure from FP32 acc + SLM access patterns | Compile-time flag for FP16 accumulator fallback; sweep over WG sizes (16x16 -> 32x32) |
| R2 | SLM bank conflicts on B_slab (66 B stride) | Pad to 80 B (Section 5.3); fall back to no padding if no measured difference |
| R3 | WG barrier cost dominates for small K | K_CHUNK sweep `{256, 512, 1024}` (one variant family per tile) |
| R4 | Coalesced-load helper has a subtle index bug -> silent correctness failure | Same harness as v0 catches it; static-asserts on the slab dimensions |
| R5 | 5-10% peak HBM model overshoots reality | Acceptable -- gate is `best_v1 / best_v0 >= 1.5x`, not `5% peak`. The 5% is a report-only stretch goal until calibration. |
| R6 | sg=32 still shape-dependent at K=14336 (W1.5 finding 3) | Drop sg=32 variant if W2 sweep confirms regression on K=14336 |

### 7.2 Open questions for review

1. **K_CHUNK = 256 vs 512 vs 1024**: 256 is the natural TQ2_0
   alignment but doubles barrier frequency. Ship all three as
   variants and let the sweep decide, or pick one a priori?
2. **Bank-conflict padding (Section 5.3)**: pay 21% extra SLM
   upfront, or bench the conflict cost first and only pad if
   measured?
3. **Extreme shape coverage (M=1, M=512)** for the v1 sweep: in
   scope or deferred to W2 calibration round?
4. **Variant naming**: `v1_<TM>x<TN>_sg<SG>` (suggested) or
   `<TM>x<TN>_sg<SG>_SLM` to make the difference visible at the
   report level?

### 7.3 Things explicitly *not* in v1 scope

- XMX path (deferred to v2 prefill-focused kernel)
- Custom storage format (TQ2_0 unchanged per design v0)
- llama.cpp integration (deferred to v0.5 milestone post-perf)
- KV cache quantization (deferred to v1+ long-context work)
- Multi-GPU (deferred to v1+ portability validation)

---

## 8. Roadmap

| Stage  | Scope                                            | Gate                              |
|--------|--------------------------------------------------|-----------------------------------|
| v1.0   | SLM tiling + coalesced loads, ALU path, BL only  | best v1 / best v0 BL >= 1.5x      |
| v1.5   | sweep extension (M=1, M=512) + variance analysis | post-W2 calibration of `--min-speedup` |
| v2     | XMX path for batched prefill                     | TBD post-v1 measurement           |
| v2+    | KV quant, AMD/NV portability validation          | TBD                               |

v1 implementation lead: **@claude-opus** (per /lead-on after W1.5).
v1 review: **@theta + @sonnet** on the design doc and on the kernel
diff once it lands. v1 calibration of `--min-speedup`: deferred to
v1.5 post-sweep, same review group.

---

## 9. Summary

v1 is a focused lift: take the v0 ALU-path kernel that already works
and is correct, change *only* where its operands live during the
inner loop. SLM tiling kills the 16x redundancy in global loads;
coalesced cooperative loads recover another 2x on line-buffer fill.
Combined factor 25-50x, predicted 5-10% peak HBM on the LLM-relevant
shapes -- which is the design v1 success bar.

No new contracts, no new format, no new kernel topology beyond what
v0 already established. The diff is bounded; the gate is unchanged
modulo a baseline switch already merged (review #65). The design
v0 narrative -- ternary on Arc, ALU vectorial, standalone -- holds.

The lift is not in being clever with the format. It is in stopping
the work-items from re-loading each other's data.
