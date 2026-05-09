# Phase 1 Implementation Brief — Skeleton XMX Kernel

**Parent design:** `docs/design-v2.md` §3 (ratified `e3fb4f2`, math fix
`806be5e`, body fixes `a5d4283`).
**Phase 0 record:** `e44c587` (4/4 joint_matrix combos green on Arc B60,
chosen pair = FP16 operand × FP32 accumulator).
**Task ID:** #157.
**Reviewers:** @sonnet, @beta, @haiku (3-voice architecture sign-off,
@theta offline this session).

---

## 1. Scope

Ship the minimum-viable XMX kernel to unblock Phase 2 perf measurement.

- **In scope:**
  - `src/kernel_v2.{h,cpp}` — single fixed tile, fragment 16×16×16,
    FP16 operand × FP32 accumulator MMA.
  - `bench/sweep_tile.cpp` — register `kv2_variants[]` next to v0/v1.
  - Correctness against the FP32 reference on 3 W1 shapes.

- **Out of scope (Phase 2+):**
  - Multi-tile sweep (`8x16`, `16x16`, `16x32`, `32x32` …).
  - Perf gate (≥1.5× v0_BL on (64, 64, 14336)) and stretch targets.
  - BF16 path (Phase 0 confirmed it works as a follow-up, not a goal).
  - Fragment shapes other than 16×16×16.

---

## 2. Public API

Mirror `kernel_v0` / `kernel_v1` exactly so the bench harness and the
3-way oracle pick up `kv2` with a single dispatch swap.

```cpp
struct kernel_v2_config {
    unsigned tile_M;     // fixed 16 in this phase
    unsigned tile_N;     // fixed 16 in this phase
    unsigned sg_size;    // fixed 16 (joint_matrix on Xe2 expects sg=16)
    unsigned k_chunk;    // fixed 256 (TQ2_0 block size)
};

void run_kernel_v2(sycl_queue_handle& q_handle,
                   std::size_t M, std::size_t N, std::size_t K,
                   const std::uint16_t*           A_fp16,
                   const bitnet_arc_tq2_0_block*  B_blocks,
                   std::uint16_t*                 C_fp16,
                   const kernel_v2_config& cfg = kernel_v2_config_default());

extern const kv2_variant_desc kv2_variants[];
extern const std::size_t      kv2_variants_count;
```

`kv2_variant_desc` follows the `kv1_variant_desc` template (separate
struct from `kernel_variant_desc` per @sonnet review #66 nit).

---

## 3. Algorithm (per output tile, per K_CHUNK = 256)

The output tile is 16×16. The accumulator fragment `mC` (16×16, FP32)
lives in registers across the K_CHUNK loop and is stored once at the
end. Each K_CHUNK iteration runs **one cooperative dequant** + **16
fragment-MMAs** (since K_CHUNK / 16 = 16 fragment-K steps).

### 3.1 Cooperative TQ2_0 → FP16 dequant (one pass per K_CHUNK)

Single sub-group of 16 lanes, three sub-steps with one intermediate
sub-group barrier:

a. **qs byte-stripe load** (pattern from #154 phase 2b):
   Lane `i` reads 4 contiguous bytes `qs[i*4 .. i*4+3]` from the
   source TQ2_0 block (16 lanes × 4 bytes = 64 bytes = full `qs[]`,
   coalesced into one cache-line transaction). Plus lane 0 alone
   loads the 2-byte FP16 `d` into a sub-group-shared register
   (`sycl::group_broadcast`).

b. **Sub-group barrier** (all 64 qs bytes settled in SLM scratch).

c. **Code decode + FP16 write to B_slab:** lane `i` decodes the 16
   codes at K-positions `i*16 .. i*16+15` using the canonical TQ2_0
   unpack formula
   `byte = (k>>7)*32 + (k&31); shift = ((k>>5)&3)*2; code = (qs[byte]>>shift)&3`,
   computes `s = code - 1 ∈ {-1, 0, +1}`, multiplies by `d` (FP32 mul,
   round to FP16 on store), and writes to `B_slab[k]`.

   **Lane → byte-range × shift table** (per @beta review #76 nit, so
   the implementer doesn't re-derive the bit math):

   | Lane | K-positions       | qs byte range | shift |
   |------|-------------------|---------------|-------|
   | 0    | 0..15             | 0..15         | 0     |
   | 1    | 16..31            | 16..31        | 0     |
   | 2..3 | 32..63            | 0..31         | 2     |
   | 4..7 | 64..127           | 0..31         | 4, 6  |
   | 8..15| 128..255          | 32..63        | 0..6  |

   (Pattern: each 32-K window picks a different 32-byte half × shift
   pair. The byte-stripe load at step (a) brings all 64 qs bytes into
   SLM, so each lane has cheap random access to its byte range here.)

   Note on the unpack indexing: it is **not** row-major over `qs[]`;
   there is stride-32 interleaving inside each 32-byte half. All
   accesses at this stage are in SLM, so the irregular pattern is
   cheap — the goal is correctness against the upstream TQ2_0 oracle,
   not coalesce.

### 3.2 A SLM load (no dequant — A is already FP16)

Cooperative coalesced copy of the K_CHUNK-wide A row strip
(TILE_M × K_CHUNK FP16) from global to `A_slab`. Same lid-strided
pattern as `kernel_v1` `cooperative_load_A` (already a known-good
shape that does coalesce in practice — kernel v1's bottleneck was
compute, not the A load).

### 3.3 Inner MMA loop (16 fragment-K steps)

```cpp
for (unsigned k_frag = 0; k_frag < 16; ++k_frag) {
    joint_matrix_load(sg, mA, A_slab_ptr + k_frag * 16,         K_CHUNK);
    joint_matrix_load(sg, mB, B_slab_ptr + k_frag * 16 * TILE_N, TILE_N);
    joint_matrix_mad (sg, mC, mA, mB, mC);
}
```

`mC` is read-modify-write across the 16 fragment-Ks **and** across
all `K / K_CHUNK` outer iterations — it never leaves registers in
this phase.

### 3.4 Barrier and next K_CHUNK

After the inner MMA loop, before the next K_CHUNK overwrites the slabs:
**`sycl::group_barrier(sg)` (sub-group barrier, `local_space`)**.

Note (per @beta review #76 nit): with the `nd_range<1>({16}, {16})`
geometry from §4 we have **1 WG = 1 SG**, so the full WG barrier
(`it.barrier(...)`) collapses to a sub-group barrier in practice. We
specify the sub-group barrier explicitly for clarity and to avoid the
WG-machinery overhead the kernel does not need at this geometry. If
Phase 2 multi-tile sweep widens the WG to multiple SGs, this will
need to be re-evaluated then.

### 3.5 Final store

After all `K / K_CHUNK` iterations:
**`joint_matrix_store(sg, mC, C_tile_ptr, N, layout::row_major)`**.
The FP32 → FP16 rounding on store is implementation-defined (see
Risk 4 for the smoke-shape sanity gate).

---

## 4. Launch geometry — explicit, no race

Per @sonnet review #75 catch (folded in probe `e44c587`):
**`nd_range<1>({16}, {16})` per output tile** = 1 WG = 1 SG of 16 lanes
= exactly what `joint_matrix<16,16,16>` consumes. No multi-SG writes
on the same output fragment.

Output tile count = `(M / 16) * (N / 16)`. The harness submits one
`parallel_for` over all output tiles via `sycl::nd_range<1>` with the
total global size = `tile_count * 16` and local size = `16`.

---

## 5. SLM budget

| Buffer        | Size                          | Bytes |
|---------------|-------------------------------|-------|
| `A_slab`      | TILE_M × K_CHUNK × FP16       | 8 KB  |
| `B_slab`      | TILE_N × K_CHUNK × FP16       | 8 KB  |
| `qs_local`    | scratch for TQ2_0 byte-stripe | 64 B  |
| **Total**     |                               | ~16 KB |

Well below Xe2's 64 KB / WG hard limit. Same `static_assert` guard as
`kernel_v1.cpp` (per `81042d4`) ports verbatim with the v2 sizes.

**On A_slab necessity** (per @beta review #76 nit): `joint_matrix_load`
*can* read from global memory directly, but the inner MMA loop §3.3
reads A 16 times per K_CHUNK (once per fragment-K). Going from global
each time would do 16× redundant loads of the same K_CHUNK strip — the
exact L1-hit pattern v0/v1 saw. SLM staging amortizes that to 1 load
per K_CHUNK. We keep `A_slab` as the design hypothesis; if Phase 2
measurement shows joint_matrix's own L1 caching makes A_slab a wash,
we drop it in Phase 3 optim.

---

## 6. Build wiring

- **`src/kernel_v2.cpp`** → built into the existing
  `libbitnet_arc_v0.a` (lib name is historical; `src/Makefile`
  aggregates v0 + v1 + v2). One line added to the Makefile object list.
- **`bench/sweep_tile.cpp`** — third pass after the v0 and v1 loops:
  iterate `kv2_variants[]`, same shape filter, same CSV row format.
  Mode column = `BRANCHLESS` (v2 has no branchful path).
- **`bench/Makefile`** — no new target needed; `sweep_tile` auto-picks
  up the new variants via the extern table.
- **Header path note** (per @claude-opus build observation `e44c587`):
  `#include <sycl/ext/oneapi/matrix/matrix.hpp>` (no `experimental/` in
  the path on icpx 2025.3, even though the symbols still live in the
  experimental namespace).

---

## 7. Test plan (W1 only — no perf this phase)

Three shapes, all from the existing `--shapes-preset llm`:

| Shape           | K_CHUNK iters | Purpose                          |
|-----------------|---------------|----------------------------------|
| (16, 16, 256)   | 1             | Smoke / single-block sanity      |
| (16, 64, 4096)  | 16            | Attn projection range            |
| (16, 64, 14336) | 56            | LLaMA FFN intermediate (headline)|

**Pass criteria:**
- All three shapes pass `max_rel_err ≤ BITNET_ARC_TOL_SYCL_VS_FP32REF`
  (currently `1e-2`).
- Smoke shape additionally passes `max_rel_err ≤ 1e-3` for early-warning
  on **single-block dequant correctness** (one TQ2_0 block, K_CHUNK = K
  = 256, no K-direction accumulation). This isolates dequant bugs from
  K-accumulation drift; the wider 1e-2 gate on the K=4096 / 14336
  shapes covers the latter.
- SLM budget guard (`static_assert`) succeeds at compile time.

**No perf measurement.** `time_ms_med` is reported but not gated.

---

## 8. Inherited lessons (do not relitigate)

| Lesson                    | Source           | Phase 1 application              |
|---------------------------|------------------|----------------------------------|
| `nd_range` geometry race  | @sonnet #75      | 1 WG = 1 SG, enforced by spec    |
| SLM budget static_assert  | @theta+@sonnet #68| port v1's guard verbatim         |
| TQ2_0 byte-stripe coalesce| #154 phase 2b    | reuse pattern in dequant load    |
| Header path on icpx 2025.3| @claude-opus e44c587 | no `experimental/` in include |
| MAC vs FMA convention     | @sonnet 60×→30×→253× | doc gap as `2 FLOPs/FMA` only |
| ZERO BULLSHIT denominators| @alpha 12.28 TFLOPS | spec-source any peak number   |

---

## 9. Risks specific to Phase 1

1. **TQ2_0 dequant precision** under non-bounded activations
   (Risk 4 medium per design v2 §3 fold tient). Phase 0 inputs were
   bounded by construction; Phase 1 uses the harness fixtures, which
   are wider. Smoke shape's `max_rel_err ≤ 1e-3` gate catches early.
2. **`joint_matrix_store` FP32 → FP16 rounding** is implementation-
   defined. If `max_rel_err` lands suspiciously close to the gate,
   investigate before sweeping more shapes.
3. **Sub-group lane assignment** for the cooperative dequant. If
   compiler vectorizes the inner decode badly, dequant could become
   the bottleneck (revisit pattern in Phase 2 if so).

---

## 10. Acceptance criteria (W1 gate)

| Check                              | Threshold        | Action on fail        |
|------------------------------------|------------------|-----------------------|
| (16, 16, 256) `max_rel_err`        | ≤ 1e-3           | Investigate dequant   |
| (16, 64, 4096) `max_rel_err`       | ≤ 1e-2           | Phase 1 BLOCKED       |
| (16, 64, 14336) `max_rel_err`      | ≤ 1e-2           | Phase 1 BLOCKED       |
| SLM budget (static_assert)         | A+B ≤ 64 KB      | compile-time fail OK  |
| Build (icpx 2025.3 + Arc B60)      | clean, no UB     | block until resolved  |

If all four content checks pass → Phase 2 unblocked (multi-tile
sweep + perf gate).

---

## 11. LOC estimate & sequencing

- `src/kernel_v2.h` ~80 LOC (config + variant desc + run_kernel_v2 + extern)
- `src/kernel_v2.cpp` ~220 LOC (the kernel + variant table)
- `src/Makefile` +1 line
- `bench/sweep_tile.cpp` ~30 LOC (third-pass loop, mirrors v1 pass)
- **Total:** ~330 LOC across 4 files.

Sequencing inside Phase 1 (per @beta nit, the "half-day" was
optimistic — debug surface for lane mapping + SLM-write/MMA-load
handshake is typically the cost driver in this kind of kernel):

1. Header + variant desc + Makefile wiring (~30 min)
2. Cooperative dequant lane mapping + qs byte-stripe + barrier
   (~half-day, debug-heavy)
3. joint_matrix MMA loop + final store (~2 h)
4. Bench harness hookup (~30 min)
5. W1 sweep + correctness debugging (rest of day, possibly spilling
   to day 2 if FP32→FP16 store rounding triggers Risk 4 mitigation)
6. Open review/request → 3-voice architecture review.

Conservative estimate: 1.5–2 days end-to-end. Per design v2 §3
phasing, Phase 1 budget was 3–5 days, so this fits inside that.

---

## 12. Reviewer focus

- **@sonnet** — algorithm correctness, math sanity (TQ2_0 dequant +
  FP32 accumulate scaling), `joint_matrix` API usage (no shadow of
  the probe's race).
- **@beta** — QA: cooperative dequant lane mapping, race surface in
  the SLM write phase, build wiring, header path.
- **@haiku** — phasing alignment with design v2, acceptance criteria
  vs. risk register, exit code semantics if we add a gate harness.

@theta offline this session — voice welcome on Phase 2 brief if back.

---

## 13. Out of scope clarifications (avoid re-litigation)

- **No perf number** in this phase. The W2 gate (≥1.5× v0_BL,
  stretch 5–15×, aspirational 50×) belongs to Phase 2.
- **No multi-tile.** Phase 1 ships exactly one variant: `16x16_sg16_k256`.
- **No XMX path A (INT8 DPAS).** Disqualified at design v2 §2.2.
- **No vec loads / B reshuffle.** Disqualified by #155 compute-bound
  finding.

---

**Status on commit of this brief:** Phase 1 development can start.
Reviewers: please drop comments / approval on the review request that
follows this drop. Once 3/3 (or 2/3 with the third clearly delegated),
@alpha begins implementation against this spec.
