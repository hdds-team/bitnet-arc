# Design v0 - SYCL ternary BitNet kernel for Arc B60

**Status:** MERGED -- review #57 approved 2/2 (@sonnet + @theta)
**Author:** @alpha
**Reviewers:** @claude-opus, @sonnet, @theta, @haiku
**Workflow:** #14
**Task:** #144
**Date merged:** 2026-05-08

---

## 1. Context & Motivation

BitNet (Ma et al. 2024, arXiv:2402.17764) ships ternary {-1, 0, +1} weights
for transformer LLMs at ~2 bits per weight, with reported parity vs FP16 at
the 8B class. The deployment story today is asymmetric: the official path
(bitnet.cpp) is CPU-only, and llama.cpp upstream has no GPU kernel for
TQ1_0 or TQ2_0 -- neither in the Vulkan backend nor in the SYCL backend.

On our reference rig (bi-Xeon E5-2699 v4 + Arc B60 + RTX 3070 Ti) we measure
BitNet 8B I2_S at 10.5 t/s decode on CPU, while Llama 8B Q4_K_M on the same
Arc B60 GPU runs at 33.4 t/s -- a 3x lift just from going to GPU on a
non-ternary format. Ternary on GPU should beat that by another 2-3x given
the read-bandwidth ratio (TQ2_0 = 2.0625 bpw vs Q4_K_M ~4.5 bpw).

This project closes the gap with a native SYCL kernel for Arc B60
(Xe2 / Battlemage), portable via Codeplay oneAPI plugins to AMD and NVIDIA.
The win condition is not merely "BitNet on GPU works" but "BitNet on GPU
beats Q4_K_M on the same hardware" -- the public-facing argument that
ternary is worth the engineering on consumer-class accelerators.

Baselines (bi-Xeon E5-2699 v4 + Arc B60 + RTX 3070 Ti):

| Workload                       | pp    | tg     |
|--------------------------------|-------|--------|
| BitNet 8B I2_S CPU 22t numactl | 109   | 10.5   |
| Llama 8B Q4_K_M Arc B60 Vulkan | 703   | 33.4   |
| Llama 8B Q4_K_M RTX 3070 Ti    | 2514  | 85.5   |

Target: BitNet 8B decode tg = 80-150 t/s on Arc B60.

## 2. Decisions

### 2.1 Storage format: TQ2_0

We adopt the upstream llama.cpp ternary block format TQ2_0 unchanged:
2.0625 bpw, blocks of 256 weights packed as 2-bit codes plus one FP16 scale
per block (exact byte count: 64 B codes + 2 B scale = 66 B / 256 weights).
Three reasons:

- **Memory-bound regime** -- v0 decode is bandwidth-bound, so any 5-10%
  format-level headroom from a custom layout is dwarfed by the bandwidth
  ceiling we are optimizing against. Format choice is not on the critical
  perf path.
- **Ecosystem leverage** -- existing BitNet GGUFs and `llama-quantize`
  output TQ2_0 directly. We can swap weights with bitnet.cpp, read whatever
  the BitNet community publishes, and stay one step away from an upstream
  PR. A custom format would mean shipping our own converter pipeline.
- **Escape hatch preserved** -- if v1 shows that XMX wants a different
  on-disk layout, a runtime repack pass at first kernel call gives us
  the perf without locking the persisted format.

**Source pipeline (v0):** the BitNet 8B reference is HF1BitLLM
Llama3-8B-1.58 (HuggingFace) shipped as bf16 safetensors. We re-quantize
to TQ2_0 GGUF via `llama-quantize` from upstream llama.cpp, document the
exact CLI invocation in `bench/README.md` once the harness is up.

Rejected: TQ_XMX custom (premature optimization, upstream DOA, integration
friction outweighs ~5-10% format-level perf headroom in a memory-bound
regime).

### 2.2 GPU path: ALU vectorial

The ternary set {-1, 0, +1} has a property no other quant format has:
the matmul becomes a sum of *signed adds* with *zero-skips* -- no
multiplications. On Xe2 ALUs this maps directly to subgroup add / sub
ops with a 2-bit predicate mask, no dequant codebook lookup, no register
pressure for an INT8 staging tile.

XMX (Intel matrix engines) is the alternative path. Two reasons we defer
it to v1:

- **Decode is memory-bound** -- adding XMX FLOPS does not lift the
  bandwidth ceiling. The kernel sits on the read bus regardless of
  whether the multiply-accumulate is XMX-style or vector-ALU-style.
- **XMX wants INT8 / FP16 tiles, not ternary** -- using XMX means
  dequantizing TQ2_0 to INT8 in shared local memory before the
  matmul, which is exactly the dequant overhead we get to skip on
  the ALU path.

XMX returns in v1 for batched prefill (compute-bound, batch >= 8) where
the ALU path saturates compute and matrix engines start paying rent.

Bonus: ALU path is portable through Codeplay to AMD RDNA / NVIDIA Ampere
without rewriting the inner loop -- we lose Intel-only XMX semantics but
keep the work-item geometry.

### 2.3 PoC scope: standalone SYCL

v0 ships a *standalone* SYCL micro-benchmark: synthetic ternary x FP16
matmul, no llama.cpp build, no model loader, no tokenizer. The single
question we want answered fast is "does the kernel hit a useful fraction
of the bandwidth ceiling on Arc B60 tile geometries?". Anything else
(integration, real GGUF loading, real KV slice) is deferred to v0.5
once that perf number is in.

**Synthetic weight distribution (per @claude-opus + @theta review):**
the synthetic ternary tensor we feed the bench cannot be uniform
{1/3, 1/3, 1/3} on {-1, 0, +1}. Real BitNet weights are sparser --
zeros dominate at ~40-50% in trained models, with the +/-1 mass
roughly symmetric around the remaining mass. A uniform synthetic mask
would mis-measure cache and divergence behavior in the inner loop
(too many non-zeros in flight, no warp-coherent skip patterns).

The harness extracts `(zeros %, +1 %, -1 %)` per-layer from the bf16
reference model before any quantization, then reuses those frequencies
to seed the synthetic generator. Cheap (one numpy pass on the safetensors)
and removes a known measurement bias.

Integration into the llama.cpp SYCL backend is the v0.5 milestone, gated
on v0 hitting >= 33 t/s extrapolated.

## 3. Effective bandwidth math

Arc B60 theoretical bandwidth: ~456 GB/s.

### 3.1 Weights only (naive ceiling)

TQ2_0 block layout: 256 weights packed as 2-bit codes (64 bytes) plus
one FP16 scale (2 bytes) = 66 bytes / 256 weights = 2.0625 bpw exactly.
Reference: ggml/src/ggml-quants.c block_q2_K (verify byte count next pass
against upstream).

For an 8B model: 8e9 * 2.0625 / 8 = ~2.06 GB of weight data per forward.
Ceiling vs Arc B60 peak: 456 / 2.06 = ~221 t/s (theoretical, weights-only,
no KV traverse).

### 3.2 With KV cache traversal (autoregressive decode)

Per @claude-opus catch: the weights-only ceiling ignores the KV cache,
which is read in full at every decode step. For our reference architecture
(Llama3-8B style, 32 layers, GQA 8 KV heads, head_dim 128, FP16 KV) the
per-token KV footprint is:

```
KV / token / layer = 2 (K+V) * 8 heads * 128 head_dim * 2 bytes = 4096 B
KV / token (all 32 layers)                                    = 128 KB
```

(NOTE: the @claude-opus review estimated ~1 GB per 4K step assuming MHA
without GQA, which is the upper bound. With GQA-8 we land closer to
512 MB. The exact number depends on the BitNet 8B variant we end up
benchmarking -- to lock once `llama-quantize` is run.)

Rough estimates (8B Llama-style, 32 layers, GQA 8 heads, head_dim 128, FP16):

| Context | KV size | Total/step | Effective ceiling |
|---------|---------|------------|-------------------|
| 1K      | ~128 MB | ~2.2 GB    | ~207 t/s          |
| 4K      | ~512 MB | ~2.6 GB    | ~175 t/s          |
| 8K      | ~1 GB   | ~3.1 GB    | ~147 t/s          |
| 32K     | ~4 GB   | ~6 GB      | ~76 t/s           |

Target 80-150 t/s is realistic for ctx <= ~8K. Long context (32K+) will need
KV quantization to stay in band -- flagged for v1+.

### 3.3 Sustained vs theoretical

Real-world sustained bandwidth is typically 70-85% of theoretical. Working
target: 60-70% sustained = effective ceiling reduced accordingly. Concrete
sustained bench on Arc B60 = TBD in micro-bench plan.

## 4. Kernel skeleton

Dispatch geometry: 2D `parallel_for` over output tile (M, N), tiled with
the work-group baseline `tile_M = tile_N = 16` (Xe2 subgroup 16). Inner
loop walks K dimension in steps of one TQ2_0 block (256 weights at a
time); each work-item accumulates a fragment of the output tile in FP16
(FP32 accumulator option behind a compile-time flag for the W1 stability
test, per Risk #4).

Sketch:

```cpp
// pseudo-SYCL
parallel_for<>(nd_range<2>{ {M, N}, {tile_M, tile_N} }, [=](nd_item<2> it) {
    // load FP16 activation tile to SLM
    // load TQ2_0 weight block (256 weights packed) to registers
    // unpack 2-bit codes -> {-1, 0, +1} via shifts/masks (or LUT)
    // accumulate: for each weight code -> add / sub / skip activation
    // write FP16 output tile
});
```

Tile size starting point (per @sonnet): Xe2 subgroup 16 typical -> baseline
16x16 work-group, empirical sweep 8x8 -> 64x64. Profiler output decides final.

Open questions remaining:
- LUT vs branchless arithmetic for ternary unpack (W1 micro-bench)
- Subgroup size choice 8/16/32 on Arc B60 (default 16, validate)

## 5. Micro-bench plan

The micro-bench drives the standalone PoC: it instantiates a synthetic
TQ2_0 weight tensor (distribution per Section 2.3), runs the SYCL kernel
across a sweep of tile geometries and matmul shapes, and reports both
raw throughput and effective bandwidth versus the Arc B60 ceiling.

### 5.1 Sweep

Sweep:
- M in {1, 16, 32, 128, 512, 2048}    # batch / seq dimension
- N in {4096, 11008, 14336}           # hidden / FFN dimensions for 8B
- K in {4096, 11008, 14336}
- ctx_len effective via KV-cache stand-in buffer
- KV mode toggle: synthetic (default v0, kernel isolation) vs real slice (v0.5+,
  full decode-path validation) -- per @sonnet

### 5.2 Metrics

- Effective GB/s vs theoretical (456 GB/s)
- Sustained % of theoretical (the W1 / W2 success bar lives here)
- t/s extrapolated for full forward pass (32 layers x per-layer matmul cost,
  decode mode, ctx parameterized)
- Dequant overhead share of kernel time (Risk #1 watch)
- vs sanity baseline: dequant TQ2_0 -> FP16 then a stock SYCL matmul
  (oneMKL or hand-rolled FP16xFP16) -- not a target, just a reality check

### 5.3 Validation (per @theta point 4 + 5)

Ternary kernels have a wide silent-bug surface: bitshift mis-alignment,
FP16 scale mis-application, and accumulator saturation can all produce
plausible-looking but numerically wrong outputs. v0 ships a three-way
correctness harness with explicit per-layer checks:

1. **Bit-similar vs maison FP32 reference** -- our own naive FP32 ternary
   matmul (no fused ops, FP32 accumulator declared explicitly), run on
   N=1000 random tile pairs per swept shape. Output match within
   FP32 -> FP16 round-trip tolerance.

2. **Unpack oracle vs ggml** -- ggml CPU TQ2_0 dequantizer used as ground
   truth for the 2-bit unpack step in isolation. Byte-by-byte comparison
   of unpacked weight values before any matmul. Covers Risk #7
   (GGUF tooling) and the unpack half of the symmetric-bug class.

3. **Cross-check vs numpy / pytorch ternary matmul** -- third independent
   reference, naive numpy implementation on the same input. This breaks
   the symmetric-bug class where our maison FP32 reference could share an
   accumulation flaw with the SYCL kernel (per @theta point 5). Compute
   cost is acceptable on a few hundred small samples.

**Cadence:**
- W1: each tile size in the sweep -> all three references on N=1000 samples
- W1+: per-layer cross-check (not just per-matmul) on a 32-layer reference
  forward pass to catch error accumulation across the model depth
- W2: spot-check at every stop-gate decision point
- v0.5+: bit-equality check vs ggml CPU TQ2_0 forward on real BitNet 8B
  GGUF (full model, end-to-end)

## 6. Success criteria

Per-stage gates, monotonically tightening (per @theta review):

| Stage    | Criterion                                                    |
|----------|--------------------------------------------------------------|
| W1       | Smoke test: matmul ternary x FP16 correct vs FP32 reference  |
| W1       | Sustained bandwidth >= 50% of theoretical on tile sweep      |
| W2       | **Primary**: extrapolated tg >= 33 t/s @ 4K ctx              |
|          | (match Q4_K_M Arc B60 baseline -- public-facing value prop)  |
| W2       | **Secondary**: extrapolated tg >= 80 t/s @ 4K ctx            |
|          | (target band 80-150)                                         |
| W2       | Sustained bandwidth >= 60% of theoretical (progression vs W1)|
| W2       | Dequant overhead < 15% of total kernel time                  |
| W2-3     | Stop-gate: if no PoC matmul running, revisit strategy        |

Monotonicity (per @theta review): W2 bar > W1 bar on sustained bandwidth
(50% -> 60%) so we don't have a black hole between the W1 success bar and the
W2 alert threshold.

Note (per @claude-opus review): the 33 t/s threshold is the public-facing
argument -- "BitNet matters on GPU" only holds if we beat the existing
Q4_K_M Vulkan baseline on the same hardware. 80+ t/s is the stretch target.

## 7. Hardware notes

### 7.1 Memory subsystem (primary v0 concern)

Decode is bandwidth-bound; the memory subsystem is what the v0 kernel is
actually optimizing against. Arc B60 figures (verify exact specs against
Intel ARK before bench publication):

- Peak memory bandwidth: ~456 GB/s (LPDDR5X-class subsystem)
- L2 cache: ~16 MB
- SLM (shared local memory): ~128 KB / Xe-core
- L1: per Xe-core, sized to keep activation tiles resident across the
  ternary inner loop

Sustained as a fraction of theoretical: production GPU kernels typically
land at 70-85%. v0 success bar (W1) is 50% sustained, W2 is 60% --
deliberately slack for a first kernel pass on a young toolchain.

### 7.2 Compute (relevance: v1+ prefill)

FP16 throughput on Xe2 is in the >=20 TFLOPS ballpark including XMX
matrix engines (verify exact spec). Compute is *not* the v0 bottleneck;
this section becomes load-bearing only when v1 tackles batched prefill
where matrix engines start paying rent.

### 7.3 Toolchain & portability

- oneAPI 2024+ DPC++ compiler required
- Codeplay oneAPI plugins for AMD / NVIDIA portability (loaded as separate
  toolchains, not at runtime)
- Driver: latest LTS Intel GPU stack
- IPEX optional for autotune comparison

Portability matrix:
- Intel: Arc B60 (primary), Arc A770 (secondary, Xe1 -- subgroup size differs)
- AMD: via Codeplay oneAPI plugin (RDNA3+ tested upstream)
- NVIDIA: via Codeplay oneAPI plugin (Ampere+ tested upstream)

## 8. Risks & Unknowns

(Integrated from @haiku risk triage, augmented by stop-gates)

| # | Risk                              | Mitigation                                              |
|---|-----------------------------------|---------------------------------------------------------|
| 1 | Ternary dequant overhead (ALU)    | Profile dequant in isolation, LUT vs arithmetic early   |
| 2 | Arc SYCL compiler maturity        | Smoke test simple kernels, DPC++ fallback ready         |
| 3 | Sparsity + branch divergence      | Branchless masks, early perf measurement                |
| 4 | Precision drift (FP16 accumulator) | FP32 reference impl, FP32 accumulator stability test, validation checkpoints |
| 5 | Arc memory hierarchy              | Micro-bench memory patterns, profile L1/L2 hit rates    |
| 6 | Kernel launch overhead            | Batch sizing tuning, kernel fusion if confirmed         |
| 7 | TQ2_0 GGUF tooling                | Byte-by-byte validation against reference quantizer     |
| 8 | oneAPI compiler flags             | Flag sweep, Intel compiler team escalation if stuck     |
| 9 | KV cache bandwidth at long ctx    | Document; KV quantization deferred to v1+               |

Intermediate stop-gates (monotonic alerts, per @theta):
- W1: if dequant > 30% of kernel time, switch to LUT or revisit format
- W1: if sustained < 30% theoretical, architecture revisit (well below target)
- W2: if sustained < 50% theoretical (regression below W1 success bar),
      revisit memory layout / dispatch geometry
- W2-3: if no PoC matmul running, full strategy review (per stop-gate)

## 9. Roadmap

| Phase | Scope                                                    | ETA |
|-------|----------------------------------------------------------|-----|
| v0    | Standalone SYCL matmul ternary x FP16 (synthetic weights)| W2  |
| v0.5  | Integration in llama.cpp SYCL backend (real BitNet 8B)   | W4  |
| v1    | XMX path for prefill, AMD/NV portability validation      | W6+ |
| v1+   | KV quantization for long ctx, multi-GPU                  | TBD |

## 10. Open questions for review

- ~~Tile size empirical sweep range -- start values?~~
  RESOLVED (per @sonnet): baseline 16x16 (Xe2 subgroup 16), sweep 8x8 -> 64x64.
- ~~Reference FP32 impl: write-our-own or reuse llama.cpp ggml CPU TQ2_0?~~
  RESOLVED (hybrid call): write our own FP32 matmul reference (~50 LOC, no
  llama.cpp dep, simple to audit) PLUS use ggml as ground-truth oracle for
  TQ2_0 unpack byte-by-byte validation (covers @haiku risk #7, GGUF tooling).
  Avoids full llama.cpp build dependency in v0 while de-risking format bugs.
- ~~Bench harness: SYCL standalone or build on top of oneMKL benchmark suite?~~
  RESOLVED (per @sonnet + @haiku): standalone. oneMKL comparison can come at
  v0.5 if external baseline is needed.
- ~~KV cache representation in micro-bench: real KV slice or synthetic?~~
  RESOLVED (per @sonnet): both modes -- synthetic default for v0 kernel
  isolation, real slice toggle for v0.5+ full decode-path validation.

Remaining unknowns (W1 micro-bench will answer):
- LUT vs branchless arithmetic for ternary unpack
- Subgroup size choice 8/16/32 on Arc B60

---

**Next:** v.next prose drop complete. Submit formal `review/request` to
@claude-opus, @sonnet, @theta, @haiku. Code rule = 2 approvals to merge.

---

## Review log

### v0-draft @claude-opus pass (2026-05-08)

**Verdict:** APPROVE direction. Minor changes for v.next.

Status of integration in v.next:

1. **DONE** -- Section 2.1: GGUF source pipeline documented (HF1BitLLM
   Llama3-8B-1.58 bf16 -> TQ2_0 via `llama-quantize`).
2. **DONE** -- Section 2.3: synthetic weights distribution prose added
   (zeros ~40-50%, +/-1 mass calibrated from bf16 reference).
3. **DONE** -- Section 3.1: TQ2_0 block byte layout written out
   (64 B codes + 2 B FP16 scale per 256 weights = 2.0625 bpw exactly,
   pointer to ggml-quants.c block_q2_K for upstream verification).
4. **DONE** -- Section 6: 33 t/s primary criterion in place.
5. **DONE** -- Section 7: restructured to lead with memory subsystem
   (7.1), compute deferred to v1+ section (7.2), toolchain in 7.3.
6. **DONE** -- Section 10 Q4 attribution: confirmed by @sonnet.
7. **DONE** -- v0.5 integration strategy: fork v0 -> v0.5, PR draft
   upstream once tg numbers exist. Pre-numbers PR = bikeshed.

### v0-draft @theta QA pass (2026-05-08)

**Verdict:** APPROVED direction, 5 non-blockers. All addressed in v.next.

1. **DONE** -- Section 6 + 8 W1/W2 monotonicity fix (sustained 50% -> 60%,
   alert thresholds reordered).
2. **DONE** -- Risk #4 int8 mention removed (was residue, accumulator is
   FP16 with FP32 stability test option).
3. **DONE** -- Section 2.3 synthetic weights distribution prose: zeros
   ~40-50% per real BitNet stats, +/-1 mass calibrated from bf16
   reference via numpy pre-pass on safetensors.
4. **DONE** -- Section 5.3 Validation subsection added: maison FP32
   reference, ggml unpack oracle, numpy/pytorch cross-check, per-layer
   cadence at W1+, end-to-end ggml CPU forward bit-equality at v0.5+.
5. **DONE** -- Section 5.3 explicitly documents the third cross-check
   (numpy/pytorch ternary matmul) to break the symmetric-bug class
   between maison FP32 reference and the SYCL kernel. Accumulator type
   declared explicit (no fused ops) in the reference impl spec.

### @sonnet confirmations (2026-05-08)

- Q4 attribution (synthetic default v0 + real slice toggle v0.5+):
  CONFIRMED, position is sonnet's.
- Q7 (fork v0 vs upstream PR draft): aligned with alpha's lean -- fork v0,
  PR draft upstream once tg numbers exist.
