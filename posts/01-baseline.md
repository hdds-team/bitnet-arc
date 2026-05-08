# BitNet on GPU: filling the ternary kernel gap on Intel Arc

**TL;DR.** BitNet's 1.58-bit ternary models work great on CPU and have no native
GPU implementation in any major inference framework. We measured the cost of
that gap on real hardware (an Intel Arc B60 / Battlemage Xe2) and decided to
close it with a standalone SYCL kernel. This post is the baseline: state of the
world, numbers, and where we're going.

---

## The pitch and the gap

Microsoft's BitNet b1.58 family stores neural network weights as ternary values
(`-1`, `0`, `+1`). At inference, this collapses matrix multiplication into a
sum of additions, subtractions, and skips — no actual multiplications on the
weight side. The advertised wins are real: a Llama-3-derived 8B model trained
in this regime fits in **~3.6 GB** of memory and runs entirely on CPU at usable
speeds.

The ecosystem has caught up partly:

- `bitnet.cpp` (Microsoft's reference implementation) ships highly tuned CPU
  kernels for AVX2 / AVX-512 / NEON, with a custom GGUF-like format `I2_S`
  (and `TL1` / `TL2` lookup-table variants).
- `llama.cpp` upstream defines two ternary GGUF types — `TQ1_0` (1.6875 bpw)
  and `TQ2_0` (2.0625 bpw) — and a working CPU kernel.

But here's what nobody has shipped:

> **There is no GPU kernel for ternary weights in any major inference framework
> as of May 2026.**

Vulkan: nothing. SYCL: nothing. CUDA in `llama.cpp` upstream: nothing.
Loading a `TQ2_0` GGUF on a GPU silently falls back to CPU for the matmul, or
forces a dequantize-to-FP16 path that throws away the entire point of ternary
storage.

This is a hole. It's also an opportunity, because the hardware is more than
ready.

## What "the gap" actually costs you

We benchmarked the same 8B-class model under three configurations on a single
machine:

| Workload                                    | Hardware                  | pp512 | tg128 |
|---------------------------------------------|---------------------------|-------|-------|
| BitNet 8B (`I2_S`) via `bitnet.cpp`         | bi-Xeon E5-2699 v4, 22T   |   109 |  10.5 |
| Llama 3.1 8B `Q4_K_M` via `llama.cpp` Vulkan| Intel Arc Pro B60 (24 GB) |   703 |  33.4 |
| Llama 3.1 8B `Q4_K_M` via `llama.cpp` Vulkan| NVIDIA RTX 3070 Ti (8 GB) |  2514 |  85.5 |

(Numbers are tokens per second, single sequence, default sampler. Build:
`bitnet.cpp` master 1f86f058, `llama.cpp` upstream Vulkan build, May 2026.)

Two observations:

**1. BitNet on CPU is genuinely competitive.** 10.5 tok/s for an 8B model on
a 2016-vintage server CPU — using only AVX2 — is a solid result. For
comparison, the same model in FP16 on the same CPU manages ~2-3 tok/s. The
ternary format earns its keep.

**2. The Arc B60 is also genuinely capable.** 33 tok/s for a Q4_K_M 8B on
an Intel discrete GPU is decent; the RTX 3070 Ti is faster, but the Arc has
3× the VRAM (24 GB vs 8 GB) and a much better $/GB ratio for inference.

Now imagine running BitNet 8B on the Arc with native ternary kernels. The
storage format is **2.5× smaller** than `Q4_K_M`. In a memory-bound regime
(which decode is), that's the dominant factor. The Arc B60's theoretical
memory bandwidth is around 456 GB/s. With ternary weights at ~2.06 GB for 8B,
the weights-only ceiling is **~221 tok/s**.

Add the KV cache traversal that an autoregressive decoder pays at every step
and you get a more realistic picture:

| Context (FP16 KV) | KV size | Total per step | Effective ceiling |
|-------------------|---------|----------------|-------------------|
| 1K tokens         | ~128 MB | ~2.2 GB        | ~207 tok/s        |
| 4K tokens         | ~512 MB | ~2.6 GB        | ~175 tok/s        |
| 8K tokens         | ~1 GB   | ~3.1 GB        | ~147 tok/s        |
| 32K tokens        | ~4 GB   | ~6 GB          | ~76 tok/s         |

At 4-8K context, sustained at 60-70% of theoretical (a normal range for
well-tuned memory-bound kernels), we'd expect **~100-130 tok/s** for ternary
8B on an Arc B60.

That's roughly **3-4× the Q4_K_M baseline on the same GPU**, **10× the
CPU baseline**, and lands in the same band as the RTX 3070 Ti — on a card
with 3× the VRAM. That's the prize.

## What we're building

A standalone SYCL kernel implementing ternary × FP16 matmul, designed for the
`TQ2_0` storage format already defined in `llama.cpp` upstream. Three design
decisions, all locked after team review:

**1. Storage: `TQ2_0`, not a custom format.**
`TQ2_0` is the upstream-blessed format (2.0625 bits per weight, 256-weight
blocks with FP16 scale). Adopting it means existing BitNet GGUFs and converters
work unchanged, the upstream PR path stays open, and we keep an escape hatch
(runtime repack to an XMX-friendly layout) for a v1 if needed. A custom format
would have given maybe 5-10% headroom in a regime where bandwidth dominates —
not worth the integration friction.

**2. GPU path: ALU vectorial, not the XMX matrix engines (yet).**
The Arc B60's XMX units are excellent at INT8 / INT4 matrix multiply, but
ternary `{-1, 0, +1}` doesn't map cleanly to them — you'd pay a dequant step
or use a codebook lookup, and the math becomes "almost INT8 matmul" instead of
"native ternary". The vectorial ALU path, by contrast, expresses the ternary
sum as `add` / `sub` / `skip` directly through subgroup operations, with zero
dequant overhead. It also stays portable: Codeplay's oneAPI plugins compile
SYCL kernels to AMD (RDNA3+) and NVIDIA (Ampere+) without XMX-specific tricks.

XMX is on the roadmap for a v1 targeted at compute-bound prefill / batched
inference, where the matrix engines actually shine.

**3. Scope: standalone PoC, not an `llama.cpp` integration on day one.**
The first goal is to validate the bandwidth thesis, fast. A standalone SYCL
benchmark with synthetic ternary weights × FP16 activations gives us a clean
measurement of what the kernel can do, without entangling it with the rest of
the inference stack. If the numbers match the ceiling math, we integrate at
v0.5 against a real `BitNet-b1.58` GGUF. If they don't, we know within days
instead of weeks and pivot before sunk cost.

## How we'll know it worked

Three success bars, in order:

1. **Correctness, week 1.** Bit-exact match against an FP32 reference plus
   a third cross-check (independent NumPy/PyTorch ternary matmul) to break
   any symmetric-bug class. A ternary kernel can produce silently wrong
   results at full speed; we'd rather see this immediately.
2. **Public-facing value, week 2.** Extrapolated decode throughput
   ≥ **33 tok/s @ 4K context** — match the Q4_K_M Arc B60 baseline. This is
   the threshold at which "BitNet matters on GPU" becomes a defensible claim
   on the same hardware.
3. **Stretch, week 2.** Extrapolated decode ≥ **80 tok/s @ 4K context** —
   land in the projected band and start eating into the NVIDIA gap.

If by week 2-3 the standalone PoC isn't running a correct matmul, we revisit
the whole strategy. That stop-gate is non-negotiable. There is no "we'll get
there eventually" mode.

## Why now, why this hardware

There are three reasons we think this is the right project at the right time.

**The hardware lives in the right gap.** Intel Arc Pro B60 ships with 24 GB
of VRAM at a price point ($600-800 in 2026) that no NVIDIA card matches. The
weakness — modest matrix throughput compared to recent RTX cards — matters
much less in a memory-bound, ternary regime. This is exactly the workload
where the Arc shines.

**SYCL is the right portability layer.** A Vulkan compute shader would have
worked, but `llama.cpp` already has a SYCL backend and Codeplay maintains
oneAPI plugins for AMD and NVIDIA. A SYCL kernel written cleanly compiles to
all three vendors. Vulkan has no equivalent matrix engine path; SYCL has
direct access via oneDNN intrinsics if we ever want them in v1.

**No 70B ternary model exists yet, and that's fine.** The largest publicly
available ternary-trained model today is roughly 10B (Falcon3-10B-1.58bit).
Microsoft's official BitNet checkpoint is 2B. We are not betting on a 70B
ternary model arriving — we're delivering the kernel that makes a future
larger ternary model viable on accessible hardware, while immediately giving
existing 8-10B models a real GPU runtime. If the larger models arrive, the
kernel is already there. If they don't, current users still benefit.

## What's next, in public

The work is happening in the open from day one.

- **Repo**: `bitnet-arc` (going public alongside this post).
- **Design doc**: `docs/design-v0.md`, locked tonight after multi-LLM review.
- **PoC milestone**: standalone SYCL matmul, target weeks 1-2.
- **Integration milestone**: `llama.cpp` SYCL backend, target weeks 3-4
  (kept on a fork until numbers are solid; upstream PR draft once they are).

If you work on `llama.cpp`, on Intel oneAPI / DevRel, or on a BitNet-adjacent
research effort, we'd love to compare notes. Find us on GitHub, or — for the
HN / Twitter / Mastodon crowd — the comments below.

The gap is real. The hardware is ready. Let's close it.

---

*Acknowledgements: this project is run as a multi-LLM collaboration via
[aIRCp](https://github.com/hdds-team/bitnet-arc). Architecture and review by `@claude-opus`,
`@alpha`, `@sonnet`, `@theta`, `@haiku`. Hardware and supervision by Olivier
(`@naskel`). Numbers and code, all of it, will be reproducible from the repo.*
