# bitnet-arc

SYCL ternary BitNet kernel for Intel Arc B60 (Xe2 / Battlemage).

## Goal

Provide a native GPU kernel for ternary weights ({-1, 0, +1}) on Arc B60, filling
the gap left by llama.cpp upstream (no GPU support for TQ1_0 / TQ2_0 today).

## Why

| Path                         | Decode tg (8B) | Notes                  |
|------------------------------|----------------|------------------------|
| BitNet I2_S CPU bi-Xeon      | 10.5 t/s       | bitnet.cpp, CPU-only   |
| Llama Q4_K_M Arc B60 Vulkan  | 33.4 t/s       | reference GPU baseline |
| Llama Q4_K_M RTX 3070 Ti     | 85.5 t/s       | upper reference        |
| **Target: BitNet 8B Arc B60**| **80-150 t/s** | this kernel            |

Memory-bound regime: TQ2_0 = 2.0625 bpw, ~5x less to read than Q8, ~2.5x less
than Q4_K_M. Bandwidth ceiling on Arc B60: ~456 GB/s theoretical.

## Design v0 (locked Workflow #14)

| Decision      | Choice               | Rationale (short)                                |
|---------------|----------------------|--------------------------------------------------|
| Storage       | TQ2_0                | upstream compat, escape hatch via runtime repack |
| GPU path      | ALU vectorial        | native add/sub/skip, portable, bandwidth-bound   |
| PoC scope     | standalone SYCL      | validate bandwidth thesis fast, low plumbing     |

XMX path kept on the table for v1 (batched prefill).

## Stop-gate

T+2-3 weeks. If no matmul ternary x FP16 PoC running, revisit strategy.

## Roadmap

- v0   : standalone SYCL matmul ternary x FP16 on synthetic weights
- v0.5 : integration in llama.cpp SYCL backend (real BitNet 8B GGUF E2E)
- v1+  : XMX path for prefill, AMD/NV portability validation

## Layout

- `docs/`       design docs, bench notes, decisions log
- `src/`        SYCL kernels (TBD)
- `bench/`      micro-benchmark harness (TBD)
- `tests/`      correctness vs FP32 reference (TBD)

## Team

- Lead: @alpha
- Reviewers: @claude-opus, @sonnet, @theta, @haiku
- Coordination: @claude-opus + @naskel

## Status

WIP. Design doc in `docs/design-v0.md` (drop ETA: tomorrow EOD).
