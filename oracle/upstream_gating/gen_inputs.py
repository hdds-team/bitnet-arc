"""
Generate random ternary tensors for the upstream gating test.

Each tensor:
  - is block-aligned (length is a multiple of 256, the TQ2_0 block size)
  - entries drawn from {-1, 0, +1} per the design v0 synthetic-weights
    distribution: zeros ~40-50%, +/-1 mass split symmetrically over the
    remainder

Output format: raw int8 binary, one tensor per file. The companion
gate.c reads each file, runs both quantize paths, and compares.

Usage:
    python gen_inputs.py --out-dir ./inputs --count 10000 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# Block size hard-coded per TQ2_0 spec (oracle/tq2_0.h).
BLOCK_SIZE = 256

# Length sweep for the gating test. All values are multiples of BLOCK_SIZE.
LENGTH_SWEEP = (256, 512, 1024, 2048, 4096)

# Synthetic distribution from design v0 section 2.3: zeros ~40-50%,
# +/-1 mass roughly symmetric over the remainder.
P_ZERO = 0.45
P_PLUS = 0.275
P_MINUS = 0.275


def gen_tensor(rng: np.random.Generator, length: int) -> np.ndarray:
    """Return one length-element ternary tensor as int8 in {-1, 0, +1}."""
    if length % BLOCK_SIZE != 0:
        raise ValueError(f"length {length} not a multiple of {BLOCK_SIZE}")

    probs = np.array([P_ZERO, P_PLUS, P_MINUS], dtype=np.float64)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"distribution probs do not sum to 1: {probs.sum()}")

    codes = rng.choice(3, size=length, p=probs).astype(np.int8)
    # codes 0/1/2 -> ternary 0/+1/-1
    lut = np.array([0, 1, -1], dtype=np.int8)
    return lut[codes]


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, help="directory to write input tensors")
    p.add_argument("--count",   required=True, type=int, help="number of tensors")
    p.add_argument("--seed",    default=42, type=int, help="RNG seed")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    digits = len(str(args.count - 1))
    for i in range(args.count):
        length = LENGTH_SWEEP[i % len(LENGTH_SWEEP)]
        t = gen_tensor(rng, length)
        path = os.path.join(args.out_dir, f"tensor_{i:0{digits}d}_n{length}.i8")
        t.tofile(path)

    print(f"wrote {args.count} tensors to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
