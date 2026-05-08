"""
Third independent voice for the bitnet-arc oracle.

Computes the same matmul as oracle/fp32_matmul.c, but via numpy.dot in
float64. NumPy dispatches to BLAS (MKL or OpenBLAS) on most installs,
so the reduction loop here is independent from our maison FP32
reference. This breaks the symmetric-bug class flagged by @theta in
the design review: a flaw in our naive m-n-k loop would affect the
SYCL kernel and the FP32 maison reference identically, but not this
voice.

Tolerance bars are NOT hardcoded here -- they are parsed from
oracle/tolerance.h at import time, which is the single source of
truth shared with the C/C++ harness (bench/harness_3way.cpp). Edit
tolerance.h, both consumers update.

Loaded by bench/harness_3way.cpp via subprocess, pybind11, or shared
buffer (the harness picks). Stand-alone CLI form below for debugging.

Usage:
    python numpy_xcheck.py --A activations.fp16 --B weights.fp32 \
                           --M 16 --K 4096 --N 4096 --out result.fp64
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np


# --- Tolerance loading (single source of truth: oracle/tolerance.h) ---

_TOLERANCE_HEADER = os.path.join(os.path.dirname(__file__), "tolerance.h")
_TOL_LINE_RE = re.compile(
    r"^#define\s+(BITNET_ARC_TOL_\w+)\s+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)f?\s*$"
)


def _load_tolerances(path: str = _TOLERANCE_HEADER) -> dict[str, float]:
    """Parse oracle/tolerance.h and return {macro_name: float} mapping.

    Refuses to silently default: any malformed `#define BITNET_ARC_TOL_*`
    line raises. Comment lines and other defines are skipped.
    """
    tols: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line.startswith("#define BITNET_ARC_TOL_"):
                continue
            m = _TOL_LINE_RE.match(line)
            if m is None:
                raise ValueError(
                    f"{path}: malformed BITNET_ARC_TOL_* define: {raw!r}"
                )
            tols[m.group(1)] = float(m.group(2))
    if not tols:
        raise RuntimeError(f"{path}: no BITNET_ARC_TOL_* defines found")
    return tols


# Public mapping. Consumers should use TOLERANCES["BITNET_ARC_TOL_..."]
# instead of literals. Cached at import time.
TOLERANCES = _load_tolerances()


def matmul_xcheck_f64(A_fp16: np.ndarray, B_fp32: np.ndarray) -> np.ndarray:
    """
    Run the cross-check matmul in float64.

    Args:
        A_fp16: shape (M, K), dtype float16. Activations as supplied to
                the SYCL kernel.
        B_fp32: shape (K, N), dtype float32. Already-dequantized ternary
                weights -- entries are exactly scale_fp16 * {-1, 0, +1}.

    Returns:
        C: shape (M, N), dtype float64. The harness compares against
           the FP32 maison reference per the tolerance model.

    Raises:
        TypeError if input dtypes are not as documented above. We
        refuse implicit promotion: the whole point of this voice is
        explicit float64.
    """
    if A_fp16.dtype != np.float16:
        raise TypeError(f"A_fp16 must be float16, got {A_fp16.dtype}")
    if B_fp32.dtype != np.float32:
        raise TypeError(f"B_fp32 must be float32, got {B_fp32.dtype}")

    # Explicit float64 cast end-to-end. The BLAS call is dgemm.
    # Worst-case rounding drift on K=14336 is bounded by ~K * eps_f64,
    # i.e. ~3e-12 relative -- well below every tolerance bar above.
    A = A_fp16.astype(np.float64, copy=False)
    B = B_fp32.astype(np.float64, copy=False)
    return A @ B


def _read_bin(path: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"{path}: expected {expected} elements of {dtype}, got {arr.size}"
        )
    return arr.reshape(shape)


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--A",   required=True, help="FP16 activations, M*K halfs")
    p.add_argument("--B",   required=True, help="FP32 dequant weights, K*N floats")
    p.add_argument("--M",   required=True, type=int)
    p.add_argument("--K",   required=True, type=int)
    p.add_argument("--N",   required=True, type=int)
    p.add_argument("--out", required=True, help="output FP64, M*N doubles")
    args = p.parse_args()

    A = _read_bin(args.A, (args.M, args.K), np.float16)
    B = _read_bin(args.B, (args.K, args.N), np.float32)
    C = matmul_xcheck_f64(A, B)
    assert C.dtype == np.float64, f"internal: C dtype is {C.dtype}, expected float64"
    C.tofile(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
