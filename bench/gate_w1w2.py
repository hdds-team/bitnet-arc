#!/usr/bin/env python3
"""
bitnet-arc #147 stop-gate logic.

Consumes a CSV produced by bench/sweep_tile.cpp (or any compatible
producer) and emits a per-shape summary, a per-variant verdict
table, and an outlier list. Reports correctness failures and FP16
overflow events independently from the performance gate.

Default mode is dry-run: the script always exits 0 and just prints a
report. Pass --strict-gates to make it exit 1 on any error-severity
violation. This matches the design v0 #147 contract (calibrate
first, gate later).

CSV format (lenient parser, header keyed by name):

    variant,M,N,K,tile_M,tile_N,sg_size,mode,
    time_ms_med,time_ms_min,bytes,bandwidth_gbs,
    correct,max_rel_err,over_threshold

Lines starting with 'device:' (env header) or 'skip ' (sweep-runner
diagnostic from stderr if the user redirected 2>&1) are recognized
and reported separately. Anything else that does not parse cleanly is
flagged as a parse error (still non-fatal in dry-run).

Gate model (post claude-opus brief, post first Arc B60 smoke -- the
absolute %peak thresholds from design v0 were invalidated by the v0
baseline being ~0.07% of HBM peak):

    W1 (correctness)   : per-variant correct=YES AND
                         over_threshold==0.  Hard gate.
    W2 (speedup)       : per-variant bw / baseline_bw >=
                         --min-speedup (default 1.5x). Warn in
                         dry-run, error in --strict-gates.
    Outlier            : bw < --outlier-frac (=0.7) * shape_mean.
                         Always a warning.
    Soft peak (info)   : best variant < --soft-peak-pct (=0.5%) of
                         HBM peak. Diagnostic only, never fails.

The baseline variant defaults to '16x16_sg16_BRANCHFUL' (matches
design v0 narrative); falls back to slowest runnable variant in the
shape if absent and emits a 'baseline_missing' warning. Override
via --baseline-variant.

Usage:
    python3 gate_w1w2.py path/to/sweep.csv
    python3 gate_w1w2.py --strict-gates path/to/sweep.csv
    python3 gate_w1w2.py --csv-out v.csv path/to/sweep.csv
    python3 gate_w1w2.py --min-speedup 2.0 --baseline-variant ... ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

# Arc Pro B60 peak HBM bandwidth, per Intel datasheet. Used for the
# soft "fraction of peak" diagnostic only -- not a gate. Override
# with --peak-gbs.
DEFAULT_PEAK_GBS = 456.0

# W2 gate: a variant must achieve at least this speedup factor over
# the designated baseline variant in its shape. The 1.5 default comes
# from claude-opus's brief in chat after the first smoke run on Arc
# B60, where the BRANCHLESS variant beat BRANCHFUL by ~1.7x at the
# same tile/sg. Overridable via --min-speedup. Calibration of this
# value is the topic of review #62 once the sweep covers more shapes.
DEFAULT_MIN_SPEEDUP = 1.5

# Soft warning threshold: variants below this fraction of HBM peak
# get a "kernel still far from peak" note. NOT a hard gate -- the v0
# baseline lives well below 1% of peak by design.
DEFAULT_SOFT_PEAK_PCT = 0.5

# Designated baseline variant. Picked to match design v0 narrative:
# smallest registered tile (16x16), default sg (16), branchful inner
# loop (matches the design v0 narrative of "native ternary semantics
# with sub/skip/add"). If this variant is missing from a given shape
# (e.g. tile_M=16 incompatible with M=8), gate_w1w2 falls back to the
# slowest runnable variant in that shape and emits a warning.
DEFAULT_BASELINE_VARIANT = "16x16_sg16_BRANCHFUL"

# Outlier rule: a variant is an outlier within its shape group if its
# bandwidth is below this fraction of the shape's mean bandwidth.
# 0.7 follows the design v0 #147 narrative ("alarm if tile < 0.7 *
# layer_avg") and is configurable via --outlier-frac.
DEFAULT_OUTLIER_FRAC = 0.7


# Data classes ------------------------------------------------------- #


@dataclass
class Row:
    """One CSV data row, parsed and typed."""
    variant: str
    M: int
    N: int
    K: int
    tile_M: int
    tile_N: int
    sg_size: int
    mode: str
    time_ms_med: float
    time_ms_min: float
    bytes_total: int
    bandwidth_gbs: float
    correct: bool
    max_rel_err: float
    over_threshold: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.M, self.N, self.K)


@dataclass
class ParseDiag:
    """Lines we skipped or could not parse, surfaced in the report."""
    device_lines: List[str] = field(default_factory=list)
    skip_lines: List[str] = field(default_factory=list)
    parse_errors: List[Tuple[int, str, str]] = field(default_factory=list)


@dataclass
class GateConfig:
    peak_gbs: float
    min_speedup: float
    soft_peak_pct: float
    outlier_frac: float
    baseline_variant: str
    strict: bool
    csv_out: Optional[Path] = None


@dataclass
class Violation:
    """One gate-relevant finding on a row or shape."""
    severity: str   # "error" | "warning"
    kind: str       # "correctness", "overflow", "outlier", "w1",
                    # "w2", "soft_peak", "baseline_missing"
    message: str


@dataclass
class VariantVerdict:
    """Per-variant pass/fail record for the machine-readable CSV out.
    Schema follows claude-opus's revised brief in chat after the
    first Arc B60 smoke (see review #62):
       variant, M, N, K, bandwidth_gbs, speedup_vs_baseline,
       is_baseline, gate_correctness_ok (W1), gate_speedup_ok (W2),
       is_outlier, pct_peak, verdict, reason
    """
    variant: str
    M: int
    N: int
    K: int
    bandwidth_gbs: float
    speedup_vs_baseline: float
    is_baseline: bool
    gate_correctness_ok: bool   # W1
    gate_speedup_ok: bool       # W2
    is_outlier: bool
    pct_peak: float
    verdict: str  # "pass" | "regress" | "outlier" | "fail_correctness"
    reason: str


# Parsing ------------------------------------------------------------ #


def _str_to_bool(s: str) -> bool:
    """Accept 'YES'/'NO' (sweep_tile.cpp) and '1'/'0' (alt schema)."""
    s = s.strip().upper()
    if s in ("YES", "1", "TRUE", "OK"):
        return True
    if s in ("NO", "0", "FALSE", "FAIL"):
        return False
    raise ValueError(f"unrecognized boolean literal: {s!r}")


def parse_csv(path: Path) -> Tuple[List[Row], ParseDiag]:
    """
    Read the sweep CSV. Resilient to:
      - 'device: ...' header lines (from sweep_tile.cpp's stderr)
      - 'skip <variant>: <reason>' lines (also stderr, sometimes
        merged into stdout via 2>&1)
      - blank lines and trailing whitespace

    Returns (rows, diag). Caller decides how to handle parse errors.
    """
    diag = ParseDiag()
    rows: List[Row] = []
    header: Optional[List[str]] = None

    with path.open("r", encoding="utf-8", newline="") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            if line.startswith("device:"):
                diag.device_lines.append(line)
                continue
            if line.startswith("skip "):
                diag.skip_lines.append(line)
                continue
            # First data-shaped line is treated as the header.
            if header is None:
                header = [c.strip() for c in line.split(",")]
                # Sanity: must contain at minimum the columns we need.
                required = {"variant", "M", "N", "K", "bandwidth_gbs",
                            "correct"}
                missing = required - set(header)
                if missing:
                    diag.parse_errors.append(
                        (lineno, line,
                         f"header missing required columns: {missing}"))
                    header = None
                continue
            # Row.
            try:
                cols = next(csv.reader([line]))
                if len(cols) != len(header):
                    raise ValueError(
                        f"got {len(cols)} cols, header has "
                        f"{len(header)}")
                d = dict(zip(header, [c.strip() for c in cols]))
                row = Row(
                    variant=d["variant"],
                    M=int(d["M"]),
                    N=int(d["N"]),
                    K=int(d["K"]),
                    tile_M=int(d["tile_M"]),
                    tile_N=int(d["tile_N"]),
                    sg_size=int(d["sg_size"]),
                    mode=d.get("mode", ""),
                    time_ms_med=float(d.get("time_ms_med", "nan")),
                    time_ms_min=float(d.get("time_ms_min", "nan")),
                    bytes_total=int(float(d.get("bytes", "0"))),
                    bandwidth_gbs=float(d["bandwidth_gbs"]),
                    correct=_str_to_bool(d["correct"]),
                    max_rel_err=float(d.get("max_rel_err", "nan")),
                    over_threshold=int(d.get("over_threshold", "0")),
                )
                rows.append(row)
            except Exception as exc:
                diag.parse_errors.append((lineno, line, str(exc)))

    return rows, diag


# Gate logic --------------------------------------------------------- #


def group_by_shape(rows: Iterable[Row]
                   ) -> List[Tuple[Tuple[int, int, int], List[Row]]]:
    """Group rows by (M, N, K), preserving first-seen shape order."""
    seen: dict[Tuple[int, int, int], List[Row]] = {}
    for r in rows:
        seen.setdefault(r.shape, []).append(r)
    return list(seen.items())


def _pick_baseline(shape_rows: List[Row], baseline_variant: str
                   ) -> Tuple[Row, bool]:
    """
    Return (baseline_row, is_canonical). The canonical baseline is
    the row matching `baseline_variant`. If no such row exists in
    this shape (e.g. that tile combo was skipped), fall back to the
    SLOWEST runnable variant -- conservative choice that makes any
    speedup metric harder, not easier, to pass.
    """
    for r in shape_rows:
        if r.variant == baseline_variant:
            return r, True
    fallback = min(shape_rows, key=lambda r: r.bandwidth_gbs)
    return fallback, False


def evaluate(rows: List[Row], cfg: GateConfig
             ) -> Tuple[List[Violation], List[Tuple], List[VariantVerdict]]:
    """
    Walk grouped rows. Returns (violations, summary_rows, verdicts).

    Gate semantics (post claude-opus brief, post first Arc B60 smoke):
      - W1 (correctness)   : per-variant, `correct == YES` AND
                             `over_threshold == 0`. Hard gate.
      - W2 (speedup)       : per-variant, `bw / baseline_bw >=
                             min_speedup`. Soft warn by default,
                             upgraded to error in --strict-gates.
      - Outlier            : per-variant, `bw < outlier_frac *
                             shape_mean_bw`. Warn (always).
      - Soft peak fraction : per-shape, best variant `bw / peak <
                             soft_peak_pct%`. Informational warn.
    """
    violations: List[Violation] = []
    summary: List[Tuple] = []
    verdicts: List[VariantVerdict] = []

    if not rows:
        return violations, summary, verdicts

    for shape, shape_rows in group_by_shape(rows):
        bws = [r.bandwidth_gbs for r in shape_rows]
        bw_mean = mean(bws)
        bw_min = min(bws)
        bw_max = max(bws)
        best = max(shape_rows, key=lambda r: r.bandwidth_gbs)

        baseline_row, baseline_canonical = _pick_baseline(
            shape_rows, cfg.baseline_variant)
        baseline_bw = baseline_row.bandwidth_gbs

        if not baseline_canonical:
            violations.append(Violation(
                "warning",
                "baseline_missing",
                f"shape {shape}: canonical baseline "
                f"'{cfg.baseline_variant}' missing, falling back "
                f"to slowest runnable variant '{baseline_row.variant}' "
                f"(bw={baseline_bw:.3f} GB/s)"))

        peak_pct_max = 100.0 * bw_max / cfg.peak_gbs

        # Soft "kernel far from peak" warn at the shape level. Pure
        # diagnostic, never an error -- the v0 baseline is expected
        # to be well below 1% of peak.
        if peak_pct_max < cfg.soft_peak_pct:
            violations.append(Violation(
                "warning",
                "soft_peak",
                f"shape {shape}: best variant {best.variant} hits "
                f"{peak_pct_max:.4f}% of peak (< "
                f"{cfg.soft_peak_pct:.2f}%), kernel still far from "
                f"HBM peak ({cfg.peak_gbs:.0f} GB/s)"))

        # Per-variant pass through.
        for r in shape_rows:
            speedup = (r.bandwidth_gbs / baseline_bw
                       if baseline_bw > 0.0 else float("nan"))
            is_baseline = (r.variant == baseline_row.variant)
            is_outlier = r.bandwidth_gbs < cfg.outlier_frac * bw_mean
            gate_corr = r.correct and r.over_threshold == 0
            # The baseline is exempt from the speedup gate (it is
            # the unit of measure -- speedup vs itself is 1.0). Also
            # exempt from outlier (it sets the floor by definition).
            gate_speed = (
                True if is_baseline
                else speedup >= cfg.min_speedup
            )

            # Pick the dominant verdict label for the human/CSV
            # summary. Order matters: correctness > regression >
            # outlier > pass.
            if not gate_corr:
                verdict = "fail_correctness"
                reason = (f"correct={r.correct}, "
                          f"over_threshold={r.over_threshold}, "
                          f"max_rel_err={r.max_rel_err:.3g}")
            elif is_baseline:
                verdict = "baseline"
                reason = "designated baseline variant"
            elif is_outlier:
                verdict = "outlier"
                reason = (f"bw={r.bandwidth_gbs:.3f} below "
                          f"{cfg.outlier_frac:.2f}x shape_mean "
                          f"({bw_mean:.3f})")
            elif not gate_speed:
                verdict = "regress"
                reason = (f"speedup={speedup:.3f}x below "
                          f"{cfg.min_speedup:.2f}x min")
            else:
                verdict = "pass"
                reason = (f"speedup={speedup:.3f}x, "
                          f"%peak={100.0*r.bandwidth_gbs/cfg.peak_gbs:.4f}")

            verdicts.append(VariantVerdict(
                variant=r.variant,
                M=r.M, N=r.N, K=r.K,
                bandwidth_gbs=r.bandwidth_gbs,
                speedup_vs_baseline=speedup,
                is_baseline=is_baseline,
                gate_correctness_ok=gate_corr,
                gate_speedup_ok=gate_speed,
                is_outlier=is_outlier,
                pct_peak=100.0 * r.bandwidth_gbs / cfg.peak_gbs,
                verdict=verdict,
                reason=reason,
            ))

            # Now translate to violations for the human report.
            if not r.correct:
                violations.append(Violation(
                    "error",
                    "correctness",
                    f"shape {shape}: variant {r.variant} failed "
                    f"correctness (max_rel_err={r.max_rel_err:.3g}, "
                    f"over_threshold={r.over_threshold})"))
            if r.over_threshold > 0:
                violations.append(Violation(
                    "warning" if r.correct else "error",
                    "overflow",
                    f"shape {shape}: variant {r.variant} has "
                    f"{r.over_threshold} elements above tolerance "
                    f"(max_rel_err={r.max_rel_err:.3g})"))
            if is_outlier:
                violations.append(Violation(
                    "warning",
                    "outlier",
                    f"shape {shape}: variant {r.variant} bw="
                    f"{r.bandwidth_gbs:.3f} GB/s is below "
                    f"{cfg.outlier_frac:.2f}x shape mean "
                    f"({bw_mean:.3f} GB/s)"))
            if (not is_baseline) and gate_corr and (not gate_speed):
                violations.append(Violation(
                    "error" if cfg.strict else "warning",
                    "w2",
                    f"shape {shape}: variant {r.variant} speedup="
                    f"{speedup:.3f}x vs baseline "
                    f"'{baseline_row.variant}' is below W2 "
                    f"min={cfg.min_speedup:.2f}x"))

        summary.append((
            shape, len(shape_rows),
            bw_mean, bw_min, bw_max,
            peak_pct_max,
            best.variant, best.bandwidth_gbs,
            baseline_row.variant, baseline_bw, baseline_canonical,
            best.bandwidth_gbs / baseline_bw if baseline_bw > 0 else float("nan"),
        ))

    return violations, summary, verdicts


# Reporting ---------------------------------------------------------- #


def render_report(rows: List[Row], diag: ParseDiag,
                  cfg: GateConfig,
                  violations: List[Violation],
                  summary: List[Tuple],
                  verdicts: List[VariantVerdict],
                  csv_path: Path) -> str:
    out: List[str] = []
    out.append("=== bitnet-arc W1/W2 stop-gate report ===")
    out.append(f"Input         : {csv_path}")
    out.append(f"Baseline var. : {cfg.baseline_variant}")
    out.append(f"Peak BW model : {cfg.peak_gbs:.1f} GB/s "
               f"(--peak-gbs, info-only)")
    out.append(f"W1 (correct)  : per-variant correct=YES AND "
               f"over_threshold==0")
    out.append(f"W2 (speedup)  : per-variant bw / baseline_bw >= "
               f"{cfg.min_speedup:.2f}x")
    out.append(f"Outlier rule  : tile_bw < "
               f"{cfg.outlier_frac:.2f} * shape_mean -> warn")
    out.append(f"Soft peak warn: best variant < "
               f"{cfg.soft_peak_pct:.2f}% of HBM peak")
    out.append(f"Mode          : "
               f"{'STRICT (exit 1 on error)' if cfg.strict else 'DRY-RUN (exit 0)'}")
    out.append("")
    if diag.device_lines:
        out.append("Device header(s):")
        for d in diag.device_lines:
            out.append(f"  {d}")
        out.append("")

    if not rows:
        out.append("No data rows parsed -- nothing to gate.")
        if diag.parse_errors:
            out.append("Parse errors:")
            for ln, raw, why in diag.parse_errors:
                out.append(f"  L{ln}: {why} :: {raw}")
        return "\n".join(out)

    # Per-shape summary table.
    out.append("Per-shape summary:")
    out.append("  " + " | ".join([
        "shape (M,N,K)".ljust(22),
        "n".rjust(3),
        "bw_min".rjust(8),
        "bw_max".rjust(8),
        "best/base".rjust(9),
        "%peak".rjust(7),
        "best variant",
    ]))
    out.append("  " + "-" * 92)
    for (shape, n, bw_mean, bw_min, bw_max, pct_max,
         best_var, best_bw, base_var, base_bw, base_canon,
         best_speedup) in summary:
        marker = "" if base_canon else " *fallback*"
        out.append("  " + " | ".join([
            f"{shape}".ljust(22),
            f"{n}".rjust(3),
            f"{bw_min:.3f}".rjust(8),
            f"{bw_max:.3f}".rjust(8),
            f"{best_speedup:.2f}x".rjust(9),
            f"{pct_max:.4f}".rjust(7),
            f"{best_var} (base={base_var}{marker})",
        ]))
    out.append("")

    # Per-variant verdict table (compact).
    out.append("Per-variant verdicts:")
    out.append("  " + " | ".join([
        "variant".ljust(28),
        "shape (M,N,K)".ljust(20),
        "bw_gbs".rjust(7),
        "speedup".rjust(7),
        "verdict".ljust(16),
    ]))
    out.append("  " + "-" * 92)
    for v in verdicts:
        out.append("  " + " | ".join([
            v.variant.ljust(28),
            f"({v.M},{v.N},{v.K})".ljust(20),
            f"{v.bandwidth_gbs:.3f}".rjust(7),
            f"{v.speedup_vs_baseline:.2f}x".rjust(7),
            v.verdict.ljust(16),
        ]))
    out.append("")

    # Skipped variants (informational).
    if diag.skip_lines:
        out.append("Skipped variants (reported by sweep_tile):")
        for s in diag.skip_lines:
            out.append(f"  {s}")
        out.append("")

    # Violations.
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]
    out.append(f"Violations: {len(errors)} error(s), "
               f"{len(warnings)} warning(s)")
    for v in violations:
        tag = "[ERROR  ]" if v.severity == "error" else "[WARNING]"
        out.append(f"  {tag} {v.kind:12s} {v.message}")
    if not violations:
        out.append("  (none)")
    out.append("")

    if diag.parse_errors:
        out.append(f"Parse errors: {len(diag.parse_errors)}")
        for ln, raw, why in diag.parse_errors:
            out.append(f"  L{ln}: {why} :: {raw}")
        out.append("")

    return "\n".join(out)


# CLI ---------------------------------------------------------------- #


def write_csv_verdict(path: Path,
                      verdicts: List[VariantVerdict]) -> None:
    """Emit a machine-readable per-variant verdict CSV. Schema is
    stable across runs so downstream tooling (review #62 calibration
    notebooks, regression diffs across sweeps) can rely on it."""
    fields = [
        "variant", "M", "N", "K",
        "bandwidth_gbs", "speedup_vs_baseline",
        "is_baseline", "gate_correctness_ok", "gate_speedup_ok",
        "is_outlier", "pct_peak", "verdict", "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for v in verdicts:
            w.writerow([
                v.variant, v.M, v.N, v.K,
                f"{v.bandwidth_gbs:.6f}",
                f"{v.speedup_vs_baseline:.6f}",
                "1" if v.is_baseline else "0",
                "1" if v.gate_correctness_ok else "0",
                "1" if v.gate_speedup_ok else "0",
                "1" if v.is_outlier else "0",
                f"{v.pct_peak:.6f}",
                v.verdict, v.reason,
            ])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="bitnet-arc W1/W2 stop-gate (dry-run by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("csv", type=Path,
                   help="path to a sweep CSV produced by sweep_tile")
    p.add_argument("--peak-gbs", type=float,
                   default=DEFAULT_PEAK_GBS,
                   help="peak HBM bandwidth in GB/s "
                        f"(default: {DEFAULT_PEAK_GBS:.1f}, Arc B60)"
                        " -- info-only, not gated")
    p.add_argument("--min-speedup", type=float,
                   default=DEFAULT_MIN_SPEEDUP,
                   help="W2 gate: minimum speedup vs baseline "
                        f"(default: {DEFAULT_MIN_SPEEDUP:.2f}x)")
    p.add_argument("--baseline-variant", type=str,
                   default=DEFAULT_BASELINE_VARIANT,
                   help=f"designated baseline variant name "
                        f"(default: {DEFAULT_BASELINE_VARIANT}); "
                        "falls back to slowest in shape if missing")
    p.add_argument("--soft-peak-pct", type=float,
                   default=DEFAULT_SOFT_PEAK_PCT,
                   help="soft warning threshold as %% of peak "
                        f"(default: {DEFAULT_SOFT_PEAK_PCT:.2f}%%)")
    p.add_argument("--outlier-frac", type=float,
                   default=DEFAULT_OUTLIER_FRAC,
                   help="outlier alarm threshold as fraction of "
                        "shape-mean BW "
                        f"(default: {DEFAULT_OUTLIER_FRAC:.2f})")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="optional path to write a per-variant "
                        "verdict CSV (machine-readable)")
    p.add_argument("--strict-gates", action="store_true",
                   help="exit 1 on any error-severity violation "
                        "(default: dry-run, exit 0)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """
    Exit code contract (per @sonnet review #62 nit):

      - dry-run mode (default) : ALWAYS exit 0, even on correctness
        errors. This is intentional -- the report is the deliverable
        in calibration mode, and a CI cron should not fail on a
        finding it cannot act on. Use --strict-gates to flip this:
        any error-severity violation will then exit 1.
      - --strict-gates        : exit 1 on any error (correctness
        fail, overflow on incorrect rows, W2 below threshold).
    """
    args = build_argparser().parse_args(argv)

    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    cfg = GateConfig(
        peak_gbs=args.peak_gbs,
        min_speedup=args.min_speedup,
        soft_peak_pct=args.soft_peak_pct,
        outlier_frac=args.outlier_frac,
        baseline_variant=args.baseline_variant,
        strict=args.strict_gates,
        csv_out=args.csv_out,
    )

    rows, diag = parse_csv(args.csv)
    violations, summary, verdicts = evaluate(rows, cfg)
    print(render_report(rows, diag, cfg,
                        violations, summary, verdicts, args.csv))

    if cfg.csv_out is not None:
        write_csv_verdict(cfg.csv_out, verdicts)
        print(f"per-variant verdict CSV written to {cfg.csv_out}")

    if cfg.strict and any(v.severity == "error" for v in violations):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
