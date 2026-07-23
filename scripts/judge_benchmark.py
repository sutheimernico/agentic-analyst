"""Quantified judge benchmark (Task C1).

Plants controlled lies into a set of verified base findings -- across every
attack class in ``planting.py`` (value-swap, claim/value mismatch, fabricated
evidence, alias-dodge) at four relative magnitudes (+-5%, +-20%, +-50%,
+-200%) -- then runs the real judge (`verify_finding`) on each and measures how
reliably it catches them. Replaces anecdotal "the judge caught the demo lie"
confidence with a reproducible precision/recall/catch-rate table + chart.

Base findings are genuinely-true quantities recomputed from the committed telco
CSV at runtime (each is asserted `verified` before any tampering), so a "catch"
is a real catch, not an artifact of a broken base. Everything is deterministic
(exhaustive enumeration, no RNG; DuckDB over the fixed CSV), so the committed
artifact and both figures are byte-stable across runs -- the SVG additionally
needs a fixed `svg.hashsalt` and a suppressed timestamp (see below), or
matplotlib would randomize its element IDs and stamp a live date. Wall-clock is
logged to stdout only, never written into the artifact numbers.

Scope choices (documented in the artifact meta):

- Evidence is SQL-only. The judge's Python-evidence path is exercised by the
  unit tests; the sweep uses fast SQL evidence so N>=200 cases run in seconds.
- The baseline retrain (`verify_baseline`, ~seconds/case) is excluded: these
  attack classes target finding-level claim verification, which never retrains.
  Baseline tamper-detection is covered separately in `tests/test_judge.py`.

Usage: uv run python scripts/judge_benchmark.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
from matplotlib.figure import Figure

from agentic_analyst.judge import verify_finding
from agentic_analyst.planting import (
    ATTACK_CLASSES,
    CATCH_VERDICT,
    DEFAULT_MAGNITUDES,
    BaseFinding,
    PlantedCase,
    generate_cases,
    honest_finding,
)
from agentic_analyst.tools import query_sql, single_value

# Deterministic SVG: fix the hash salt so matplotlib's element/clip-path IDs are
# stable across runs (randomized otherwise), so the committed SVG is byte-stable
# (the embedded <dc:date> timestamp is suppressed at savefig time -- see main).
matplotlib.rcParams["svg.hashsalt"] = "agentic-analyst-judge-benchmark"

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "telco-customer-churn.csv"
OUT_JSON = REPO_ROOT / "results" / "judge_benchmark.json"
OUT_PNG = REPO_ROOT / "results" / "figures" / "judge_benchmark.png"
OUT_SVG = REPO_ROOT / "results" / "figures" / "judge_benchmark.svg"

REL_TOL = 0.01
ABS_TOL = 0.01

# House chart palette (warm paper + ink tokens), matching the other pieces.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CLASS_COLORS = {
    "value_swap": "#2a78d6",
    "claim_mismatch": "#1baf7a",
    "fabricated_evidence": "#eda100",
    "alias_dodge": "#a15bd0",
}
CLASS_LABELS = {
    "value_swap": "value-swap",
    "claim_mismatch": "claim/value mismatch",
    "fabricated_evidence": "fabricated evidence",
    "alias_dodge": "alias-dodge",
}

# Base findings: (label, evidence SQL, claim template with one {n}, is-count).
# A deliberate spread of magnitudes -- low rates (~0.16-0.43), mid means
# (~32-65), and large counts (11-3096) -- so the catch-rate curve reveals the
# judge's abs_tol=0.01 floor honestly (a small distortion of a small value can
# fall under it) rather than by accident of a single value scale.
_BASE_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "churn_rate",
        "SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data",
        "The overall churn rate is {n} (share of customers with Churn = Yes).",
        False,
    ),
    (
        "senior_fraction",
        "SELECT avg(SeniorCitizen) FROM data",
        "The share of senior citizens is {n}.",
        False,
    ),
    (
        "m2m_churn_rate",
        "SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data "
        "WHERE Contract='Month-to-month'",
        "Among month-to-month customers, the churn rate is {n}.",
        False,
    ),
    (
        "avg_tenure",
        "SELECT avg(tenure) FROM data",
        "The average customer tenure is {n} months.",
        False,
    ),
    (
        "avg_monthly_charges",
        "SELECT avg(MonthlyCharges) FROM data",
        "The average monthly charge is {n} dollars.",
        False,
    ),
    (
        "blank_total_charges",
        "SELECT sum(CASE WHEN trim(TotalCharges)='' THEN 1 ELSE 0 END) FROM data",
        "{n} rows have a blank (whitespace-only) TotalCharges value.",
        True,
    ),
    (
        "count_senior",
        "SELECT sum(SeniorCitizen) FROM data",
        "{n} customers are senior citizens.",
        True,
    ),
    (
        "count_churned",
        "SELECT count(*) FROM data WHERE Churn='Yes'",
        "{n} customers churned in total (Churn = Yes).",
        True,
    ),
    (
        "count_fiber",
        "SELECT count(*) FROM data WHERE InternetService='Fiber optic'",
        "{n} customers subscribe to fiber-optic internet.",
        True,
    ),
)


def _true_value(sql: str, as_int: bool) -> float:
    result = query_sql(sql, CSV_PATH)
    if not result.ok:
        raise SystemExit(f"base query failed: {sql!r}: {result.error}")
    raw = single_value(result.stdout)
    return int(raw) if as_int else float(raw)


def build_bases() -> list[BaseFinding]:
    """Recompute each base finding's true value from the real CSV and assert it
    verifies untampered -- a broken base would invalidate every catch built on
    it."""
    bases: list[BaseFinding] = []
    with TemporaryDirectory(prefix="agentic-analyst-bench-base-") as tmp:
        workdir = Path(tmp)
        for label, sql, template, as_int in _BASE_SPECS:
            base = BaseFinding(
                label=label,
                evidence_sql=sql,
                claim_template=template,
                as_int=as_int,
                value=_true_value(sql, as_int),
            )
            judged = verify_finding(
                honest_finding(base), CSV_PATH, workdir, rel_tol=REL_TOL, abs_tol=ABS_TOL
            )
            if judged.verdict != "verified":
                raise SystemExit(
                    f"base finding {label!r} does not verify untampered "
                    f"({judged.verdict}: {judged.detail}) -- benchmark aborted"
                )
            bases.append(base)
    return bases


def _rate(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


def run_benchmark(cases: list[PlantedCase]) -> tuple[dict, float, float]:
    """Judge every case. Returns (results_by_case, first_case_seconds,
    total_seconds). ``results_by_case`` maps case index -> (case, verdict)."""
    per_case: dict[int, tuple[PlantedCase, str]] = {}
    first_case_seconds = 0.0
    total_start = time.perf_counter()
    with TemporaryDirectory(prefix="agentic-analyst-bench-") as tmp:
        workdir = Path(tmp)
        for i, case in enumerate(cases):
            t0 = time.perf_counter()
            judged = verify_finding(
                case.finding, CSV_PATH, workdir, rel_tol=REL_TOL, abs_tol=ABS_TOL
            )
            if i == 0:
                first_case_seconds = time.perf_counter() - t0
            per_case[i] = (case, judged.verdict)
    return per_case, first_case_seconds, time.perf_counter() - total_start


def aggregate(per_case: dict[int, tuple[PlantedCase, str]]) -> dict:
    tp = fn = fp = tn = 0
    # Of the caught tampered cases, how many used the mechanism the attack class
    # is meant to trip (CATCH_VERDICT) rather than some incidental verdict. This
    # is a diagnostic only -- it never enters catch-rate/recall, which stay a
    # bare `verdict != "verified"`.
    mechanism_matches = 0
    by_class_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [caught, n]
    by_mag_counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    by_cell: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    missed: list[dict] = []

    for case, verdict in per_case.values():
        caught = verdict != "verified"
        if case.tampered:
            if caught:
                tp += 1
                if verdict == CATCH_VERDICT[case.attack_class]:
                    mechanism_matches += 1
            else:
                fn += 1
                missed.append(
                    {
                        "base_label": case.base_label,
                        "attack_class": case.attack_class,
                        "magnitude": case.magnitude,
                        "direction": case.direction,
                    }
                )
            by_class_counts[case.attack_class][0] += int(caught)
            by_class_counts[case.attack_class][1] += 1
            by_mag_counts[case.magnitude][0] += int(caught)
            by_mag_counts[case.magnitude][1] += 1
            cell = by_cell[case.attack_class][f"{case.magnitude:g}"]
            cell[0] += int(caught)
            cell[1] += 1
        else:
            if caught:
                fp += 1
            else:
                tn += 1

    overall = {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "false_positive_rate": _rate(fp, fp + tn),
    }
    by_class = {
        cls: {"caught": c[0], "n": c[1], "recall": _rate(c[0], c[1])}
        for cls, c in ((k, by_class_counts[k]) for k in ATTACK_CLASSES)
    }
    by_magnitude = {
        f"{mag:g}": {"caught": by_mag_counts[mag][0], "n": by_mag_counts[mag][1],
                     "recall": _rate(*by_mag_counts[mag])}
        for mag in DEFAULT_MAGNITUDES
    }
    by_class_magnitude = {
        cls: {
            f"{mag:g}": {
                "caught": by_cell[cls][f"{mag:g}"][0],
                "n": by_cell[cls][f"{mag:g}"][1],
                "catch_rate": _rate(*by_cell[cls][f"{mag:g}"]),
            }
            for mag in DEFAULT_MAGNITUDES
        }
        for cls in ATTACK_CLASSES
    }
    mechanism_match = {
        "caught": tp,
        "used_intended_mechanism": mechanism_matches,
        "rate": _rate(mechanism_matches, tp),
    }
    return {
        "overall": overall,
        "mechanism_match": mechanism_match,
        "by_class": by_class,
        "by_magnitude": by_magnitude,
        "by_class_magnitude": by_class_magnitude,
        "missed_cases": missed,
    }


def build_meta(bases: list[BaseFinding], cases: list[PlantedCase]) -> dict:
    n_tampered = sum(c.tampered for c in cases)
    return {
        "generated_by": "scripts/judge_benchmark.py",
        "dataset": "telco-customer-churn (7,043 rows)",
        "n_cases": len(cases),
        "n_tampered": n_tampered,
        "n_clean_controls": len(cases) - n_tampered,
        "attack_classes": list(ATTACK_CLASSES),
        "magnitudes": list(DEFAULT_MAGNITUDES),
        "tolerances": {"rel_tol": REL_TOL, "abs_tol": ABS_TOL},
        "deterministic": True,
        "rng": "none -- exhaustive factorial enumeration",
        "evidence_kind": "SQL only (the Python-evidence path is covered by unit tests)",
        "negative_direction_note": (
            "negative direction is applied only for magnitudes < 100%; at >=100% it "
            "would flip the number's sign rather than distort its magnitude"
        ),
        "fabricated_and_alias_are_magnitude_invariant": (
            "fabricated_evidence and alias_dodge are rejected before any value compare "
            "(the evidence cannot have touched the data), so their catch rate is 100% "
            "at every magnitude by construction"
        ),
        "baseline_excluded": (
            "verify_baseline (the ~seconds/case retrain) is excluded -- these attack "
            "classes target finding-level claim verification, which never retrains; "
            "baseline tamper-detection is covered in tests/test_judge.py"
        ),
        "base_findings": [
            {"label": b.label, "evidence_sql": b.evidence_sql, "true_value": b.value}
            for b in bases
        ],
    }


def headline(agg: dict) -> str:
    """One honest sentence derived from the measured numbers."""
    ge20_caught = sum(
        agg["by_magnitude"][f"{m:g}"]["caught"] for m in DEFAULT_MAGNITUDES if m >= 0.20
    )
    ge20_n = sum(agg["by_magnitude"][f"{m:g}"]["n"] for m in DEFAULT_MAGNITUDES if m >= 0.20)
    recall_5 = agg["by_magnitude"]["0.05"]["recall"]
    ge20 = _rate(ge20_caught, ge20_n)
    return (
        f"The judge catches {ge20:.0%} of distortions >=20%; "
        f"at 5% catch-rate dips to {recall_5:.0%} (small distortions of small "
        f"values fall under the abs_tol={ABS_TOL} floor)."
    )


def make_figure(agg: dict, meta: dict) -> Figure:
    """Grouped bar chart: catch-rate (%) per magnitude, one bar per attack
    class. Title states the measured finding."""
    fig = Figure(figsize=(10, 5.8), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    magnitudes = list(DEFAULT_MAGNITUDES)
    n_classes = len(ATTACK_CLASSES)
    group_width = 0.8
    bar_width = group_width / n_classes

    for ci, cls in enumerate(ATTACK_CLASSES):
        rates = [agg["by_class_magnitude"][cls][f"{m:g}"]["catch_rate"] * 100 for m in magnitudes]
        offsets = [x + (ci - (n_classes - 1) / 2) * bar_width for x in range(len(magnitudes))]
        ax.bar(offsets, rates, bar_width, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])
        for xo, r in zip(offsets, rates, strict=True):
            if r < 99.5:  # annotate only the bars that fall short of a clean sweep
                ax.text(xo, r + 1.5, f"{r:.0f}%", ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY)

    ax.axhline(100, color=BASELINE, linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels([f"+-{int(m * 100)}%" for m in magnitudes], color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("catch-rate (% of planted lies flagged)", color=INK_MUTED, fontsize=10)
    ax.set_xlabel("distortion magnitude", color=INK_MUTED, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax.set_title(headline(agg), color=INK_PRIMARY, fontsize=12.5, loc="left", pad=26)
    ov = agg["overall"]
    ax.text(
        0, 1.02,
        f"{meta['n_tampered']} planted lies x {len(ATTACK_CLASSES)} attack classes - "
        f"precision {ov['precision']:.0%}, FPR {ov['false_positive_rate']:.0%} "
        f"over {meta['n_clean_controls']} honest controls",
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom",
    )
    return fig


def main() -> None:
    bases = build_bases()
    cases = generate_cases(bases)

    per_case, first_s, total_s = run_benchmark(cases)
    agg = aggregate(per_case)
    meta = build_meta(bases, cases)

    artifact = {"meta": meta, **agg}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(artifact, indent=2) + "\n")

    fig = make_figure(agg, meta)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    # metadata={"Date": None} drops the live <dc:date> stamp so the SVG is
    # byte-stable across runs (the fixed svg.hashsalt above handles element IDs).
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=SURFACE, metadata={"Date": None})

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")
    print()
    print(headline(agg))
    print()
    print(f"{'class':<22} " + " ".join(f"{f'+-{int(m * 100)}%':>7}" for m in DEFAULT_MAGNITUDES))
    for cls in ATTACK_CLASSES:
        row = " ".join(
            f"{agg['by_class_magnitude'][cls][f'{m:g}']['catch_rate']:>7.0%}"
            for m in DEFAULT_MAGNITUDES
        )
        print(f"{cls:<22} {row}")
    ov = agg["overall"]
    mm = agg["mechanism_match"]
    print()
    print(
        f"overall: precision={ov['precision']:.0%} recall={ov['recall']:.0%} "
        f"FPR={ov['false_positive_rate']:.0%} "
        f"(TP={ov['true_positives']} FN={ov['false_negatives']} "
        f"FP={ov['false_positives']} TN={ov['true_negatives']})"
    )
    print(
        f"mechanism-match: {mm['used_intended_mechanism']}/{mm['caught']} caught lies "
        f"used the intended detection mechanism ({mm['rate']:.0%})"
    )
    print()
    # Wall-clock: stdout only, never written into the artifact numbers.
    print(
        f"timing: first case {first_s * 1000:.0f} ms; "
        f"{len(cases)} cases in {total_s:.1f} s "
        f"(~{total_s / len(cases) * 1000:.0f} ms/case)"
    )


if __name__ == "__main__":
    main()
