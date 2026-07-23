"""Quantified judge benchmark (Task C1).

Plants controlled lies into a set of verified base findings -- across every
attack class in ``planting.py``. The four magnitude-swept classes (value-swap,
claim/value mismatch, fabricated evidence, alias-dodge) are planted at four
relative magnitudes (+-5%, +-20%, +-50%, +-200%); the fifth, population_switch
(Task C3: an honest subpopulation number under a full-population claim), has no
magnitude dimension and is planted once per base that declares a subset query.
The real judge (`verify_finding`) is run on each, measuring how reliably it
catches them. Replaces anecdotal "the judge caught the demo lie" confidence
with a reproducible precision/recall/catch-rate table + chart.

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
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
from matplotlib.figure import Figure

from agentic_analyst.benchmark_stats import aggregate
from agentic_analyst.judge import verify_finding
from agentic_analyst.planting import (
    ATTACK_CLASSES,
    DEFAULT_MAGNITUDES,
    SINGLE_CASE_CLASSES,
    BaseFinding,
    PlantedCase,
    generate_cases,
    honest_finding,
)
from agentic_analyst.tools import query_sql, single_value

# The magnitude-swept attack classes (everything except the single-case
# population_switch): these are the bars in the catch-rate-by-magnitude chart
# and the rows of the class x magnitude table. population_switch has no
# magnitude dimension and is reported separately from `by_class`.
SWEPT_CLASSES = tuple(c for c in ATTACK_CLASSES if c not in SINGLE_CASE_CLASSES)

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

# Base findings: (label, evidence SQL, claim template with one {n}, is-count,
# subset SQL | None). A deliberate spread of magnitudes -- low rates
# (~0.16-0.43), mid means (~32-65), and large counts (11-3096) -- so the
# catch-rate curve reveals the judge's abs_tol=0.01 floor honestly (a small
# distortion of a small value can fall under it) rather than by accident of a
# single value scale.
#
# The 5th element is the honest subpopulation query for the population_switch
# class (None = no such case for that base): a real filter whose column/value
# the full-population claim template above deliberately does NOT name, so the
# planted case is "full-population framing over a genuinely-correct subset
# number" -- caught only by the C3 population-mismatch heuristic. The three
# subsets below (month-to-month churn, fiber-optic charge, two-year tenure) all
# hide their filter from their claim; note m2m_churn_rate / count_churned /
# count_fiber below carry no subset because their OWN claims disclose their
# filter -- they double as the heuristic's false-positive controls.
_BASE_SPECS: tuple[tuple[str, str, str, bool, str | None], ...] = (
    (
        "churn_rate",
        "SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data",
        "The overall churn rate is {n} (share of customers with Churn = Yes).",
        False,
        "SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data "
        "WHERE Contract='Month-to-month'",
    ),
    (
        "senior_fraction",
        "SELECT avg(SeniorCitizen) FROM data",
        "The share of senior citizens is {n}.",
        False,
        None,
    ),
    (
        "m2m_churn_rate",
        "SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data "
        "WHERE Contract='Month-to-month'",
        "Among month-to-month customers, the churn rate is {n}.",
        False,
        None,
    ),
    (
        "avg_tenure",
        "SELECT avg(tenure) FROM data",
        "The average customer tenure is {n} months.",
        False,
        "SELECT avg(tenure) FROM data WHERE Contract='Two year'",
    ),
    (
        "avg_monthly_charges",
        "SELECT avg(MonthlyCharges) FROM data",
        "The average monthly charge is {n} dollars.",
        False,
        "SELECT avg(MonthlyCharges) FROM data WHERE InternetService='Fiber optic'",
    ),
    (
        "blank_total_charges",
        "SELECT sum(CASE WHEN trim(TotalCharges)='' THEN 1 ELSE 0 END) FROM data",
        "{n} rows have a blank (whitespace-only) TotalCharges value.",
        True,
        None,
    ),
    (
        "count_senior",
        "SELECT sum(SeniorCitizen) FROM data",
        "{n} customers are senior citizens.",
        True,
        None,
    ),
    (
        "count_churned",
        "SELECT count(*) FROM data WHERE Churn='Yes'",
        "{n} customers churned in total (Churn = Yes).",
        True,
        None,
    ),
    (
        "count_fiber",
        "SELECT count(*) FROM data WHERE InternetService='Fiber optic'",
        "{n} customers subscribe to fiber-optic internet.",
        True,
        None,
    ),
)


def _true_value(sql: str, as_int: bool) -> float:
    result = query_sql(sql, CSV_PATH)
    if not result.ok:
        raise SystemExit(f"base query failed: {sql!r}: {result.error}")
    raw = single_value(result.stdout)
    return int(raw) if as_int else float(raw)


def build_bases() -> list[BaseFinding]:
    """Recompute each base finding's true value (and its subset value, where a
    subset query is declared) from the real CSV and assert the honest finding
    verifies untampered -- a broken base would invalidate every catch built on
    it."""
    bases: list[BaseFinding] = []
    with TemporaryDirectory(prefix="agentic-analyst-bench-base-") as tmp:
        workdir = Path(tmp)
        for label, sql, template, as_int, subset_sql in _BASE_SPECS:
            base = BaseFinding(
                label=label,
                evidence_sql=sql,
                claim_template=template,
                as_int=as_int,
                value=_true_value(sql, as_int),
                subset_evidence_sql=subset_sql,
                subset_value=_true_value(subset_sql, as_int) if subset_sql else None,
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
        "population_switch_note": (
            "population_switch has no magnitude dimension: one case per base that "
            "declares a subset query, tagged magnitude 0.0, an honest subset number "
            "under a full-population claim -- caught by the C3 population-mismatch "
            "heuristic as 'unverified'. Its catch rate lives in by_class (NOT "
            "by_magnitude / by_class_magnitude, whose per-magnitude cells for it are "
            "empty by construction). The heuristic's own false-positive rate is the "
            "overall FPR: a false positive could only be a clean control it wrongly "
            "flagged, including the 3 controls whose claims DO disclose their WHERE "
            "filter (m2m_churn_rate, count_churned, count_fiber)."
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
    ge20 = ge20_caught / ge20_n if ge20_n else 0.0
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
    n_classes = len(SWEPT_CLASSES)
    group_width = 0.8
    bar_width = group_width / n_classes

    for ci, cls in enumerate(SWEPT_CLASSES):
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
    ps = agg["by_class"]["population_switch"]
    ax.text(
        0, 1.02,
        f"{meta['n_tampered']} planted lies - precision {ov['precision']:.0%}, "
        f"FPR {ov['false_positive_rate']:.0%} over {meta['n_clean_controls']} honest "
        f"controls - population-switch heuristic caught {ps['caught']}/{ps['n']} "
        "(shown separately: no magnitude dimension)",
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
    for cls in SWEPT_CLASSES:
        row = " ".join(
            f"{agg['by_class_magnitude'][cls][f'{m:g}']['catch_rate']:>7.0%}"
            for m in DEFAULT_MAGNITUDES
        )
        print(f"{cls:<22} {row}")
    ps = agg["by_class"]["population_switch"]
    print(
        f"population_switch (single-case, no magnitude sweep): "
        f"caught {ps['caught']}/{ps['n']} = {ps['recall']:.0%}"
    )
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
