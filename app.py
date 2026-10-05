"""Streamlit report UI (M5): renders the agent's report with inline judge
verdict badges (verified / unverified / contradicted).

No `ANTHROPIC_API_KEY` is needed: the report comes from `run_agent` driven by
`FakeLLM` (a deterministic, scripted stand-in for the real Anthropic API --
see agent.py) run live against the real telco CSV, so every number on screen
is computed by the real M1 tools, not invented. The real-Claude path
(`AnthropicClient`) needs an API key -- Needs owner.

The sidebar has two "demo the judge" toggles, each planting a different class
of lie into the churn-rate finding before re-judging, so the honest all-green
report and both of the judge's failure modes are visible from the same UI:
- `inject_planted_false_claim` overstates the claim/value but leaves the
  evidence honest -- the judge's recompute genuinely disagrees, so this comes
  back `contradicted`.
- `inject_consistent_lie` fabricates the claim, value, AND evidence together
  (evidence that never reads the real data, e.g. `SELECT 0.75 AS
  churn_rate`) -- a self-proving lie the value<->evidence compare alone
  cannot see. The judge catches this a *different* way: the
  evidence-plausibility precondition rejects it before ever comparing
  values, so it comes back `unverified: evidence_does_not_touch_data`, not
  `contradicted` -- see judge.py's `verify_finding` docstring and README's
  "What the judge does NOT catch".

Both are plain functions (not buried in a Streamlit callback) so they -- and
the resulting verdicts -- are directly unit-testable; see tests/test_app.py.

Run: uv run streamlit run app.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from agentic_analyst.agent import CHURN_RATE_QUERY, FakeLLM, run_agent
from agentic_analyst.judge import JudgedBaseline, JudgedFinding, JudgedReport, verify_report
from agentic_analyst.report import Report

REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "data" / "telco-customer-churn.csv"

# The demo lie: overstate the churn rate. Matches the project plan's worked example
# (claimed 0.75 vs. the real ~0.2654) so the README and the UI tell the same
# story. Clearly labeled in the claim text as an injected demonstration --
# never presented as a real number from the data.
TAMPERED_VALUE = 0.75
TAMPERED_CLAIM = (
    "[DEMO INJECTED LIE] 75% of customers churned -- almost everyone leaves! "
    "(planted for this demo; the judge below recomputes the real figure)"
)

# The second demo lie: fabricate the evidence too, not just the claim/value.
# `CONSISTENT_LIE_EVIDENCE` never references the `data` view (no FROM clause
# at all) -- review notes finding A-1, attack 1: evidence that recomputes to
# exactly its own literal always "matches" whatever value is claimed next to
# it, a lie that carries its own proof. `judge.py`'s evidence-plausibility
# precondition (`_evidence_implausibility_reason`) is what catches this, not
# the value comparison -- see `inject_consistent_lie` below.
CONSISTENT_LIE_VALUE = 0.75
CONSISTENT_LIE_EVIDENCE = "SELECT 0.75 AS churn_rate"
CONSISTENT_LIE_CLAIM = (
    "[DEMO INJECTED LIE, EVIDENCE FABRICATED TOO] 75% of customers churned -- "
    "almost everyone leaves! (planted for this demo; even the evidence below "
    "is fake -- it never reads the real data)"
)

STATUS_META: dict[str, tuple[str, str, str]] = {
    # verdict -> (color, icon, label). Colors are the fixed status palette --
    # reserved for verdicts only, never reused for anything categorical.
    "verified": ("#0ca30c", "✓", "verified"),
    "contradicted": ("#d03b3b", "✗", "contradicted"),
    "unverified": ("#fab219", "?", "unverified"),
}

_INK_SECONDARY = "#52514e"

# Verdict-independent accent for the narrow-subset provenance note (same hex
# as STATUS_META["unverified"]'s amber, but a SEPARATE constant on purpose:
# STATUS_META's colors are reserved for verdicts only -- this caption can sit
# on a `verified` or `contradicted` card, and reusing STATUS_META's own
# color there would make a green "verified" badge read as downgraded by an
# amber accent underneath it, exactly the misreading
# `_provenance_caption_html` exists to avoid).
_ACCENT_AMBER = "#fab219"


def inject_planted_false_claim(report: Report) -> Report:
    """Return a copy of `report` with the churn-rate finding's claimed value
    overstated to `TAMPERED_VALUE`, for the "demo the judge" toggle.

    Pure and side-effect-free (uses `dataclasses.replace`, never mutates
    `report`) so it is directly testable without Streamlit. Only the
    finding whose evidence is the churn-rate query is touched; every other
    finding and the baseline pass through unchanged. This never touches the
    underlying CSV or the evidence query itself -- the judge re-executes
    that same query against the real data, which is exactly why it catches
    the lie.
    """
    tampered_findings = [
        dataclasses.replace(finding, claim=TAMPERED_CLAIM, value=TAMPERED_VALUE)
        if finding.evidence_sql_or_code == CHURN_RATE_QUERY
        else finding
        for finding in report.findings
    ]
    return dataclasses.replace(report, findings=tampered_findings)


def inject_consistent_lie(report: Report) -> Report:
    """Return a copy of `report` with the churn-rate finding's claim, value,
    AND evidence all fabricated together, for the second "demo the judge"
    toggle.

    Unlike `inject_planted_false_claim` (which keeps the real evidence and
    only lies about `claim`/`value`, so the recompute genuinely disagrees --
    `contradicted`), this plants a self-consistent lie: `CONSISTENT_LIE_
    EVIDENCE` is `SELECT 0.75 AS churn_rate` -- no `FROM data` at all -- so
    it recomputes to exactly its own literal and would "match" any value
    claimed next to it. `verify_finding`'s value comparison alone cannot see
    this; the judge instead catches it via the evidence-plausibility
    precondition, returning `unverified: evidence_does_not_touch_data`
    *before* any value is ever compared (see judge.py). Pure and
    side-effect-free (uses `dataclasses.replace`, never mutates `report`),
    mirroring `inject_planted_false_claim`.
    """
    tampered_findings = [
        dataclasses.replace(
            finding,
            claim=CONSISTENT_LIE_CLAIM,
            value=CONSISTENT_LIE_VALUE,
            evidence_sql_or_code=CONSISTENT_LIE_EVIDENCE,
        )
        if finding.evidence_sql_or_code == CHURN_RATE_QUERY
        else finding
        for finding in report.findings
    ]
    return dataclasses.replace(report, findings=tampered_findings)


@st.cache_data(show_spinner="Running the FakeLLM demo agent over the telco CSV...")
def _run_demo_report() -> Report:
    with TemporaryDirectory(prefix="agentic-analyst-app-agent-") as tmp:
        return run_agent(FakeLLM(CSV_PATH), CSV_PATH, Path(tmp))


@st.cache_data(show_spinner="Judge is independently re-verifying every claim...")
def get_judged_report(tamper: bool, consistent_lie: bool) -> tuple[Report, JudgedReport]:
    """Build the (report, judged_report) pair for the given toggle state.

    If both toggles are on, `consistent_lie` wins: it fully overwrites the
    same churn-rate finding `inject_planted_false_claim` would have touched,
    so applying both in sequence would just silently discard whichever ran
    first -- the sidebar documents this precedence rather than leaving it
    implicit.

    Cached per (tamper, consistent_lie) so flipping either toggle back and
    forth in the running app doesn't re-run the sandboxed subprocess
    pipeline (SQL profiling + a fresh LogisticRegression retrain in the
    judge) every time.
    """
    report = _run_demo_report()
    if consistent_lie:
        report = inject_consistent_lie(report)
    elif tamper:
        report = inject_planted_false_claim(report)
    with TemporaryDirectory(prefix="agentic-analyst-app-judge-") as tmp:
        judged = verify_report(report, CSV_PATH, Path(tmp))
    return report, judged


def _inject_style() -> None:
    st.markdown(
        f"""
        <style>
        .aa-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            white-space: nowrap;
        }}
        .aa-stat {{
            text-align: center;
            padding: 14px 8px;
            border-radius: 10px;
            border: 1px solid rgba(137, 135, 129, 0.4);
        }}
        .aa-stat-value {{
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .aa-stat-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {_INK_SECONDARY};
        }}
        .aa-value-box {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.95rem;
            margin-right: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge_html(verdict: str) -> str:
    color, icon, label = STATUS_META[verdict]
    text_color = "#0b0b0b" if verdict == "unverified" else "#fcfcfb"
    return (
        f'<span class="aa-badge" style="background:{color};color:{text_color};">'
        f"{icon} {label}</span>"
    )


def _value_box_html(label: str, value: object, color: str) -> str:
    return (
        f'<span class="aa-value-box" style="background:{color}22;border:1px solid {color};'
        f'color:{color};"><b>{label}:</b> {value}</span>'
    )


def _evidence_language(evidence: str) -> str:
    leading = evidence.strip().split(None, 1)[0].upper() if evidence.strip() else ""
    return "sql" if leading in ("SELECT", "WITH") else "python"


def render_summary(summary: dict[str, int]) -> None:
    cols = st.columns(3)
    for col, key in zip(cols, ("verified", "unverified", "contradicted"), strict=True):
        color, icon, label = STATUS_META[key]
        count = summary.get(key, 0)
        col.markdown(
            f'<div class="aa-stat" style="border-color:{color};">'
            f'<div class="aa-stat-value" style="color:{color};">{icon} {count}</div>'
            f'<div class="aa-stat-label">{label}</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def _provenance_caption_html(
    evidence_row_count: int, total_rows: int | None, provenance_note: str | None
) -> str:
    """Render the population-provenance line for a finding card (Task A5's
    `evidence_row_count`/`provenance_note` -- pure metadata, computed
    independently of `verdict`, see judge.py). Ordinary (non-narrow)
    provenance is a plain, muted caption: "this claim rests on N of M rows".
    `provenance_note == "narrow_subset"` gets `_ACCENT_AMBER` (not a loud
    st.warning box, and NOT `STATUS_META["unverified"]`'s color -- see
    `_ACCENT_AMBER`'s comment for why: this caption can sit underneath a
    `verified` or `contradicted` badge, and STATUS_META's colors are
    reserved for verdicts only) -- a nudge to read the claim's wording
    against the population size, not a verdict downgrade; a narrow subset
    is not itself evidence of a lie (review notes attack 4 remains a
    documented, separate gap -- this only surfaces the number).
    """
    ratio = f" of {total_rows}" if total_rows is not None else ""
    text = f"Evidence touched {evidence_row_count}{ratio} rows."
    if provenance_note != "narrow_subset":
        return f'<span style="color:{_INK_SECONDARY};font-size:0.85rem;">{text}</span>'
    text += (
        " Narrow subset (<10% of the dataset) -- read the claim's wording "
        "against this population."
    )
    return f'<span style="color:{_ACCENT_AMBER};font-size:0.85rem;">⚠️ {text}</span>'


def render_finding_card(judged_finding: JudgedFinding, total_rows: int | None = None) -> None:
    finding = judged_finding.finding
    st.markdown(f"**Claim:** {finding.claim}")
    st.code(finding.evidence_sql_or_code, language=_evidence_language(finding.evidence_sql_or_code))
    st.markdown(_badge_html(judged_finding.verdict), unsafe_allow_html=True)

    if judged_finding.verdict == "contradicted":
        st.markdown(
            _value_box_html("Claimed", finding.value, STATUS_META["contradicted"][0])
            + _value_box_html("Recomputed (real)", judged_finding.recomputed_value, "#0ca30c"),
            unsafe_allow_html=True,
        )
    else:
        # `unverified` (e.g. the consistent-lie toggle) legitimately has
        # recomputed_value=None -- the module invariant judge.py documents
        # ("unverified => recomputed_value is None"). Render that as "n/a",
        # matching render_baseline_card's existing None-handling, not the
        # literal string "None".
        recomputed_value = judged_finding.recomputed_value
        recomputed_display = recomputed_value if recomputed_value is not None else "n/a"
        st.markdown(
            f'<span style="color:{_INK_SECONDARY};">Claimed: <code>{finding.value}</code> '
            f"&nbsp;&middot;&nbsp; Recomputed: <code>{recomputed_display}</code>"
            "</span>",
            unsafe_allow_html=True,
        )
    st.caption(judged_finding.detail)

    if judged_finding.evidence_row_count is not None:
        st.markdown(
            _provenance_caption_html(
                judged_finding.evidence_row_count, total_rows, judged_finding.provenance_note
            ),
            unsafe_allow_html=True,
        )


def render_baseline_card(judged_baseline: JudgedBaseline) -> None:
    baseline = judged_baseline.baseline
    st.markdown(f"**Baseline model:** `{baseline.model}`")
    st.code(f"features = {baseline.features}", language="python")
    st.markdown(_badge_html(judged_baseline.verdict), unsafe_allow_html=True)

    claimed = f"{baseline.metric_value:.4f}"
    recomputed_value = judged_baseline.recomputed_value
    recomputed = f"{recomputed_value:.4f}" if recomputed_value is not None else "n/a"
    contradicted_color = STATUS_META["contradicted"][0]
    if judged_baseline.verdict == "contradicted":
        st.markdown(
            _value_box_html(f"Claimed {baseline.metric_name}", claimed, contradicted_color)
            + _value_box_html(f"Recomputed {baseline.metric_name} (real)", recomputed, "#0ca30c"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{_INK_SECONDARY};">Claimed {baseline.metric_name}: '
            f"<code>{claimed}</code> &nbsp;&middot;&nbsp; "
            f"Recomputed: <code>{recomputed}</code></span>",
            unsafe_allow_html=True,
        )
    if baseline.notes:
        st.caption(baseline.notes)
    st.caption(judged_baseline.detail)


def main() -> None:
    st.set_page_config(
        page_title="Agentic Analyst -- Honest Report", page_icon="\U0001f50e", layout="wide"
    )
    _inject_style()

    st.title("\U0001f50e Agentic Analyst")
    st.caption(
        "An LLM agent runs EDA + a baseline model on the telco churn dataset. "
        "A judge layer recomputes every claim from its own evidence and flags "
        "mismatches -- it catches a misreported number or evidence that never "
        "touches the data, not a consistently fabricated result (see the two "
        "demo toggles below)."
    )
    st.info(
        "**Demo mode:** this report comes from the deterministic `FakeLLM` agent "
        "(scripted tool-call sequence, but every number is computed live by the real "
        "sandboxed tools against the real telco CSV) -- no `ANTHROPIC_API_KEY` needed. "
        "The real-Claude agent path (`AnthropicClient`) needs an API key in `.env` -- Needs owner.",
        icon="ℹ️",
    )

    with st.sidebar:
        st.header("Demo controls")
        tamper = st.toggle(
            "\U0001f52c Inject a planted false claim (demo the judge)",
            value=False,
            help=(
                "DEMONSTRATION ONLY. When on, overstates the churn-rate finding's claim "
                "and value while leaving its evidence honest, so you can watch the judge "
                "catch it: a red 'contradicted' badge with the claimed value next to the "
                "real recomputed one. This demos ONE of the judge's two failure classes "
                "-- see the second toggle below for the other. The planted claim is "
                "clearly labeled in the report as an injected demo lie -- it is never "
                "presented as a real number. (Overridden by the toggle below if both "
                "are on.)"
            ),
        )
        consistent_lie = st.toggle(
            "\U0001f52c Inject a *consistent* lie (evidence fabricated too)",
            value=False,
            help=(
                "DEMONSTRATION ONLY. Fabricates the churn-rate finding's claim, value, "
                "AND evidence together -- the evidence becomes `SELECT 0.75 AS "
                "churn_rate`, which never reads the real data. The judge cannot catch "
                "this by comparing values (the fake evidence always 'matches' whatever "
                "is claimed next to it); it catches it a different way, by rejecting the "
                "evidence itself BEFORE any value is compared: a yellow 'unverified' "
                "badge with reason `evidence_does_not_touch_data`, not 'contradicted'. "
                "Takes precedence over the toggle above if both are on."
            ),
        )
        st.divider()
        st.file_uploader(
            "Upload your own CSV",
            type="csv",
            disabled=True,
            help=(
                "Disabled in this demo. FakeLLM's tool-call sequence is scripted for the "
                "telco schema and can't analyze an arbitrary CSV -- that needs the real "
                "Claude agent (ANTHROPIC_API_KEY, real tool-use loop). This demo always "
                "analyzes the committed telco-customer-churn.csv."
            ),
        )

    _report, judged = get_judged_report(tamper, consistent_lie)

    if consistent_lie:
        st.warning(
            "\U0001f52c **Demo tamper active (consistent lie)** -- the churn-rate "
            "finding's claim, value, AND evidence have all been fabricated together. "
            "The judge flags it 'unverified: evidence_does_not_touch_data' below, not "
            "'contradicted' -- it never claims to have disproven the number, only that "
            "this evidence could not have legitimately produced any number about the "
            "real data.",
            icon="\U0001f52c",
        )
    elif tamper:
        st.warning(
            "\U0001f52c **Demo tamper active** -- the churn-rate finding below has been "
            "deliberately overstated to demonstrate the judge catching a lie. That "
            "claimed value is NOT real; the recomputed value next to it is.",
            icon="\U0001f52c",
        )

    st.subheader("Summary")
    render_summary(judged.summary)

    st.subheader(f"Dataset: {judged.dataset.name}")
    st.caption(f"{judged.dataset.rows} rows × {judged.dataset.cols} columns")

    st.subheader("Findings")
    for judged_finding in judged.findings:
        with st.container(border=True):
            render_finding_card(judged_finding, judged.dataset.rows)

    st.subheader("Baseline")
    with st.container(border=True):
        render_baseline_card(judged.baseline)


if __name__ == "__main__":
    main()
