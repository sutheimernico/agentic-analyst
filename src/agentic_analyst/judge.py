"""Judge / verification layer (M4): independently re-verifies every factual
claim in a `Report` against the real data.

DESIGN DECISION -- read before extending. The judge does NOT use an LLM to
extract or check claims, and that is deliberate, not a shortcut. Every
`Finding` already carries its own `evidence_sql_or_code` and the single
`value` it claims (see report.py and agent.py's one-query-per-finding
convention). That means the judge can simply re-execute that exact evidence
through the same M1 tools (`query_sql`/`run_python`) the agent used, extract
the one recomputed number, and compare it to the claimed value within a
tolerance. This is strictly stronger than an LLM-based free-text claim
checker: it cannot hallucinate a verdict, it needs no API key, it is fully
deterministic, and it is directly testable (see tests/test_judge.py, which
plants false claims and proves the judge catches them). An LLM-based
extractor for less-structured, free-text report claims is explicitly out of
scope / future work -- it would be a *weaker* verification method for a
report this structured, and there is no second use case yet to justify the
extra machinery (YAGNI).

`verify_baseline` follows the same recompute-don't-trust principle, with one
honest limitation: `Baseline` only records model description, features,
metric name/value -- not the full training recipe (target definition, split
strategy, preprocessing). The judge can only faithfully retrain the one
recipe this project's agent uses (LogisticRegression, stratified 80/20 split
on `Churn`, StandardScaler), parameterized by the report's declared features
and the random_state/max_iter parsed out of `baseline.model`. A baseline it
doesn't recognize is marked `unverified` with a clear reason -- never
rubber-stamped. The retraining code below is deliberately NOT shared with
agent.py's `BASELINE_CODE_TEMPLATE`: an independent check should not run the
exact same code path that produced the number in the first place, or a bug
in the original training code would go uncaught.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Reuse the markdown-table single-value parser: it is the exact inverse of
# `tools._render_markdown_table`, which is what `query_sql` output already
# is. Re-implementing table parsing here would risk a subtly different
# parser disagreeing with agent.py's about what "a 1x1 result" means.
from agentic_analyst.agent import _single_value
from agentic_analyst.report import Baseline, Dataset, Finding, Report
from agentic_analyst.tools import query_sql, run_python

Verdict = Literal["verified", "unverified", "contradicted"]

_SQL_LEADING_KEYWORDS = ("SELECT", "WITH")


def _is_sql(evidence: str) -> bool:
    """SQL evidence starts with SELECT/WITH (case-insensitive); anything
    else is treated as Python evidence."""
    stripped = evidence.strip()
    if not stripped:
        return False
    leading_word = re.match(r"[A-Za-z]+", stripped)
    return bool(leading_word) and leading_word.group(0).upper() in _SQL_LEADING_KEYWORDS


def _parse_number(raw: str) -> int | float | None:
    """Parse a recomputed value's text form as a number, or None if it isn't
    one (e.g. a categorical value like 'Month-to-month')."""
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _extract_sql_value(evidence: str, csv_path: Path) -> tuple[str | None, str | None]:
    """Run `evidence` via query_sql and extract its single 1x1 cell.

    Returns (raw_value, error_detail); error_detail is None on success.
    """
    result = query_sql(evidence, csv_path)
    if not result.ok:
        return None, f"query_sql failed to execute evidence: {result.error}"
    try:
        return str(_single_value(result.stdout)), None
    except ValueError as exc:
        return None, f"could not extract a single value from the query result: {exc}"


def _extract_python_value(evidence: str, workdir: Path) -> tuple[str | None, str | None]:
    """Run `evidence` via run_python and extract the recomputed value.

    Convention (documented here since there is no structured return value
    for Python evidence): the code must print the single recomputed value
    as the last non-blank stdout line -- a bare number or short string, no
    surrounding text. This mirrors the convention `agent.py`'s baseline
    training code already uses (last stdout line = a JSON metrics blob);
    for a single-value finding it is simplified to "last line = the value".

    Returns (raw_value, error_detail); error_detail is None on success.
    """
    result = run_python(evidence, workdir)
    if not result.ok:
        return None, f"run_python failed to execute evidence: {result.error}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None, "run_python produced no stdout to extract a value from"
    return lines[-1].strip(), None


def _compare(
    claimed: str | int | float, raw: str, rel_tol: float, abs_tol: float
) -> tuple[Verdict, str | int | float, str]:
    """Compare a recomputed raw value (always text) against the claimed
    value. Numeric claims are compared with math.isclose; everything else
    falls back to a normalized (stripped, casefolded) string compare."""
    claimed_is_numeric = isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
    recomputed_number = _parse_number(raw)

    if claimed_is_numeric:
        if recomputed_number is None:
            return (
                "unverified",
                raw,
                f"claimed value {claimed!r} is numeric but the recomputed value "
                f"{raw!r} is not -- values are not comparable",
            )
        if math.isclose(recomputed_number, claimed, rel_tol=rel_tol, abs_tol=abs_tol):
            return (
                "verified",
                recomputed_number,
                f"recomputed {recomputed_number} matches claimed {claimed} "
                f"(rel_tol={rel_tol}, abs_tol={abs_tol})",
            )
        return (
            "contradicted",
            recomputed_number,
            f"recomputed {recomputed_number} does NOT match claimed {claimed} "
            f"(rel_tol={rel_tol}, abs_tol={abs_tol})",
        )

    claimed_norm = str(claimed).strip().casefold()
    raw_norm = raw.strip().casefold()
    if claimed_norm == raw_norm:
        return ("verified", raw, f"recomputed {raw!r} matches claimed {claimed!r} (string compare)")
    return (
        "contradicted",
        raw,
        f"recomputed {raw!r} does NOT match claimed {claimed!r} (string compare)",
    )


@dataclass
class JudgedFinding:
    finding: Finding
    verdict: Verdict
    recomputed_value: str | int | float | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "claim": self.finding.claim,
            "evidence_sql_or_code": self.finding.evidence_sql_or_code,
            "claimed_value": self.finding.value,
            "verdict": self.verdict,
            "recomputed_value": self.recomputed_value,
            "detail": self.detail,
        }


@dataclass
class JudgedBaseline:
    baseline: Baseline
    verdict: Verdict
    recomputed_value: float | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "model": self.baseline.model,
            "features": self.baseline.features,
            "metric_name": self.baseline.metric_name,
            "claimed_value": self.baseline.metric_value,
            "verdict": self.verdict,
            "recomputed_value": self.recomputed_value,
            "detail": self.detail,
        }


@dataclass
class JudgedReport:
    dataset: Dataset
    findings: list[JudgedFinding]
    baseline: JudgedBaseline
    summary: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "dataset": asdict(self.dataset),
            "findings": [f.to_dict() for f in self.findings],
            "baseline": self.baseline.to_dict(),
            "summary": self.summary,
        }


def verify_finding(
    finding: Finding,
    csv_path: Path,
    workdir: Path,
    rel_tol: float = 0.01,
    abs_tol: float = 0.5,
) -> JudgedFinding:
    """Re-execute `finding.evidence_sql_or_code` against the real data and
    compare the recomputed value to `finding.value`.

    `unverified` covers every case where the evidence could not be executed
    or no single comparable value could be extracted from it -- it is
    distinct from `contradicted`, which means the evidence DID run and
    produced a value, but that value disagrees with the claim.
    """
    evidence = finding.evidence_sql_or_code
    if _is_sql(evidence):
        raw_value, error_detail = _extract_sql_value(evidence, csv_path)
    else:
        raw_value, error_detail = _extract_python_value(evidence, workdir)

    if error_detail is not None:
        return JudgedFinding(
            finding=finding, verdict="unverified", recomputed_value=None, detail=error_detail
        )

    verdict, recomputed_value, detail = _compare(finding.value, raw_value, rel_tol, abs_tol)
    return JudgedFinding(
        finding=finding, verdict=verdict, recomputed_value=recomputed_value, detail=detail
    )


# --- baseline re-training ------------------------------------------------------
# The only recipe this judge knows how to reproduce: this project's churn
# baseline (target = Churn, TotalCharges blank->0.0, stratified 80/20 split,
# StandardScaler, LogisticRegression), parameterized by the report's declared
# features and the random_state/max_iter parsed from `baseline.model`.

_SUPPORTED_BASELINE_METRICS = frozenset({"roc_auc", "accuracy"})
_MODEL_CLASS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_RANDOM_STATE_RE = re.compile(r"random_state\s*=\s*(\d+)")
_MAX_ITER_RE = re.compile(r"max_iter\s*=\s*(\d+)")
_DEFAULT_MAX_ITER = 1000
# Small floor so near-zero metric differences don't fall through a purely
# relative tolerance; metrics here are bounded to [0, 1] so this is tight,
# unlike the deliberately loose abs_tol used for findings of any magnitude.
_BASELINE_ABS_TOL = 0.01

_BASELINE_REPRO_TEMPLATE = """
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(__CSV_PATH__)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["Churn_target"] = (df["Churn"] == "Yes").astype(int)

features = __FEATURES__
X = df[features]
y = df["Churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=__RANDOM_STATE__, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=__RANDOM_STATE__, max_iter=__MAX_ITER__)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print(json.dumps(metrics))
"""


def _render_baseline_repro_code(
    csv_path: Path, features: list[str], random_state: int, max_iter: int
) -> str:
    return (
        _BASELINE_REPRO_TEMPLATE.replace("__CSV_PATH__", repr(str(csv_path)))
        .replace("__FEATURES__", repr(list(features)))
        .replace("__RANDOM_STATE__", str(random_state))
        .replace("__MAX_ITER__", str(max_iter))
    )


def _unverified_baseline(baseline: Baseline, reason: str) -> JudgedBaseline:
    return JudgedBaseline(
        baseline=baseline, verdict="unverified", recomputed_value=None, detail=reason
    )


def verify_baseline(
    baseline: Baseline,
    csv_path: Path,
    workdir: Path,
    rel_tol: float = 0.05,
) -> JudgedBaseline:
    """Re-train the baseline's model spec on its declared features and
    compare the recomputed metric to `baseline.metric_value`.

    If the spec isn't reproducible from the report fields alone (unknown
    model class, no parseable random_state, unsupported metric, empty
    feature list, a feature/column that doesn't exist), this returns
    `unverified` with a specific reason instead of faking a verdict.
    """
    class_match = _MODEL_CLASS_RE.match(baseline.model)
    model_class = class_match.group(1) if class_match else None
    if model_class != "LogisticRegression":
        return _unverified_baseline(
            baseline,
            "cannot reproduce baseline: judge only knows how to retrain "
            f"LogisticRegression from the report fields, got model spec {baseline.model!r}",
        )

    random_state_match = _RANDOM_STATE_RE.search(baseline.model)
    if random_state_match is None:
        return _unverified_baseline(
            baseline,
            "cannot reproduce baseline deterministically: no random_state found "
            f"in model spec {baseline.model!r}",
        )
    random_state = int(random_state_match.group(1))

    max_iter_match = _MAX_ITER_RE.search(baseline.model)
    max_iter = int(max_iter_match.group(1)) if max_iter_match else _DEFAULT_MAX_ITER

    if baseline.metric_name not in _SUPPORTED_BASELINE_METRICS:
        return _unverified_baseline(
            baseline,
            f"unsupported metric_name {baseline.metric_name!r}: judge can only "
            f"recompute {sorted(_SUPPORTED_BASELINE_METRICS)} for the reproducible "
            "LogisticRegression recipe",
        )

    if not baseline.features:
        return _unverified_baseline(baseline, "baseline.features is empty; nothing to retrain on")

    code = _render_baseline_repro_code(csv_path, baseline.features, random_state, max_iter)
    result = run_python(code, workdir, timeout_s=60)
    if not result.ok:
        detail = result.error or "unknown error"
        if result.stderr:
            detail = f"{detail}: {result.stderr.strip().splitlines()[-1]}"
        return _unverified_baseline(baseline, f"retraining failed: {detail}")

    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    try:
        metrics = json.loads(lines[-1]) if lines else {}
        recomputed = float(metrics[baseline.metric_name])
    except (KeyError, ValueError, IndexError) as exc:
        return _unverified_baseline(
            baseline,
            f"could not extract metric {baseline.metric_name!r} from retrain "
            f"output: {exc}",
        )

    spec = f"{model_class}(random_state={random_state}, max_iter={max_iter}) on {baseline.features}"
    if math.isclose(recomputed, baseline.metric_value, rel_tol=rel_tol, abs_tol=_BASELINE_ABS_TOL):
        return JudgedBaseline(
            baseline=baseline,
            verdict="verified",
            recomputed_value=recomputed,
            detail=(
                f"retrained {spec}: recomputed {baseline.metric_name}={recomputed:.4f} "
                f"matches claimed {baseline.metric_value:.4f} (rel_tol={rel_tol})"
            ),
        )
    return JudgedBaseline(
        baseline=baseline,
        verdict="contradicted",
        recomputed_value=recomputed,
        detail=(
            f"retrained {spec}: recomputed {baseline.metric_name}={recomputed:.4f} "
            f"does NOT match claimed {baseline.metric_value:.4f} (rel_tol={rel_tol})"
        ),
    )


def verify_report(
    report: Report,
    csv_path: Path,
    workdir: Path,
    rel_tol: float = 0.01,
    abs_tol: float = 0.5,
    baseline_rel_tol: float = 0.05,
) -> JudgedReport:
    """Verify every finding and the baseline in `report`, returning the full
    judged report plus a {verified, unverified, contradicted} summary."""
    judged_findings = [
        verify_finding(finding, csv_path, workdir, rel_tol=rel_tol, abs_tol=abs_tol)
        for finding in report.findings
    ]
    judged_baseline = verify_baseline(report.baseline, csv_path, workdir, rel_tol=baseline_rel_tol)

    summary = {"verified": 0, "unverified": 0, "contradicted": 0}
    for judged_finding in judged_findings:
        summary[judged_finding.verdict] += 1
    summary[judged_baseline.verdict] += 1

    return JudgedReport(
        dataset=report.dataset,
        findings=judged_findings,
        baseline=judged_baseline,
        summary=summary,
    )
