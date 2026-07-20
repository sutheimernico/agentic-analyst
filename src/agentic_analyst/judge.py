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

`verify_baseline` follows the same recompute-don't-trust principle, but is
honestly weaker than `verify_finding`: it retrains this project's declared
recipe (LogisticRegression, stratified 80/20 split on `Churn`, TotalCharges
blank->0.0, StandardScaler) in a fresh process, parameterized by the report's
declared features and the random_state/max_iter parsed out of
`baseline.model`. This DOES catch a *misreported* metric value -- a number
that disagrees with what the declared recipe actually produces when run
(REVIEW.md finding A-3). The net comparison tolerance is `math.isclose`'s
`max(rel_tol * max(|a|,|b|), abs_tol)`, i.e. whichever of the two bounds is
looser for the pair being compared. `rel_tol` (`verify_baseline`'s default,
1e-6) is the bound that actually governs for real AUC/accuracy metrics: it
is a caller-overridable parameter, exactly like `verify_finding`'s rel_tol.
`_BASELINE_ABS_TOL` is deliberately much tighter (~1e-9) so it only ever
matters for a metric near 0, where `rel_tol * value` itself shrinks toward
0 -- it is a near-zero-only floor, not a second general-purpose tolerance.
An earlier version of this fix set both to 1e-6: for metrics bounded to
[0, 1], `rel_tol * value` never exceeds `rel_tol`, so an equally-sized
`_BASELINE_ABS_TOL` tied-or-won every comparison, silently making the
overridable `rel_tol` param inert and a private constant the real decision
-- see `_BASELINE_ABS_TOL`'s comment below for why it is ~1e-9 now, not 0.
It is explicitly NOT an independent methodology check:
`_BASELINE_REPRO_TEMPLATE` below reproduces the exact same recipe as
agent.py's `BASELINE_CODE_TEMPLATE` -- same fillna, same split strategy, same
seed, same model -- so a methodological bug shared by both (e.g. a leakage
mistake baked into the template itself) reproduces identically here and
still passes as `verified` (REVIEW.md finding A-2). `Baseline` also only
records model description, features, metric name/value -- not the full
training recipe (target definition, split strategy, preprocessing) -- so the
judge can only faithfully retrain the one recipe this project's agent uses.
A baseline it doesn't recognize is marked `unverified` with a clear reason --
never rubber-stamped.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from agentic_analyst.report import Baseline, Dataset, Finding, Report

# `single_value` is the exact inverse of `tools._render_markdown_table`, which
# is what `query_sql` output already is. Reusing tools.py's parser (rather than
# re-implementing table parsing here) keeps this module's idea of "a 1x1
# result" from drifting apart from the one that produced the table.
from agentic_analyst.tools import query_sql, run_python, single_value

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


# --- evidence plausibility (REVIEW.md finding A-1, attacks 1+2) ---------------
# `verify_finding` re-executes whatever evidence a Finding carries and trusts
# the recomputed value against `finding.value` -- but nothing so far checked
# that the evidence could have legitimately touched the real dataset at all.
# `SELECT 0.75 AS churn_rate` (no FROM clause) or `print(0.90)` (no CSV read)
# recompute to exactly their own literal, so they always "match" whatever
# value is claimed next to them: a lie that carries its own proof. This is
# the minimal deterministic version of the "witness allow-list" pattern
# (require evidence to reference a registered source rather than a
# free-floating literal; see research/2026-07-19-agentic-judge-sota.md
# section 4). It does not parse SQL or Python -- it only checks that the
# evidence's source text names the one legitimate data source it was given.

_SQL_DATA_REFERENCE_RE = re.compile(r'\b(?:FROM|JOIN)\s+"?data"?\b', re.IGNORECASE)


def _sql_evidence_touches_data(query: str) -> bool:
    """True if `query` references the `data` view as a FROM/JOIN source.

    Token-level, not a real SQL parser (this repo's evidence queries are the
    simple single-statement SELECT/WITH forms tools.py's own
    `_validate_single_select` already restricts them to -- this deliberately
    does not duplicate that parsing). Requires `data` to appear immediately
    after FROM or JOIN, not merely anywhere in the query text: `SELECT 0.75
    AS data` contains the bare token `data` but never reads the view, so a
    naive "does the token `data` appear anywhere" check would wrongly call it
    legitimate (see
    test_sql_evidence_aliasing_a_column_as_data_without_reading_the_view_is_unverified
    in tests/test_judge.py). A `WITH cte AS (SELECT ... FROM data) SELECT
    ... FROM cte` CTE still matches, because the search scans the whole
    query text, not just its outermost FROM clause.

    What this CANNOT distinguish: a query that legitimately has `FROM data`
    but filters to a subpopulation the claim's text doesn't mention still
    passes (REVIEW.md attack 4 -- see verify_finding's docstring). This only
    answers "did the evidence at least read the real table", never "did it
    read the right rows of it".
    """
    return bool(_SQL_DATA_REFERENCE_RE.search(query))


def _python_evidence_touches_data(code: str, csv_path: Path) -> bool:
    """True if `code`'s source text references the dataset's actual file
    path.

    Python evidence receives no injected "already-loaded dataframe" variable
    -- `run_python` (tools.py) only sets cwd/env for the subprocess, nothing
    dataset-specific. The only way legitimate evidence code can read the
    real CSV is by embedding the literal path in something like
    `pd.read_csv(csv_path)`, exactly as the agent's own system prompt hands
    the model that path ("The dataset lives at: {csv_path}",
    `agent.py.SYSTEM_PROMPT`) and as the baseline retraining templates in
    `agent.py`/`judge.py` already do. A substring check on the evidence
    source for that exact path string is therefore both the simplest and
    (for this repo's single-CSV setup) sufficient signal that the code at
    least NAMES the real file.

    What this CANNOT distinguish: code that names the path but never
    actually uses it (`pd.read_csv(csv_path); print(0.9)`) still passes --
    the same static-precondition limitation the SQL check above documents,
    not something a source-text check can close.
    """
    return str(csv_path) in code


def _evidence_implausibility_reason(evidence: str, csv_path: Path) -> str | None:
    """None if `evidence` could legitimately have touched the real dataset;
    otherwise a human-readable reason it could not have."""
    if _is_sql(evidence):
        if _sql_evidence_touches_data(evidence):
            return None
        return (
            "SQL evidence never references the `data` view in a FROM/JOIN clause "
            "-- it cannot have queried the real dataset"
        )
    if _python_evidence_touches_data(evidence, csv_path):
        return None
    return (
        f"Python evidence never references the dataset path ({csv_path}) "
        "-- it cannot have loaded the real dataset"
    )


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
        return str(single_value(result.stdout)), None
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
) -> tuple[Verdict, str | int | float | None, str]:
    """Compare a recomputed raw value (always text) against the claimed
    value. Numeric claims are compared with math.isclose; everything else
    falls back to a normalized (stripped, casefolded) string compare.

    Upholds the module invariant `unverified => recomputed_value is None`:
    the one `unverified` branch here (numeric claim, non-numeric recompute)
    returns None, not the raw string, so a caller can rely on
    `recomputed_value is None` meaning "couldn't check". The un-parseable
    text is surfaced in `detail` instead.
    """
    claimed_is_numeric = isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
    recomputed_number = _parse_number(raw)

    if claimed_is_numeric:
        if recomputed_number is None:
            return (
                "unverified",
                None,
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


# Matches a numeral in a claim's free text: digits with optional thousands
# separators and an optional decimal part, optionally followed by a percent
# marker -- either the `%` sign or the spelled-out word ("percent" / "per
# cent", any casing). `per\s*cent\b` matches both spellings at once: with
# zero whitespace it's exactly "percent"; the trailing `\b` stops it from
# firing inside an unrelated word ("percentage", "per centimeter"). Anchored
# on the mandatory leading digit, so it only ever matches at a numeral's
# start position -- no spurious empty matches.
_CLAIM_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(%|per\s*cent\b)?", re.IGNORECASE)


def _claim_numbers(claim: str) -> list[float]:
    """Extract every numeral in `claim`'s free text as float candidates.

    A `26.5%` token (or "26.5 percent" / "26.5 per cent") yields BOTH `26.5`
    and `0.265` as candidates, since a Finding's `value` may be stored either
    way (a rate finding here typically claims "26.5%" in text but carries
    `value=0.265`). Thousands separators ("1,234") are stripped before
    parsing. Returns an empty list for a claim with no numeral at all -- not
    every claim quotes its number, and the caller must treat that as
    "nothing to check", not a mismatch.
    """
    candidates: list[float] = []
    for digits, percent_marker in _CLAIM_NUMBER_RE.findall(claim):
        number = float(digits.replace(",", ""))
        if percent_marker:
            candidates.append(number)
            candidates.append(number / 100)
        else:
            candidates.append(number)
    return candidates


def _claim_agrees_with_value(claim: str, value: float, rel_tol: float, abs_tol: float) -> bool:
    """True if at least one numeral parsed from `claim` matches `value`
    within tolerance, or if `claim` quotes no numeral at all.

    Only ONE candidate needs to match, not all of them: a claim naming a
    year or an unrelated count that legitimately differs from `value` is
    exactly why this isn't an "every candidate must match" rule.

    KNOWN LIMITATION (accepted, not a bug): this recognizes a claim that
    quotes `value` literally or in %-form, but NOT one phrased as "count of
    total" -- e.g. "1,869 of 7,043 customers churned" is just as honest as
    "26.5% of customers churned" for the same finding (value=0.2654), but
    neither 1869 nor 7043 is close to 0.2654, so it currently produces a
    false `contradicted` (see
    test_count_over_total_claim_is_a_known_false_positive_for_a_rate_finding
    in tests/test_judge.py). This is deliberately not patched with a
    magnitude/ratio-aware guard: any tolerance loose enough to accept
    "1,869 of 7,043" as agreeing with 0.2654 would also have to accept some
    genuinely wrong number related to `value` by other arithmetic, reopening
    the hole this check exists to close (a lie like "7500% churned" must
    keep failing). The real fix is a report-writing contract on the agent
    side -- a claim must quote either `value` itself or its %-form -- not a
    smarter comparison here.
    """
    candidates = _claim_numbers(claim)
    if not candidates:
        return True
    return any(math.isclose(c, value, rel_tol=rel_tol, abs_tol=abs_tol) for c in candidates)


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
    abs_tol: float = 0.01,
) -> JudgedFinding:
    """Re-execute `finding.evidence_sql_or_code` against the real data and
    compare the recomputed value to `finding.value`.

    Before executing anything, the evidence must pass a plausibility check
    (REVIEW.md finding A-1, attacks 1+2): SQL evidence must reference the
    `data` view in a FROM/JOIN clause; Python evidence must reference the
    dataset's file path (see `_evidence_implausibility_reason`). Evidence
    that fails this -- `SELECT 0.75 AS churn_rate` (no FROM at all),
    `print(0.90)` (never reads the CSV) -- recomputes to exactly its own
    literal and therefore always "matches" whatever value is claimed next to
    it, a self-proving lie the value<->evidence compare below cannot see on
    its own. This check runs BEFORE execution, deliberately: it is a static
    precondition on the evidence's source text (same spirit as tools.py's
    `_validate_single_select`, which also rejects a submission before DuckDB
    ever runs it), so there is no reason to spend a subprocess/DuckDB
    execution on evidence that could not have been legitimate regardless of
    what it happens to print. Failing it returns `unverified`, not
    `contradicted`, with reason `evidence_does_not_touch_data`: the evidence
    didn't produce a disprovable number about the data at all, so "couldn't
    legitimately check" is the honest verdict, not "checked and it's wrong".

    KNOWN GAP THIS DOES NOT CLOSE (REVIEW.md attack 4, intentionally not
    mechanically detectable): evidence that legitimately reads `FROM data`
    but silently narrows the population -- e.g. filtering to
    `Contract='Month-to-month'` while the claim's text says "ALL customers"
    -- passes both this check and the value compare, because the recomputed
    number is genuinely, correctly derived from a real (if differently
    scoped) query against the real table. VeriGraph (arXiv 2606.16603) names
    this failure class "executability can mask weak semantic transitions" --
    see research/2026-07-19-agentic-judge-sota.md section 4. Closing it
    would require comparing the claim text's stated population against the
    query's actual filter predicates -- a semantic check this deterministic,
    execution-based judge does not attempt (see
    test_population_switch_evidence_is_a_documented_known_gap_still_verified
    in tests/test_judge.py).

    `unverified` also covers every case where the evidence could not be
    executed or no single comparable value could be extracted from it -- it
    is distinct from `contradicted`, which means the evidence DID run and
    produced a value, but that value disagrees with the claim.

    After a successful, value-matching recompute, `finding.claim`'s own text
    is additionally checked for a disagreeing numeral (REVIEW.md finding A-1,
    attack 3: honest evidence + honest `value`, but a claim sentence quoting
    a different number -- e.g. "75% of customers churned" next to
    `value=0.2654` -- used to pass as `verified` because nothing ever read
    the sentence). A disagreement downgrades `verified` to `contradicted`
    with reason `claim_text_disagrees_with_value`. This only ever
    strengthens the verdict: it runs solely on the `verified` branch, so an
    `unverified` or already-`contradicted` outcome is never masked or
    weakened by it.

    Tolerances are tight on purpose. `math.isclose` passes if EITHER rel_tol
    OR abs_tol is satisfied, and the dominant claim type here is a rate /
    proportion in [0, 1]; a loose abs_tol (e.g. 0.5) would let almost any
    non-extreme lie about a proportion slip through on the abs_tol branch
    (claim 0.75 vs real 0.2654 is only 0.48 apart). abs_tol=0.01 matches the
    baseline metric tolerance and only rescues near-zero claims where rel_tol
    collapses; real disagreement is caught by rel_tol at 1%.
    """
    evidence = finding.evidence_sql_or_code

    implausibility = _evidence_implausibility_reason(evidence, csv_path)
    if implausibility is not None:
        return JudgedFinding(
            finding=finding,
            verdict="unverified",
            recomputed_value=None,
            detail=f"evidence_does_not_touch_data: {implausibility}",
        )

    if _is_sql(evidence):
        raw_value, error_detail = _extract_sql_value(evidence, csv_path)
    else:
        raw_value, error_detail = _extract_python_value(evidence, workdir)

    if error_detail is not None:
        return JudgedFinding(
            finding=finding, verdict="unverified", recomputed_value=None, detail=error_detail
        )

    # On the success path an extractor always returns a raw value; assert it
    # so the implicit "error_detail is None => raw_value is not None" contract
    # between the extractors and _compare is checked, not just assumed.
    assert raw_value is not None
    verdict, recomputed_value, detail = _compare(finding.value, raw_value, rel_tol, abs_tol)

    value_is_numeric = isinstance(finding.value, (int, float)) and not isinstance(
        finding.value, bool
    )
    if (
        verdict == "verified"
        and value_is_numeric
        and not _claim_agrees_with_value(finding.claim, finding.value, rel_tol, abs_tol)
    ):
        verdict = "contradicted"
        detail = (
            f"claim_text_disagrees_with_value: claim {finding.claim!r} quotes no number "
            f"matching the claimed value {finding.value!r} (rel_tol={rel_tol}, abs_tol={abs_tol})"
        )

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
# math.isclose passes if EITHER bound below is satisfied:
# abs(a-b) <= max(rel_tol * max(|a|,|b|), abs_tol). For a metric bounded to
# [0, 1], rel_tol*value can never exceed abs_tol once abs_tol is not itself
# tiny -- an earlier version of this fix set _BASELINE_ABS_TOL to 1e-6 (same
# as rel_tol), which meant abs_tol tied-or-won EVERY comparison and the
# overridable rel_tol param was inert, decided instead by this private
# constant (quality-review finding). _BASELINE_ABS_TOL is now ~1e-9 -- small
# enough that rel_tol (see verify_baseline's rel_tol=1e-6 default) is the
# bound that actually governs for any real metric value, restoring rel_tol
# as a genuinely overridable, dominant tolerance (matching verify_finding's
# design, where rel_tol/abs_tol are both real caller-facing knobs). This
# floor is NOT the loose, deliberately-wide abs_tol used for findings of
# any magnitude -- it exists only to guard the one case rel_tol structurally
# cannot: a metric near 0, where rel_tol*max(|a|,|b|) itself shrinks toward
# 0, so abs_tol=0 would reject genuine cross-platform float noise (different
# BLAS/sklearn builds producing e.g. 1e-9 of drift) as a false
# "contradicted". 1e-9 absorbs exactly that drift magnitude without being
# anywhere near large enough to mask a real embellishment (REVIEW.md A-3's
# smallest, 0.0095).
_BASELINE_ABS_TOL = 1e-9

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
    rel_tol: float = 1e-6,
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
                f"matches claimed {baseline.metric_value:.4f} "
                f"(rel_tol={rel_tol}, abs_tol={_BASELINE_ABS_TOL})"
            ),
        )
    return JudgedBaseline(
        baseline=baseline,
        verdict="contradicted",
        recomputed_value=recomputed,
        detail=(
            f"retrained {spec}: recomputed {baseline.metric_name}={recomputed:.4f} "
            f"does NOT match claimed {baseline.metric_value:.4f} "
            f"(rel_tol={rel_tol}, abs_tol={_BASELINE_ABS_TOL})"
        ),
    )


def verify_report(
    report: Report,
    csv_path: Path,
    workdir: Path,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
    baseline_rel_tol: float = 1e-6,
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
