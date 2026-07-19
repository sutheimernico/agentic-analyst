"""Tests for the judge / verification layer (M4).

These are the milestone-critical tests: every test here uses the REAL telco
CSV and the REAL M1 tools (query_sql, run_python) -- no API key, no LLM. The
whole point of M4 is to prove the judge actually recomputes and catches a
planted lie, rather than trusting whatever `value` a Finding claims -- the
planted-false-claim and planted-wrong-baseline tests below are that proof.
"""

from pathlib import Path

import pytest

from agentic_analyst.judge import verify_baseline, verify_finding, verify_report
from agentic_analyst.report import Baseline, Dataset, Finding, Report

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"

CHURN_RATE_QUERY = "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate FROM data"
REAL_CHURN_RATE = 0.2654  # 1869 / 7043 -- known fact about this CSV (see test_tools.py)

REAL_BASELINE_MODEL = "LogisticRegression(random_state=42, max_iter=1000)"
REAL_BASELINE_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
REAL_BASELINE_AUC = 0.8105272675605157  # from results/report.json, same recipe/seed


# --- verify_finding: correct claim ---------------------------------------------


def test_correct_finding_is_verified(tmp_path):
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


# --- verify_finding: THE milestone-critical test -------------------------------


def test_planted_false_churn_rate_claim_is_contradicted(tmp_path):
    finding = Finding(
        claim="90% of customers churned -- almost everyone leaves!",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=0.90,  # planted lie: the real churn rate is ~26.5%
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.finding.value == 0.90  # claim preserved for the report/UI
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)
    # This is already contradicted by the value<->evidence compare; the
    # claim-text check (see below) must not also fire and overwrite the
    # detail with its own reason -- the original mismatch explanation stays
    # authoritative.
    assert "claim_text_disagrees_with_value" not in judged.detail


def test_moderate_proportion_lie_is_contradicted_under_default_tolerance(tmp_path):
    # Regression for the abs_tol bug: with abs_tol=0.5, a claim of 0.75 vs the
    # real 0.2654 (abs diff 0.48, well under 0.5) sailed through as VERIFIED --
    # gutting the "catch lies" promise for the dominant claim type (rates in
    # [0, 1]). With the tightened default tolerance it must be contradicted.
    finding = Finding(
        claim="75% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=0.75,  # abs diff from real ~0.2654 is ~0.48; rel diff ~83%
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)  # default tolerances

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


# --- verify_finding: unverified (couldn't check) is distinct from contradicted


def test_finding_with_unexecutable_query_is_unverified(tmp_path):
    finding = Finding(
        claim="Some claim backed by a query against a table that doesn't exist.",
        evidence_sql_or_code="SELECT * FROM this_table_does_not_exist",
        value=42,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert judged.detail  # a reason must be present


def test_finding_with_nonexistent_column_is_unverified(tmp_path):
    finding = Finding(
        claim="Some claim backed by a query referencing a column that doesn't exist.",
        evidence_sql_or_code="SELECT this_column_does_not_exist FROM data",
        value=42,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_finding_with_multi_value_query_is_unverified(tmp_path):
    # Two columns, one row -- not the 1x1 single-value result a finding's
    # evidence is supposed to produce, so no comparable value exists.
    finding = Finding(
        claim="Some claim backed by a query that returns more than one value.",
        evidence_sql_or_code="SELECT tenure, MonthlyCharges FROM data LIMIT 1",
        value=1,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_numeric_claim_vs_non_numeric_recompute_is_unverified_with_none(tmp_path):
    # A numeric claim whose evidence returns a categorical (non-numeric) value
    # is not comparable -- it must be unverified, and (invariant) the
    # recomputed_value must be None, not the raw string that was found.
    finding = Finding(
        claim="The most common contract type, as a number, is 5.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value=5,  # numeric claim, but recompute yields the string 'Month-to-month'
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert "Month-to-month" in judged.detail  # what was actually found is surfaced


# --- verify_finding: non-numeric claims ----------------------------------------


def test_non_numeric_claim_verified_by_normalized_string_compare(tmp_path):
    finding = Finding(
        claim="Month-to-month is the most common contract type.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value="  month-to-month  ",  # deliberately mixed case + surrounding whitespace
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == "Month-to-month"


def test_non_numeric_claim_contradicted_when_wrong(tmp_path):
    finding = Finding(
        claim="Two year is the most common contract type.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value="Two year",  # planted lie: Month-to-month is actually the mode
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == "Month-to-month"


# --- verify_finding: Python evidence path (documents the last-line convention)


def test_python_evidence_last_stdout_line_is_the_value(tmp_path):
    finding = Finding(
        claim="2 + 2 is 4 (sanity check for the Python evidence convention).",
        evidence_sql_or_code="print('some noise line')\nprint(2 + 2)",
        value=4,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == 4


# --- verify_finding: tolerance boundary ----------------------------------------


def test_tolerance_boundary_just_inside_verifies_just_outside_contradicts(tmp_path):
    query = "SELECT 1000.0 AS v"  # constant -- isolates the tolerance math itself

    just_inside = Finding(claim="approx 1000", evidence_sql_or_code=query, value=1009)  # 0.9% off
    just_outside = Finding(claim="approx 1000", evidence_sql_or_code=query, value=1011)  # 1.1% off

    inside_result = verify_finding(just_inside, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.5)
    outside_result = verify_finding(just_outside, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.5)

    assert inside_result.verdict == "verified"
    assert outside_result.verdict == "contradicted"


# --- verify_finding: claim text vs claimed value (REVIEW.md A-1, attack 3) -----
# `_compare` above only checks value<->evidence consistency -- both fields are
# agent-authored. The natural-language `claim` is the sentence a reader
# actually trusts and was never read at all: a claim that lies in its text
# while carrying an honest, evidence-matching `value` used to pass as
# `verified`. These tests prove the judge now reads the claim text too.


def test_claim_text_lying_about_percentage_is_contradicted_even_with_honest_value(tmp_path):
    finding = Finding(
        claim="75% of customers churned -- almost everyone leaves!",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,  # honest value, honest evidence -- only the claim text lies
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)
    assert "claim_text_disagrees_with_value" in judged.detail


def test_claim_text_quoting_the_correct_percentage_stays_verified(tmp_path):
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.01)

    assert judged.verdict == "verified"


def test_claim_text_quoting_a_plain_float_matching_value_stays_verified(tmp_path):
    finding = Finding(
        claim="The churn rate is 0.2654, which is a concerning figure.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_with_thousands_separator_matching_value_stays_verified(tmp_path):
    finding = Finding(
        claim="1,869 customers churned in total.",
        evidence_sql_or_code="SELECT count(*) AS n FROM data WHERE Churn = 'Yes'",
        value=1869,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_with_wrong_thousands_separator_count_is_contradicted(tmp_path):
    finding = Finding(
        claim="1,234 customers churned in total.",  # wrong count in the sentence
        evidence_sql_or_code="SELECT count(*) AS n FROM data WHERE Churn = 'Yes'",
        value=1869,  # honest value, honest evidence -- only the claim text lies
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert "claim_text_disagrees_with_value" in judged.detail


def test_claim_text_with_no_numeric_token_passes_through_silently(tmp_path):
    # Not every claim quotes its number -- a claim about *why* customers churn
    # carries no numeral to check against the rate finding's value, so the new
    # check must not interfere with the pre-existing verified verdict.
    finding = Finding(
        claim="Most customers who churn are on a month-to-month contract.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


# --- verify_baseline ------------------------------------------------------------


def test_real_baseline_recomputes_to_verified(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


def test_planted_wrong_baseline_metric_is_contradicted(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.99,  # planted lie: the real ROC-AUC is ~0.81
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


def test_baseline_with_unrecognized_model_class_is_unverified(tmp_path):
    baseline = Baseline(
        model="RandomForestClassifier(random_state=42)",
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.85,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_baseline_without_random_state_is_unverified(tmp_path):
    baseline = Baseline(
        model="LogisticRegression(max_iter=1000)",  # no random_state -> not reproducible
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.81,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_baseline_with_nonexistent_feature_column_is_unverified(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=[*REAL_BASELINE_FEATURES, "this_column_does_not_exist"],
        metric_name="roc_auc",
        metric_value=0.81,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_baseline_metric_depends_on_feature_set(tmp_path):
    # Proves verify_baseline actually re-trains on the declared features rather
    # than echoing the claim: the real ~0.8105 AUC came from 3 features; a
    # single-feature model produces a materially different metric, so claiming
    # the 3-feature AUC while declaring only ['tenure'] must be contradicted,
    # and the recomputed value must differ from the claim.
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=["tenure"],  # fewer features than produced REAL_BASELINE_AUC
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value is not None
    assert judged.recomputed_value != pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


# --- verify_report: wiring + summary counts ------------------------------------


def test_verify_report_summarizes_findings_and_baseline(tmp_path):
    report = Report(
        dataset=Dataset(name="telco-customer-churn", rows=7043, cols=21),
        findings=[
            Finding(
                claim="26.5% of customers churned.",
                evidence_sql_or_code=CHURN_RATE_QUERY,
                value=REAL_CHURN_RATE,
            ),
            Finding(
                claim="90% of customers churned!",  # planted lie
                evidence_sql_or_code=CHURN_RATE_QUERY,
                value=0.90,
            ),
        ],
        data_quality_issues=["n/a"],
        baseline=Baseline(
            model=REAL_BASELINE_MODEL,
            features=REAL_BASELINE_FEATURES,
            metric_name="roc_auc",
            metric_value=REAL_BASELINE_AUC,
        ),
    )

    judged = verify_report(report, TELCO_CSV, tmp_path)

    assert judged.summary == {"verified": 2, "unverified": 0, "contradicted": 1}
    assert len(judged.findings) == 2
    assert judged.baseline.verdict == "verified"
