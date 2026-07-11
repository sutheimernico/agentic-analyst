"""Tests for the structured report schema and its validator."""

from agentic_analyst.report import (
    Baseline,
    Dataset,
    Finding,
    Report,
    report_from_dict,
    validate_report,
)

VALID_REPORT = {
    "dataset": {"name": "telco-customer-churn", "rows": 7043, "cols": 21},
    "findings": [
        {
            "claim": "26.5% of customers churned.",
            "evidence_sql_or_code": "SELECT avg((Churn='Yes')::INT) FROM data",
            "value": 0.265,
        },
    ],
    "data_quality_issues": ["TotalCharges has 11 blank rows for tenure-0 customers."],
    "baseline": {
        "model": "LogisticRegression",
        "features": ["tenure", "MonthlyCharges", "TotalCharges"],
        "metric_name": "roc_auc",
        "metric_value": 0.82,
        "notes": "held-out 20% split, random_state=42",
    },
    "caveats": ["only numeric features used"],
}


def test_valid_report_passes():
    assert validate_report(VALID_REPORT) == []


def test_missing_baseline_fails_with_clear_message():
    report = {k: v for k, v in VALID_REPORT.items() if k != "baseline"}

    problems = validate_report(report)

    assert any("baseline" in p for p in problems)


def test_empty_findings_fails():
    report = {**VALID_REPORT, "findings": []}

    problems = validate_report(report)

    assert any("findings" in p and "empty" in p for p in problems)


def test_findings_not_a_list_fails():
    report = {**VALID_REPORT, "findings": "not a list"}

    problems = validate_report(report)

    assert any("findings" in p for p in problems)


def test_finding_missing_claim_fails():
    report = {
        **VALID_REPORT,
        "findings": [{"evidence_sql_or_code": "SELECT 1", "value": 1}],
    }

    problems = validate_report(report)

    assert any("claim" in p for p in problems)


def test_non_numeric_metric_value_fails():
    report = {
        **VALID_REPORT,
        "baseline": {**VALID_REPORT["baseline"], "metric_value": "high"},
    }

    problems = validate_report(report)

    assert any("metric_value" in p for p in problems)


def test_boolean_metric_value_fails():
    # bool is a subclass of int in Python -- must not slip through as numeric.
    report = {
        **VALID_REPORT,
        "baseline": {**VALID_REPORT["baseline"], "metric_value": True},
    }

    problems = validate_report(report)

    assert any("metric_value" in p for p in problems)


def test_missing_dataset_rows_fails():
    report = {**VALID_REPORT, "dataset": {"name": "telco", "cols": 21}}

    problems = validate_report(report)

    assert any("dataset.rows" in p for p in problems)


def test_non_dict_report_fails():
    assert validate_report("not a dict") != []


def test_report_from_dict_roundtrip():
    report = report_from_dict(VALID_REPORT)

    assert isinstance(report, Report)
    assert report.dataset == Dataset(name="telco-customer-churn", rows=7043, cols=21)
    assert report.findings[0] == Finding(**VALID_REPORT["findings"][0])
    assert report.baseline == Baseline(**VALID_REPORT["baseline"])
    assert report.to_dict() == VALID_REPORT
