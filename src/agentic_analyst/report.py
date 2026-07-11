"""Structured report schema emitted by the agent loop.

The judge layer (M4, not built yet) consumes this: each finding carries the
claim, the SQL or Python evidence that produced it, and the resulting value,
so the judge can re-run the evidence and check the claim against the
recomputed number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Dataset:
    name: str
    rows: int
    cols: int


@dataclass
class Finding:
    claim: str
    evidence_sql_or_code: str
    value: str | int | float


@dataclass
class Baseline:
    model: str
    features: list[str]
    metric_name: str
    metric_value: float
    notes: str = ""


@dataclass
class Report:
    dataset: Dataset
    findings: list[Finding]
    data_quality_issues: list[str]
    baseline: Baseline
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def report_from_dict(data: dict) -> Report:
    """Build a `Report` from a dict that has already passed `validate_report`.

    Does not re-validate -- callers should call `validate_report` first and
    handle problems before constructing a `Report` from untrusted input.
    """
    return Report(
        dataset=Dataset(**data["dataset"]),
        findings=[Finding(**f) for f in data["findings"]],
        data_quality_issues=list(data["data_quality_issues"]),
        baseline=Baseline(**data["baseline"]),
        caveats=list(data.get("caveats", [])),
    )


def _is_number(value: object) -> bool:
    # bool is a subclass of int in Python; exclude it so `value: true` isn't
    # accepted as a numeric metric.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def validate_report(data: dict) -> list[str]:
    """Validate a report dict (e.g. a submit_report tool input, or parsed
    results/report.json) against the schema documented on `Report`.

    Returns a list of human-readable problems; an empty list means the
    report is valid. Accumulates every problem found rather than stopping at
    the first, so a caller (or the agent, on a rejected submit_report call)
    gets the full picture in one pass.
    """
    problems: list[str] = []

    if not isinstance(data, dict):
        return ["report must be a JSON object"]

    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        problems.append("'dataset' must be an object with name, rows, cols")
    else:
        if not isinstance(dataset.get("name"), str) or not dataset.get("name"):
            problems.append("'dataset.name' must be a non-empty string")
        if not isinstance(dataset.get("rows"), int) or isinstance(dataset.get("rows"), bool):
            problems.append("'dataset.rows' must be an integer")
        if not isinstance(dataset.get("cols"), int) or isinstance(dataset.get("cols"), bool):
            problems.append("'dataset.cols' must be an integer")

    findings = data.get("findings")
    if not isinstance(findings, list):
        problems.append("'findings' must be a list")
    elif len(findings) == 0:
        problems.append("'findings' must not be empty")
    else:
        for i, finding in enumerate(findings):
            if not isinstance(finding, dict):
                problems.append(f"findings[{i}] must be an object")
                continue
            if not isinstance(finding.get("claim"), str) or not finding.get("claim"):
                problems.append(f"findings[{i}].claim must be a non-empty string")
            evidence = finding.get("evidence_sql_or_code")
            if not isinstance(evidence, str) or not evidence:
                problems.append(f"findings[{i}].evidence_sql_or_code must be a non-empty string")
            value = finding.get("value")
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                problems.append(f"findings[{i}].value must be a string or number")

    data_quality_issues = data.get("data_quality_issues")
    if not _is_str_list(data_quality_issues):
        problems.append("'data_quality_issues' must be a list of strings")

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        problems.append(
            "'baseline' must be an object with model, features, metric_name, metric_value"
        )
    else:
        if not isinstance(baseline.get("model"), str) or not baseline.get("model"):
            problems.append("'baseline.model' must be a non-empty string")
        if not _is_str_list(baseline.get("features")):
            problems.append("'baseline.features' must be a list of strings")
        if not isinstance(baseline.get("metric_name"), str) or not baseline.get("metric_name"):
            problems.append("'baseline.metric_name' must be a non-empty string")
        if not _is_number(baseline.get("metric_value")):
            problems.append("'baseline.metric_value' must be numeric")

    caveats = data.get("caveats", [])
    if not _is_str_list(caveats):
        problems.append("'caveats' must be a list of strings")

    return problems
