"""Demo run of the judge layer (M4) against the real, committed report.

Loads results/report.json -- the real demo report produced by
scripts/demo_fake_run.py -- and independently re-verifies every finding plus
the baseline against the real telco churn CSV: pure structured recompute via
the M1 tools (query_sql/run_python), no API key, no LLM. Writes
results/judged_report.json and prints a verdict table.

Usage: uv run python scripts/demo_judge_run.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_analyst.judge import verify_report
from agentic_analyst.report import report_from_dict, validate_report

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "telco-customer-churn.csv"
REPORT_PATH = REPO_ROOT / "results" / "report.json"
OUT_PATH = REPO_ROOT / "results" / "judged_report.json"


def main() -> None:
    data = json.loads(REPORT_PATH.read_text())
    problems = validate_report(data)
    if problems:
        raise SystemExit(f"{REPORT_PATH} fails validate_report: {problems}")
    report = report_from_dict(data)

    with TemporaryDirectory(prefix="agentic-analyst-judge-") as tmp:
        judged = verify_report(report, CSV_PATH, Path(tmp))

    OUT_PATH.write_text(json.dumps(judged.to_dict(), indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")
    print()

    print(f"{'verdict':<13} {'claimed':>10} {'recomputed':>12}  claim")
    for judged_finding in judged.findings:
        claimed = judged_finding.finding.value
        recomputed = judged_finding.recomputed_value
        claim = judged_finding.finding.claim
        print(f"{judged_finding.verdict:<13} {claimed!s:>10} {recomputed!s:>12}  {claim}")

    jb = judged.baseline
    print(
        f"{jb.verdict:<13} {jb.baseline.metric_value!s:>10} {jb.recomputed_value!s:>12}  "
        f"baseline {jb.baseline.model} [{jb.baseline.metric_name}]"
    )
    print()
    print("Details:")
    for judged_finding in judged.findings:
        print(f"  - [{judged_finding.verdict}] {judged_finding.detail}")
    print(f"  - [{jb.verdict}] {jb.detail}")
    print()
    print(f"Summary: {judged.summary}")


if __name__ == "__main__":
    main()
