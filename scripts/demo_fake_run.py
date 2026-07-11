"""Deterministic demo run of the agent loop, with no API key required.

Drives `run_agent` with `FakeLLM` -- a scripted stand-in for the real
Anthropic API (see agent.py) -- against the real telco churn CSV. FakeLLM's
tool-call sequence is hardcoded, but every number that ends up in the report
is computed by the real M1 tools (query_sql, run_python) running against the
real data; only the "which tool to call next" decision is faked, because
there is no ANTHROPIC_API_KEY in this environment.

Writes results/report.json and prints a summary.

Usage: uv run python scripts/demo_fake_run.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_analyst.agent import FakeLLM, run_agent

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "telco-customer-churn.csv"
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with TemporaryDirectory(prefix="agentic-analyst-demo-") as tmp:
        report = run_agent(FakeLLM(CSV_PATH), CSV_PATH, Path(tmp))

    out_path = RESULTS_DIR / "report.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    print(f"Wrote {out_path}")
    print()
    dataset = report.dataset
    print(f"Dataset: {dataset.name} ({dataset.rows} rows, {dataset.cols} cols)")
    print()
    print("Findings:")
    for finding in report.findings:
        print(f"  - {finding.claim} [value={finding.value}]")
    print()
    print("Data quality issues:")
    for issue in report.data_quality_issues:
        print(f"  - {issue}")
    print()
    print(
        f"Baseline: {report.baseline.model} | features={report.baseline.features} | "
        f"{report.baseline.metric_name}={report.baseline.metric_value:.4f}"
    )
    print(f"  notes: {report.baseline.notes}")
    print()
    print("Caveats:")
    for caveat in report.caveats:
        print(f"  - {caveat}")


if __name__ == "__main__":
    main()
