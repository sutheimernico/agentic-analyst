"""Tests for the report UI (M5).

`inject_planted_false_claim` is tested directly as a pure function (no
Streamlit involved) since that -- and the judge actually catching what it
plants -- is the milestone-critical behavior. `streamlit.testing.v1.AppTest`
then smoke-tests the full app in both toggle states, including checking that
the tampered state's rendered output contains a "contradicted" verdict.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from streamlit.testing.v1 import AppTest

from agentic_analyst.agent import CHURN_RATE_QUERY, FakeLLM, run_agent
from agentic_analyst.judge import verify_report
from agentic_analyst.report import Report
from app import CSV_PATH, TAMPERED_VALUE, inject_planted_false_claim

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def honest_report() -> Report:
    with TemporaryDirectory(prefix="agentic-analyst-test-app-") as tmp:
        return run_agent(FakeLLM(CSV_PATH), CSV_PATH, Path(tmp))


# --- inject_planted_false_claim: pure function, no Streamlit ------------------


def test_inject_planted_false_claim_overstates_only_the_churn_finding(honest_report):
    tampered = inject_planted_false_claim(honest_report)

    assert len(tampered.findings) == len(honest_report.findings)
    for original, tampered_finding in zip(honest_report.findings, tampered.findings, strict=True):
        if original.evidence_sql_or_code == CHURN_RATE_QUERY:
            assert tampered_finding.value == TAMPERED_VALUE
            assert "DEMO" in tampered_finding.claim.upper()
            assert tampered_finding.evidence_sql_or_code == original.evidence_sql_or_code
        else:
            # Every other finding is untouched.
            assert tampered_finding == original

    assert tampered.baseline == honest_report.baseline
    assert tampered.dataset == honest_report.dataset


def test_inject_planted_false_claim_does_not_mutate_the_original_report(honest_report):
    original_findings = list(honest_report.findings)

    inject_planted_false_claim(honest_report)

    assert honest_report.findings == original_findings


def test_planted_false_claim_is_caught_as_contradicted(honest_report):
    tampered = inject_planted_false_claim(honest_report)

    with TemporaryDirectory(prefix="agentic-analyst-test-app-") as tmp:
        judged = verify_report(tampered, CSV_PATH, Path(tmp))

    churn_judged = next(jf for jf in judged.findings if jf.finding.value == TAMPERED_VALUE)
    assert churn_judged.verdict == "contradicted"
    assert churn_judged.recomputed_value == pytest.approx(0.2654, abs=1e-3)
    assert judged.summary["contradicted"] >= 1


# --- Full app smoke tests via AppTest ------------------------------------------


def test_app_runs_without_exception_in_honest_state():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)

    assert not at.exception


def test_app_toggle_on_shows_contradicted_verdict_without_exception():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    assert not at.exception

    at.toggle[0].set_value(True)
    at.run(timeout=120)

    assert not at.exception
    rendered = "\n".join(el.value for el in at.markdown)
    assert "contradicted" in rendered.lower()
