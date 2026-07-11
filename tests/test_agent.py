"""Tests for the agent loop: tool dispatch, the FakeLLM-driven full loop
(against the real M1 tools and the real CSV), the max_iters guard, and
determinism. No test in this file calls the real Anthropic API.
"""

from pathlib import Path

import pytest

from agentic_analyst.agent import (
    AgentIncompleteError,
    FakeLLM,
    LLMResponse,
    ToolUseBlock,
    dispatch_tool,
    run_agent,
)
from agentic_analyst.report import Report

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"


# --- dispatch_tool ------------------------------------------------------------


def test_dispatch_tool_routes_read_schema(tmp_path):
    result = dispatch_tool("read_schema", {}, TELCO_CSV, tmp_path)

    assert result.ok
    assert "row_count: 7043" in result.stdout


def test_dispatch_tool_routes_query_sql(tmp_path):
    query = {"query": "SELECT count(*) AS n FROM data"}
    result = dispatch_tool("query_sql", query, TELCO_CSV, tmp_path)

    assert result.ok
    assert "7043" in result.stdout


def test_dispatch_tool_routes_run_python(tmp_path):
    result = dispatch_tool("run_python", {"code": "print(1 + 1)"}, TELCO_CSV, tmp_path)

    assert result.ok
    assert "2" in result.stdout


def test_dispatch_tool_query_sql_error_is_reported(tmp_path):
    result = dispatch_tool("query_sql", {"query": "DROP TABLE data"}, TELCO_CSV, tmp_path)

    assert not result.ok
    assert result.error is not None


def test_dispatch_tool_unknown_name_returns_error_result_not_raise(tmp_path):
    result = dispatch_tool("not_a_real_tool", {"x": 1}, TELCO_CSV, tmp_path)

    assert not result.ok
    assert "unknown tool" in result.error


# --- full loop with FakeLLM ---------------------------------------------------


def test_fake_llm_drives_full_loop_and_returns_valid_report(tmp_path):
    report = run_agent(FakeLLM(TELCO_CSV), TELCO_CSV, tmp_path)

    assert isinstance(report, Report)
    # These can only be correct if read_schema/query_sql/run_python actually
    # ran against the real CSV -- they are not hardcoded anywhere in FakeLLM.
    assert report.dataset.rows == 7043
    assert report.dataset.cols == 21
    assert len(report.findings) >= 1
    assert report.baseline.metric_name == "roc_auc"
    assert 0.0 <= report.baseline.metric_value <= 1.0
    assert report.data_quality_issues  # the TotalCharges gotcha must be reported


def test_report_reflects_real_totalcharges_blank_count(tmp_path):
    report = run_agent(FakeLLM(TELCO_CSV), TELCO_CSV, tmp_path)

    blank_findings = [f for f in report.findings if "blank" in f.claim.lower()]

    assert blank_findings
    assert blank_findings[0].value == 11  # known gotcha count for this CSV


def test_same_fake_llm_script_yields_identical_report(tmp_path):
    workdir1 = tmp_path / "run1"
    workdir2 = tmp_path / "run2"

    report1 = run_agent(FakeLLM(TELCO_CSV), TELCO_CSV, workdir1)
    report2 = run_agent(FakeLLM(TELCO_CSV), TELCO_CSV, workdir2)

    assert report1.to_dict() == report2.to_dict()


# --- error tool_results are actually fed back into the conversation ----------


class _RejectedQueryThenGiveUpLLM:
    """Emits one query_sql call the validator will reject, reads the
    resulting is_error tool_result back on the next turn, then gives up --
    used to prove tool errors are genuinely round-tripped, not swallowed."""

    def __init__(self) -> None:
        self.fed_back_tool_result: dict | None = None

    def create_message(self, *, system, messages, tools, max_tokens=4096):
        if len(messages) == 1:
            block = ToolUseBlock(id="t1", name="query_sql", input={"query": "DROP TABLE data"})
            return LLMResponse(content=[block], stop_reason="tool_use")
        self.fed_back_tool_result = messages[-1]["content"][0]
        return LLMResponse(content=[], stop_reason="end_turn")


def test_tool_errors_are_fed_back_as_is_error_tool_result(tmp_path):
    fake = _RejectedQueryThenGiveUpLLM()

    with pytest.raises(AgentIncompleteError):
        run_agent(fake, TELCO_CSV, tmp_path, max_iters=5)

    assert fake.fed_back_tool_result is not None
    assert fake.fed_back_tool_result["tool_use_id"] == "t1"
    assert fake.fed_back_tool_result["is_error"] is True
    assert "allowed" in fake.fed_back_tool_result["content"]


# --- max_iters guard -----------------------------------------------------------


class _NeverSubmitsLLM:
    """Always requests another read_schema call; never calls submit_report."""

    def create_message(self, *, system, messages, tools, max_tokens=4096):
        block = ToolUseBlock(id="loop", name="read_schema", input={})
        return LLMResponse(content=[block], stop_reason="tool_use")


def test_max_iters_guard_stops_without_infinite_loop(tmp_path):
    with pytest.raises(AgentIncompleteError, match="max_iters=3"):
        run_agent(_NeverSubmitsLLM(), TELCO_CSV, tmp_path, max_iters=3)


def test_loop_ends_cleanly_when_model_stops_without_submitting(tmp_path):
    class _ImmediatelyGivesUpLLM:
        def create_message(self, *, system, messages, tools, max_tokens=4096):
            return LLMResponse(content=[], stop_reason="end_turn")

    with pytest.raises(AgentIncompleteError):
        run_agent(_ImmediatelyGivesUpLLM(), TELCO_CSV, tmp_path, max_iters=5)
