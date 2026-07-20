"""Manual Anthropic tool-use agent loop over the M1 tools (`read_schema`,
`query_sql`, `run_python`), plus a client-side `submit_report` tool that ends
the loop with a validated structured report.

There is no ANTHROPIC_API_KEY in this environment, so the loop is built
against a small `LLMClient` protocol instead of being hard-wired to the real
SDK:

- `AnthropicClient` is a thin wrapper around the real SDK. It imports
  `anthropic` lazily, inside `__init__`, so this module (and every test) is
  importable without the package installed or a key set.
- `FakeLLM` is a deterministic, scripted stand-in used by all tests and by
  `scripts/demo_fake_run.py`. It still calls the real M1 tools against the
  real CSV -- only the "which tool to call next" decision is scripted, so
  the numbers in the resulting report are genuine.

Needs Nico: the real-API path (`AnthropicClient` against the live model) is
untested here by design -- set `ANTHROPIC_API_KEY` in `.env` to try it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentic_analyst.report import Report, report_from_dict, validate_report
from agentic_analyst.tools import (
    ToolResult,
    parse_markdown_table,
    query_sql,
    read_schema,
    run_python,
    single_value,
)

logger = logging.getLogger(__name__)

# Default loop model and an escalation target for harder reasoning (per
# PLAN.md). Only DEFAULT_MODEL is wired into AnthropicClient today --
# escalation logic (when/how to switch mid-loop) is not built because there
# is no concrete trigger for it yet (YAGNI); the constant documents the
# intended target for when M4/M5 need it.
DEFAULT_MODEL = "claude-sonnet-5"
ESCALATION_MODEL = "claude-opus-4-8"


class AgentIncompleteError(RuntimeError):
    """Raised when the loop ends (max_iters exhausted, or the model stops
    without calling a tool) without a valid submitted report."""


# --- LLM response shapes -----------------------------------------------------
# Duck-type the real Anthropic SDK's Message/content-block shape (`.type`,
# and `.text` or `.id`/`.name`/`.input`) so the loop's block-handling code
# below works unchanged whether `response` came from the real SDK
# (AnthropicClient) or these local dataclasses (FakeLLM).


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class LLMResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str


@runtime_checkable
class LLMClient(Protocol):
    """Minimal `create_message`-style interface both LLM implementations share."""

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Thin wrapper around the real Anthropic SDK.

    `anthropic` is imported inside `__init__`, not at module level, so
    importing `agentic_analyst.agent` never requires the package or an API
    key to be present.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic  # local import: see class docstring

        self._client = anthropic.Anthropic()
        self.model = model

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ):
        return self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )


# --- Tool schemas -------------------------------------------------------------
# read_schema / query_sql / run_python map 1:1 onto the M1 functions in
# tools.py; submit_report is a 4th, client-side-only tool that never reaches
# a real executor -- run_agent intercepts it to validate and end the loop.

READ_SCHEMA_TOOL = {
    "name": "read_schema",
    "description": (
        "Describe the telco churn CSV: columns, inferred dtypes, row count, "
        "null counts per column, and a preview of rows. Call this first, "
        "before writing any SQL or Python, to see what you're working with."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n_preview": {
                "type": "integer",
                "description": "Number of preview rows to include (default 5).",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}

QUERY_SQL_TOOL = {
    "name": "query_sql",
    "description": (
        "Run a single read-only SQL query (SELECT / WITH / SUMMARIZE / "
        "DESCRIBE) over the telco CSV, exposed as a view named `data`. Use "
        "this to profile columns, counts, and distributions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to run against the `data` view.",
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum rows to return (default 200).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code in a sandboxed subprocess (pandas, duckdb, and "
        "scikit-learn are available). Use this for statistics beyond what "
        "SQL can express, and to train and evaluate the churn baseline "
        "model. Read the CSV from the exact path given in the system "
        "prompt -- there is no network access. When this code is a "
        "finding's evidence (the code a separate judge process will "
        "re-execute to check the finding's claimed value), print the "
        "recomputed value on its own line in the exact form `RESULT: "
        "<value>`, e.g. `print(f\"RESULT: {value}\")` -- other print "
        "statements (labels, intermediate diagnostics) are fine anywhere "
        "around it, and the match is not case-sensitive. Without a "
        "`RESULT:` line, the judge falls back to treating the last "
        "non-blank stdout line as the value, so unlabelled bare-print "
        "output still works, but `RESULT:` is the reliable convention for "
        "anything with decorated output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

SUBMIT_REPORT_TOOL = {
    "name": "submit_report",
    "description": (
        "Submit the final structured analysis report and end the run. Call "
        "this exactly once, after profiling the data and training the "
        "baseline, with a report matching the required schema. If the "
        "report is rejected, fix the reported problems and call it again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rows": {"type": "integer"},
                    "cols": {"type": "integer"},
                },
                "required": ["name", "rows", "cols"],
                "additionalProperties": False,
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_sql_or_code": {"type": "string"},
                        "value": {"type": ["string", "number"]},
                    },
                    "required": ["claim", "evidence_sql_or_code", "value"],
                    "additionalProperties": False,
                },
            },
            "data_quality_issues": {"type": "array", "items": {"type": "string"}},
            "baseline": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "metric_name": {"type": "string"},
                    "metric_value": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["model", "features", "metric_name", "metric_value"],
                "additionalProperties": False,
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["dataset", "findings", "data_quality_issues", "baseline"],
        "additionalProperties": False,
    },
}

TOOLS = [READ_SCHEMA_TOOL, QUERY_SQL_TOOL, RUN_PYTHON_TOOL, SUBMIT_REPORT_TOOL]


SYSTEM_PROMPT = """You are an autonomous data analyst investigating a customer churn dataset.

The dataset lives at: {csv_path}

Work deterministically and back every claim with a tool call -- never report
a number you have not actually computed. Follow this sequence:

1. Call `read_schema` first to see columns, inferred dtypes, row count, and
   null counts.
2. Profile the data via `query_sql` and/or `run_python`: class balance of the
   target `Churn`, and any distributions you plan to use as features.
3. Look specifically for data quality issues. In particular, check whether
   every column that looks numeric actually parses as numeric -- a column
   can be stored as text with blank (or whitespace) values for a subset of
   rows.
4. Propose features and train a baseline churn classifier with `run_python`
   (e.g. scikit-learn LogisticRegression). Use a fixed random_state, an
   explicit held-out test split, and report ROC-AUC and accuracy on that
   held-out split.
5. Once you have profiled the data, identified data quality issues, and
   trained and evaluated a baseline, call `submit_report` exactly once with
   the structured report: findings (each with a claim, the SQL or Python
   evidence that produced it, and the resulting value), data quality
   issues, the baseline model description and metrics, and caveats.

Keep findings and evidence terse and reproducible -- another process will
recompute them against the same data to check your work. There is no
network access; do not attempt to reach one."""


def dispatch_tool(name: str, tool_input: dict, csv_path: Path, workdir: Path) -> ToolResult:
    """Route a model-requested tool call to the matching M1 function.

    An unrecognized tool name (e.g. a hallucinated call) is reported as an
    ordinary failed `ToolResult` rather than raised, so the caller can feed
    it back to the model as an `is_error` tool_result like any other failure.
    """
    start = time.monotonic()
    if name == "read_schema":
        n_preview = tool_input.get("n_preview", 5)
        return read_schema(csv_path, n_preview=n_preview)
    if name == "query_sql":
        query = tool_input.get("query", "")
        max_rows = tool_input.get("max_rows", 200)
        return query_sql(query, csv_path, max_rows=max_rows)
    if name == "run_python":
        code = tool_input.get("code", "")
        return run_python(code, workdir)
    return ToolResult(
        ok=False,
        stdout="",
        stderr="",
        elapsed_s=time.monotonic() - start,
        error=f"unknown tool: {name}",
    )


def _tool_result_message(tool_use_id: str, result: ToolResult) -> dict:
    if result.ok:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.stdout}
    error_text = result.error or "unknown error"
    if result.stderr:
        error_text = f"{error_text}\n{result.stderr}"
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": error_text,
        "is_error": True,
    }


def run_agent(llm: LLMClient, csv_path: Path, workdir: Path, max_iters: int = 12) -> Report:
    """Drive the tool-use loop until the model submits a valid report.

    Loops until `submit_report` is called with a report that passes
    `validate_report`, or `max_iters` iterations pass without that
    happening -- whichever comes first. Every iteration and tool call is
    logged.
    """
    csv_path = Path(csv_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    system = SYSTEM_PROMPT.format(csv_path=csv_path)
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Profile the churn dataset, identify data quality issues, "
                "train a baseline churn classifier, and submit the "
                "structured report."
            ),
        }
    ]

    for iteration in range(1, max_iters + 1):
        logger.info("agent iteration %d/%d", iteration, max_iters)
        response = llm.create_message(system=system, messages=messages, tools=TOOLS)

        if response.stop_reason != "tool_use":
            logger.info(
                "loop stopped with stop_reason=%s (no report submitted)", response.stop_reason
            )
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        results: list[dict] = []
        submitted: Report | None = None
        for block in tool_use_blocks:
            logger.info("tool call: %s(%r)", block.name, block.input)

            if block.name == "submit_report":
                problems = validate_report(block.input)
                if problems:
                    logger.warning("submit_report rejected: %s", problems)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "report rejected: " + "; ".join(problems),
                            "is_error": True,
                        }
                    )
                else:
                    submitted = report_from_dict(block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "report accepted",
                        }
                    )
                continue

            result = dispatch_tool(block.name, block.input, csv_path, workdir)
            results.append(_tool_result_message(block.id, result))

        messages.append({"role": "user", "content": results})

        if submitted is not None:
            logger.info("report submitted and accepted after %d iteration(s)", iteration)
            return submitted

    raise AgentIncompleteError(
        f"agent did not submit a valid report within max_iters={max_iters} iterations"
    )


# --- FakeLLM ------------------------------------------------------------------
# Deterministic stand-in used by every test and by scripts/demo_fake_run.py.
# The sequence of tool calls is hardcoded, but the submit_report call is
# built from the actual tool_result text produced by those calls -- so the
# report's numbers are computed by the real M1 tools against the real CSV;
# only "which tool to call next" is scripted.

# One profiling query per claimed value, so a judge (M4) can recompute the
# single scalar each finding's `evidence_sql_or_code` returns and compare it
# to that finding's `value` -- no need to parse the claim text to pick a
# column out of a multi-column result.
CHURN_RATE_QUERY = "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate FROM data"
BLANK_TOTAL_CHARGES_QUERY = (
    "SELECT sum(CASE WHEN trim(TotalCharges) = '' THEN 1 ELSE 0 END) AS n_blank FROM data"
)

# run_python's sandbox pins OPENBLAS_NUM_THREADS/OMP_NUM_THREADS=1 in the
# child env (see tools.py), so model-generated training code -- including
# this scripted baseline -- does not need to set them itself.
BASELINE_CODE_TEMPLATE = """
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(__CSV_PATH__)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["Churn_target"] = (df["Churn"] == "Yes").astype(int)

features = ["tenure", "MonthlyCharges", "TotalCharges"]
X = df[features]
y = df["Churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "features": features,
}
print(json.dumps(metrics))
"""


def _render_baseline_code(csv_path: Path) -> str:
    return BASELINE_CODE_TEMPLATE.replace("__CSV_PATH__", repr(str(csv_path)))


def _get_block_attr(block: object, name: str) -> object:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _collect_tool_calls(messages: list[dict]) -> list[dict]:
    """Return each completed tool call as {name, input, content}, pairing
    every tool_result back to the tool_use block that requested it via
    `tool_use_id`. A list (not a name->content dict) so multiple query_sql
    calls in one run stay distinct. Handles both the dataclass blocks in
    assistant turns and the plain dicts in the user turns we build."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = _get_block_attr(block, "type")
            if block_type == "tool_use":
                block_id = _get_block_attr(block, "id")
                if block_id:
                    by_id[block_id] = {
                        "name": _get_block_attr(block, "name"),
                        "input": _get_block_attr(block, "input"),
                        "content": None,
                    }
                    order.append(block_id)
            elif block_type == "tool_result":
                tool_use_id = _get_block_attr(block, "tool_use_id")
                if tool_use_id in by_id:
                    by_id[tool_use_id]["content"] = _get_block_attr(block, "content")
    return [by_id[block_id] for block_id in order]


def _first_result(calls: list[dict], name: str, query: str | None = None) -> str:
    """The content of the first completed call to `name` (optionally the one
    whose `query` input matches). Raises if no such call produced a result."""
    for call in calls:
        if call["name"] != name or call["content"] is None:
            continue
        if query is not None and (call["input"] or {}).get("query") != query:
            continue
        return call["content"]
    raise ValueError(f"no completed {name} call found" + (f" for query {query!r}" if query else ""))


def _build_report_from_messages(messages: list[dict]) -> dict:
    """Build the submit_report input purely from the real tool_result text
    already present in `messages` -- every number here was computed by the
    actual M1 tools, not invented. Each finding's evidence query returns
    exactly the one value the finding claims (see the query constants), so
    M4's judge can recompute and compare without parsing the claim text."""
    calls = _collect_tool_calls(messages)

    schema_text = _first_result(calls, "read_schema")
    row_count_match = re.search(r"row_count:\s*(\d+)", schema_text)
    if not row_count_match:
        raise ValueError(f"could not find row_count in read_schema output: {schema_text!r}")
    row_count = int(row_count_match.group(1))

    schema_section = schema_text.split("schema:\n", 1)[1].split("\n\nnull_counts:")[0]
    _, schema_rows = parse_markdown_table(schema_section)
    col_count = len(schema_rows)

    churn_rate = float(single_value(_first_result(calls, "query_sql", CHURN_RATE_QUERY)))
    n_blank = int(single_value(_first_result(calls, "query_sql", BLANK_TOTAL_CHARGES_QUERY)))

    metrics_line = _first_result(calls, "run_python").strip().splitlines()[-1]
    baseline_metrics = json.loads(metrics_line)

    return {
        "dataset": {"name": "telco-customer-churn", "rows": row_count, "cols": col_count},
        "findings": [
            {
                "claim": (
                    f"{churn_rate:.1%} of customers churned (target class "
                    "Churn = 'Yes'); the dataset is imbalanced toward non-churners."
                ),
                "evidence_sql_or_code": CHURN_RATE_QUERY,
                "value": round(churn_rate, 4),
            },
            {
                "claim": (
                    f"{n_blank} rows have a blank (whitespace-only) TotalCharges "
                    "value -- these are the tenure-0 customers who have not been "
                    "billed yet."
                ),
                "evidence_sql_or_code": BLANK_TOTAL_CHARGES_QUERY,
                "value": n_blank,
            },
        ],
        "data_quality_issues": [
            f"TotalCharges is read as VARCHAR, not numeric, because {n_blank} rows "
            "hold a single-space string instead of a number; it must be coerced "
            "with pd.to_numeric(..., errors='coerce') and the resulting NaNs "
            "handled before it can be used as a model feature.",
        ],
        "baseline": {
            "model": "LogisticRegression(random_state=42, max_iter=1000)",
            "features": baseline_metrics["features"],
            "metric_name": "roc_auc",
            "metric_value": baseline_metrics["roc_auc"],
            "notes": (
                f"accuracy={baseline_metrics['accuracy']:.4f} on a stratified "
                "80/20 train/test split (random_state=42), features standardized."
            ),
        },
        "caveats": [
            "Baseline uses 3 numeric features only; categorical service and "
            "contract features (Contract, InternetService, PaymentMethod, ...) "
            "were profiled but not yet included, and would likely improve the "
            "model.",
            "Single train/test split, no cross-validation -- metric_value has "
            "some variance across seeds.",
        ],
    }


class FakeLLM:
    """Deterministic, scripted stand-in for the real Anthropic API.

    Emits a fixed sequence of tool calls -- read_schema, one query_sql per
    profiling value (churn rate, then blank-TotalCharges count), then a
    run_python call that trains the churn baseline -- and finally a
    submit_report call whose content is built from the real tool_result text
    those calls produced (see `_build_report_from_messages`).
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        self._step = 0

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self._step += 1

        if self._step == 1:
            return self._tool_call("read_schema", {})
        if self._step == 2:
            return self._tool_call("query_sql", {"query": CHURN_RATE_QUERY})
        if self._step == 3:
            return self._tool_call("query_sql", {"query": BLANK_TOTAL_CHARGES_QUERY})
        if self._step == 4:
            return self._tool_call("run_python", {"code": _render_baseline_code(self._csv_path)})
        if self._step == 5:
            return self._tool_call("submit_report", _build_report_from_messages(messages))

        # Scripted sequence is exhausted -- stop instead of looping forever.
        return LLMResponse(content=[], stop_reason="end_turn")

    def _tool_call(self, name: str, tool_input: dict) -> LLMResponse:
        block = ToolUseBlock(id=f"toolu_{self._step}_{name}", name=name, input=tool_input)
        return LLMResponse(content=[block], stop_reason="tool_use")
