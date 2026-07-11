# Agentic Analyst

> Give the agent a raw CSV, get back a reproducible analysis report — and a judge that shows you where it lies.

## Why this isn't just another GPT wrapper

An LLM agent that profiles a dataset, proposes features, and trains a baseline model is easy
to build and easy to fake: the model can hallucinate a churn rate, round a metric in its own
favor, or simply not check whether the number it just typed is true. Nothing in a typical
"agent writes a report" pipeline catches that.

This project adds a second, independent layer: **the judge**. Every claim (`Finding`) in the
report carries its own `evidence_sql_or_code` — the exact SQL or Python that produced it — plus
the single value it claims. The judge re-executes that evidence against the real data, extracts
the recomputed value, and compares it to the claim with a tight tolerance. It never asks another
LLM whether the text "looks right"; it recomputes the number and checks. That is what separates
this from a wrapper around a chat model: **the report is only trusted as far as the judge can
independently verify it**, claim by claim.

## The demo: honest report, then a caught lie

Run against the real Telco Customer Churn CSV (7,043 rows, 21 columns), the agent's report comes
back and the judge verifies it 3 for 3:

| verdict | claim | claimed | recomputed |
|---|---|---|---|
| ✓ verified | 26.5% of customers churned (imbalanced target) | `0.2654` | `0.26537...` |
| ✓ verified | 11 rows have a blank `TotalCharges` (tenure-0 customers, not yet billed) | `11` | `11` |
| ✓ verified | `LogisticRegression` baseline, ROC-AUC | `0.8105` | `0.8105` |

That's an honest report — good, but it doesn't *show* the judge doing its job, since every badge
is green. So the app has a **"🔬 Inject a planted false claim (demo the judge)" toggle**: turning
it on overstates the churn-rate claim to `0.75` ("almost everyone leaves!") before re-judging.
The claim text is clearly labeled `[DEMO INJECTED LIE]` — it is never presented as a real number.
What comes back:

| verdict | claim | claimed | recomputed (real) |
|---|---|---|---|
| ✗ contradicted | [DEMO INJECTED LIE] 75% of customers churned | `0.75` | `0.2654` |

This is the money moment: the judge doesn't know the claim is fake, doesn't get told which
finding to distrust — it just re-runs the same `SELECT avg(...) FROM data` query the finding
cites, gets the real ~26.5%, and flags the mismatch. Same mechanism that verified the honest
report catches the planted lie.

## Architecture

```
CSV ──► agent loop (Anthropic tool-use) ──► structured Report ──► judge (recompute + compare) ──► JudgedReport
         read_schema / query_sql / run_python                     query_sql / run_python (again, independently)
```

- **Agent loop** (`src/agentic_analyst/agent.py`) — a manual Anthropic tool-use loop against an
  `LLMClient` protocol: `AnthropicClient` (real SDK, imported lazily so the module is importable
  without a key) and `FakeLLM` (a deterministic, scripted stand-in used by every test and by this
  demo — it still calls the *real* tools against the *real* CSV; only "which tool to call next"
  is scripted). Tools: `read_schema` (DuckDB `DESCRIBE` + null counts + preview), `query_sql`
  (single-statement `SELECT`/`WITH`/`SUMMARIZE`/`DESCRIBE` over the CSV via DuckDB), `run_python`
  (sandboxed subprocess: pandas/duckdb/scikit-learn available). The loop ends when the model
  calls `submit_report` with a report that passes schema validation.
- **Report schema** (`src/agentic_analyst/report.py`) — `Report` = `Dataset` + `Finding[]` +
  `data_quality_issues` + `Baseline` + `caveats`. Each `Finding` is `{claim, evidence_sql_or_code,
  value}` — the evidence is what makes independent re-verification possible at all.
- **Judge** (`src/agentic_analyst/judge.py`) — see below; this is the differentiator.
- **App** (`app.py`, Streamlit) — runs the `FakeLLM` demo pipeline live, renders the report with
  inline verdict badges, and hosts the tamper toggle.

### The judge is a deterministic structured recompute, not an LLM pass — deliberately

`judge.py` never calls an LLM. A free-text "LLM-as-judge" would first have to *parse* the
report's prose back into a checkable claim — an extra, fallible step. Because this project's
report is already structured (every `Finding` carries its own evidence and the single value it
claims), the judge skips extraction entirely: it **re-executes the evidence** through the same
tools the agent used (`query_sql`/`run_python`), pulls out the one recomputed number, and
compares it with `math.isclose` (numeric claims) or a normalized string compare (categorical
claims). Verdicts are three-valued, not two:

- `verified` — recompute succeeded and matches the claim.
- `contradicted` — recompute succeeded and disagrees beyond tolerance.
- `unverified` — the evidence couldn't be executed, didn't produce one comparable value, or
  named a model spec the judge doesn't know how to reproduce. This is never collapsed into
  `verified` or `contradicted` — "couldn't check" is its own outcome, and a `recomputed_value`
  of `None` always means exactly that.

`verify_baseline` goes further than trusting the claimed metric: it **re-trains** the same
`LogisticRegression` spec (features + `random_state`/`max_iter` parsed out of `baseline.model`)
and recomputes the metric from scratch, on a training code path deliberately *not* shared with
the agent's own baseline code — an independent check that ran the exact same code as the
original would miss a bug in that code, not just a lie about its output.

Tolerances are tight on purpose: the dominant claim type is a rate/proportion in `[0, 1]`, and
`math.isclose`'s OR logic (`rel_tol` *or* `abs_tol`) means a loose `abs_tol` (e.g. `0.5`) would
let a moderate lie about a proportion slip through (a claim of `0.75` vs. the real `0.2654` is
only `0.48` apart — comfortably under a loose bound). Both default to `0.01`, so real
disagreement is caught by `rel_tol` and `abs_tol` only rescues near-zero claims.

### Sandbox: defense-in-depth, not a hard boundary — stated honestly

`run_python` executes model-generated code in a subprocess with a memory rlimit (`RLIMIT_AS`), a
wall-clock timeout, a scrubbed environment (no ambient secrets), and a network kill-switch that
monkeypatches `socket.socket`/`socket.create_connection`. This is enough to survive "an LLM being
sloppy" — an infinite loop, a memory bomb, an accidental network call — but it is **not** a
security boundary against a determined adversary, and the code says so rather than pretending
otherwise. Documented, unpatched gaps:

- **No filesystem jail.** `cwd=workdir` only confines *relative* writes; any absolute path the
  OS user can read or write is still reachable — there is no chroot/namespace.
- **DNS and raw sockets bypass the kill-switch.** Only `socket.socket` and
  `socket.create_connection` are patched. `socket.getaddrinfo` still resolves DNS, and code that
  imports the raw `_socket` C extension directly can build a connection that never touches the
  patched names.
- **`os.fork`** can spawn children before/around the parent's rlimits and timeout bookkeeping.

If this ever ran on multi-tenant infrastructure or executed anything riskier than "an LLM
occasionally being careless," it would need a real sandbox (container, gVisor, Firecracker, or a
hosted code-execution service) — not what's here.

## No API key needed for the demo

The Streamlit app (`app.py`) and both demo scripts run entirely on `FakeLLM` — a deterministic,
scripted tool-call sequence that still executes the real sandboxed tools against the real CSV, so
every number on screen is genuinely computed, not invented. **No `ANTHROPIC_API_KEY` is required**
for any of this.

The real-Claude agent path (`AnthropicClient`, the actual tool-use loop against `claude-sonnet-5`)
is wired and importable but untested here by design — set `ANTHROPIC_API_KEY` in `.env` to try it.
**Needs Nico.** The CSV uploader in the app is disabled for the same reason: `FakeLLM`'s tool-call
script is scripted specifically for the telco schema and can't analyze an arbitrary upload; that
needs the real agent.

## Reproduce it

```bash
uv sync

# Agent demo: FakeLLM run over the real telco CSV -> results/report.json
uv run python scripts/demo_fake_run.py

# Judge demo: independently re-verify results/report.json -> results/judged_report.json
uv run python scripts/demo_judge_run.py

# App: report UI with verification badges + the tamper toggle
uv run streamlit run app.py

# Gate
uv run pytest -q
uv run ruff check .
```

## Tech

Anthropic SDK tool-use loop · DuckDB (via `query_sql`) · a sandboxed `run_python` subprocess ·
scikit-learn (`LogisticRegression` churn baseline) · Streamlit (report UI) · uv + ruff + pytest.

## Dataset

Telco Customer Churn (`data/telco-customer-churn.csv`) — 7,043 customers, 21 columns, target
`Churn`. `TotalCharges` is stored as text with 11 blank (whitespace-only) values for tenure-0
customers who haven't been billed yet — a real data-quality gotcha the agent is expected to catch
and the judge re-verifies.

## Status

Milestones 1–6 (tools, agent loop, baseline modelling, judge layer, report UI, this write-up) are
done — see `PLAN.md` for the milestone-by-milestone detail and `AUTOPILOT_LOG.md` for the full
build history, including two design fixes the judge's own tests forced (a tolerance bug that let
moderate proportion lies through, and an invariant leak on non-numeric recomputes).
