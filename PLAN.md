# Agentic Data Analyst

> Give the agent a raw CSV, get back a reproducible analysis report — and I show you where it lies.

## Goal / Narrative
An **LLM agent** that autonomously runs EDA on a tabular dataset (profiling → hypotheses →
feature ideas → baseline model → markdown report), wrapped in a **verifier / LLM-as-judge
layer** that checks every factual claim in the report against the actual data and flags
unverified or hallucinated statements. The judge layer is the whole point — it's what
separates this from "another GPT wrapper".

## Dataset
- **Telco Customer Churn** — `data/telco-customer-churn.csv` (7,043 customers, 21 columns), target `Churn`.
- Source: IBM sample data (public GitHub mirror). Same dataset as the well-known Kaggle version; pulled GitHub (no Kaggle auth).
- Why this one: rich business context (contract, tenure, services, charges — mixed categorical/numeric), obvious real relationships for the agent to discover and the judge to verify. Known gotcha: `TotalCharges` has blank strings for tenure-0 customers — a good honesty test for the agent.

## Techniques & Stack
- **Anthropic SDK tool-use loop** — agent with tools: `run_python` (sandboxed), `query_sql` (DuckDB over the CSV), `read_schema`. Default model `claude-sonnet-5` for the loop; escalate hard reasoning to Opus if needed. **Confirm exact model IDs + tool-use patterns via the `claude-api` skill before building.**
- **Sandboxed execution** — restricted Python/DuckDB, no network, resource-limited. Security matters here (untrusted code path).
- **Judge layer** — deterministic structured recompute, not an LLM pass: because each `Finding` already carries its own `evidence_sql_or_code` and claimed `value`, the judge re-executes that evidence via the M1 tools (`query_sql`/`run_python`), extracts the recomputed value, and compares it to the claim within a tolerance, marking `verified` / `unverified` / `contradicted`. No API key, no LLM call. See "Judge layer" section below.
- Baseline model: scikit-learn (the agent proposes + trains a churn baseline).
- App: FastAPI or Streamlit — upload → live agent run → annotated report.
- Tooling: uv + ruff + pytest.

## Milestones (verifiable)
1. **Tools** — `run_python` (sandboxed), `query_sql` (DuckDB), schema reader. Test each tool in isolation. **DONE.**
2. **Agent loop** — tool-use loop that profiles the data and emits a structured report (findings + numbers + a proposed baseline). Deterministic-enough via low temperature + seeds where possible. **DONE** — manual tool-use loop against an `LLMClient` protocol (`AnthropicClient` real-SDK wrapper + `FakeLLM` deterministic harness, no API key available); `results/report.json` is a committed demo artifact from `scripts/demo_fake_run.py`. Real-API run is Needs Nico (`ANTHROPIC_API_KEY` in `.env`).
3. **Baseline modelling** — agent proposes features, trains + evaluates a churn classifier, reports metrics. **DONE** — LogisticRegression on tenure/MonthlyCharges/TotalCharges, stratified 80/20 split, random_state=42; ROC-AUC ≈0.81, accuracy ≈0.77 in the demo run.
4. **Judge layer** — claim extraction + independent re-verification against the data; per-claim flag. **DONE** — no LLM: `judge.py`'s `verify_finding` re-runs each finding's `evidence_sql_or_code` through `query_sql`/`run_python` and compares the recomputed value to the claim (`math.isclose` for numerics, normalized string compare otherwise); `verify_baseline` re-trains the same LogisticRegression spec (features + random_state/max_iter parsed from `baseline.model`) and compares the recomputed metric; unreproducible specs are `unverified`, never rubber-stamped. `tests/test_judge.py` plants a false churn-rate claim (0.90 vs real ≈0.2654) and a false baseline AUC (0.99 vs real ≈0.8105) and proves both come back `contradicted`, plus unverified cases (broken query, nonexistent column, multi-value result, unrecognized model spec) and a tolerance-boundary case — 15 tests, all against the real CSV. `scripts/demo_judge_run.py` → `results/judged_report.json`: all 3/3 claims (2 findings + baseline) on the real committed report come back `verified`. **This is the differentiator.** Test with planted false claims → judge must catch them.
5. **App** — upload CSV → run agent → render report with inline `verified / unverified / contradicted` badges. **DONE** — `app.py` (Streamlit): runs the `FakeLLM` demo pipeline live against the real telco CSV (no API key), renders the summary stat row, per-finding cards (claim, evidence code, claimed value, verdict badge, recomputed value, detail), and the baseline card, all in the fixed status palette (verified `#0ca30c`, contradicted `#d03b3b`, unverified `#fab219`, each with a text label + icon, never color alone). Sidebar toggle "🔬 Inject a planted false claim (demo the judge)" calls the pure, directly-testable `inject_planted_false_claim` helper to overstate the churn-rate claim (0.75 vs. real ≈0.2654) before re-judging, so the viewer sees a red `contradicted` badge with claimed-vs-recomputed side by side — clearly labeled `[DEMO INJECTED LIE]`, never presented as real. CSV upload is a disabled stub with an honest note (FakeLLM is scripted for the telco schema; arbitrary uploads need the real agent). `tests/test_app.py` (5 tests): the tamper helper's leave-everything-else-untouched behavior and non-mutation, the planted lie coming back `contradicted` via `verify_report` directly, plus `streamlit.testing.v1.AppTest` smoke tests of the full app in both toggle states, asserting no exception and that the tampered state's rendered output contains a `contradicted` verdict. 72 tests green, ruff clean. Manually verified the app starts (`streamlit run app.py`, HTTP 200 + healthy) and stopped it — no server left running.
6. **Write-up** — README framing it as "honest agentic analysis", with a demo run on the churn data showing caught vs verified claims. **DONE** — `README.md` (repo had none before): leads with why the judge — not the agent — is the differentiator from a GPT wrapper; shows the honest 3/3-verified demo table (churn 26.5%, 11 blank `TotalCharges` rows, ROC-AUC 0.8105) and the tamper-demo table (claimed 0.75 → contradicted, recomputed ≈0.2654) as the money moment; documents the judge as a deliberate deterministic structured-recompute (no LLM extractor) with the reasoning from `judge.py`'s docstring; states the sandbox honestly as defense-in-depth, not a hard boundary, listing the documented gaps (no filesystem jail / absolute-path escape, `getaddrinfo` + raw `_socket` bypass the network kill-switch, `os.fork` around the rlimits); states no API key is needed for the demo and the real-Claude path is Needs Nico; gives exact reproduce steps (`uv run` the two demo scripts + `streamlit run app.py` + the test/lint gate).

## Judge layer

`src/agentic_analyst/judge.py` verifies a `Report` without ever calling an LLM.
This is a deliberate design choice, not a shortcut:

- **Why no LLM claim extractor.** A free-text "LLM-as-judge" would have to
  *parse* the report's prose back into a checkable claim before it could even
  start verifying — an extra, fallible step. This project's report is already
  structured: every `Finding` carries its own `evidence_sql_or_code` (the
  exact SQL or Python that produced it) and the single `value` it claims. So
  the judge skips extraction entirely and just **re-executes the evidence**
  through the same M1 tools (`query_sql`, `run_python`) the agent used,
  pulls out the one recomputed number, and compares it to the claim with
  `math.isclose` (numeric) or a normalized string compare (categorical).
  That recompute is the whole verification — nothing is judged by an LLM's
  opinion of the text. This needs no API key, is fully deterministic, and is
  directly testable, which an LLM-as-judge would not be. An LLM-based
  extractor for less-structured, free-text claims is explicitly out of scope
  / future work: it would be a *weaker* check for a report this structured,
  and there's no second use case yet to justify the extra machinery (YAGNI).
- **Three verdicts, not two.** `verified` = recompute succeeded and matches;
  `contradicted` = recompute succeeded and disagrees beyond tolerance;
  `unverified` = the evidence couldn't be executed, didn't produce a single
  comparable value, or claimed a spec the judge can't reproduce. The judge
  never collapses "couldn't check" into either verified or contradicted, and
  a recomputed value of `None` always means "couldn't check".
- **Tight tolerances.** `math.isclose` passes if *either* `rel_tol` or
  `abs_tol` is satisfied. Since the dominant claim type is a rate/proportion
  in [0, 1], both default to `0.01`: a loose `abs_tol` (e.g. 0.5) would let
  almost any non-extreme lie about a proportion slip through the abs_tol
  branch (a claim of 0.75 vs a real 0.2654 is only 0.48 apart). `abs_tol`
  only rescues near-zero claims where `rel_tol` collapses; real disagreement
  is caught by `rel_tol` at 1%.
- **Baseline verification actually retrains.** `verify_baseline` re-trains
  the same model (features + `random_state`/`max_iter` parsed out of
  `baseline.model`) and recomputes the metric — it does not trust
  `metric_value`. If the model class or metric isn't one the judge knows how
  to reproduce from the report fields alone, it returns `unverified` with a
  specific reason rather than faking a verdict.
- **Proof it isn't a rubber stamp.** `tests/test_judge.py` plants a false
  churn-rate claim (0.90 vs the real ≈0.2654) and a false baseline AUC (0.99
  vs the real ≈0.8105) and asserts both come back `contradicted` with the
  correct recomputed value — the milestone's whole point.
- **Honest demo.** `scripts/demo_judge_run.py` re-verifies the real,
  committed `results/report.json` and writes `results/judged_report.json`;
  all 3 claims (2 findings + the baseline) come back `verified`, because the
  committed report is, in fact, honest.

## Deliverables
- The agent + tools + judge as a reusable harness.
- Annotated report UI with verification badges.
- A demo run showing the judge catching at least one real overreach.

## "Krass" factor & honest framing
- Very 2026 (agentic) — but the closest of the three to what Nico has already built (the many `-scout` projects), and a reviewer spots "LLM wrapper" fast.
- The judge/verification layer is what rescues it into "real engineering". **Without the judge, skip this project.**

## Risks & limits
- Biggest risk: reads as "yet another GPT app". The verification layer must be genuinely load-bearing, not decorative.
- Sandbox security: executing model-generated code is the real hard part — treat as untrusted (no net, cpu/mem/time limits, no fs escape).
- Cost: agent + judge = many tokens per run. Cache, cap iterations, log spend.

## Open / Needs Nico
- Only worth it if the judge layer excites you — otherwise it's too close to existing projects (my read).
- Anthropic API key needed (put in `.env`, never commit; rotate if pasted anywhere).
