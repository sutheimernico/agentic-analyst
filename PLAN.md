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
- **Judge layer** — a separate LLM pass that extracts each claim from the report and re-checks it against the data (recompute the number, mark `verified` / `unverified` / `contradicted`).
- Baseline model: scikit-learn (the agent proposes + trains a churn baseline).
- App: FastAPI or Streamlit — upload → live agent run → annotated report.
- Tooling: uv + ruff + pytest.

## Milestones (verifiable)
1. **Tools** — `run_python` (sandboxed), `query_sql` (DuckDB), schema reader. Test each tool in isolation.
2. **Agent loop** — tool-use loop that profiles the data and emits a structured report (findings + numbers + a proposed baseline). Deterministic-enough via low temperature + seeds where possible.
3. **Baseline modelling** — agent proposes features, trains + evaluates a churn classifier, reports metrics.
4. **Judge layer** — claim extraction + independent re-verification against the data; per-claim flag. **This is the differentiator.** Test with planted false claims → judge must catch them.
5. **App** — upload CSV → run agent → render report with inline `verified / unverified / contradicted` badges.
6. **Write-up** — README framing it as "honest agentic analysis", with a demo run on the churn data showing caught vs verified claims.

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
