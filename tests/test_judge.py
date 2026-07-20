"""Tests for the judge / verification layer (M4).

These are the milestone-critical tests: every test here uses the REAL telco
CSV and the REAL M1 tools (query_sql, run_python) -- no API key, no LLM. The
whole point of M4 is to prove the judge actually recomputes and catches a
planted lie, rather than trusting whatever `value` a Finding claims -- the
planted-false-claim and planted-wrong-baseline tests below are that proof.
"""

from pathlib import Path

import pytest

from agentic_analyst.judge import verify_baseline, verify_finding, verify_report
from agentic_analyst.report import Baseline, Dataset, Finding, Report

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"

CHURN_RATE_QUERY = "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate FROM data"
REAL_CHURN_RATE = 0.2654  # 1869 / 7043 -- known fact about this CSV (see test_tools.py)
# Churn rate among Month-to-month customers only -- known fact about this CSV,
# recomputed directly via duckdb for the attack-4 test below.
REAL_MONTH_TO_MONTH_CHURN_RATE = 0.4270967741935484

REAL_BASELINE_MODEL = "LogisticRegression(random_state=42, max_iter=1000)"
REAL_BASELINE_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
REAL_BASELINE_AUC = 0.8105272675605157  # from results/report.json, same recipe/seed


# --- verify_finding: correct claim ---------------------------------------------


def test_correct_finding_is_verified(tmp_path):
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


# --- verify_finding: THE milestone-critical test -------------------------------


def test_planted_false_churn_rate_claim_is_contradicted(tmp_path):
    finding = Finding(
        claim="90% of customers churned -- almost everyone leaves!",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=0.90,  # planted lie: the real churn rate is ~26.5%
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.finding.value == 0.90  # claim preserved for the report/UI
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)
    # This is already contradicted by the value<->evidence compare; the
    # claim-text check (see below) must not also fire and overwrite the
    # detail with its own reason -- the original mismatch explanation stays
    # authoritative.
    assert "claim_text_disagrees_with_value" not in judged.detail


def test_moderate_proportion_lie_is_contradicted_under_default_tolerance(tmp_path):
    # Regression for the abs_tol bug: with abs_tol=0.5, a claim of 0.75 vs the
    # real 0.2654 (abs diff 0.48, well under 0.5) sailed through as VERIFIED --
    # gutting the "catch lies" promise for the dominant claim type (rates in
    # [0, 1]). With the tightened default tolerance it must be contradicted.
    finding = Finding(
        claim="75% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=0.75,  # abs diff from real ~0.2654 is ~0.48; rel diff ~83%
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)  # default tolerances

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


# --- verify_finding: unverified (couldn't check) is distinct from contradicted


def test_finding_with_unexecutable_query_is_unverified(tmp_path):
    finding = Finding(
        claim="Some claim backed by a query against a table that doesn't exist.",
        evidence_sql_or_code="SELECT * FROM this_table_does_not_exist",
        value=42,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert judged.detail  # a reason must be present


def test_finding_with_nonexistent_column_is_unverified(tmp_path):
    finding = Finding(
        claim="Some claim backed by a query referencing a column that doesn't exist.",
        evidence_sql_or_code="SELECT this_column_does_not_exist FROM data",
        value=42,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_finding_with_multi_value_query_is_unverified(tmp_path):
    # Two columns, one row -- not the 1x1 single-value result a finding's
    # evidence is supposed to produce, so no comparable value exists.
    finding = Finding(
        claim="Some claim backed by a query that returns more than one value.",
        evidence_sql_or_code="SELECT tenure, MonthlyCharges FROM data LIMIT 1",
        value=1,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_numeric_claim_vs_non_numeric_recompute_is_unverified_with_none(tmp_path):
    # A numeric claim whose evidence returns a categorical (non-numeric) value
    # is not comparable -- it must be unverified, and (invariant) the
    # recomputed_value must be None, not the raw string that was found.
    finding = Finding(
        claim="The most common contract type, as a number, is 5.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value=5,  # numeric claim, but recompute yields the string 'Month-to-month'
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert "Month-to-month" in judged.detail  # what was actually found is surfaced


# --- verify_finding: non-numeric claims ----------------------------------------


def test_non_numeric_claim_verified_by_normalized_string_compare(tmp_path):
    finding = Finding(
        claim="Month-to-month is the most common contract type.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value="  month-to-month  ",  # deliberately mixed case + surrounding whitespace
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == "Month-to-month"


def test_non_numeric_claim_contradicted_when_wrong(tmp_path):
    finding = Finding(
        claim="Two year is the most common contract type.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value="Two year",  # planted lie: Month-to-month is actually the mode
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == "Month-to-month"


# --- verify_finding: Python evidence path (documents the last-line convention)


def test_python_evidence_last_stdout_line_is_the_value(tmp_path):
    finding = Finding(
        claim="2 + 2 is 4 (sanity check for the Python evidence convention).",
        # The `pd.read_csv(...)` line is not part of what this test is
        # checking (that's the last-stdout-line extraction convention below)
        # -- it's here only so this evidence passes the evidence-plausibility
        # check (Task A2): Python evidence must reference the dataset's real
        # path or it is rejected before extraction is even attempted.
        evidence_sql_or_code=(
            f"import pandas as pd\npd.read_csv({str(TELCO_CSV)!r})\n"
            "print('some noise line')\nprint(2 + 2)"
        ),
        value=4,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == 4


# --- verify_finding: tolerance boundary ----------------------------------------


def test_tolerance_boundary_just_inside_verifies_just_outside_contradicts(tmp_path):
    # "FROM data LIMIT 1" is only here to satisfy the evidence-plausibility
    # check (Task A2) -- it does not change the constant, which is what
    # isolates the tolerance math itself from real-data variability.
    query = "SELECT 1000.0 AS v FROM data LIMIT 1"

    just_inside = Finding(claim="approx 1000", evidence_sql_or_code=query, value=1009)  # 0.9% off
    just_outside = Finding(claim="approx 1000", evidence_sql_or_code=query, value=1011)  # 1.1% off

    inside_result = verify_finding(just_inside, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.5)
    outside_result = verify_finding(just_outside, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.5)

    assert inside_result.verdict == "verified"
    assert outside_result.verdict == "contradicted"


# --- verify_finding: claim text vs claimed value (REVIEW.md A-1, attack 3) -----
# `_compare` above only checks value<->evidence consistency -- both fields are
# agent-authored. The natural-language `claim` is the sentence a reader
# actually trusts and was never read at all: a claim that lies in its text
# while carrying an honest, evidence-matching `value` used to pass as
# `verified`. These tests prove the judge now reads the claim text too.


def test_claim_text_lying_about_percentage_is_contradicted_even_with_honest_value(tmp_path):
    finding = Finding(
        claim="75% of customers churned -- almost everyone leaves!",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,  # honest value, honest evidence -- only the claim text lies
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)
    assert "claim_text_disagrees_with_value" in judged.detail


def test_claim_text_quoting_the_correct_percentage_stays_verified(tmp_path):
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path, rel_tol=0.01, abs_tol=0.01)

    assert judged.verdict == "verified"


def test_claim_text_quoting_a_plain_float_matching_value_stays_verified(tmp_path):
    finding = Finding(
        claim="The churn rate is 0.2654, which is a concerning figure.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_with_thousands_separator_matching_value_stays_verified(tmp_path):
    finding = Finding(
        claim="1,869 customers churned in total.",
        evidence_sql_or_code="SELECT count(*) AS n FROM data WHERE Churn = 'Yes'",
        value=1869,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_with_wrong_thousands_separator_count_is_contradicted(tmp_path):
    finding = Finding(
        claim="1,234 customers churned in total.",  # wrong count in the sentence
        evidence_sql_or_code="SELECT count(*) AS n FROM data WHERE Churn = 'Yes'",
        value=1869,  # honest value, honest evidence -- only the claim text lies
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert "claim_text_disagrees_with_value" in judged.detail


def test_claim_text_with_no_numeric_token_passes_through_silently(tmp_path):
    # Not every claim quotes its number -- a claim about *why* customers churn
    # carries no numeral to check against the rate finding's value, so the new
    # check must not interfere with the pre-existing verified verdict.
    finding = Finding(
        claim="Most customers who churn are on a month-to-month contract.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_spelling_out_percent_matches_value_stays_verified(tmp_path):
    # "percent" (no %-sign) must be recognized exactly like "%" -- previously
    # only the symbol was, so this honest claim wrongly came back
    # `contradicted` (26.5 vs 0.2654 with no %-candidate to try 0.265).
    finding = Finding(
        claim="26.5 percent of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_text_spelling_out_per_cent_two_words_matches_value_stays_verified(tmp_path):
    # The two-word spelling ("per cent") must be recognized too.
    finding = Finding(
        claim="26.5 per cent of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_claim_with_unrelated_decoy_number_next_to_correct_one_stays_verified(tmp_path):
    # The docstring's own example: a claim naming a year alongside the
    # correctly-quoted rate must still verify -- only ONE candidate has to
    # match, so the decoy (2021) doesn't matter.
    finding = Finding(
        claim="In 2021, 26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


def test_count_over_total_claim_is_a_known_false_positive_for_a_rate_finding(tmp_path):
    # KNOWN LIMITATION, pinned deliberately (see _claim_agrees_with_value's
    # docstring): "1,869 of 7,043 customers churned" is exactly as honest as
    # "26.5% of customers churned" for this finding (1869/7043 == the real
    # churn rate, value=0.2654) -- but the check only recognizes a claim that
    # quotes `value` literally or in %-form, not a count-over-total phrasing.
    # Neither 1869 nor 7043 is close to 0.2654 under any tolerance, so this
    # currently -- and knowingly -- comes back `contradicted`.
    #
    # This is NOT patched with a magnitude/ratio-aware guard: a tolerance
    # loose enough to accept "1,869 of 7,043" as agreeing with 0.2654 would
    # also have to accept a genuinely wrong number related to `value` by some
    # other arithmetic, reopening the hole this check exists to close (e.g. a
    # planted "7500% churned" lie must keep failing). The real fix belongs in
    # the agent's report-writing contract -- a claim must quote either
    # `value` itself or its %-form -- not a smarter comparison in the judge.
    # Must be listed in the README's "What the judge does NOT catch" section
    # (tracked as a later task, not part of this one).
    finding = Finding(
        claim="1,869 of 7,043 customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"  # known false positive, see comment above
    assert "claim_text_disagrees_with_value" in judged.detail


def test_non_numeric_value_with_numeric_token_in_claim_stays_verified_and_does_not_crash(tmp_path):
    # Regression for the `value_is_numeric` guard in verify_finding: the
    # claim-text check must never run math.isclose against a categorical
    # (string) value, even when the claim text itself contains a numeral
    # ("55%") unrelated to the recomputed contract type. Without the guard
    # this raises TypeError: must be real number, not str -- and no existing
    # categorical test has a digit in its claim, so the suite would stay
    # green if the guard were silently deleted.
    finding = Finding(
        claim="Month-to-month is the most common contract, held by 55% of customers.",
        evidence_sql_or_code=(
            "SELECT Contract FROM data GROUP BY Contract ORDER BY count(*) DESC LIMIT 1"
        ),
        value="Month-to-month",
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == "Month-to-month"


def test_bool_value_falls_through_to_string_compare_without_crashing(tmp_path):
    # Optional corner case: validate_report rejects a bool `value` upstream,
    # but verify_finding itself must not crash if one slips through via
    # direct Finding construction. bool is a subclass of int, so the
    # `value_is_numeric` guard must exclude it explicitly, mirroring
    # `_compare`'s own `claimed_is_numeric` check.
    finding = Finding(
        claim="This claim mentions 42 in passing, but the value is a boolean.",
        # `pd.read_csv(...)` line: satisfies the evidence-plausibility check
        # (Task A2), not part of what this test is about.
        evidence_sql_or_code=f"import pandas as pd\npd.read_csv({str(TELCO_CSV)!r})\nprint(True)",
        value=True,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"


# --- verify_finding: evidence plausibility (REVIEW.md A-1, attacks 1+2+4) -----
# The claim-text check above (Task A1) closes attack 3 (honest evidence,
# honest value, lying claim text). It does nothing for evidence that never
# touched the real dataset at all: `SELECT 0.75 AS churn_rate` (no FROM
# clause) or `print(0.90)` (no CSV read) always recompute to exactly their
# own literal, so they "match" whatever value is claimed next to them -- a
# self-proving lie. These tests prove the judge now rejects evidence that
# could not possibly have queried/read the real data, and pin the one attack
# (4) that remains a documented, not mechanically closable, gap.


def test_sql_evidence_that_never_touches_the_data_view_is_unverified(tmp_path):
    # REVIEW.md A-1, attack 1: `SELECT 0.75 AS churn_rate` has no FROM clause
    # at all -- it can't have queried anything -- yet it recomputes to
    # exactly 0.75 and used to pass as `verified`.
    finding = Finding(
        claim="75% of customers churned.",
        evidence_sql_or_code="SELECT 0.75 AS churn_rate",
        value=0.75,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert "evidence_does_not_touch_data" in judged.detail


def test_sql_evidence_aliasing_a_column_as_data_without_reading_the_view_is_unverified(tmp_path):
    # Guards the check's own design against the exact dodge REVIEW.md warns
    # about: naming an *output* column/alias `data` instead of actually
    # reading the view in a FROM/JOIN clause. A check that merely asked "does
    # the token `data` appear anywhere in the query" would be fooled by this;
    # the real check requires `data` to appear where a table reference
    # belongs.
    finding = Finding(
        claim="75% of customers churned.",
        evidence_sql_or_code="SELECT 0.75 AS data",
        value=0.75,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert "evidence_does_not_touch_data" in judged.detail


def test_python_evidence_that_never_reads_the_dataset_is_unverified(tmp_path):
    # REVIEW.md A-1, attack 2: `print(0.90)` never opens the CSV -- it can't
    # have computed anything about the real data -- yet it recomputes to
    # exactly 0.90 and used to pass as `verified`.
    finding = Finding(
        claim="90% of customers churned -- almost everyone leaves!",
        evidence_sql_or_code="print(0.90)",
        value=0.90,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None
    assert "evidence_does_not_touch_data" in judged.detail


def test_with_cte_evidence_referencing_data_is_still_verified(tmp_path):
    # False-positive guard: a legitimate query.py's own single-statement
    # convention allows a WITH-CTE whose FROM data reference sits inside the
    # CTE body, not the outermost FROM. The plausibility check must not
    # reject this shape.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            "WITH churn_flags AS ("
            "SELECT CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END AS flag FROM data"
            ") SELECT avg(flag) AS churn_rate FROM churn_flags"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


def test_python_evidence_reading_the_real_csv_path_is_still_verified(tmp_path):
    # False-positive guard: legitimate Python evidence that actually loads
    # the dataset via its real path must not be rejected.
    finding = Finding(
        claim="There are 7043 rows in the dataset.",
        evidence_sql_or_code=(
            f"import pandas as pd\ndf = pd.read_csv({str(TELCO_CSV)!r})\nprint(len(df))"
        ),
        value=7043,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == 7043


def test_population_switch_evidence_is_a_documented_known_gap_still_verified(tmp_path):
    # REVIEW.md A-1, attack 4 -- KNOWN GAP, deliberately NOT fixed by the
    # evidence-plausibility check above (see verify_finding's docstring).
    # The evidence below is a completely legitimate, executable query
    # against the real `data` view -- it genuinely computes the churn rate
    # among Month-to-month customers. The claim text, however, describes the
    # result as being about "ALL customers", silently switching the
    # population the number is actually about. Both the plausibility check
    # (it DOES reference `FROM data`) and the value<->evidence compare (the
    # recomputed number DOES match the claimed value) pass -- there is
    # nothing execution-based to catch here, because nothing about the
    # execution is dishonest; only the claim's framing of the population is.
    #
    # This is exactly the failure VeriGraph (arXiv 2606.16603) names
    # "executability can mask weak semantic transitions": see
    # research/2026-07-19-agentic-judge-sota.md section 4. Mechanically
    # closing this would require comparing the claim text's stated
    # population ("ALL customers") against the query's actual WHERE
    # predicates -- a semantic check well beyond what a deterministic,
    # execution-based judge does. Documented here as a known limitation, not
    # silently left undiscovered.
    finding = Finding(
        claim="42.7% of ALL customers churned.",
        evidence_sql_or_code=(
            "SELECT avg(CASE WHEN \"Churn\"='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate "
            "FROM data WHERE \"Contract\"='Month-to-month'"
        ),
        value=REAL_MONTH_TO_MONTH_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"  # known gap -- see comment above
    assert judged.recomputed_value == pytest.approx(REAL_MONTH_TO_MONTH_CHURN_RATE, abs=1e-6)


# --- verify_baseline ------------------------------------------------------------


def test_real_baseline_recomputes_to_verified(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


def test_planted_wrong_baseline_metric_is_contradicted(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.99,  # planted lie: the real ROC-AUC is ~0.81
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value == pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


def test_baseline_with_unrecognized_model_class_is_unverified(tmp_path):
    baseline = Baseline(
        model="RandomForestClassifier(random_state=42)",
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.85,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_baseline_without_random_state_is_unverified(tmp_path):
    baseline = Baseline(
        model="LogisticRegression(max_iter=1000)",  # no random_state -> not reproducible
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.81,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


def test_baseline_with_nonexistent_feature_column_is_unverified(tmp_path):
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=[*REAL_BASELINE_FEATURES, "this_column_does_not_exist"],
        metric_name="roc_auc",
        metric_value=0.81,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.recomputed_value is None


# --- verify_baseline: tolerance boundary (REVIEW.md A-3) -----------------------
# The retrain is fully deterministic (fixed seed, verified byte-identical
# reproduction) -- there is no legitimate source of run-to-run variance to
# excuse a loose tolerance. This first case is caught by tightening rel_tol
# alone (0.85 vs the real ~0.8105 is a 4.87% relative gap, comfortably above
# even the OLD abs_tol=0.01 floor); the next test below pins the case
# rel_tol alone does NOT fix -- a smaller embellishment that used to hide
# under that floor.


def test_baseline_tolerance_boundary_matches_deterministic_recompute(tmp_path):
    review_a3_lie = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.85,  # REVIEW.md A-3: embellished from the real 0.8105
    )
    float_noise = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC + 1e-9,  # well inside float-noise epsilon
    )

    lie_result = verify_baseline(review_a3_lie, TELCO_CSV, tmp_path)
    noise_result = verify_baseline(float_noise, TELCO_CSV, tmp_path)

    assert lie_result.verdict == "contradicted"
    assert noise_result.verdict == "verified"


def test_baseline_abs_tol_floor_does_not_mask_a_sub_floor_embellishment(tmp_path):
    # Regression: tightening rel_tol alone (see the test above) is not
    # sufficient. `_BASELINE_ABS_TOL` is an unconditional floor in
    # math.isclose's max(rel_tol*value, abs_tol) -- for a metric bounded to
    # [0, 1], rel_tol*value can never exceed abs_tol once abs_tol is
    # non-trivial, so abs_tol alone governs the comparison. A metric
    # embellished from the real ~0.8105 to 0.82 (diff ~0.0095) sits just
    # under the old abs_tol=0.01 floor and used to pass as "verified" even
    # with rel_tol=1e-6 -- this is the exact gap the spec review flagged.
    sub_floor_lie = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=0.82,  # diff from REAL_BASELINE_AUC ~0.0095 -- below the old 0.01 floor
    )
    float_noise = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC + 1e-9,  # genuine float-noise scale
    )

    lie_result = verify_baseline(sub_floor_lie, TELCO_CSV, tmp_path)
    noise_result = verify_baseline(float_noise, TELCO_CSV, tmp_path)

    assert lie_result.verdict == "contradicted"
    assert noise_result.verdict == "verified"


def test_baseline_metric_depends_on_feature_set(tmp_path):
    # Proves verify_baseline actually re-trains on the declared features rather
    # than echoing the claim: the real ~0.8105 AUC came from 3 features; a
    # single-feature model produces a materially different metric, so claiming
    # the 3-feature AUC while declaring only ['tenure'] must be contradicted,
    # and the recomputed value must differ from the claim.
    baseline = Baseline(
        model=REAL_BASELINE_MODEL,
        features=["tenure"],  # fewer features than produced REAL_BASELINE_AUC
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC,
    )

    judged = verify_baseline(baseline, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.recomputed_value is not None
    assert judged.recomputed_value != pytest.approx(REAL_BASELINE_AUC, abs=1e-3)


# --- verify_report: wiring + summary counts ------------------------------------


def test_verify_report_summarizes_findings_and_baseline(tmp_path):
    report = Report(
        dataset=Dataset(name="telco-customer-churn", rows=7043, cols=21),
        findings=[
            Finding(
                claim="26.5% of customers churned.",
                evidence_sql_or_code=CHURN_RATE_QUERY,
                value=REAL_CHURN_RATE,
            ),
            Finding(
                claim="90% of customers churned!",  # planted lie
                evidence_sql_or_code=CHURN_RATE_QUERY,
                value=0.90,
            ),
        ],
        data_quality_issues=["n/a"],
        baseline=Baseline(
            model=REAL_BASELINE_MODEL,
            features=REAL_BASELINE_FEATURES,
            metric_name="roc_auc",
            metric_value=REAL_BASELINE_AUC,
        ),
    )

    judged = verify_report(report, TELCO_CSV, tmp_path)

    assert judged.summary == {"verified": 2, "unverified": 0, "contradicted": 1}
    assert len(judged.findings) == 2
    assert judged.baseline.verdict == "verified"
