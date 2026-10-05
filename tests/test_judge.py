"""Tests for the judge / verification layer (M4).

These are the milestone-critical tests: every test here uses the REAL telco
CSV and the REAL M1 tools (query_sql, run_python) -- no API key, no LLM. The
whole point of M4 is to prove the judge actually recomputes and catches a
planted lie, rather than trusting whatever `value` a Finding claims -- the
planted-false-claim and planted-wrong-baseline tests below are that proof.
"""

from pathlib import Path

import pytest

from agentic_analyst import judge
from agentic_analyst.judge import verify_baseline, verify_finding, verify_report
from agentic_analyst.report import Baseline, Dataset, Finding, Report

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"

CHURN_RATE_QUERY = "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate FROM data"
REAL_CHURN_RATE = 0.2654  # 1869 / 7043 -- known fact about this CSV (see test_tools.py)
# Churn rate among Month-to-month customers only -- known fact about this CSV,
# recomputed directly via duckdb for the attack-4 test below.
REAL_MONTH_TO_MONTH_CHURN_RATE = 0.4270967741935484
# avg(MonthlyCharges) among Fiber-optic customers only -- known fact about this
# CSV, for the population-switch heuristic tests below.
REAL_FIBER_AVG_MONTHLY_CHARGE = 91.50012919896615
# avg(MonthlyCharges) among tenure>60 customers -- a numeric-range filter the
# heuristic's simple-predicate parser deliberately cannot parse (no-fire case).
REAL_TENURE_GT60_AVG_MONTHLY_CHARGE = 75.95270078180513

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


def test_python_evidence_result_sentinel_is_extracted_from_decorated_stdout(tmp_path):
    # Task A4: a `RESULT: <value>` line is the structured sentinel now
    # documented as the real contract (see RUN_PYTHON_TOOL's description in
    # agent.py and _extract_python_value's docstring) -- evidence code may
    # print explanatory text around the value as long as it also emits this
    # line. Proven here by printing a decorated noise line AFTER the
    # sentinel too: a naive "last non-blank line is the value" extractor
    # would wrongly return "done." instead of the real churn rate.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "rate = (df['Churn'] == 'Yes').mean()\n"
            "print('the rate is:')\n"
            "print(f'RESULT: {rate:.4f}')\n"
            "print('done.')\n"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


def test_python_evidence_lowercase_result_sentinel_is_still_recognized(tmp_path):
    # Reviewer follow-up: the sentinel match must be case-insensitive. LLMs
    # drop case details routinely -- before this fix, a lowercase `result:
    # 0.2654` line silently fell through to the legacy fallback, which then
    # returned the ENTIRE decorated line ("result: 0.2654") as the raw
    # value; that string fails to parse as a number, so the finding came
    # back `unverified` instead of `verified` with no loud signal that the
    # sentinel was even attempted.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "rate = (df['Churn'] == 'Yes').mean()\n"
            "print(f'result: {rate:.4f}')\n"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


def test_python_evidence_multiple_disagreeing_result_lines_pins_last_one_wins(tmp_path):
    # Reviewer follow-up: _extract_python_value's docstring claims the last
    # `RESULT:` line wins when more than one is printed, but nothing pinned
    # that behavior. This is a deliberately malformed evidence script (two
    # disagreeing RESULT lines) -- pinning the exact chosen tie-break here
    # rather than leaving it as an untested docstring claim. Disagreement
    # between multiple RESULT lines is a documented, accepted limitation
    # (see the docstring) rather than a separately flagged case: doing so
    # would require widening the extractor's two-element return contract.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "print('RESULT: 0.9')\n"
            "rate = (df['Churn'] == 'Yes').mean()\n"
            "print(f'RESULT: {rate:.4f}')\n"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)


def test_broken_extractor_contract_raises_loud_value_error(tmp_path, monkeypatch):
    # Regression guard for the bare `assert raw_value is not None` this task
    # replaced (review notes A-4): if an extractor ever reports success
    # (error_detail=None) but returns no raw_value -- an internal contract
    # break that no real evidence string can trigger through normal
    # execution -- the failure must be a loud, diagnosable ValueError naming
    # the extractor and the evidence, not a bare AssertionError with no
    # context (and one that silently vanishes if Python is ever run with
    # -O, which strips asserts).
    # Named (not lambda) so its __name__ matches the real extractor's --
    # verify_finding now builds the diagnostic name via
    # `_extract_python_value.__name__` (reviewer follow-up, replacing a
    # hand-typed string literal), so a lambda stand-in here would report
    # itself as "<lambda>" and this test would stop proving anything about
    # the real extractor's name appearing in the error.
    def _extract_python_value(evidence, workdir):
        return (None, None, None)

    monkeypatch.setattr(judge, "_extract_python_value", _extract_python_value)
    finding = Finding(
        claim="broken extractor contract regression guard.",
        evidence_sql_or_code=f"import pandas as pd\npd.read_csv({str(TELCO_CSV)!r})\nprint(1)",
        value=1,
    )

    with pytest.raises(ValueError, match="_extract_python_value"):
        verify_finding(finding, TELCO_CSV, tmp_path)


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


# --- verify_finding: claim text vs claimed value (review notes A-1, attack 3) -----
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


# --- verify_finding: evidence plausibility (review notes A-1, attacks 1+2+4) -----
# The claim-text check above (Task A1) closes attack 3 (honest evidence,
# honest value, lying claim text). It does nothing for evidence that never
# touched the real dataset at all: `SELECT 0.75 AS churn_rate` (no FROM
# clause) or `print(0.90)` (no CSV read) always recompute to exactly their
# own literal, so they "match" whatever value is claimed next to them -- a
# self-proving lie. These tests prove the judge now rejects evidence that
# could not possibly have queried/read the real data, and pin the one attack
# (4) that remains a documented, not mechanically closable, gap.


def test_sql_evidence_that_never_touches_the_data_view_is_unverified(tmp_path):
    # review notes A-1, attack 1: `SELECT 0.75 AS churn_rate` has no FROM clause
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
    # Guards the check's own design against the exact dodge review notes warns
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
    # review notes A-1, attack 2: `print(0.90)` never opens the CSV -- it can't
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


def test_population_switch_evidence_is_downgraded_to_unverified(tmp_path):
    # review notes A-1, attack 4 -- the CANONICAL known-gap case, now PARTIALLY
    # mitigated by the population-switch heuristic (Task C3). The evidence
    # below is a completely legitimate, executable query against the real
    # `data` view -- it genuinely computes the churn rate among
    # Month-to-month customers, and the value<->evidence compare passes. The
    # claim text, however, describes the result as being about "ALL
    # customers", silently switching the population the number is actually
    # about. Nothing execution-based catches this: the query IS honest about
    # the data, only the claim's framing of the population is not.
    #
    # The heuristic closes the canonical case conservatively (see
    # `_population_mismatch_reason` in judge.py): the top-level WHERE filters
    # on `Contract='Month-to-month'`, and NEITHER the column name token
    # ("contract") NOR any filtered-value token ("month"/"to") appears in the
    # claim text -- so the filter is hidden from the reader and the verdict is
    # downgraded `verified` -> `unverified` with reason
    # `population_mismatch_suspected`. This is a heuristic, not a proof: it
    # only fires on a single simple `col = 'literal'` / `col IN (...)`
    # predicate whose tokens are absent from the claim, and prefers a false
    # negative (do nothing) to a false positive everywhere else -- see the
    # dedicated no-fire tests below and `verify_finding`'s docstring.
    finding = Finding(
        claim="42.7% of ALL customers churned.",
        evidence_sql_or_code=(
            "SELECT avg(CASE WHEN \"Churn\"='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate "
            "FROM data WHERE \"Contract\"='Month-to-month'"
        ),
        value=REAL_MONTH_TO_MONTH_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert "population_mismatch_suspected" in judged.detail
    # Invariant `unverified => recomputed_value is None`: the (correct-for-the-
    # subset) number is surfaced in the detail, not as a recomputed_value that
    # would read as a checked, trustworthy figure for the claimed population.
    assert judged.recomputed_value is None


# --- verify_finding: population-switch heuristic (Task C3) --------------------
# Partial mitigation for review notes attack 4. The heuristic extracts a single
# top-level `col = 'literal'` / `col IN (...)` WHERE predicate and, if NEITHER
# the column-name token(s) NOR the filtered-value token(s) appear in the claim
# text (case-insensitive, alphanumeric-token match), downgrades a still-
# `verified` verdict to `unverified` (`population_mismatch_suspected`). It is
# deliberately conservative: it prefers a false negative to a false positive,
# never weakens `contradicted`/`unverified`, and does nothing at all on a
# WHERE clause it cannot parse as one simple predicate.

_FIBER_AVG_CHARGE_EVIDENCE = (
    "SELECT avg(MonthlyCharges) AS avg_charge FROM data WHERE InternetService='Fiber optic'"
)


def test_population_switch_hidden_where_filter_is_flagged_unverified(tmp_path):
    # FIRE case: the number is genuinely the Fiber-optic average, but the claim
    # frames it as "all customers" and mentions neither the InternetService
    # column nor the "Fiber optic" value -> population_mismatch_suspected.
    finding = Finding(
        claim="The average monthly charge across all customers is 91.5 dollars.",
        evidence_sql_or_code=_FIBER_AVG_CHARGE_EVIDENCE,
        value=REAL_FIBER_AVG_MONTHLY_CHARGE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert "population_mismatch_suspected" in judged.detail
    assert judged.recomputed_value is None


def test_population_switch_no_fire_when_claim_mentions_the_filtered_value(tmp_path):
    # NO-FIRE (value disclosed): the claim mentions the "fiber-optic" value, so
    # a reader can see the number is scoped -- the heuristic must not fire.
    finding = Finding(
        claim="Among fiber-optic customers, the average monthly charge is 91.5 dollars.",
        evidence_sql_or_code=_FIBER_AVG_CHARGE_EVIDENCE,
        value=REAL_FIBER_AVG_MONTHLY_CHARGE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_FIBER_AVG_MONTHLY_CHARGE, abs=1e-6)


def test_population_switch_no_fire_when_claim_mentions_the_filter_column(tmp_path):
    # NO-FIRE (column disclosed): the claim names the InternetService column
    # (even without the value), signalling the number is broken out by it.
    finding = Finding(
        claim="Broken out by InternetService, the average monthly charge is 91.5 dollars.",
        evidence_sql_or_code=_FIBER_AVG_CHARGE_EVIDENCE,
        value=REAL_FIBER_AVG_MONTHLY_CHARGE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_FIBER_AVG_MONTHLY_CHARGE, abs=1e-6)


def test_population_switch_heuristic_never_weakens_a_contradicted_verdict(tmp_path):
    # CONSERVATIVITY: a value lie on evidence that ALSO hides its population.
    # The value<->evidence compare contradicts first; the population heuristic
    # (which only ever runs on a still-`verified` verdict) must NOT soften that
    # caught lie into a milder `unverified`.
    finding = Finding(
        claim="The average monthly charge across all customers is 50 dollars.",
        evidence_sql_or_code=_FIBER_AVG_CHARGE_EVIDENCE,
        value=50.0,  # real Fiber-optic avg is ~91.5 -- a lie, must stay contradicted
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert "population_mismatch_suspected" not in judged.detail


def test_population_switch_no_fire_when_where_clause_is_not_a_simple_predicate(tmp_path):
    # NO-FIRE (unparseable): a numeric-range filter (`tenure > 60`) is not a
    # `col = 'literal'` / `col IN (...)` predicate, so the parser bails and does
    # nothing -- the finding stays `verified` on the honest value compare even
    # though the claim ("all customers") does hide the filter. A false negative
    # by design: the heuristic never guesses at a predicate it can't parse.
    finding = Finding(
        claim="The average monthly charge across all customers is 75.95 dollars.",
        evidence_sql_or_code="SELECT avg(MonthlyCharges) AS avg_charge FROM data WHERE tenure > 60",
        value=REAL_TENURE_GT60_AVG_MONTHLY_CHARGE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_TENURE_GT60_AVG_MONTHLY_CHARGE, abs=1e-6)


# --- verify_finding: population provenance (Task A5, SOTA rec #5) -------------
# `evidence_row_count` records how many rows of the `data` view actually fed
# the recomputed value -- pure metadata, never a verdict input. A finding
# whose evidence's population is under 10% of the full dataset additionally
# gets `provenance_note = "narrow_subset"`, so a reader can see "this claim
# rests on 11 of 7,043 rows" instead of trusting an unqualified aggregate.


def test_sql_finding_over_the_whole_table_records_full_row_count(tmp_path):
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=CHURN_RATE_QUERY,  # no WHERE -- scans the whole table
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"  # this is metadata, not a verdict change
    assert judged.evidence_row_count == 7043
    assert judged.provenance_note is None  # 100% of the dataset -- not narrow


def test_sql_finding_filtered_to_narrow_subset_gets_provenance_note(tmp_path):
    # tenure=0 customers are the 11 (of 7043) rows that have never been
    # billed yet -- 0.16% of the dataset, well under the 10% threshold.
    finding = Finding(
        claim="No tenure-0 customers have churned.",
        evidence_sql_or_code=(
            "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate "
            "FROM data WHERE tenure = 0"
        ),
        value=0.0,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.evidence_row_count == 11
    assert judged.provenance_note == "narrow_subset"


def test_narrow_subset_note_does_not_mask_a_contradicted_verdict(tmp_path):
    # Provenance is informational, never a verdict input in either direction:
    # a lie built on a narrow population must still come back `contradicted`,
    # WITH the narrow-subset note attached, not instead of it.
    finding = Finding(
        claim="90% of tenure-0 customers churned.",
        evidence_sql_or_code=(
            "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate "
            "FROM data WHERE tenure = 0"
        ),
        value=0.90,  # planted lie: the real rate among tenure=0 customers is 0.0
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "contradicted"
    assert judged.evidence_row_count == 11
    assert judged.provenance_note == "narrow_subset"


def test_sql_evidence_where_clause_inside_a_cte_is_still_counted_correctly(tmp_path):
    # The population-counting query reuses the evidence's own CTE (and
    # whatever WHERE it contains) rather than re-deriving the filter by hand
    # -- proven here with a WHERE that lives inside the CTE body, not the
    # outer query.
    finding = Finding(
        claim="No tenure-0 customers have churned.",
        evidence_sql_or_code=(
            "WITH tenure0 AS (SELECT * FROM data WHERE tenure = 0) "
            "SELECT avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate FROM tenure0"
        ),
        value=0.0,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.evidence_row_count == 11
    assert judged.provenance_note == "narrow_subset"


def test_python_evidence_rows_sentinel_is_recorded_as_evidence_row_count(tmp_path):
    finding = Finding(
        claim="No tenure-0 customers have churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "subset = df[df['tenure'] == 0]\n"
            "rate = (subset['Churn'] == 'Yes').mean()\n"
            "print(f'RESULT: {rate:.4f}')\n"
            "print(f'ROWS: {len(subset)}')\n"
        ),
        value=0.0,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.evidence_row_count == 11
    assert judged.provenance_note == "narrow_subset"


def test_python_evidence_without_rows_sentinel_leaves_evidence_row_count_none(tmp_path):
    # ROWS: is an OPTIONAL second sentinel -- absent, evidence_row_count must
    # be None (not 0, not a guess), and no provenance_note can be derived.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "rate = (df['Churn'] == 'Yes').mean()\n"
            "print(f'RESULT: {rate:.4f}')\n"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.evidence_row_count is None
    assert judged.provenance_note is None


def test_sql_companion_row_count_query_failure_degrades_to_none(tmp_path):
    # Quality-review follow-up (Task A5): the fail-safe design of the
    # row-count companion query has its own real, present-day gap --
    # _mask_parenthesized only hides content inside PARENTHESES, not string
    # literals, so a string literal containing the word "FROM" ahead of the
    # query's real FROM clause defeats the top-level FROM search: it matches
    # the literal's FROM first, slicing the substituted query mid-quote and
    # producing a malformed, unbalanced-quote companion query. DuckDB's
    # parser rejects it (see _row_count_query's docstring for the exact
    # substituted text). This must fail SAFE -- evidence_row_count degrades
    # to None, not a wrong number or an unhandled exception -- and the
    # verdict path must still behave sanely.
    #
    # The original evidence also returns 2 columns (label, y), not the 1x1
    # a finding's evidence must produce, so it is `unverified` for that
    # reason regardless -- proving the row-count companion query fails safe
    # ALONGSIDE that, not that it's the only thing going on here.
    finding = Finding(
        claim="testing the row-count companion query's failure path.",
        evidence_sql_or_code=(
            "SELECT 'FROM' AS label, avg(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS y "
            "FROM data"
        ),
        value=0.2654,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "unverified"
    assert judged.evidence_row_count is None
    assert judged.provenance_note is None


def test_python_evidence_legacy_fallback_picks_value_line_over_trailing_rows_line(tmp_path):
    # Quality-review follow-up (Task A5): no RESULT: sentinel here -- the
    # legacy last-line fallback applies. The bare value is printed BEFORE an
    # optional trailing ROWS: line; the fallback must pick the value line,
    # not the ROWS: line, as "the value" (see the ROWS: exclusion in
    # _extract_python_value's fallback pool). Without that exclusion, the
    # fallback's naive "last non-blank line" would wrongly return the
    # literal text "ROWS: 7043" as the raw value, which does not parse as a
    # number close to REAL_CHURN_RATE and would come back `unverified`
    # instead of `verified`.
    finding = Finding(
        claim="26.5% of customers churned.",
        evidence_sql_or_code=(
            f"import pandas as pd\n"
            f"df = pd.read_csv({str(TELCO_CSV)!r})\n"
            "rate = (df['Churn'] == 'Yes').mean()\n"
            "print(rate)\n"
            "print(f'ROWS: {len(df)}')\n"
        ),
        value=REAL_CHURN_RATE,
    )

    judged = verify_finding(finding, TELCO_CSV, tmp_path)

    assert judged.verdict == "verified"
    assert judged.recomputed_value == pytest.approx(REAL_CHURN_RATE, abs=1e-3)
    assert judged.evidence_row_count == 7043


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


# --- verify_baseline: tolerance boundary (review notes A-3) -----------------------
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
        metric_value=0.85,  # review notes A-3: embellished from the real 0.8105
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


def test_baseline_rel_tol_is_the_dominant_bound_for_real_metric_values(tmp_path):
    # Quality-review follow-up: with _BASELINE_ABS_TOL == rel_tol == 1e-6 (the
    # prior state of this fix), abs_tol tied-or-won for every metric in [0, 1]
    # (rel_tol*value <= rel_tol always), so the overridable rel_tol param was
    # inert -- a private constant silently decided every verdict instead.
    # _BASELINE_ABS_TOL is now a near-zero-only floor (~1e-9); rel_tol*value
    # (~1e-6 * 0.8105 =~ 8.1e-7 here) is the bound that actually governs.
    # diff=5e-7 sits below that rel_tol bound -> verified; diff=9e-7 sits just
    # above it -> contradicted. Both diffs are comfortably above the abs_tol
    # floor, so abs_tol cannot rescue either one -- this isolates rel_tol as
    # the deciding bound.
    within_rel_tol = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC + 5e-7,
    )
    beyond_rel_tol = Baseline(
        model=REAL_BASELINE_MODEL,
        features=REAL_BASELINE_FEATURES,
        metric_name="roc_auc",
        metric_value=REAL_BASELINE_AUC + 9e-7,
    )

    within_result = verify_baseline(within_rel_tol, TELCO_CSV, tmp_path)
    beyond_result = verify_baseline(beyond_rel_tol, TELCO_CSV, tmp_path)

    assert within_result.verdict == "verified"
    assert beyond_result.verdict == "contradicted"


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
