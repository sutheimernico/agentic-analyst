"""Tests for the error injector (`planting.py`) used by the judge benchmark.

These prove each attack class plants the intended, *detectable* perturbation
into a verified base finding, and that case generation is deterministic. The
structural tests are pure (no data); the detectability tests run the real
judge against the real telco CSV -- the same recompute-don't-trust proof the
judge's own tests use -- to confirm a planted lie actually comes back
`contradicted`/`unverified`, not merely that a field changed.
"""

from pathlib import Path

from agentic_analyst import planting
from agentic_analyst.judge import verify_finding
from agentic_analyst.planting import (
    ATTACK_CLASSES,
    CATCH_VERDICT,
    BaseFinding,
    distort,
    generate_cases,
    honest_finding,
    plant_alias_dodge,
    plant_claim_mismatch,
    plant_fabricated_evidence,
    plant_value_swap,
)

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"

# Real, judge-verifiable base findings (values recomputed from the committed CSV).
CHURN_BASE = BaseFinding(
    label="churn_rate",
    evidence_sql="SELECT avg(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END) FROM data",
    claim_template="The overall churn rate is {n}.",
    as_int=False,
    value=0.2653698707936959,
)
# A genuinely low-magnitude rate: distortions here fall under the judge's
# abs_tol=0.01 floor at small magnitudes, so it pins the documented "5%
# distortions are not all caught" limitation.
SENIOR_BASE = BaseFinding(
    label="senior_fraction",
    evidence_sql="SELECT avg(SeniorCitizen) FROM data",
    claim_template="The share of senior citizens is {n}.",
    as_int=False,
    value=0.1621468124378816,
)
COUNT_BASE = BaseFinding(
    label="count_churned",
    evidence_sql="SELECT count(*) FROM data WHERE Churn='Yes'",
    claim_template="{n} customers churned in total.",
    as_int=True,
    value=1869,
)


# --- distort ------------------------------------------------------------------


def test_distort_scales_by_signed_magnitude():
    assert distort(100.0, 0.20, 1) == 120.0
    assert distort(100.0, 0.20, -1) == 80.0
    assert distort(0.5, 2.00, 1) == 1.5


# --- per-class structural perturbation ----------------------------------------


def test_value_swap_changes_only_the_value():
    planted = plant_value_swap(CHURN_BASE, 0.20, 1)
    honest = honest_finding(CHURN_BASE)

    assert planted.claim == honest.claim  # claim untouched
    assert planted.evidence_sql_or_code == CHURN_BASE.evidence_sql  # evidence untouched
    assert planted.value == distort(CHURN_BASE.value, 0.20, 1)
    assert planted.value != CHURN_BASE.value


def test_claim_mismatch_changes_only_the_claim():
    planted = plant_claim_mismatch(COUNT_BASE, 0.50, 1)

    assert planted.value == COUNT_BASE.value  # value untouched (honest)
    assert planted.evidence_sql_or_code == COUNT_BASE.evidence_sql  # evidence untouched
    # distorted number rendered into the claim text (1869 -> ~2804)
    assert "2804" in planted.claim
    assert planted.claim != honest_finding(COUNT_BASE).claim


def test_fabricated_evidence_has_no_from_data():
    planted = plant_fabricated_evidence(CHURN_BASE, 0.50, 1)

    assert planted.evidence_sql_or_code.startswith("SELECT")
    assert "FROM" not in planted.evidence_sql_or_code.upper()  # cannot have touched the data
    # the lie carries its own "proof": the literal is the distorted value
    assert planted.value == distort(CHURN_BASE.value, 0.50, 1)


def test_alias_dodge_aliases_output_as_data_without_reading_the_view():
    planted = plant_alias_dodge(CHURN_BASE, 0.50, 1)

    assert planted.evidence_sql_or_code.endswith("AS data")
    assert "FROM" not in planted.evidence_sql_or_code.upper()


# --- detectability: the planted lie actually gets caught by the real judge ----


def test_value_swap_large_magnitude_is_contradicted():
    judged = verify_finding(plant_value_swap(CHURN_BASE, 0.50, 1), TELCO_CSV, TELCO_CSV.parent)
    assert judged.verdict == "contradicted"


def test_claim_mismatch_large_magnitude_is_contradicted():
    judged = verify_finding(plant_claim_mismatch(CHURN_BASE, 0.50, 1), TELCO_CSV, TELCO_CSV.parent)
    assert judged.verdict == "contradicted"
    assert "claim_text_disagrees_with_value" in judged.detail


def test_fabricated_evidence_is_unverified():
    judged = verify_finding(
        plant_fabricated_evidence(CHURN_BASE, 0.50, 1), TELCO_CSV, TELCO_CSV.parent
    )
    assert judged.verdict == "unverified"
    assert "evidence_does_not_touch_data" in judged.detail


def test_alias_dodge_is_unverified():
    judged = verify_finding(plant_alias_dodge(CHURN_BASE, 0.50, 1), TELCO_CSV, TELCO_CSV.parent)
    assert judged.verdict == "unverified"
    assert "evidence_does_not_touch_data" in judged.detail


def test_honest_base_finding_is_verified():
    # Proves the base findings are legitimate (not accidentally contradicted),
    # so a caught tampered case is a real catch, not a broken base.
    for base in (CHURN_BASE, SENIOR_BASE, COUNT_BASE):
        judged = verify_finding(honest_finding(base), TELCO_CSV, TELCO_CSV.parent)
        assert judged.verdict == "verified", f"{base.label}: {judged.detail}"


def test_small_magnitude_distortion_on_a_low_value_slips_under_the_abs_tol_floor():
    # Documented limitation, pinned: a 5% distortion of a ~0.16 rate is ~0.008,
    # under the judge's abs_tol=0.01 floor, so it is NOT caught. This is the
    # honest reason the benchmark's 5% catch-rate is below 100%.
    swap = verify_finding(plant_value_swap(SENIOR_BASE, 0.05, 1), TELCO_CSV, TELCO_CSV.parent)
    mismatch = verify_finding(
        plant_claim_mismatch(SENIOR_BASE, 0.05, 1), TELCO_CSV, TELCO_CSV.parent
    )
    assert swap.verdict == "verified"
    assert mismatch.verdict == "verified"


# --- case generation ----------------------------------------------------------


def test_generate_cases_is_deterministic():
    bases = [CHURN_BASE, SENIOR_BASE, COUNT_BASE]
    assert generate_cases(bases) == generate_cases(bases)


def test_generate_cases_has_one_clean_control_per_base_and_only_registered_classes():
    bases = [CHURN_BASE, SENIOR_BASE]
    cases = generate_cases(bases)

    clean = [c for c in cases if not c.tampered]
    assert len(clean) == len(bases)
    assert all(c.attack_class == "clean" for c in clean)

    tampered_classes = {c.attack_class for c in cases if c.tampered}
    assert tampered_classes == set(ATTACK_CLASSES)


def test_generate_cases_skips_negative_direction_for_magnitudes_at_or_above_one():
    cases = generate_cases([CHURN_BASE])
    for c in cases:
        if c.tampered and c.magnitude >= 1.0:
            assert c.direction == 1, "no sign-flipping negative direction at magnitude >= 100%"
    # a sub-100% magnitude still exercises both directions
    half = {c.direction for c in cases if c.tampered and c.magnitude == 0.50}
    assert half == {1, -1}


def test_generate_cases_yields_exactly_seven_tampered_cases_per_class_per_base():
    # 3 magnitudes x 2 directions + 1 magnitude (>=100%) x 1 direction = 7.
    # Pins the total case count (1 clean + 4*7 = 29 per base -> 261 for the
    # benchmark's 9 bases) against an off-by-one in _directions().
    cases = generate_cases([CHURN_BASE, SENIOR_BASE])
    for base in ("churn_rate", "senior_fraction"):
        for cls in ATTACK_CLASSES:
            n = sum(1 for c in cases if c.base_label == base and c.attack_class == cls)
            assert n == 7, f"{base}/{cls}: {n}"
    assert len(cases) == 2 * (1 + len(ATTACK_CLASSES) * 7)


def test_generate_cases_rejects_a_zero_value_base():
    zero_base = BaseFinding(
        label="always_zero",
        evidence_sql="SELECT 0 FROM data",
        claim_template="The value is {n}.",
        as_int=True,
        value=0,
    )
    try:
        generate_cases([zero_base])
    except ValueError as exc:
        assert "always_zero" in str(exc)
    else:
        raise AssertionError("expected ValueError for a zero-value base")


def test_attack_class_registry_and_catch_verdicts_are_consistent():
    # C3 extends this benchmark by adding one injector + one CATCH_VERDICT
    # entry; guard that the two stay in lockstep.
    assert set(ATTACK_CLASSES) == set(planting._INJECTORS)
    assert set(CATCH_VERDICT) == set(ATTACK_CLASSES)
