"""Tests for the judge-benchmark aggregation math (`benchmark_stats.py`).

Builds hand-crafted per-case mappings (no data, no judge) so the precision /
recall / FPR / mechanism-match arithmetic and the zero-denominator edge cases
are pinned independently of a real benchmark run.
"""

from agentic_analyst.benchmark_stats import aggregate
from agentic_analyst.planting import ATTACK_CLASSES, DEFAULT_MAGNITUDES, PlantedCase
from agentic_analyst.report import Finding

_DUMMY = Finding(claim="c", evidence_sql_or_code="e", value=1)


def _case(attack_class, magnitude, direction, tampered, base_label="b1"):
    return PlantedCase(
        base_label=base_label,
        attack_class=attack_class,
        magnitude=magnitude,
        direction=direction,
        tampered=tampered,
        finding=_DUMMY,
    )


def _per_case(pairs):
    return {i: pair for i, pair in enumerate(pairs)}


# A mixed scenario exercising every count: TP (right + wrong mechanism), FN, FP, TN.
_MIXED = _per_case(
    [
        (_case("clean", 0.0, 0, False), "verified"),  # TN
        (_case("value_swap", 0.20, 1, True), "contradicted"),  # TP, intended mechanism
        (_case("value_swap", 0.05, 1, True), "verified"),  # FN (missed)
        (_case("fabricated_evidence", 0.20, 1, True), "unverified"),  # TP, intended mechanism
        (_case("value_swap", 0.50, 1, True), "unverified"),  # TP, WRONG mechanism for value_swap
        (_case("clean", 0.0, 0, False, base_label="b2"), "contradicted"),  # FP
    ]
)


def test_overall_confusion_matrix_and_rates():
    agg = aggregate(_MIXED)["overall"]
    assert agg == {
        "true_positives": 3,
        "false_negatives": 1,
        "false_positives": 1,
        "true_negatives": 1,
        "precision": 0.75,  # 3 / (3 + 1)
        "recall": 0.75,  # 3 / (3 + 1)
        "false_positive_rate": 0.5,  # 1 / (1 + 1)
    }


def test_mechanism_match_counts_only_intended_verdicts():
    mm = aggregate(_MIXED)["mechanism_match"]
    # 3 caught, but the value_swap caught as `unverified` used the wrong mechanism.
    assert mm == {"caught": 3, "used_intended_mechanism": 2, "rate": 0.6667}


def test_by_class_recall_and_empty_classes_are_zero():
    by_class = aggregate(_MIXED)["by_class"]
    assert by_class["value_swap"] == {"caught": 2, "n": 3, "recall": 0.6667}
    assert by_class["fabricated_evidence"] == {"caught": 1, "n": 1, "recall": 1.0}
    # classes with no cases must be present with a 0/0 -> 0.0 recall, not missing
    assert by_class["claim_mismatch"] == {"caught": 0, "n": 0, "recall": 0.0}
    assert by_class["alias_dodge"] == {"caught": 0, "n": 0, "recall": 0.0}
    assert set(by_class) == set(ATTACK_CLASSES)


def test_by_magnitude_recall_and_missed_cases():
    agg = aggregate(_MIXED)
    by_mag = agg["by_magnitude"]
    assert by_mag["0.05"] == {"caught": 0, "n": 1, "recall": 0.0}
    assert by_mag["0.2"] == {"caught": 2, "n": 2, "recall": 1.0}
    assert by_mag["0.5"] == {"caught": 1, "n": 1, "recall": 1.0}
    assert by_mag["2"] == {"caught": 0, "n": 0, "recall": 0.0}  # empty magnitude cell
    assert agg["missed_cases"] == [
        {"base_label": "b1", "attack_class": "value_swap", "magnitude": 0.05, "direction": 1}
    ]


def test_magnitude_keys_use_g_formatting_consistently():
    # write side (`f"{m:g}"` in aggregate) must match the read side used
    # everywhere: 0.20 -> "0.2", 0.50 -> "0.5", 2.00 -> "2", never "0.20"/"2.0".
    expected = [f"{m:g}" for m in DEFAULT_MAGNITUDES]
    assert expected == ["0.05", "0.2", "0.5", "2"]
    agg = aggregate(_MIXED)
    assert list(agg["by_magnitude"]) == expected
    for cls in ATTACK_CLASSES:
        assert list(agg["by_class_magnitude"][cls]) == expected


def test_zero_denominator_precision_when_no_positives_or_false_positives():
    # Only clean, un-flagged controls: tp+fp == 0 -> precision must be 0.0, not
    # a ZeroDivisionError; recall's tp+fn == 0 likewise.
    only_tn = _per_case([(_case("clean", 0.0, 0, False), "verified") for _ in range(3)])
    ov = aggregate(only_tn)["overall"]
    assert ov["true_negatives"] == 3
    assert ov["precision"] == 0.0
    assert ov["recall"] == 0.0
    assert ov["false_positive_rate"] == 0.0


def test_zero_denominator_mechanism_rate_when_nothing_caught():
    # Tampered cases that all slip through: tp == 0 -> mechanism rate 0/0 -> 0.0.
    all_missed = _per_case(
        [(_case("value_swap", 0.05, 1, True), "verified") for _ in range(3)]
    )
    agg = aggregate(all_missed)
    assert agg["overall"]["true_positives"] == 0
    assert agg["overall"]["false_negatives"] == 3
    assert agg["overall"]["recall"] == 0.0
    assert agg["mechanism_match"] == {"caught": 0, "used_intended_mechanism": 0, "rate": 0.0}


def test_single_case_class_counts_in_totals_but_not_in_the_magnitude_breakdown():
    # A population_switch case is magnitude-invariant: generate_cases tags it
    # with magnitude 0.0 (not one of DEFAULT_MAGNITUDES). It must count as a
    # normal true positive (overall + by_class recall + mechanism_match, since
    # CATCH_VERDICT["population_switch"] == "unverified"), yet contribute
    # NOTHING to the per-magnitude views -- 0.0 is not a swept magnitude, so it
    # never lands in by_magnitude, and by_class_magnitude's population_switch
    # cells stay empty (n=0). This is why the README reads its catch rate from
    # by_class, not by_class_magnitude.
    per_case = _per_case(
        [
            (_case("clean", 0.0, 0, False), "verified"),  # TN
            (_case("population_switch", 0.0, 0, True), "unverified"),  # TP, intended mechanism
            (_case("value_swap", 0.20, 1, True), "contradicted"),  # TP, swept
        ]
    )
    agg = aggregate(per_case)

    assert agg["overall"]["true_positives"] == 2
    assert agg["by_class"]["population_switch"] == {"caught": 1, "n": 1, "recall": 1.0}
    assert agg["mechanism_match"] == {"caught": 2, "used_intended_mechanism": 2, "rate": 1.0}
    # magnitude 0.0 is not a swept magnitude -> absent from by_magnitude entirely
    assert "0" not in agg["by_magnitude"]
    assert set(agg["by_magnitude"]) == {f"{m:g}" for m in DEFAULT_MAGNITUDES}
    # ...and population_switch's per-magnitude cells are all empty (n == 0)
    assert all(cell["n"] == 0 for cell in agg["by_class_magnitude"]["population_switch"].values())
