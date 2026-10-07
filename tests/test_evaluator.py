import pytest

from data import DocumentInput, TrainingExample, prediction_for
from evaluator import score
from tests.conftest import entity


@pytest.mark.parametrize(
    "start,end,label,expected",
    [
        (0, 2, "D:AR", 1.0),
        (0, 2, "T:FR", 0.0),
        (0, 1, "D:AR", 0.0),
        (2, 3, "D:AR", 0.0),
    ],
)
def test_strict_matching(
    example: TrainingExample, start: int, end: int, label: str, expected: float
) -> None:
    pred = prediction_for(example.input, [entity(start, end, label)], "test")
    assert score([pred], [example]).strict_micro.f1 == expected


def test_missing_and_extra_entities(example: TrainingExample) -> None:
    empty = prediction_for(example.input, [], "test")
    assert score([empty], [example]).strict_micro.recall == 0
    extra = prediction_for(example.input, [entity(), entity(3, 5, "T:FR", "e2")], "test")
    result = score([extra], [example]).strict_micro
    assert result.precision == 0.5 and result.recall == 1
    example.gold.result.append(entity(3, 5, "T:FR", "e2"))
    missing = prediction_for(example.input, [entity()], "test")
    assert score([missing], [example]).strict_micro.recall == 0.5


def test_missing_documents_are_not_silently_dropped(example: TrainingExample) -> None:
    with pytest.raises(ValueError, match="IDs must match"):
        score([], [example])


def test_different_entity_ids_do_not_affect_matching(example: TrainingExample) -> None:
    pred = prediction_for(example.input, [entity(id="different")], "test")
    assert score([pred], [example]).strict_micro.f1 == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_overlap_error_cannot_steal_later_exact_match(
    example: TrainingExample, reverse: bool
) -> None:
    example.gold.result.append(entity(0, 5, "T:FR", "nested"))
    results = [entity(0, 1, id="wrong"), entity(), entity(0, 5, "T:FR", "nested")]
    if reverse:
        results.reverse()
    report = score([prediction_for(example.input, results, "test")], [example])
    assert report.strict_micro.precision == pytest.approx(2 / 3)
    assert report.strict_micro.recall == 1
    assert report.strict_micro.f1 == pytest.approx(0.8)
    assert report.per_label["D:AR"].precision == 0.5
    assert report.per_label["T:FR"].f1 == 1


def test_duplicate_predictions_do_not_add_correct_matches(example: TrainingExample) -> None:
    pred = prediction_for(example.input, [entity(), entity(id="duplicate")], "test")
    report = score([pred], [example])
    assert report.strict_micro.precision == 0.5
    assert report.strict_micro.recall == 1


def test_overlap_gives_full_credit_for_one_character_even_below_one_percent(
    example: TrainingExample,
) -> None:
    example.gold.result[0]["value"]["end"] = 200
    pred = prediction_for(example.input, [entity(199, 200)], "test")
    report = score([pred], [example])
    assert report.strict_micro.f1 == 0
    assert report.overlap_micro is not None and report.overlap_micro.f1 == 1


def test_wrong_label_cannot_consume_gold_before_correct_overlap(example: TrainingExample) -> None:
    pred = prediction_for(example.input, [entity(0, 1, "T:FR", "wrong"), entity(1, 2)], "test")
    report = score([pred], [example])
    assert report.overlap_micro is not None
    assert report.overlap_micro.precision == 0.5
    assert report.overlap_micro.recall == 1


def test_overlap_is_one_to_one_and_touching_exclusive_ends_do_not_match(
    example: TrainingExample,
) -> None:
    pred = prediction_for(
        example.input, [entity(0, 1), entity(1, 2, id="second"), entity(2, 3, id="touch")], "test"
    )
    report = score([pred], [example])
    assert report.overlap_micro is not None
    assert report.overlap_micro.precision == pytest.approx(1 / 3)
    assert report.overlap_micro.recall == 1


def test_macro_averages_gold_supported_labels_and_reports_prediction_only_labels(
    example: TrainingExample,
) -> None:
    example.gold.result.extend([entity(3, 5, id="second"), entity(6, 8, "T:FR", "third")])
    pred = prediction_for(example.input, [entity(), entity(9, 11, "EXTRA", "extra")], "test")
    report = score([pred], [example])
    assert report.macro_labels == ("D:AR", "T:FR")
    assert report.strict_macro is not None
    assert report.strict_macro.precision == 0.5
    assert report.strict_macro.recall == 0.25
    assert report.strict_macro.f1 == pytest.approx(1 / 3)
    assert report.per_label["EXTRA"].support == 0
    assert report.strict_micro.f1 == pytest.approx(0.4)


def test_inference_failures_remain_in_gold_denominator(example: TrainingExample) -> None:
    second = TrainingExample(DocumentInput(2, example.input.text), example.gold)
    correct = prediction_for(example.input, [entity()], "test")
    failed = prediction_for(second.input, [], "test")
    failed["meta"] = {"extraction_error": "timeout"}
    report = score([correct, failed], [example, second])
    assert report.documents == 2 and report.failed_documents == 1
    assert report.strict_micro.support == 2 and report.strict_micro.recall == 0.5


def test_empty_entities_report_zero_scores(example: TrainingExample) -> None:
    example.gold.result.clear()
    report = score([prediction_for(example.input, [], "test")], [example])
    assert report.strict_micro.f1 == 0
    assert report.overlap_micro is not None and report.overlap_micro.f1 == 0
    assert report.strict_macro is not None and report.strict_macro.f1 == 0
    assert report.macro_labels == ()
