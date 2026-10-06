import pytest

from data import TrainingExample, prediction_for
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
