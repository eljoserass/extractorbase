"""Character-span NER scoring through nervaluate, independent of fitting and splitting."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from nervaluate import Evaluator
from nervaluate.entities import Entity, EvaluationResult
from nervaluate.strategies import EntityTypeEvaluation, StrictEvaluation

from data import LabelStudioTaskPrediction, ResultItem, TrainingExample


@dataclass(frozen=True)
class PRFScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricsReport:
    strict_micro: PRFScore
    per_label: dict[str, PRFScore]
    documents: int
    failed_documents: int = 0
    relations: PRFScore | None = None  # Parsed and retained; relation scoring comes later.
    overlap_micro: PRFScore | None = None
    strict_macro: PRFScore | None = None
    overlap_macro: PRFScore | None = None
    per_label_overlap: dict[str, PRFScore] = field(default_factory=dict)
    macro_labels: tuple[str, ...] = ()  # Only labels with gold support in this test set.


class LabelOverlapEvaluation(EntityTypeEvaluation):
    """Full credit for any character overlap with the same label, one match per entity."""

    def _has_sufficient_overlap(self, pred: Entity, true: Entity) -> bool:
        # Default nervaluate uses >=1% overlap and lets wrong labels consume gold spans.
        return pred.label == true.label and pred.start <= true.end and pred.end >= true.start


def _prf(result: EvaluationResult) -> PRFScore:
    return PRFScore(result.precision, result.recall, result.f1, result.possible)


def _macro(per_label: dict[str, PRFScore], labels: tuple[str, ...]) -> PRFScore:
    if not labels:
        return PRFScore(0.0, 0.0, 0.0, 0)
    values = [per_label[label] for label in labels]
    return PRFScore(
        sum(value.precision for value in values) / len(values),
        sum(value.recall for value in values) / len(values),
        sum(value.f1 for value in values) / len(values),
        sum(value.support for value in values),
    )


class EvalEntity(TypedDict):
    label: str
    start: int
    end: int


def _entities(results: list[ResultItem]) -> list[EvalEntity]:
    # Label Studio uses exclusive ends; nervaluate uses inclusive positions for overlap.
    return [
        EvalEntity(label=label, start=item["value"]["start"], end=item["value"]["end"] - 1)
        for item in results
        if item["type"] == "labels"
        for label in item["value"]["labels"]
    ]


def _exact_first(true: list[EvalEntity], pred: list[EvalEntity]) -> list[EvalEntity]:
    """Prevent nervaluate's greedy overlap matching from consuming later exact matches."""
    remaining = Counter((e["label"], e["start"], e["end"]) for e in true)
    exact: list[EvalEntity] = []
    other: list[EvalEntity] = []
    for entity in pred:
        key = (entity["label"], entity["start"], entity["end"])
        if remaining[key]:
            exact.append(entity)
            remaining[key] -= 1
        else:
            other.append(entity)
    return exact + other


def score(
    predictions: Sequence[LabelStudioTaskPrediction], gold: Sequence[TrainingExample]
) -> MetricsReport:
    by_id = {prediction["id"]: prediction for prediction in predictions}
    gold_ids = [example.input.id for example in gold]
    if len(by_id) != len(predictions) or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Scoring requires unique document IDs.")
    if set(by_id) != set(gold_ids):
        raise ValueError("Prediction and gold document IDs must match exactly.")
    true = [_entities(example.gold.result) for example in gold]
    pred: list[list[EvalEntity]] = []
    for example, true_entities in zip(gold, true, strict=True):
        task_predictions = by_id[example.input.id]["predictions"]
        if len(task_predictions) != 1:
            raise ValueError("Scoring expects exactly one prediction per document.")
        pred.append(_exact_first(true_entities, _entities(task_predictions[0]["result"])))
    tags = sorted({entity["label"] for document in [*true, *pred] for entity in document})
    failures = sum(bool(p.get("meta", {}).get("extraction_error")) for p in predictions)
    if not tags:
        empty = PRFScore(0.0, 0.0, 0.0, 0)
        return MetricsReport(
            empty,
            {},
            len(gold),
            failures,
            overlap_micro=empty,
            strict_macro=empty,
            overlap_macro=empty,
        )
    evaluator = Evaluator(true, pred, tags=tags, loader="dict")
    evaluator.strategies = {"strict": StrictEvaluation(), "overlap": LabelOverlapEvaluation()}
    results = evaluator.evaluate()
    per_label = {
        label: _prf(value["strict"]) for label, value in sorted(results["entities"].items())
    }
    per_label_overlap = {
        label: _prf(value["overlap"]) for label, value in sorted(results["entities"].items())
    }
    macro_labels = tuple(label for label, value in per_label.items() if value.support > 0)
    return MetricsReport(
        _prf(results["overall"]["strict"]),
        per_label,
        len(gold),
        failures,
        overlap_micro=_prf(results["overall"]["overlap"]),
        strict_macro=_macro(per_label, macro_labels),
        overlap_macro=_macro(per_label_overlap, macro_labels),
        per_label_overlap=per_label_overlap,
        macro_labels=macro_labels,
    )
