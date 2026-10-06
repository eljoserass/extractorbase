"""Character-span NER scoring through nervaluate, independent of fitting and splitting."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from nervaluate import Evaluator

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
    for example in gold:
        task_predictions = by_id[example.input.id]["predictions"]
        if len(task_predictions) != 1:
            raise ValueError("Scoring expects exactly one prediction per document.")
        pred.append(_entities(task_predictions[0]["result"]))
    tags = sorted({entity["label"] for document in [*true, *pred] for entity in document})
    failures = sum(bool(p.get("meta", {}).get("extraction_error")) for p in predictions)
    if not tags:
        return MetricsReport(PRFScore(0.0, 0.0, 0.0, 0), {}, len(gold), failures)
    results = Evaluator(true, pred, tags=tags, loader="dict").evaluate()
    overall = results["overall"]["strict"]
    per_label = {
        label: PRFScore(
            value["strict"].precision,
            value["strict"].recall,
            value["strict"].f1,
            value["strict"].possible,
        )
        for label, value in results["entities"].items()
    }
    return MetricsReport(
        PRFScore(overall.precision, overall.recall, overall.f1, overall.possible),
        per_label,
        len(gold),
        failures,
    )
