"""Typed Label Studio adapters. The raw exports remain the source of truth."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict
from xml.etree import ElementTree

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

type DocumentId = int | str
type JsonObject = dict[str, JsonValue]


class SpanValue(TypedDict):
    start: int
    end: int
    text: str
    labels: list[str]


class EntityResult(TypedDict):
    id: str
    type: Literal["labels"]
    from_name: str
    to_name: str
    value: SpanValue
    meta: NotRequired[JsonObject | None]


class RelationResult(TypedDict):
    type: Literal["relation"]
    from_id: str
    to_id: str
    labels: NotRequired[list[str]]
    direction: NotRequired[str]


type ResultItem = Annotated[EntityResult | RelationResult, Field(discriminator="type")]


class LabelStudioPrediction(TypedDict):
    result: list[ResultItem]
    model_version: str


class LabelStudioTaskPrediction(TypedDict):
    id: DocumentId
    predictions: list[LabelStudioPrediction]
    data: NotRequired[dict[str, str]]
    meta: NotRequired[JsonObject]


class Annotator(BaseModel):
    id: int


class GoldAnnotation(BaseModel):
    id: int
    result: list[ResultItem]
    completed_by: int | Annotator | None = None
    updated_by: int | Annotator | None = None
    parent_annotation: int | None = None
    ground_truth: bool = False
    was_cancelled: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class TextData(BaseModel):
    text: str


class LabelStudioTask(BaseModel):
    id: DocumentId
    data: TextData
    annotations: list[GoldAnnotation] = Field(default_factory=list)


@dataclass(frozen=True)
class DocumentInput:
    id: DocumentId
    text: str


@dataclass(frozen=True)
class TrainingExample:
    input: DocumentInput
    gold: GoldAnnotation


@dataclass(frozen=True)
class TaskSpec:
    entity_labels: tuple[str, ...]
    relation_labels: tuple[str, ...]
    from_name: str = "labels"
    to_name: str = "text"
    instructions: str | None = None

    @classmethod
    def from_xml(cls, path: Path, instructions: str | None = None) -> "TaskSpec":
        root = ElementTree.fromstring(path.read_text())
        control = root.find("Labels")
        if control is None:
            raise ValueError("The schema needs a Label Studio <Labels> control.")
        labels = tuple(node.attrib["value"] for node in control.findall("Label"))
        relations = tuple(node.attrib["value"] for node in root.findall("./Relations/Relation"))
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Entity labels must be nonempty and unique.")
        return cls(
            labels, relations, control.attrib["name"], control.attrib["toName"], instructions
        )


class GoldPolicy(StrEnum):
    CLINICIAN = "clinician"
    LATEST = "latest"
    GROUND_TRUTH = "ground-truth"


@dataclass(frozen=True)
class GoldSelection:
    annotation: GoldAnnotation
    reason: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataIssue:
    task_id: DocumentId
    annotation_id: int
    severity: Literal["warning", "error"]
    message: str


@dataclass
class DatasetReport:
    tasks: int
    annotations_per_task: dict[int, int]
    ground_truth: dict[bool, int]
    parent_annotation: dict[str, int]
    completed_by: dict[str, int]
    selected_by: dict[str, int]
    entity_labels: dict[str, int]
    relation_labels: dict[str, int]
    entities: int = 0
    relations: int = 0
    issues: list[DataIssue] = field(default_factory=list)


def load_labelstudio(paths: list[Path]) -> list[LabelStudioTask]:
    tasks: list[LabelStudioTask] = []
    adapter = TypeAdapter(list[LabelStudioTask])
    for path in paths:
        try:
            tasks.extend(adapter.validate_json(path.read_bytes(), strict=True))
        except ValidationError as error:
            details = error.errors(include_input=False, include_url=False)[:3]
            raise ValueError(f"Invalid Label Studio export {path}: {details}") from error
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs across the selected exports; load them separately.")
    return tasks


def annotator_id(value: int | Annotator | None) -> int | None:
    return value.id if isinstance(value, Annotator) else value


def _updated(annotation: GoldAnnotation) -> datetime:
    value = annotation.updated_at or annotation.created_at
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def select_gold(
    task: LabelStudioTask,
    policy: GoldPolicy = GoldPolicy.CLINICIAN,
    reviewer_ids: tuple[int, ...] = (3, 4),
) -> GoldSelection:
    annotations = [a for a in task.annotations if not a.was_cancelled]
    warnings: list[str] = []
    if policy == GoldPolicy.CLINICIAN:
        candidates = [a for a in annotations if annotator_id(a.completed_by) in reviewer_ids]
        reason = "clinician-created annotation"
        if not candidates:
            candidates = [a for a in annotations if annotator_id(a.updated_by) in reviewer_ids]
            reason = "original updated by clinician"
            warnings.append("Review inferred from updated_by; verify this provenance.")
    elif policy == GoldPolicy.GROUND_TRUTH:
        candidates = [a for a in annotations if a.ground_truth]
        reason = "explicit ground_truth policy"
    else:
        candidates = annotations
        reason = "explicit latest-update policy"
    if not candidates:
        raise ValueError(f"Task {task.id}: no annotation satisfies gold policy {policy}.")
    candidates.sort(key=_updated, reverse=True)
    if len(candidates) > 1 and _updated(candidates[0]) == _updated(candidates[1]):
        raise ValueError(f"Task {task.id}: ambiguous gold annotations with identical timestamps.")
    selected = candidates[0]
    if policy == GoldPolicy.CLINICIAN:
        for other in annotations:
            if (
                other.id != selected.id
                and annotator_id(other.updated_by) in reviewer_ids
                and _updated(other) > _updated(selected)
            ):
                warnings.append(
                    f"Annotation {other.id} was updated later; policy prefers the clinician's "
                    "revision. Inspect if final provenance is uncertain."
                )
    return GoldSelection(selected, reason, tuple(warnings))


def make_input(task: LabelStudioTask) -> DocumentInput:
    return DocumentInput(task.id, task.data.text)


def annotation_issues(
    task: LabelStudioTask, annotation: GoldAnnotation, spec: TaskSpec
) -> list[DataIssue]:
    issues: list[DataIssue] = []
    ids: set[str] = set()
    spans: set[tuple[int, int, tuple[str, ...]]] = set()

    def add(severity: Literal["warning", "error"], message: str) -> None:
        issues.append(DataIssue(task.id, annotation.id, severity, message))

    for item in annotation.result:
        if item["type"] != "labels":
            continue
        if item["id"] in ids:
            add("error", f"Duplicate entity ID {item['id']}.")
        ids.add(item["id"])
        value = item["value"]
        start, end = value["start"], value["end"]
        if not 0 <= start < end <= len(task.data.text):
            add("error", f"Entity {item['id']} has invalid span [{start}, {end}).")
        elif value["text"] != task.data.text[start:end]:
            add(
                "warning",
                f"Entity {item['id']} text differs from its offsets; offsets are retained.",
            )
        if not value["labels"] or set(value["labels"]) - set(spec.entity_labels):
            add("error", f"Entity {item['id']} has missing or unknown labels.")
        key = (start, end, tuple(value["labels"]))
        if key in spans:
            add("warning", f"Duplicate span/label entity {item['id']} is retained in gold.")
        spans.add(key)
    for item in annotation.result:
        if item["type"] != "relation":
            continue
        if item["from_id"] not in ids or item["to_id"] not in ids:
            add("error", "Relation references an unknown entity ID.")
        if not item.get("labels"):
            add("warning", "Unlabeled relation is retained; its type is not inferred.")
        elif set(item.get("labels", [])) - set(spec.relation_labels):
            add("error", "Relation has an unknown label.")
    return issues


def make_example(
    task: LabelStudioTask, spec: TaskSpec, policy: GoldPolicy, reviewer_ids: tuple[int, ...]
) -> TrainingExample:
    selected = select_gold(task, policy, reviewer_ids)
    errors = [
        issue
        for issue in annotation_issues(task, selected.annotation, spec)
        if issue.severity == "error"
    ]
    if errors:
        raise ValueError(f"Task {task.id}: {' '.join(issue.message for issue in errors)}")
    return TrainingExample(make_input(task), selected.annotation)


def inspect_dataset(
    tasks: list[LabelStudioTask], spec: TaskSpec, policy: GoldPolicy, reviewer_ids: tuple[int, ...]
) -> DatasetReport:
    annotations = [a for task in tasks for a in task.annotations]
    report = DatasetReport(
        len(tasks),
        dict(Counter(len(t.annotations) for t in tasks)),
        dict(Counter(a.ground_truth for a in annotations)),
        dict(
            Counter("child" if a.parent_annotation is not None else "original" for a in annotations)
        ),
        dict(Counter(str(annotator_id(a.completed_by)) for a in annotations)),
        {},
        {},
        {},
    )
    for task in tasks:
        try:
            selection = select_gold(task, policy, reviewer_ids)
        except ValueError as error:
            report.issues.append(DataIssue(task.id, -1, "error", str(error)))
            continue
        report.selected_by[selection.reason] = report.selected_by.get(selection.reason, 0) + 1
        report.issues.extend(
            DataIssue(task.id, selection.annotation.id, "warning", warning)
            for warning in selection.warnings
        )
        report.issues.extend(annotation_issues(task, selection.annotation, spec))
        for item in selection.annotation.result:
            if item["type"] == "labels":
                report.entities += 1
                for label in item["value"]["labels"]:
                    report.entity_labels[label] = report.entity_labels.get(label, 0) + 1
            else:
                report.relations += 1
                for label in item.get("labels", []) or ["UNLABELED"]:
                    report.relation_labels[label] = report.relation_labels.get(label, 0) + 1
    return report


def prediction_for(
    document: DocumentInput, results: list[ResultItem], model_version: str
) -> LabelStudioTaskPrediction:
    return {
        "id": document.id,
        "data": {"text": document.text},
        "predictions": [{"model_version": model_version, "result": results}],
    }
