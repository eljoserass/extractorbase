from dataclasses import asdict
from pathlib import Path

import pytest

from data import (
    GoldAnnotation,
    GoldPolicy,
    LabelStudioTask,
    TaskSpec,
    TextData,
    annotation_issues,
    load_labelstudio,
    make_input,
    select_gold,
)
from tests.conftest import entity


def test_clinician_revision_beats_original_ground_truth() -> None:
    original = GoldAnnotation(
        id=1, result=[], completed_by=1, ground_truth=True, updated_at="2026-06-30T00:00:00Z"
    )
    review = GoldAnnotation(
        id=2, result=[entity()], completed_by=3, updated_at="2026-06-20T00:00:00Z"
    )
    task = LabelStudioTask(id=1, data=TextData(text="AR"), annotations=[original, review])
    assert select_gold(task).annotation.id == 2
    assert select_gold(task, GoldPolicy.GROUND_TRUTH).annotation.id == 1
    assert select_gold(task, GoldPolicy.LATEST).annotation.id == 1
    assert asdict(make_input(task)) == {"id": 1, "text": "AR"}


def test_in_place_review_and_missing_review() -> None:
    annotation = GoldAnnotation(id=1, result=[], completed_by=1, updated_by=3)
    task = LabelStudioTask(id=1, data=TextData(text="AR"), annotations=[annotation])
    assert select_gold(task).warnings
    annotation.updated_by = None
    with pytest.raises(ValueError, match="no annotation"):
        select_gold(task)


def test_ambiguous_reviews_are_rejected() -> None:
    task = LabelStudioTask(
        id=1,
        data=TextData(text="AR"),
        annotations=[
            GoldAnnotation(id=1, result=[], completed_by=3),
            GoldAnnotation(id=2, result=[], completed_by=4),
        ],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        select_gold(task)


def test_relations_resolve_and_unlabeled_relations_are_retained(spec: TaskSpec) -> None:
    annotation = GoldAnnotation(
        id=1,
        result=[
            entity(),
            entity(3, 5, "T:FR", "e2"),
            {"type": "relation", "from_id": "e1", "to_id": "e2", "labels": []},
        ],
    )
    task = LabelStudioTask(id=1, data=TextData(text="AR FR"), annotations=[annotation])
    issues = annotation_issues(task, annotation, spec)
    assert all(issue.severity == "warning" for issue in issues)
    assert any("Unlabeled relation" in issue.message for issue in issues)
    annotation.result[-1]["to_id"] = "missing"  # type: ignore[typeddict-unknown-key]
    assert any(issue.severity == "error" for issue in annotation_issues(task, annotation, spec))


def test_loader_rejects_malformed_input_without_dumping_document_text(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"id": 1, "data": {"text": 123}}]')
    with pytest.raises(ValueError, match="Invalid Label Studio export"):
        load_labelstudio([path])


def test_nullable_metadata_and_omitted_relation_labels() -> None:
    item = entity()
    item["meta"] = None
    annotation = GoldAnnotation(
        id=1,
        result=[
            item,
            {
                "type": "relation",
                "from_id": "e1",
                "to_id": "e1",
            },
        ],
    )
    assert annotation.result[0].get("meta") is None
    assert annotation.result[1].get("labels", []) == []
