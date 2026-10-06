from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

from data import DocumentInput, EntityResult, GoldAnnotation, TaskSpec, TrainingExample


def entity(start: int = 0, end: int = 2, label: str = "D:AR", id: str = "e1") -> EntityResult:
    return {
        "id": id,
        "type": "labels",
        "from_name": "labels",
        "to_name": "text",
        "value": {"start": start, "end": end, "text": "AR", "labels": [label]},
    }


@pytest.fixture
def spec() -> TaskSpec:
    return TaskSpec(("D:AR", "T:FR"), ("valor",))


@pytest.fixture
def schema_path(tmp_path: Path, spec: TaskSpec) -> Path:
    """Build the tiny test vocabulary locally; never depend on the private schema."""
    root = Element("View")
    labels = SubElement(root, "Labels", name=spec.from_name, toName=spec.to_name)
    for label in spec.entity_labels:
        SubElement(labels, "Label", value=label)
    relations = SubElement(root, "Relations")
    for label in spec.relation_labels:
        SubElement(relations, "Relation", value=label)
    SubElement(root, "Text", name=spec.to_name, value="$text")
    path = tmp_path / "test_schema.xml"
    path.write_bytes(tostring(root, encoding="utf-8"))
    return path


@pytest.fixture
def example() -> TrainingExample:
    return TrainingExample(
        DocumentInput(1, "AR FR"), GoldAnnotation(id=1, result=[entity()], completed_by=3)
    )
