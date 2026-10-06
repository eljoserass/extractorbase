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
def example() -> TrainingExample:
    return TrainingExample(
        DocumentInput(1, "AR FR"), GoldAnnotation(id=1, result=[entity()], completed_by=3)
    )
