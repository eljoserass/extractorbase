from pathlib import Path

from data import TaskSpec, TrainingExample
from methods.dummy import DummyMethod


def test_dummy_artifact_reload(tmp_path: Path, example: TrainingExample, spec: TaskSpec) -> None:
    method = DummyMethod()
    method.dump(method.fit([example], spec), tmp_path)
    fresh = DummyMethod()
    artifact = fresh.load(tmp_path)
    assert artifact.task_spec == spec
    assert fresh.predict(artifact, [example.input])[0]["predictions"][0]["result"] == []
