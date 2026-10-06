"""An empty predictor for exercising the harness without a model or server."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from data import DocumentInput, LabelStudioTaskPrediction, TaskSpec, TrainingExample, prediction_for
from methods.base import Method, Scorer


@dataclass(frozen=True)
class DummyArtifact:
    task_spec: TaskSpec


class DummyMethod(Method[DummyArtifact]):
    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> DummyArtifact:
        return DummyArtifact(task_spec)

    def dump(self, artifact: DummyArtifact, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "method.json").write_text(
            json.dumps({"method": "dummy", "task_spec": asdict(artifact.task_spec)}, indent=2)
        )

    def load(self, directory: Path) -> DummyArtifact:
        raw = json.loads((directory / "method.json").read_text())["task_spec"]
        raw["entity_labels"] = tuple(raw["entity_labels"])
        raw["relation_labels"] = tuple(raw["relation_labels"])
        return DummyArtifact(TaskSpec(**raw))

    def predict(
        self,
        artifact: DummyArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        return [prediction_for(document, [], "dummy") for document in inputs]
