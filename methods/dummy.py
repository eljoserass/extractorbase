"""An empty predictor for exercising the harness without a model or server."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from data import DocumentInput, LabelStudioTaskPrediction, TaskSpec, TrainingExample, prediction_for
from methods.base import Method, Scorer


@dataclass(frozen=True)
class DummyArtifact:
    task_spec: TaskSpec


class DummyManifest(BaseModel):
    method: Literal["dummy"] = "dummy"
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
            DummyManifest(task_spec=artifact.task_spec).model_dump_json(indent=2)
        )

    def load(self, directory: Path) -> DummyArtifact:
        manifest = DummyManifest.model_validate_json((directory / "method.json").read_text())
        return DummyArtifact(manifest.task_spec)

    def predict(
        self,
        artifact: DummyArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        return [prediction_for(document, [], "dummy") for document in inputs]
