"""Each method owns its learning procedure and artifact representation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

from data import DocumentInput, LabelStudioTaskPrediction, TaskSpec, TrainingExample
from evaluator import MetricsReport

ArtifactT = TypeVar("ArtifactT")
type Scorer = Callable[
    [Sequence[LabelStudioTaskPrediction], Sequence[TrainingExample]], MetricsReport
]


class Method(ABC, Generic[ArtifactT]):
    @abstractmethod
    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> ArtifactT: ...

    @abstractmethod
    def dump(self, artifact: ArtifactT, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> ArtifactT: ...

    @abstractmethod
    def predict(
        self,
        artifact: ArtifactT,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]: ...
