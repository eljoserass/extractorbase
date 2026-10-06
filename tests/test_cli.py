import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cli import app
from data import DocumentInput, LabelStudioTaskPrediction, prediction_for
from methods.dummy import DummyArtifact, DummyMethod
from methods.llm import EntityMention, Extraction, LLMMethod
from tests.conftest import entity


def dataset(path: Path) -> Path:
    tasks = [
        {
            "id": index,
            "data": {"text": f"AR case {index}"},
            "annotations": [
                {"id": index, "completed_by": 3, "result": [entity()]},
            ],
        }
        for index in range(10)
    ]
    path.write_text(json.dumps(tasks))
    return path


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_prints_options_without_requiring_data(flag: str) -> None:
    result = CliRunner().invoke(app, [flag])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.stdout
    assert "--data" in result.stdout and "--stage" in result.stdout
    assert "Missing option" not in result.output


def test_first_five_fit_next_five_predict_and_reload(tmp_path: Path, schema_path: Path) -> None:
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    output = tmp_path / "evaluation"
    result = CliRunner().invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "dummy",
            "--stage",
            "run",
            "--dump",
            str(artifact),
            "--output",
            str(output),
        ],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert "Approve this fitted artifact" in result.output
    assert "HELD-OUT" in result.output
    config = json.loads((output / "config.json").read_text())
    assert config["train_ids"] == [0, 1, 2, 3, 4]
    assert config["evaluation_ids"] == [5, 6, 7, 8, 9]
    assert json.loads((artifact / "fit.json").read_text())["approved"]
    assert len(json.loads((output / "predictions.json").read_text())) == 5
    assert json.loads((output / "metrics.json").read_text())["strict_micro"]["f1"] == 0
    reload = CliRunner().invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--stage",
            "predict",
            "--load",
            str(artifact),
            "--output",
            str(tmp_path / "reloaded"),
        ],
    )
    assert reload.exit_code == 0, reload.output
    assert "Approve" not in reload.output


def test_declining_keeps_artifact_unapproved_and_blocks_prediction(
    tmp_path: Path, schema_path: Path
) -> None:
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    result = CliRunner().invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "dummy",
            "--stage",
            "run",
            "--dump",
            str(artifact),
        ],
        input="n\n",
    )
    assert result.exit_code != 0
    assert not json.loads((artifact / "fit.json").read_text())["approved"]
    assert not (tmp_path / "evaluation" / "predictions.json").exists()


def test_training_data_cannot_be_used_as_held_out(tmp_path: Path, schema_path: Path) -> None:
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "dummy",
            "--stage",
            "fit",
            "--dump",
            str(artifact),
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--stage",
            "predict",
            "--load",
            str(artifact),
            "--offset",
            "0",
            "--yes",
        ],
    )
    assert result.exit_code == 1 and "fitting documents" in result.output


def test_unlabeled_inference_has_no_fabricated_metrics(tmp_path: Path, schema_path: Path) -> None:
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "dummy",
            "--stage",
            "fit",
            "--dump",
            str(artifact),
            "--yes",
        ],
    )
    assert result.exit_code == 0
    unlabeled = tmp_path / "unlabeled.json"
    unlabeled.write_text(json.dumps([{"id": 11, "data": {"text": "new FR document"}}]))
    output = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(unlabeled),
            "--stage",
            "predict",
            "--load",
            str(artifact),
            "--offset",
            "0",
            "--count",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((output / "metrics.json").read_text()) is None


def test_failed_training_preview_cannot_be_approved_in_predict_stage(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_predict(
        self: DummyMethod, artifact: DummyArtifact, inputs: Sequence[DocumentInput]
    ) -> list[LabelStudioTaskPrediction]:
        predictions = [prediction_for(document, [], "dummy") for document in inputs]
        predictions[0]["meta"] = {"extraction_error": "simulated service failure"}
        return predictions

    monkeypatch.setattr(DummyMethod, "predict", failed_predict)
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "dummy",
            "--stage",
            "fit",
            "--dump",
            str(artifact),
            "--yes",
        ],
    )
    assert result.exit_code == 1 and "unapproved" in result.output
    result = runner.invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--stage",
            "predict",
            "--load",
            str(artifact),
            "--yes",
        ],
    )
    assert result.exit_code == 1 and "failed training preview" in result.output
    assert not json.loads((artifact / "fit.json").read_text())["approved"]


def test_grounding_warnings_are_shown_before_approval_and_saved(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeAgent:
        def run_sync(self, text: str, *, deps: DocumentInput) -> SimpleNamespace:
            assert "DOCUMENTO A ANOTAR" in text and not hasattr(deps, "gold")
            return SimpleNamespace(
                output=Extraction(
                    entities=[
                        EntityMention(id="valid", label="D:AR", text="AR", occurrence=0),
                        EntityMention(id="absent", label="D:AR", text="arthritis", occurrence=0),
                    ],
                    relations=[],
                )
            )

    monkeypatch.setattr(LLMMethod, "_agent", lambda self, artifact: FakeAgent())
    data = dataset(tmp_path / "data.json")
    artifact = tmp_path / "artifact"
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "--task-spec",
            str(schema_path),
            "--data",
            str(data),
            "--method",
            "llm",
            "--stage",
            "run",
            "--dump",
            str(artifact),
            "--output",
            str(output),
        ],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert result.output.index("Omitted absent") < result.output.index("Approve this fitted")
    predictions = json.loads((output / "predictions.json").read_text())
    assert all(prediction["meta"]["extraction_warnings"] for prediction in predictions)
    assert all(len(prediction["predictions"][0]["result"]) == 1 for prediction in predictions)
    assert json.loads((output / "metrics.json").read_text())["failed_documents"] == 0
