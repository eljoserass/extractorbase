import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.usage import RunUsage

from data import DocumentInput
from evaluate_zero_shot import parser, performance, run
from methods.llm import EntityMention, Extraction, LLMArtifact, LLMMethod
from tests.conftest import entity


def arguments(tmp_path: Path, schema_path: Path) -> list[str]:
    data = tmp_path / "tasks.json"
    data.write_text(
        json.dumps(
            [
                {
                    "id": index,
                    "data": {"text": f"AR synthetic case {index}"},
                    "annotations": [{"id": index, "completed_by": 3, "result": [entity()]}],
                }
                for index in range(3)
            ]
        )
    )
    return [
        "--data",
        str(data),
        "--task-spec",
        str(schema_path),
        "--model",
        "synthetic-model",
        "--output",
        str(tmp_path / "evaluation"),
        "--offset",
        "0",
        "--count",
        "3",
    ]


def install_agent(
    monkeypatch: pytest.MonkeyPatch, *, interrupt: bool = False, fail: bool = False
) -> list[int | str]:
    calls: list[int | str] = []

    class FakeAgent:
        def run_sync(self, text: str, *, deps: DocumentInput) -> SimpleNamespace:
            assert not hasattr(deps, "gold")
            assert text.endswith(deps.text)
            calls.append(deps.id)
            if deps.id == 1 and interrupt:
                raise KeyboardInterrupt
            if deps.id == 1 and fail:
                raise ValueError("Synthetic model failure")
            return SimpleNamespace(
                output=Extraction(
                    entities=[EntityMention(id="e1", label="D:AR", text="AR", occurrence=0)],
                    relations=[],
                ),
                usage=RunUsage(input_tokens=10, output_tokens=5, requests=1),
            )

    def agent(self: LLMMethod, artifact: LLMArtifact) -> FakeAgent:
        assert artifact.demonstrations == ()
        assert artifact.task_spec.relation_labels == ()
        assert artifact.config.backend == "mlx"
        assert artifact.config.thinking is False
        assert "synthetic case" not in artifact.prompt
        return FakeAgent()

    monkeypatch.setattr(LLMMethod, "_agent", agent)
    return calls


def test_zero_shot_probe_resumes_to_full_scores_without_repeating_calls(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = arguments(tmp_path, schema_path)
    calls = install_agent(monkeypatch)
    output = tmp_path / "evaluation"
    assert run(parser().parse_args([*argv, "--max-documents", "1"])) == 0
    assert calls == [0]
    assert not (output / "metrics.json").exists()
    assert json.loads((output / "progress.json").read_text())["completed"] == 1
    assert run(parser().parse_args(argv)) == 0
    assert calls == [0, 1, 2]
    timings = json.loads((output / "performance.json").read_text())
    assert timings["reported_input_tokens"] == 30
    assert timings["reported_output_tokens"] == 15
    assert timings["reported_api_requests"] == 3
    report = json.loads((output / "metrics.json").read_text())
    assert report["documents"] == 3 and report["strict_micro"]["f1"] == 1
    assert report["overlap_micro"]["f1"] == 1
    assert report["relations"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["train_ids"] == [] and manifest["demonstration_count"] == 0
    assert len(json.loads((output / "predictions.json").read_text())) == 3
    assert run(parser().parse_args(argv)) == 0
    assert calls == [0, 1, 2]


def test_interrupted_zero_shot_run_keeps_completed_documents(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = arguments(tmp_path, schema_path)
    first_calls = install_agent(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        run(parser().parse_args(argv))
    assert first_calls == [0, 1]
    assert (tmp_path / "evaluation" / "documents" / "000000.json").exists()
    resumed_calls = install_agent(monkeypatch)
    assert run(parser().parse_args(argv)) == 0
    assert resumed_calls == [1, 2]


def test_failed_documents_are_saved_and_remain_in_full_score(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = arguments(tmp_path, schema_path)
    calls = install_agent(monkeypatch, fail=True)
    assert run(parser().parse_args([*argv, "--max-documents", "2"])) == 1
    assert calls == [0, 1]
    assert run(parser().parse_args(argv)) == 1
    output = tmp_path / "evaluation"
    report = json.loads((output / "metrics.json").read_text())
    assert report["failed_documents"] == 1 and report["documents"] == 3
    assert report["strict_micro"]["recall"] == pytest.approx(2 / 3)
    timings = json.loads((output / "performance.json").read_text())
    assert timings["documents_with_token_usage"] == 2
    assert timings["documents_without_token_usage"] == 1
    assert timings["reported_input_tokens"] == 20
    assert run(parser().parse_args(argv)) == 1
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("change", ["model", "source", "prompt"])
def test_resume_rejects_changed_run_before_new_requests(
    tmp_path: Path, schema_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    argv = arguments(tmp_path, schema_path)
    calls = install_agent(monkeypatch)
    assert run(parser().parse_args([*argv, "--max-documents", "1"])) == 0
    args = parser().parse_args(argv)
    if change == "model":
        args.model = "different-model"
    elif change == "source":
        args.data[0].write_text(args.data[0].read_text() + "\n")
    else:
        (tmp_path / "evaluation" / "artifact" / "prompt.txt").write_text("Different prompt")
    with pytest.raises(ValueError, match="choose a new --output"):
        run(args)
    assert calls == [0]


def test_performance_includes_repairs_and_reports_unknown_token_coverage() -> None:
    receipts = [
        {"seconds": seconds, "prediction": {"meta": {"inference_usage": usage}}}
        for seconds, usage in [
            (2, {"input_tokens": 100, "output_tokens": 20, "requests": 1}),
            (4, {"input_tokens": 120, "output_tokens": 10, "requests": 2}),
            (5, {"input_tokens": None, "output_tokens": None, "requests": None}),
        ]
    ]
    result = performance(receipts, 5)
    assert result["prediction_seconds"] == 11
    assert result["median_seconds_per_document"] == 4
    assert result["p95_seconds_per_document"] == 5
    assert result["reported_input_tokens"] == 220
    assert result["reported_output_tokens"] == 30
    assert result["reported_repair_requests"] == 1
    assert result["documents_without_token_usage"] == 1
    assert result["output_tokens_per_second_end_to_end"] == 5
