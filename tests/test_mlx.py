"""Exercise the real client/parser boundary without requiring Apple hardware."""

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI
from typer.testing import CliRunner

from cli import app
from data import DocumentInput, TaskSpec
from methods import llm
from methods.llm import LLMConfig, LLMMethod


def fake_server(
    monkeypatch: pytest.MonkeyPatch, replies: list[str]
) -> tuple[list[dict[str, Any]], list[AsyncOpenAI]]:
    requests: list[dict[str, Any]] = []
    clients: list[AsyncOpenAI] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-synthetic-test",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": replies.pop(0)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    def client(**kwargs: Any) -> AsyncOpenAI:
        instance = AsyncOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)), **kwargs
        )
        clients.append(instance)
        return instance

    monkeypatch.setattr(llm, "AsyncOpenAI", client)
    return requests, clients


def extraction_json() -> str:
    return json.dumps(
        {
            "entities": [{"id": "e1", "label": "MARKER", "text": "marker", "occurrence": 0}],
            "relations": [],
        }
    )


@pytest.mark.parametrize("backend", ["lmstudio", "mlx"])
@pytest.mark.parametrize("thinking", [False, True])
def test_backend_request_and_reloaded_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: Literal["lmstudio", "mlx"],
    thinking: bool,
) -> None:
    requests, clients = fake_server(monkeypatch, [extraction_json()])
    method = LLMMethod(
        LLMConfig(
            backend=backend,
            base_url="http://127.0.0.1:8080/v1",
            model="synthetic-model",
            thinking=thinking,
        )
    )
    method.dump(method.fit([], TaskSpec(("MARKER",), ())), tmp_path)
    artifact = LLMMethod().load(tmp_path)
    try:
        predictions = method.predict(artifact, [DocumentInput(11, "a marker b")])
    finally:
        for client in clients:
            asyncio.run(client.close())

    assert artifact.config.backend == backend
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "synthetic-model"
    result = predictions[0]["predictions"][0]["result"]
    assert len(result) == 1
    entity = result[0]
    assert entity["type"] == "labels"
    assert entity["value"]["start"] == 2
    assert entity["value"]["end"] == 8
    assert "extraction_error" not in predictions[0].get("meta", {})

    if backend == "mlx":
        assert "response_format" not in request
        assert "reasoning_effort" not in request
        assert request["chat_template_kwargs"]["enable_thinking"] is thinking
        # The server ignores response_format, so its prompt must contain the actual schema.
        messages = json.dumps(request["messages"])
        assert '"properties"' in messages.replace('\\"', '"')
        assert "occurrence" in messages and "entities" in messages and "relations" in messages
    else:
        assert request["response_format"]["type"] == "json_schema"
        assert "chat_template_kwargs" not in request
        if not thinking:
            assert request["reasoning_effort"] == "none"


def test_mlx_repairs_invalid_json_through_plain_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, clients = fake_server(monkeypatch, ['{"entities": "invalid"}', extraction_json()])
    method = LLMMethod(LLMConfig(backend="mlx"))
    artifact = method.fit([], TaskSpec(("MARKER",), ()))
    try:
        predictions = method.predict(artifact, [DocumentInput(11, "a marker b")])
    finally:
        for client in clients:
            asyncio.run(client.close())

    assert len(requests) == 2
    assert all("response_format" not in request for request in requests)
    assert len(requests[1]["messages"]) > len(requests[0]["messages"])
    assert predictions[0]["predictions"][0]["result"]
    assert "extraction_error" not in predictions[0].get("meta", {})


def test_old_artifact_defaults_to_lmstudio(tmp_path: Path) -> None:
    method = LLMMethod()
    method.dump(method.fit([], TaskSpec(("MARKER",), ())), tmp_path)
    path = tmp_path / "method.json"
    manifest = json.loads(path.read_text())
    manifest["config"].pop("backend")
    path.write_text(json.dumps(manifest))
    assert method.load(tmp_path).config.backend == "lmstudio"


def test_cli_mlx_fit_then_reload_predict_without_private_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests, clients = fake_server(monkeypatch, [extraction_json(), extraction_json()])
    schema = tmp_path / "task.xml"
    schema.write_text(
        '<View><Labels name="labels" toName="text"><Label value="MARKER" /></Labels>'
        '<Text name="text" value="$text" /></View>'
    )
    tasks = []
    for id, text in [(1, "one marker"), (2, "two marker")]:
        tasks.append(
            {
                "id": id,
                "data": {"text": text},
                "annotations": [
                    {
                        "id": id,
                        "completed_by": 3,
                        "result": [
                            {
                                "id": "e1",
                                "type": "labels",
                                "from_name": "labels",
                                "to_name": "text",
                                "value": {
                                    "start": 4,
                                    "end": 10,
                                    "text": "marker",
                                    "labels": ["MARKER"],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    data = tmp_path / "tasks.json"
    data.write_text(json.dumps(tasks))
    artifact = tmp_path / "artifact"
    output = tmp_path / "evaluation"
    common = ["--data", str(data), "--task-spec", str(schema), "--count", "1"]
    runner = CliRunner()
    try:
        fit = runner.invoke(
            app,
            [
                *common,
                "--stage",
                "fit",
                "--method",
                "llm",
                "--dump",
                str(artifact),
                "--llm-backend",
                "mlx",
                "--endpoint",
                "http://127.0.0.1:8080/v1",
                "--model",
                "synthetic-model",
                "--yes",
            ],
        )
        assert fit.exit_code == 0, fit.output
        predict = runner.invoke(
            app,
            [
                *common,
                "--stage",
                "predict",
                "--load",
                str(artifact),
                "--offset",
                "1",
                "--output",
                str(output),
            ],
        )
        assert predict.exit_code == 0, predict.output
    finally:
        for client in clients:
            asyncio.run(client.close())

    assert len(requests) == 2
    assert all("response_format" not in request for request in requests)
    saved = json.loads((artifact / "method.json").read_text())
    assert saved["config"]["backend"] == "mlx"
    assert saved["config"]["model"] == "synthetic-model"
    report = json.loads((output / "metrics.json").read_text())
    assert report["strict_micro"]["f1"] == 1.0
