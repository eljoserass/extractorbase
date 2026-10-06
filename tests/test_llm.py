from pathlib import Path
from types import SimpleNamespace

import pytest

from data import DocumentInput, TaskSpec, TrainingExample
from methods.llm import (
    EntityMention,
    Extraction,
    LLMMethod,
    RelationMention,
    ground_quotes,
    to_results,
)


def test_llm_fit_dump_reload_without_server(
    tmp_path: Path, example: TrainingExample, spec: TaskSpec
) -> None:
    method = LLMMethod()
    artifact = method.fit([example], spec)
    assert "AR FR" in artifact.prompt
    method.dump(artifact, tmp_path)
    assert LLMMethod().load(tmp_path) == artifact


def test_quotes_resolve_repeated_spans_and_relations(spec: TaskSpec) -> None:
    document = DocumentInput(1, "AR FR AR")
    extraction = Extraction(
        entities=[
            EntityMention(id="a", label="D:AR", text="AR", occurrence=1),
            EntityMention(id="b", label="T:FR", text="FR", occurrence=0),
        ],
        relations=[RelationMention(from_id="b", to_id="a", labels=["valor"])],
    )
    results = to_results(extraction, document, spec)
    assert results[0]["value"]["start"] == 6
    assert results[-1]["from_id"] == "b"
    extraction.entities[0].occurrence = 2
    with pytest.raises(ValueError, match="between 0 and 1"):
        to_results(extraction, document, spec)


def test_grounding_feedback_reports_all_invalid_quotes(spec: TaskSpec) -> None:
    extraction = Extraction(
        entities=[
            EntityMention(id="a", label="D:AR", text="arthritis", occurrence=0),
            EntityMention(id="b", label="T:FR", text="factor", occurrence=0),
        ],
        relations=[],
    )
    with pytest.raises(ValueError) as error:
        to_results(extraction, DocumentInput(1, "AR FR"), spec)
    assert "arthritis" in str(error.value) and "factor" in str(error.value)
    assert "Omitted" in str(error.value)


def test_grounding_keeps_valid_mentions_and_abstains_without_guessing(spec: TaskSpec) -> None:
    extraction = Extraction(
        entities=[
            EntityMention(id="a", label="D:AR", text="AR", occurrence=0),
            EntityMention(id="b", label="T:FR", text="FR", occurrence=1),
            EntityMention(id="c", label="D:AR", text="arthritis", occurrence=0),
            EntityMention(id="d", label="D:AR", text="AR", occurrence=3),
        ],
        relations=[
            RelationMention(from_id="b", to_id="a", labels=["valor"]),
            RelationMention(from_id="c", to_id="a", labels=["valor"]),
        ],
    )
    grounded = ground_quotes(extraction, DocumentInput(1, "AR FR AR"), spec)
    assert [item["id"] for item in grounded.results if item["type"] == "labels"] == ["a", "b"]
    assert grounded.results[1]["value"]["start"] == 3
    assert len(grounded.results) == 3  # Only the relation with grounded endpoints remains.
    assert len(grounded.warnings) == 4
    assert any("Corrected b" in warning for warning in grounded.warnings)
    assert any("Omitted d" in warning for warning in grounded.warnings)


def test_grounding_omits_unknown_labels_duplicate_ids_and_dangling_relations(
    spec: TaskSpec,
) -> None:
    extraction = Extraction(
        entities=[
            EntityMention(id="a", label="D:AR", text="AR", occurrence=0),
            EntityMention(id="a", label="D:AR", text="AR", occurrence=0),
            EntityMention(id="b", label="T:FR", text="FR", occurrence=0),
            EntityMention(id="c", label="unknown", text="FR", occurrence=0),
        ],
        relations=[
            RelationMention(from_id="missing", to_id="b", labels=["valor"]),
            RelationMention(from_id="b", to_id="b", labels=["unknown"]),
        ],
    )
    grounded = ground_quotes(extraction, DocumentInput(1, "AR FR"), spec)
    assert len(grounded.results) == 1 and grounded.results[0]["id"] == "b"
    assert len(grounded.warnings) == 5


def test_inference_failure_retains_document_and_uses_text_only_inputs(
    monkeypatch: pytest.MonkeyPatch,
    spec: TaskSpec,
) -> None:
    class FakeAgent:
        def run_sync(self, text: str, *, deps: DocumentInput) -> SimpleNamespace:
            assert not hasattr(deps, "gold")
            if deps.id == 2:
                raise ValueError("simulated inference failure")
            return SimpleNamespace(
                output=Extraction(
                    entities=[
                        EntityMention(id="a", label="D:AR", text="AR", occurrence=0),
                    ],
                    relations=[],
                )
            )

    method = LLMMethod()
    monkeypatch.setattr(method, "_agent", lambda artifact: FakeAgent())
    predictions = method.predict(
        method.fit([], spec), [DocumentInput(1, "AR"), DocumentInput(2, "AR")]
    )
    assert len(predictions) == 2
    assert predictions[0]["predictions"][0]["result"]
    assert predictions[1]["predictions"][0]["result"] == []
    assert "simulated inference failure" in str(predictions[1]["meta"]["extraction_error"])
