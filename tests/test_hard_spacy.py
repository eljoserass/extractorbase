"""Public contract checks with synthetic data, independent of the private pilot notes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("spacy")

from data import DocumentInput, GoldAnnotation, TaskSpec, TrainingExample  # noqa: E402
from methods.hard_spacy import HardSpacyMethod  # noqa: E402


def fitted_method():
    text = "MTX; NAD 4"
    mentions = [
        ("drug", 0, 3, "MED:MTX"),
        ("index", 5, 8, "AC:NAD"),
        ("value", 9, 10, "V:VALOR"),
    ]
    results = [
        {
            "id": entity_id,
            "type": "labels",
            "from_name": "clinical",
            "to_name": "note",
            "value": {"start": start, "end": end, "text": text[start:end], "labels": [label]},
        }
        for entity_id, start, end, label in mentions
    ]
    results.append({"type": "relation", "from_id": "index", "to_id": "value", "labels": ["valor"]})
    spec = TaskSpec(
        ("MED:MTX", "AC:NAD", "V:VALOR"), ("valor",), from_name="clinical", to_name="note"
    )
    example = TrainingExample(DocumentInput(1, text), GoldAnnotation(id=1, result=results))
    method = HardSpacyMethod()
    return method, method.fit([example], spec)


def test_new_text_offsets_repetitions_and_relation_endpoints():
    method, artifact = fitted_method()
    document = DocumentInput("new-id", "MTX MTX; NAD12")
    prediction = method.predict(artifact, [document])[0]
    assert prediction["id"] == "new-id"
    assert prediction["data"] == {"text": document.text}
    results = prediction["predictions"][0]["result"]
    entities = {item["id"]: item for item in results if item["type"] == "labels"}
    observed = {
        (label, item["value"]["start"], item["value"]["end"])
        for item in entities.values()
        for label in item["value"]["labels"]
    }
    assert observed == {
        ("MED:MTX", 0, 3),
        ("MED:MTX", 4, 7),
        ("AC:NAD", 9, 12),
        ("V:VALOR", 12, 14),
    }
    for item in entities.values():
        value = item["value"]
        assert value["text"] == document.text[value["start"] : value["end"]]
        assert (item["from_name"], item["to_name"]) == ("clinical", "note")
    relations = [item for item in results if item["type"] == "relation"]
    assert len(relations) == 1
    relation = relations[0]
    assert entities[relation["from_id"]]["value"]["labels"] == ["AC:NAD"]
    assert entities[relation["to_id"]]["value"]["text"] == "12"
    assert method.predict(artifact, [document]) == [prediction]
    assert method.predict(artifact, []) == []


def test_artifact_reloads_in_fresh_process_and_directory(tmp_path):
    method, artifact = fitted_method()
    directory = tmp_path / "artifact"
    method.dump(artifact, directory)
    document = DocumentInput("reload", "NAD 27; MTX")
    expected = method.predict(artifact, [document])
    script = """
import json
import sys
from pathlib import Path
from data import DocumentInput
from methods.hard_spacy import HardSpacyMethod
method = HardSpacyMethod()
artifact = method.load(Path(sys.argv[1]))
document = DocumentInput(**json.load(sys.stdin))
print(json.dumps(method.predict(artifact, [document])))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", script, str(directory)],
        input=json.dumps({"id": document.id, "text": document.text}),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == expected
