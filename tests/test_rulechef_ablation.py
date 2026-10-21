"""Exercise the runner's save/load/score path without private inputs or a provider."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

pytest.importorskip("rulechef")

import experiments.rulechef_ablation as runner  # noqa: E402
from experiments import evaluate_rulechef as pilot  # noqa: E402
from methods.rulechef import RuleChefConfig, RuleChefMethod  # noqa: E402


def test_cells_share_initial_rules_and_score_all_test_notes(tmp_path, monkeypatch):
    model_calls = []
    rule = {
        "name": "pcr",
        "description": "PCR",
        "format": "regex",
        "content": r"\bPCR\b",
        "priority": 5,
        "output_key": "entities",
        "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "T:PCR"},
    }

    class Client:
        model = "synthetic"

        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            model_calls.append(kwargs)
            self.calls.append({"usage": {"input_tokens": 5, "output_tokens": 3}, "seconds": 0})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=json.dumps({"rules": [rule]})))
                ]
            )

    monkeypatch.setattr(
        runner, "RuleChefMethod", lambda config=None: RuleChefMethod(config, client=Client())
    )
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            [
                {
                    "id": index,
                    "data": {"text": f"PCR note {index}"},
                    "annotations": [
                        {
                            "id": index,
                            "completed_by": 3,
                            "result": [
                                {
                                    "id": "gold",
                                    "type": "labels",
                                    "from_name": "labels",
                                    "to_name": "text",
                                    "value": {
                                        "start": 0,
                                        "end": 3,
                                        "text": "PCR",
                                        "labels": ["T:PCR"],
                                    },
                                }
                            ],
                        }
                    ],
                }
                for index in range(1, 7)
            ]
        )
    )
    schema = tmp_path / "schema.xml"
    schema.write_text(
        '<View><Labels name="labels" toName="text"><Label value="T:PCR"/>'
        '</Labels><Text name="text" value="$text"/></View>'
    )
    glosses = tmp_path / "instructions.md"
    glosses.write_text("Find PCR.")
    manifest = {
        **{
            name + "_path": str(path)
            for name, path in (("data", data), ("schema", schema), ("glosses", glosses))
        },
        **{
            name + "_sha256": runner.digest(path)
            for name, path in (("data", data), ("schema", schema), ("glosses", glosses))
        },
        "synthesis": asdict(RuleChefConfig(model="synthetic", refinement_iterations=0)),
        "splits": [
            {
                "learning_ids": [1, 2, 3, 4],
                "train_ids": [1, 2, 3],
                "dev_ids": [4],
                "test_ids": [5, 6],
            }
        ],
        "rounds": 1,
        "profiles": ["oneshot", "train_refine", "holdout"],
    }
    for profile in manifest["profiles"]:
        runner.fit_cell(manifest, tmp_path, 1, profile)
        count = len(model_calls)
        runner.test_cell(manifest, tmp_path, 1, profile)
        assert len(model_calls) == count  # Inference never calls the provider.
        result = json.loads((tmp_path / "round_1" / profile / "result.json").read_text())
        assert result["metrics"]["strict_micro"]["f1"] == 1.0
        assert result["test_notes"] == 2 and result["metrics"]["failed_documents"] == 0
        assert result["tokens"] == {"input_tokens": 5, "output_tokens": 3}
    assert len(model_calls) == 2  # One initial fit for full train, one for train minus dev.
    runner.report(tmp_path, manifest)
    assert len(json.loads((tmp_path / "summary.json").read_text())["profiles"]) == 3


def test_pilot_snapshot_preserves_helper_modules_without_private_inputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for name in pilot.SOURCE_FILES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source for {name}\n")
    private = project / "data/private.json"
    private.parent.mkdir()
    private.write_text('{"text": "synthetic private input"}')
    monkeypatch.setattr(pilot, "ROOT", project)
    snapshot = tmp_path / "snapshot"
    hashes = pilot.snapshot_sources(snapshot)
    assert (snapshot / "methods/rulechef_client.py").is_file()
    assert (snapshot / "methods/rulechef_learning.py").is_file()
    assert hashes["methods/rulechef.py"] == pilot.sha256(project / "methods/rulechef.py")
    assert not (snapshot / "data").exists()
