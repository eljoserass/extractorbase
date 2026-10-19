"""Synthetic contract and position regressions; no credentials or clinical data."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("rulechef")

from rulechef.core import Dataset, Example, Rule, Task, TaskType  # noqa: E402

from data import DocumentInput, GoldAnnotation, TaskSpec, TrainingExample  # noqa: E402
from evaluator import score  # noqa: E402
from methods.rulechef import RuleChefConfig, RuleChefMethod  # noqa: E402

RULE = {
    "name": "pcr",
    "description": "PCR",
    "format": "regex",
    "content": r"\bPCR\b",
    "output_key": "entities",
    "priority": 5,
    "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "T:PCR"},
}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"rules": [RULE]}),
                    )
                )
            ]
        )


def test_fit_unicode_repeated_mentions_and_reload_without_client(tmp_path):
    client = FakeClient()
    method = RuleChefMethod(RuleChefConfig(refinement_iterations=1), client=client)
    spec = TaskSpec(("T:PCR",), (), from_name="clinical", to_name="note")
    example = TrainingExample(
        DocumentInput(1, "PCR"),
        GoldAnnotation(
            id=1,
            result=[
                {
                    "id": "gold",
                    "type": "labels",
                    "from_name": "clinical",
                    "to_name": "note",
                    "value": {"start": 0, "end": 3, "text": "PCR", "labels": ["T:PCR"]},
                }
            ],
        ),
    )
    artifact = method.fit([example], spec, scorer=score)
    assert len(client.calls) == 1
    assert artifact.provenance["training_metrics"]["strict_micro"]["f1"] == 1.0
    assert artifact.rules[0].validated_precision == 1.0
    document = DocumentInput("new", "ñ: PCR PCR")
    prediction = method.predict(artifact, [document])
    entities = prediction[0]["predictions"][0]["result"]
    assert [(e["value"]["start"], e["value"]["end"]) for e in entities] == [(3, 6), (7, 10)]
    assert all(e["from_name"] == "clinical" and e["to_name"] == "note" for e in entities)
    method.dump(artifact, tmp_path)
    restored = RuleChefMethod()
    assert restored.client is None
    assert restored.predict(restored.load(tmp_path), [document]) == prediction
    assert len(client.calls) == 1


def test_refinement_and_rule_priority_do_not_credit_wrong_occurrence():
    from methods.rulechef import ExactRuleLearner

    learner = ExactRuleLearner(FakeClient(), use_grex=False)
    task = Task("positions", "PCR", {"text": "str"}, {"entities": "list"}, TaskType.NER)
    dataset = Dataset(
        name="positions",
        task=task,
        examples=[
            Example(
                id="1",
                input={"text": "PCR PCR"},
                source="synthetic",
                expected_output={
                    "entities": [{"text": "PCR", "start": 4, "end": 7, "type": "T:PCR"}]
                },
            )
        ],
    )
    wrong = Rule.from_dict({**RULE, "content": "^PCR"})
    assert learner._evaluate_rules([wrong], dataset).micro_f1 == 0.0
    learner._stamp_validated_stats([wrong], dataset)
    assert wrong.validated_precision == 0.0


def test_invalid_predictions_fail_instead_of_silently_moving_offsets():
    from methods.rulechef import RuleChefArtifact

    method = RuleChefMethod()
    artifact = RuleChefArtifact(
        TaskSpec(("T:PCR",), ()),
        [
            Rule.from_dict(
                {
                    **RULE,
                    "output_template": {**RULE["output_template"], "start": 999},
                }
            )
        ],
    )
    with pytest.raises(ValueError, match="Invalid RuleChef entity"):
        method.predict(artifact, [DocumentInput(1, "PCR")])
