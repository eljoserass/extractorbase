# /// script
# requires-python = ">=3.12"
# dependencies = ["rulechef==0.2.2"]
# ///
"""Offline RuleChef integration probe; synthetic data and a fake model only."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel
from rulechef import RuleChef, RuleFormat, Task, TaskType, split_dataset
from rulechef.core import Dataset, Example, Rule
from rulechef.evaluation import evaluate_dataset
from rulechef.executor import RuleExecutor


class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["T:PCR", "V:VALOR"]


class Entities(BaseModel):
    entities: list[Entity]


RULES = [
    {
        "name": "pcr",
        "description": "Synthetic PCR mention",
        "format": "regex",
        "content": r"\bPCR\b",
        "output_key": "entities",
        "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "T:PCR"},
    },
    {
        "name": "pcr_value",
        "description": "Synthetic PCR numeric value",
        "format": "regex",
        "content": r"(?<=PCR )\d+(?:[.,]\d+)?",
        "output_key": "entities",
        "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "V:VALOR"},
    },
]


class FakeClient:
    """The small client surface a future Codex/MLX adapter needs to implement."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"rules": RULES})))]
        )


def run(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    client = FakeClient()
    task = Task(
        name="Synthetic PCR",
        description="Extract PCR test names and their separate numeric values.",
        input_schema={"text": "str"},
        output_schema=Entities,
        type=TaskType.NER,
        matching_mode="exact",
        text_field="text",
    )
    chef = RuleChef(
        task,
        client=client,
        storage_path=str(output / "synthetic_dataset"),
        allowed_formats=[RuleFormat.REGEX],
        synthesis_strategy="bulk",
        use_grex=False,
        llm_fallback=False,
    )
    for index in range(10):
        text = f"PCR {index}"
        chef.add_example(
            {"text": text},
            {
                "entities": [
                    {"text": "PCR", "start": 0, "end": 3, "type": "T:PCR"},
                    {"text": str(index), "start": 4, "end": len(text), "type": "V:VALOR"},
                ]
            },
        )
    learned = chef.learn_rules(run_evaluation=False, max_refinement_iterations=0)
    assert learned is not None
    rules, _ = learned
    assert len(rules) == 2 and len(client.calls) == 1
    saved = output / "rules.json"
    saved.write_text(json.dumps([rule.to_dict() for rule in rules], indent=2) + "\n")

    # Deploy only serialized rules and the executor; no client or credentials.
    restored = [Rule.from_dict(value) for value in json.loads(saved.read_text())]
    executor = RuleExecutor()
    text = "ñ: PCR 12; PCR 4"
    prediction = executor.apply_rules(restored, {"text": text}, TaskType.NER, "text")
    actual = {
        (entity["text"], entity["start"], entity["end"], entity["type"])
        for entity in prediction["entities"]
    }
    assert actual == {
        ("PCR", 3, 6, "T:PCR"),
        ("12", 7, 9, "V:VALOR"),
        ("PCR", 11, 14, "T:PCR"),
        ("4", 15, 16, "V:VALOR"),
    }
    assert all(text[e["start"] : e["end"]] == e["text"] for e in prediction["entities"])
    assert chef.extract({"text": text}) == prediction
    assert len(client.calls) == 1

    # Check a material protocol caveat with the same ten synthetic examples.
    _, development = split_dataset(chef.dataset, holdout_fraction=0.2)
    assert development is None  # The library requires at least five dev notes.
    conflict = {**RULES[0], "name": "same_span_other_type"}
    conflict["output_template"] = {**RULES[0]["output_template"], "type": "V:VALOR"}
    competing = executor.apply_rules(
        [restored[0], Rule.from_dict(conflict)], {"text": "PCR"}, TaskType.NER, "text"
    )
    assert len(competing["entities"]) == 1
    wrong_position = Rule.from_dict({**RULES[0], "content": "^PCR"})
    position_dataset = Dataset(
        name="synthetic_positions",
        task=task,
        examples=[
            Example(
                id="positions",
                input={"text": "PCR PCR"},
                expected_output={
                    "entities": [{"text": "PCR", "start": 4, "end": 7, "type": "T:PCR"}]
                },
                source="synthetic",
            )
        ],
    )
    native = chef.learner._evaluate_rules([wrong_position], position_dataset)
    exact = evaluate_dataset([wrong_position], position_dataset, executor.apply_rules, mode="exact")
    assert task.matching_mode == "exact"
    assert native.micro_f1 == 1.0 and exact.micro_f1 == 0.0
    result = {
        "rulechef_version": version("rulechef"),
        "scope": "synthetic integration plumbing; not model synthesis or clinical evaluation",
        "real_api_calls": 0,
        "fake_synthesis_calls": len(client.calls),
        "synthesized_rules_from_fake_response": len(restored),
        "save_reload_equal": True,
        "unicode_and_repeated_mention_offsets_correct": True,
        "runtime_llm_calls": 0,
        "runtime_client_or_api_key_required": False,
        "ten_example_20_percent_holdout_created": development is not None,
        "same_span_different_types_retained": len(competing["entities"]),
        "wrong_occurrence_refinement_f1": native.micro_f1,
        "wrong_occurrence_exact_f1": exact.micro_f1,
        "clinical_data_read": False,
    }
    (output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
