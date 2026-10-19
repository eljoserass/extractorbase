"""Self-contained RuleChef NER adapter; model access is needed only for fit."""

import hashlib
import json
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import create_model
from rulechef.core import Dataset, Example, Rule, RuleFormat, Task, TaskType
from rulechef.evaluation import EvalResult, evaluate_dataset, evaluate_rules_individually
from rulechef.executor import RuleExecutor
from rulechef.learner import RuleLearner

from data import (
    DocumentInput,
    LabelStudioTaskPrediction,
    ResultItem,
    TaskSpec,
    TrainingExample,
    prediction_for,
)
from methods.base import Method, Scorer


@dataclass(frozen=True)
class RuleChefConfig:
    backend: str = "codex"
    model: str | None = None  # None uses the user's configured Codex model.
    endpoint: str | None = None
    max_rules: int = 64
    refinement_iterations: int = 3
    call_timeout_seconds: int = 1200
    trace_directory: str | None = None


@dataclass
class RuleChefArtifact:
    task_spec: TaskSpec
    rules: list[Rule]
    provenance: dict[str, Any] = field(default_factory=dict)


class CodexClient:
    """Translate RuleChef's completion calls to the authenticated Codex CLI.

    Each request is independent and runs outside the repository with shell,
    apps, web search and delegation disabled. Save and audit every event.
    """

    def __init__(self, config: RuleChefConfig):
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        config_file = Path.home() / ".codex/config.toml"
        defaults = tomllib.loads(config_file.read_text()) if config_file.exists() else {}
        self.model = config.model or defaults.get("model")
        self.reasoning = defaults.get("model_reasoning_effort", "high")

    @staticmethod
    def _schema() -> dict:
        def obj(properties: dict[str, Any]) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }

        string = {"type": "string"}
        rule = obj(
            {
                "name": string,
                "description": string,
                "format": {"type": "string", "enum": ["regex"]},
                "content": string,
                "priority": {"type": "integer"},
                "output_key": {"type": "string", "enum": ["entities"]},
                "output_template": obj({key: string for key in ("text", "start", "end", "type")}),
            }
        )
        return obj(
            {
                "analysis": string,
                "rules": {"type": "array", "items": rule},
                "deleted_rules": {"type": "array", "items": string},
            }
        )

    def create(self, **kwargs: Any) -> SimpleNamespace:
        index = len(self.calls) + 1
        with tempfile.TemporaryDirectory(prefix="rulechef-completion-") as temporary:
            root = Path(temporary)
            trace = (
                Path(self.config.trace_directory).resolve() / f"call_{index:03d}"
                if self.config.trace_directory
                else root
            )
            trace.mkdir(parents=True, exist_ok=True)
            (root / "AGENTS.md").write_text(
                "Return the requested JSON using only the supplied messages. "
                "Do not use tools, read files, browse, or delegate.\n"
            )
            schema_file = root / "response_schema.json"
            schema_file.write_text(json.dumps(self._schema()))
            response_file = trace / "response.json"
            prompt = (
                "Act only as the completion model for RuleChef. Follow the messages below. "
                "Return JSON only. Use regex rules, no tools. "
                "For a patch, return deleted_rules as requested; otherwise use [].\n\n"
                + json.dumps(kwargs["messages"], ensure_ascii=False)
            )
            (trace / "request.json").write_text(json.dumps(kwargs, ensure_ascii=False, indent=2))
            command = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--cd",
                str(root),
                "--output-schema",
                str(schema_file),
                "--output-last-message",
                str(response_file),
                "-c",
                'web_search="disabled"',
                "-c",
                "features.shell_tool=false",
                "-c",
                "features.apps=false",
                "-c",
                "features.multi_agent=false",
                "-c",
                "features.skill_search=false",
                "-c",
                f'model_reasoning_effort="{self.reasoning}"',
            ]
            if self.model:
                command += ["--model", self.model]
            started = time.perf_counter()
            print(f"RuleChef model call {index}: {self.model or 'Codex default'}", flush=True)
            with (
                (trace / "events.jsonl").open("w") as events,
                (trace / "stderr.log").open("w") as errors,
            ):
                completed = subprocess.run(
                    [*command, "-"],
                    input=prompt,
                    text=True,
                    stdout=events,
                    stderr=errors,
                    timeout=self.config.call_timeout_seconds,
                    check=False,
                )
            record: dict[str, Any] = {
                "index": index,
                "model": self.model,
                "reasoning_effort": self.reasoning,
                "seconds": time.perf_counter() - started,
                "returncode": completed.returncode,
                "requested_output_tokens": kwargs.get("max_completion_tokens"),
                "output_token_limit_enforced": False,
                "usage": {},
                "tool_calls": [],
            }
            for line in (trace / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event.get("type") == "turn.completed":
                    for key, count in event.get("usage", {}).items():
                        record["usage"][key] = record["usage"].get(key, 0) + count
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("type") not in {
                    "agent_message",
                    "reasoning",
                    None,
                }:
                    record["tool_calls"].append(item.get("type"))
            self.calls.append(record)
            (trace / "metadata.json").write_text(json.dumps(record, indent=2))
            if completed.returncode or not response_file.exists():
                raise RuntimeError(f"Codex completion failed; inspect {trace / 'stderr.log'}")
            if record["tool_calls"]:
                raise RuntimeError("Synthesis attempted tools; refusing this training result.")
            content = response_file.read_text()
            json.loads(content)
            print(f"RuleChef model call {index} finished in {record['seconds']:.1f}s", flush=True)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )


class ExactRuleLearner(RuleLearner):
    """Keep upstream synthesis/refinement; correct its two text-only score paths."""

    def _evaluate_rules(self, rules: list[Rule], dataset: Dataset) -> EvalResult:
        return evaluate_dataset(rules, dataset, self._apply_rules, mode="exact")

    def _stamp_validated_stats(self, rules: list[Rule], dataset: Dataset) -> None:
        metrics = evaluate_rules_individually(rules, dataset, self._apply_rules, mode="exact")
        by_id = {metric.rule_id: metric for metric in metrics}
        for rule in rules:
            metric = by_id[rule.id]
            rule.validated_precision = metric.precision
            rule.validated_support = metric.true_positives + metric.false_positives


class RuleChefMethod(Method[RuleChefArtifact]):
    def __init__(self, config: RuleChefConfig | None = None, *, client: Any = None):
        self.config = config or RuleChefConfig()
        self.client = client
        self.executor = RuleExecutor(use_spacy_ner=False)

    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> RuleChefArtifact:
        if not train_examples:
            raise ValueError("RuleChef needs training examples.")
        if dev_examples is not None:
            raise ValueError("This pilot refines only on train; explicit dev is not implemented.")
        entity = create_model(
            "ClinicalEntity",
            text=(str, ...),
            start=(int, ...),
            end=(int, ...),
            type=(Literal[task_spec.entity_labels], ...),
        )
        output = create_model("ClinicalEntities", entities=(list.__class_getitem__(entity), ...))
        task = Task(
            name="Spanish clinical NER",
            type=TaskType.NER,
            description=(
                "Extract the annotated clinical entities from Spanish rheumatology notes. "
                "Preserve the original text and exact character boundaries. Values are separate "
                "entities from their test/index names. Emit only entities, no relations. "
                "Generalize from the supplied training annotations; do not memorize entire notes. "
                + (task_spec.instructions or "")
            ),
            input_schema={"text": "str"},
            output_schema=output,
            matching_mode="exact",
            text_field="text",
        )
        dataset = Dataset(name="clinical_train", task=task)
        for example in train_examples:
            entities = []
            for item in example.gold.result:
                if item["type"] == "labels":
                    value = item["value"]
                    if not 0 <= value["start"] < value["end"] <= len(example.input.text):
                        raise ValueError("Invalid training offsets.")
                    for label in value["labels"]:
                        if label not in task_spec.entity_labels:
                            raise ValueError(f"Unknown training label: {label}")
                        entities.append(
                            {
                                "text": example.input.text[value["start"] : value["end"]],
                                "start": value["start"],
                                "end": value["end"],
                                "type": label,
                            }
                        )
            dataset.examples.append(
                Example(
                    id=str(example.input.id),
                    input={"text": example.input.text},
                    expected_output={"entities": entities},
                    source="human",
                )
            )
        client = self.client
        if client is None:
            if self.config.backend == "codex":
                client = CodexClient(self.config)
            elif self.config.backend == "openai":
                from openai import OpenAI

                client = OpenAI(base_url=self.config.endpoint)
            else:
                raise ValueError(f"Unknown synthesis backend: {self.config.backend}")
        learner = ExactRuleLearner(
            client,
            model=self.config.model or getattr(client, "model", None) or "gpt-4o-mini",
            allowed_formats=[RuleFormat.REGEX],
            use_grex=False,
            use_spacy_ner=False,
            max_rules=self.config.max_rules,
            max_samples=len(train_examples),
        )
        started = time.perf_counter()
        rules = learner.synthesize_ruleset(dataset)
        if not rules:
            raise RuntimeError("RuleChef synthesized no valid rules; inspect the model traces.")
        if self.config.refinement_iterations:
            rules, _ = learner.evaluate_and_refine(
                rules,
                dataset,
                max_iterations=self.config.refinement_iterations,
                holdout_fraction=0.0,
                audit_interval=0,
            )
        else:
            learner._stamp_validated_stats(rules, dataset)
        artifact = RuleChefArtifact(
            task_spec,
            rules,
            {
                "rulechef_version": version("rulechef"),
                "config": asdict(self.config),
                "train_ids": [example.input.id for example in train_examples],
                "train_annotation_ids": [example.gold.id for example in train_examples],
                "train_text_sha256": [
                    hashlib.sha256(example.input.text.encode()).hexdigest()
                    for example in train_examples
                ],
                "dev_used": False,
                "test_used_during_fit": False,
                "selection": "upstream exact micro F1 on training data",
                "synthesis_strategy": "bulk",
                "refinement_seconds": time.perf_counter() - started,
                "span_conflicts": "native RuleChef: same start/end keeps one entity across labels",
                "llm_fallback": False,
                "model_calls": getattr(client, "calls", []),
            },
        )
        if scorer:
            artifact.provenance["training_metrics"] = asdict(
                scorer(
                    self.predict(artifact, [example.input for example in train_examples]),
                    train_examples,
                )
            )
        return artifact

    def dump(self, artifact: RuleChefArtifact, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "artifact.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "task_spec": asdict(artifact.task_spec),
                    "rules": [rule.to_dict() for rule in artifact.rules],
                    "provenance": artifact.provenance,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    def load(self, directory: Path) -> RuleChefArtifact:
        state = json.loads((directory / "artifact.json").read_text())
        if state["format_version"] != 1:
            raise ValueError("Unsupported RuleChef artifact version.")
        spec = state["task_spec"]
        spec["entity_labels"] = tuple(spec["entity_labels"])
        spec["relation_labels"] = tuple(spec["relation_labels"])
        rules = [Rule.from_dict(rule) for rule in state["rules"]]
        if any(rule.format != RuleFormat.REGEX for rule in rules):
            raise ValueError("This adapter supports regex artifacts only.")
        return RuleChefArtifact(TaskSpec(**spec), rules, state["provenance"])

    def predict(
        self,
        artifact: RuleChefArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        predictions = []
        for document in inputs:
            output = self.executor.apply_rules(
                artifact.rules,
                {"text": document.text},
                TaskType.NER,
                "text",
            )
            results: list[ResultItem] = []
            for entity in output.get("entities", []):
                start, end, label = entity.get("start"), entity.get("end"), entity.get("type")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(document.text)
                    or label not in artifact.task_spec.entity_labels
                    or entity.get("text") != document.text[start:end]
                ):
                    raise ValueError(f"Invalid RuleChef entity in document {document.id}: {entity}")
                results.append(
                    {
                        "id": f"rc_{len(results):04d}",
                        "type": "labels",
                        "from_name": artifact.task_spec.from_name,
                        "to_name": artifact.task_spec.to_name,
                        "value": {
                            "start": start,
                            "end": end,
                            "text": document.text[start:end],
                            "labels": [label],
                        },
                    }
                )
            predictions.append(prediction_for(document, results, "rulechef-0.2.2-regex"))
        return predictions
