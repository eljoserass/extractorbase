"""Self-contained RuleChef NER adapter; LLM access is needed only for fit."""

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import create_model
from rulechef import executor as upstream_executor
from rulechef.coordinator import AgenticCoordinator
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
from methods.rulechef_client import CodexClient


@dataclass(frozen=True)
class RuleChefConfig:
    backend: str = "codex"
    model: str | None = None  # None uses the user's configured Codex model.
    endpoint: str | None = None
    max_rules: int = 64
    refinement_iterations: int = 3
    call_timeout_seconds: int = 1200
    trace_directory: str | None = None
    rule_formats: tuple[str, ...] = ("regex",)
    spacy_model: str | None = None
    codex_isolation: str = "bubblewrap"
    reasoning_effort: str | None = None
    synthesis_strategy: str = "bulk"
    max_rules_per_class: int = 5
    agentic: bool = False
    learning_instructions: str = ""


@dataclass
class RuleChefArtifact:
    task_spec: TaskSpec
    rules: list[Rule]
    provenance: dict[str, Any] = field(default_factory=dict)
    spacy_model: str | None = None


class SpanishPipeline:
    """Reuse the annotated document across rules without changing upstream code."""

    def __init__(self, model: str):
        import spacy

        self.language = spacy.blank("es") if model == "blank:es" else spacy.load(model)
        if self.language.lang != "es":
            raise ValueError("RuleChef's Spanish variant requires an es pipeline.")
        self.vocab = self.language.vocab
        self._parse = lru_cache(maxsize=16)(self.language.__call__)

    def disable_pipes(self, *names: str) -> Any:
        return self.language.select_pipes(
            disable=[name for name in names if name in self.language.pipe_names]
        )

    def __call__(self, text: str) -> Any:
        return self._parse(text)


class CaptureSpanExecutor(RuleExecutor):
    """Resolve lookaround captures from Match.span, not an empty full match.

    RuleChef 0.2.2 searches for captured text inside match.group(0). That cannot
    recover a span captured by a zero-width lookaround. Preserve normal native
    results and explicit offsets; only repair this provable capture case.
    """

    def _execute_regex_rule(
        self,
        rule: Rule,
        input_data: dict,
        text_field: str | None = None,
    ) -> Any:
        results = super()._execute_regex_rule(rule, input_data, text_field)
        template = rule.output_template or {}
        capture = re.fullmatch(r"\$(\d+)", str(template.get("text", "")))
        if (
            not capture
            or template.get("start") != "$start"
            or template.get("end") != "$end"
            or not any(
                isinstance(item, dict) and item.get("start") == item.get("end") for item in results
            )
            or upstream_executor._regex_mod is None
        ):
            return results
        text = self._select_text(input_data, text_field)
        try:
            pattern = upstream_executor._regex_mod.compile(rule.content)
            matches = list(pattern.finditer(text, timeout=upstream_executor.RULE_TIMEOUT_S))
        except (TimeoutError, upstream_executor._regex_mod.error):
            return results
        for item, match in zip(results, matches, strict=True):
            if match.start() != match.end():
                continue
            start, end = match.span(int(capture.group(1)))
            if start >= 0 and end > start and item.get("text") == text[start:end]:
                item.update(start=start, end=end)
        return results


def _executor(model: str | None) -> RuleExecutor:
    executor = CaptureSpanExecutor(use_spacy_ner=False)
    if model:
        # Upstream otherwise lazily loads/downloads en_core_web_sm. Preloading
        # the per-instance pipeline keeps both fitting and inference in Spanish.
        cast(Any, executor)._nlp = SpanishPipeline(model)
    return executor


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
        if not self.config.rule_formats or set(self.config.rule_formats) - {"regex", "spacy"}:
            raise ValueError("Supported rule formats are regex and spacy.")
        if "spacy" in self.config.rule_formats and not self.config.spacy_model:
            raise ValueError("Select a Spanish spacy_model when enabling spaCy rules.")
        if self.config.synthesis_strategy not in {"bulk", "per_class"}:
            raise ValueError("synthesis_strategy must be bulk or per_class.")
        if self.config.refinement_iterations < 0:
            raise ValueError("refinement_iterations must be nonnegative.")
        self.client = client
        self.executor = _executor(self.config.spacy_model)

    def _task(self, spec: TaskSpec) -> Task:
        entity = create_model(
            "Entity",
            text=(str, ...),
            start=(int, ...),
            end=(int, ...),
            type=(Literal[spec.entity_labels], ...),
        )
        entity_list = list[entity]  # type: ignore[valid-type]
        output = create_model("Entities", entities=(entity_list, ...))
        description = (
            "Extract the annotated entities with exact character offsets in the original text. "
            "Emit only entities, no relations. Generalize from the training annotations. "
            + (spec.instructions or "")
            + "\n"
            + self.config.learning_instructions
        )
        if "spacy" in self.config.rule_formats:
            description += (
                f"\nspaCy runtime: {self.config.spacy_model}. Statistical NER is disabled. "
                "Token patterns support LOWER/TEXT, IN, REGEX, OP, FUZZY and FUZZY1. "
                'For example [{"LOWER": {"FUZZY1": "example"}}] allows one edit; '
                "consider fuzzy matching for spelling variants, but short patterns can be noisy. "
                "Combine regex and spaCy when useful; regex handles sub-token boundaries. "
                "spaCy content must be a JSON array of token dictionaries. "
                "Templates use $0/$start/$end for a whole match or $1.text/$1.start/$1.end "
                "for the first token, $2 for the second, etc. Context tokens must not be "
                "included in the extracted span unless annotated. "
            )
            if self.config.spacy_model == "blank:es":
                description += (
                    "This runtime has only a tokenizer: do not use POS, LEMMA, MORPH or DEP."
                )
            else:
                description += (
                    "The trained Spanish pipeline also provides POS, LEMMA, MORPH and DEP."
                )
        return Task(
            name="Entity extraction",
            type=TaskType.NER,
            description=description,
            input_schema={"text": "str"},
            output_schema=output,
            matching_mode="exact",
            text_field="text",
        )

    @staticmethod
    def _dataset(
        examples: Sequence[TrainingExample],
        task: Task,
        labels: Sequence[str],
        name: str,
    ) -> Dataset:
        dataset = Dataset(name=name, task=task)
        for example in examples:
            entities = []
            for item in example.gold.result:
                if item["type"] != "labels":
                    continue
                value = item["value"]
                if not 0 <= value["start"] < value["end"] <= len(example.input.text):
                    raise ValueError("Invalid training offsets.")
                for label in value["labels"]:
                    if label not in labels:
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
        return dataset

    def _learner(self, train_size: int) -> tuple[ExactRuleLearner, Any]:
        client = self.client
        if client is None:
            if self.config.backend == "codex":
                client = CodexClient(self.config)
            elif self.config.backend == "openai":
                from openai import OpenAI

                from methods.rulechef_client import RecordedClient

                client = RecordedClient(OpenAI(base_url=self.config.endpoint), self.config)
            else:
                raise ValueError(f"Unknown synthesis backend: {self.config.backend}")
        model = self.config.model or getattr(client, "model", None)
        if not model:
            raise ValueError("Choose an explicit synthesis model for this backend.")
        learner = ExactRuleLearner(
            client,
            model=model,
            allowed_formats=[RuleFormat(value) for value in self.config.rule_formats],
            use_grex=False,
            use_spacy_ner=False,
            max_rules=self.config.max_rules,
            max_rules_per_class=self.config.max_rules_per_class,
            max_samples=train_size,
        )
        learner.executor = self.executor
        return learner, client

    @staticmethod
    def _check_separation(
        train: Sequence[TrainingExample],
        dev: Sequence[TrainingExample],
    ) -> None:
        train_ids = {str(example.input.id) for example in train}
        train_texts = {" ".join(example.input.text.split()).casefold() for example in train}
        if train_ids & {str(example.input.id) for example in dev}:
            raise ValueError("Train and dev IDs overlap.")
        if train_texts & {" ".join(example.input.text.split()).casefold() for example in dev}:
            raise ValueError("Train and dev contain duplicate texts.")

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
        dev = list(dev_examples or [])
        self._check_separation(train_examples, dev)
        task = self._task(task_spec)
        dataset = self._dataset(train_examples, task, task_spec.entity_labels, "train")
        learner, client = self._learner(len(train_examples))
        started = time.perf_counter()
        if self.config.synthesis_strategy == "per_class":
            rules = learner.synthesize_ruleset_per_class(dataset)
        else:
            rules = learner.synthesize_ruleset(dataset)
        if getattr(client, "failure", None):
            raise RuntimeError(f"Synthesis completion failed: {client.failure}")
        if not rules:
            raise RuntimeError("RuleChef synthesized no valid rules; inspect the model traces.")
        provenance: dict[str, Any] = {
            "rulechef_version": version("rulechef"),
            "runtime": "rulechef-0.2.2+zero-width-capture-spans",
            "config": asdict(self.config),
            "train_ids": [example.input.id for example in train_examples],
            "train_annotation_ids": [example.gold.id for example in train_examples],
            "train_text_sha256": [
                hashlib.sha256(e.input.text.encode()).hexdigest() for e in train_examples
            ],
            "dev_ids": [example.input.id for example in dev],
            "dev_used": bool(dev),
            "test_used_during_fit": False,
            "selection": "exact micro F1 on dev" if dev else "exact micro F1 on train",
            "synthesis_strategy": self.config.synthesis_strategy,
            "model_calls": deepcopy(getattr(client, "calls", [])),
            "synthesis_seconds": time.perf_counter() - started,
            "llm_fallback": False,
            "prediction_model_version": "rulechef",
            "span_conflicts": "native RuleChef: same start/end keeps one entity across labels",
        }
        artifact = RuleChefArtifact(task_spec, rules, provenance, self.config.spacy_model)
        return self._finish(artifact, learner, client, train_examples, dev, scorer)

    def refine(
        self,
        artifact: RuleChefArtifact,
        train_examples: Sequence[TrainingExample],
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> RuleChefArtifact:
        """Refine an immutable initial artifact; useful for paired ablations and feedback."""
        dev = list(dev_examples or [])
        self._check_separation(train_examples, dev)
        if [e.input.id for e in train_examples] != artifact.provenance["train_ids"]:
            raise ValueError("Refinement train IDs differ from the initial synthesis examples.")
        if self.config.rule_formats != tuple(artifact.provenance["config"]["rule_formats"]):
            raise ValueError("Refinement must use the initial artifact's rule formats.")
        if self.config.spacy_model != artifact.provenance["config"].get("spacy_model"):
            raise ValueError("Refinement must use the initial artifact's spaCy runtime.")
        learner, client = self._learner(len(train_examples))
        return self._finish(
            deepcopy(artifact),
            learner,
            client,
            train_examples,
            dev,
            scorer,
            prior_calls=artifact.provenance.get("model_calls", []),
        )

    def _finish(
        self,
        artifact: RuleChefArtifact,
        learner: ExactRuleLearner,
        client: Any,
        train_examples: Sequence[TrainingExample],
        dev_examples: Sequence[TrainingExample],
        scorer: Scorer | None,
        prior_calls: list[dict[str, Any]] | None = None,
    ) -> RuleChefArtifact:
        from methods.rulechef_learning import refine_on_dev

        task = self._task(artifact.task_spec)
        train = self._dataset(train_examples, task, artifact.task_spec.entity_labels, "train")
        dev = (
            self._dataset(dev_examples, task, artifact.task_spec.entity_labels, "dev")
            if dev_examples
            else None
        )
        started = time.perf_counter()
        coordinator = None
        if self.config.agentic:
            coordinator = AgenticCoordinator(
                client,
                model=learner.model,
                prune_after_learn=True,
                enable_critic=True,
                critic_interval=2,
                audit_interval=3,
            )
        history: list[dict[str, Any]] = []
        if dev is not None:
            artifact.rules, history = refine_on_dev(
                learner,
                artifact.rules,
                train,
                dev,
                self.config.refinement_iterations,
                coordinator,
            )
        elif self.config.refinement_iterations:
            artifact.rules, _ = learner.evaluate_and_refine(
                artifact.rules,
                train,
                max_iterations=self.config.refinement_iterations,
                coordinator=coordinator,
                holdout_fraction=0.0,
                audit_interval=0,
            )
        else:
            learner._stamp_validated_stats(artifact.rules, train)
        if getattr(client, "failure", None):
            raise RuntimeError(f"Refinement completion failed: {client.failure}")
        new_calls = deepcopy(prior_calls or []) + deepcopy(getattr(client, "calls", []))
        uses_spacy = any(rule.format == RuleFormat.SPACY for rule in artifact.rules)
        pipeline = cast(Any, self.executor._nlp)
        artifact.provenance.update(
            {
                "config": asdict(self.config),
                "dev_ids": [example.input.id for example in dev_examples],
                "dev_used": dev is not None,
                "selection": "exact micro F1 on dev"
                if dev is not None
                else "exact micro F1 on train",
                "refinement_seconds": time.perf_counter() - started,
                "refinement_history": history,
                "model_calls": new_calls,
                "rule_formats": dict(Counter(rule.format.value for rule in artifact.rules)),
                "spacy_pipeline": {
                    "name": self.config.spacy_model,
                    "version": pipeline.language.meta.get("version"),
                    "language": pipeline.language.lang,
                    "components": pipeline.language.pipe_names,
                    "ner_enabled": False,
                    "document_cache_size": 16,
                }
                if self.config.spacy_model
                else None,
                "inference_spacy_required": uses_spacy,
            }
        )
        if not uses_spacy:
            artifact.spacy_model = None
        if scorer:
            artifact.provenance["training_metrics"] = asdict(
                scorer(
                    self.predict(artifact, [example.input for example in train_examples]),
                    train_examples,
                )
            )
            if dev_examples:
                artifact.provenance["dev_metrics"] = asdict(
                    scorer(
                        self.predict(artifact, [example.input for example in dev_examples]),
                        dev_examples,
                    )
                )
        return artifact

    def dump(self, artifact: RuleChefArtifact, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "artifact.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "task_spec": asdict(artifact.task_spec),
                    "rules": [rule.to_dict() for rule in artifact.rules],
                    "provenance": artifact.provenance,
                    "spacy_model": artifact.spacy_model,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    def load(self, directory: Path) -> RuleChefArtifact:
        state = json.loads((directory / "artifact.json").read_text())
        if state["format_version"] not in {1, 2}:
            raise ValueError("Unsupported RuleChef artifact version.")
        spec = state["task_spec"]
        spec["entity_labels"] = tuple(spec["entity_labels"])
        spec["relation_labels"] = tuple(spec["relation_labels"])
        rules = [Rule.from_dict(rule) for rule in state["rules"]]
        if any(rule.format not in {RuleFormat.REGEX, RuleFormat.SPACY} for rule in rules):
            raise ValueError("This adapter supports regex and spaCy artifacts only.")
        model = state.get("spacy_model")
        if any(rule.format == RuleFormat.SPACY for rule in rules) and not model:
            raise ValueError("spaCy artifacts must record their Spanish pipeline.")
        self.executor = _executor(model)
        expected = state["provenance"].get("spacy_pipeline")
        if (
            expected
            and model
            and cast(Any, self.executor._nlp).language.meta.get("version") != expected["version"]
        ):
            raise ValueError("Installed spaCy model version differs from the saved artifact.")
        return RuleChefArtifact(TaskSpec(**spec), rules, state["provenance"], model)

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
            model_version = (
                "rulechef-0.2.2-spacy-es" if artifact.spacy_model else "rulechef-0.2.2-regex"
            )
            model_version = artifact.provenance.get("prediction_model_version", model_version)
            predictions.append(prediction_for(document, results, model_version))
        return predictions
