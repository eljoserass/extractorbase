"""A frozen few-shot extraction recipe, served locally through LM Studio."""

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pydantic_ai
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

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
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "google/gemma-4-e2b"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_prompt_characters: int = 120_000
    retries: int = 1
    timeout: float = 120.0
    thinking: bool = False

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")) or not self.model:
            raise ValueError("LLM configuration requires a valid endpoint and model ID.")
        if self.max_output_tokens < 1 or self.max_prompt_characters < 1 or self.retries < 0:
            raise ValueError("Invalid LLM token limit, prompt limit, or retry count.")
        if self.timeout <= 0:
            raise ValueError("LLM request timeout must be positive.")


class EntityMention(BaseModel):
    id: str
    label: str
    text: str = Field(min_length=1)
    occurrence: int = Field(
        ge=0, description="Zero-based occurrence of this exact text in the document"
    )


class RelationMention(BaseModel):
    from_id: str
    to_id: str
    labels: list[str] = Field(min_length=1)


class Extraction(BaseModel):
    """An ephemeral model response; persisted outputs always use Label Studio results."""

    entities: list[EntityMention]
    relations: list[RelationMention]


class Demonstration(BaseModel):
    text: str
    extraction: Extraction


@dataclass(frozen=True)
class LLMArtifact:
    task_spec: TaskSpec
    config: LLMConfig
    prompt: str
    demonstrations: tuple[Demonstration, ...]


class LLMManifest(BaseModel):
    method: Literal["llm"] = "llm"
    task_spec: TaskSpec
    config: LLMConfig


class Demonstrations(BaseModel):
    examples: list[Demonstration]


def occurrences(text: str, quote: str) -> list[int]:
    starts: list[int] = []
    start = text.find(quote)
    while start >= 0:
        starts.append(start)
        start = text.find(quote, start + 1)
    return starts


def validate_structure(extraction: Extraction, spec: TaskSpec) -> None:
    ids = [entity.id for entity in extraction.entities]
    if len(ids) != len(set(ids)):
        raise ValueError("Entity IDs must be unique.")
    if any(entity.label not in spec.entity_labels for entity in extraction.entities):
        raise ValueError("Entity labels must belong to the schema.")
    for relation in extraction.relations:
        if relation.from_id not in ids or relation.to_id not in ids:
            raise ValueError("Relation endpoints must refer to extracted entity IDs.")
        if set(relation.labels) - set(spec.relation_labels):
            raise ValueError("Relation labels must belong to the schema.")


@dataclass(frozen=True)
class GroundedResults:
    results: list[ResultItem]
    warnings: list[str]


def ground_quotes(
    extraction: Extraction, document: DocumentInput, spec: TaskSpec
) -> GroundedResults:
    """Abstain on absent/ambiguous quotes; never invent character offsets."""
    results: list[ResultItem] = []
    ids: set[str] = set()
    warnings: list[str] = []
    id_counts = Counter(entity.id for entity in extraction.entities)
    for entity in extraction.entities:
        if id_counts[entity.id] > 1 or entity.label not in spec.entity_labels:
            warnings.append(
                f"Omitted {entity.id} ({entity.label}): duplicate ID or unknown entity label."
            )
            continue
        starts = occurrences(document.text, entity.text)
        occurrence = entity.occurrence
        if not starts:
            warnings.append(
                f"Omitted {entity.id} ({entity.label}): quote {entity.text!r} is absent "
                "from the document."
            )
            continue
        if occurrence >= len(starts):
            if len(starts) == 1:
                warnings.append(
                    f"Corrected {entity.id}: unique quote {entity.text!r} uses occurrence 0, "
                    f"not {occurrence}."
                )
                occurrence = 0
            else:
                warnings.append(
                    f"Omitted {entity.id}: quote {entity.text!r} requires an occurrence "
                    f"between 0 and {len(starts) - 1}, not {occurrence}."
                )
                continue
        start = starts[occurrence]
        ids.add(entity.id)
        results.append(
            {
                "id": entity.id,
                "type": "labels",
                "from_name": spec.from_name,
                "to_name": spec.to_name,
                "value": {
                    "start": start,
                    "end": start + len(entity.text),
                    "text": entity.text,
                    "labels": [entity.label],
                },
            }
        )
    for relation in extraction.relations:
        if relation.from_id not in ids or relation.to_id not in ids:
            warnings.append(
                f"Omitted relation {relation.from_id}->{relation.to_id}: "
                "endpoint is missing or was omitted."
            )
            continue
        if set(relation.labels) - set(spec.relation_labels):
            warnings.append(
                f"Omitted relation {relation.from_id}->{relation.to_id}: unknown relation label."
            )
            continue
        results.append(
            {
                "type": "relation",
                "from_id": relation.from_id,
                "to_id": relation.to_id,
                "labels": relation.labels,
                "direction": "right",
            }
        )
    return GroundedResults(results, warnings)


def to_results(extraction: Extraction, document: DocumentInput, spec: TaskSpec) -> list[ResultItem]:
    """Strict conversion for callers that require every mention to resolve unchanged."""
    validate_structure(extraction, spec)
    grounded = ground_quotes(extraction, document, spec)
    if grounded.warnings:
        raise ValueError("\n".join(grounded.warnings))
    return grounded.results


def demonstration(example: TrainingExample) -> Demonstration:
    entities: list[EntityMention] = []
    ids: dict[str, str] = {}
    for item in example.gold.result:
        if item["type"] != "labels":
            continue
        value = item["value"]
        quote = example.input.text[value["start"] : value["end"]]
        if not quote:
            raise ValueError("LLM demonstrations require nonempty entity spans.")
        ids[item["id"]] = f"e{len(entities)}"
        for label in value["labels"]:
            entities.append(
                EntityMention(
                    id=f"e{len(entities)}",
                    label=label,
                    text=quote,
                    occurrence=occurrences(example.input.text, quote).index(value["start"]),
                )
            )
    relations = [
        RelationMention(
            from_id=ids[item["from_id"]], to_id=ids[item["to_id"]], labels=item.get("labels", [])
        )
        for item in example.gold.result
        if item["type"] == "relation" and item.get("labels")
    ]
    return Demonstration(
        text=example.input.text, extraction=Extraction(entities=entities, relations=relations)
    )


class LLMMethod(Method[LLMArtifact]):
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> LLMArtifact:
        demos = tuple(demonstration(example) for example in train_examples)
        prompt = (
            "Extrae entidades y relaciones de textos clinicos en espanol siguiendo los ejemplos. "
            "No inventes datos ni traduzcas o normalices las citas. Cada entidad necesita un id "
            "unico, una etiqueta, text copiado literalmente del documento, y occurrence: indice "
            "desde cero de esa cita exacta entre todas sus apariciones, contando de izquierda a "
            "derecha (incluidas apariciones dentro de palabras). Si aparece una vez, "
            "occurrence=0. Respeta mayusculas, tildes, espacios y abreviaturas del documento: "
            "no expandas 'med' a 'medico'. Omite cualquier entidad cuya cita literal no exista. "
            "No calcules offsets. "
            "Las relaciones referencian los ids de las entidades extraidas. Usa listas vacias "
            "si no hay entidades o relaciones. Las entidades pueden solaparse.\n"
            f"Etiquetas de entidades: {json.dumps(task_spec.entity_labels)}\n"
            f"Etiquetas de relaciones: {json.dumps(task_spec.relation_labels)}\n"
            "Los ejemplos omiten relaciones sin etiqueta; no infieras su tipo.\n"
        )
        if task_spec.instructions:
            prompt += task_spec.instructions + "\n"
        for index, demo in enumerate(demos):
            prompt += f"\nEJEMPLO {index + 1}\nDOCUMENTO:\n{demo.text}\nSALIDA:\n"
            prompt += demo.extraction.model_dump_json() + "\n"
        if len(prompt) > self.config.max_prompt_characters:
            raise ValueError(
                "Few-shot prompt exceeds the configured character budget; use fewer examples."
            )
        return LLMArtifact(task_spec, self.config, prompt, demos)

    def dump(self, artifact: LLMArtifact, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = LLMManifest(task_spec=artifact.task_spec, config=artifact.config)
        (directory / "method.json").write_text(manifest.model_dump_json(indent=2))
        (directory / "prompt.txt").write_text(artifact.prompt)
        (directory / "examples.json").write_text(
            Demonstrations(examples=list(artifact.demonstrations)).model_dump_json(indent=2)
        )

    def load(self, directory: Path) -> LLMArtifact:
        manifest = LLMManifest.model_validate_json((directory / "method.json").read_text())
        demos = Demonstrations.model_validate_json((directory / "examples.json").read_text())
        return LLMArtifact(
            manifest.task_spec,
            manifest.config,
            (directory / "prompt.txt").read_text(),
            tuple(demos.examples),
        )

    def _agent(self, artifact: LLMArtifact) -> Agent[DocumentInput, Extraction]:
        pydantic_ai.BANNER_ENABLED = False
        model = OpenAIChatModel(
            artifact.config.model,
            provider=OpenAIProvider(
                openai_client=AsyncOpenAI(
                    base_url=artifact.config.base_url, api_key="lm-studio", max_retries=0
                )
            ),
            profile=OpenAIModelProfile(
                supports_json_schema_output=True,
                openai_chat_supports_max_completion_tokens=False,
                openai_chat_supports_multiple_system_messages=False,
            ),
        )
        settings = OpenAIChatModelSettings(
            temperature=artifact.config.temperature,
            max_tokens=artifact.config.max_output_tokens,
            timeout=artifact.config.timeout,
        )
        if not artifact.config.thinking:
            settings["openai_reasoning_effort"] = "none"
        return Agent(
            model,
            output_type=NativeOutput(Extraction),
            deps_type=DocumentInput,
            instructions=artifact.prompt,
            retries=artifact.config.retries,
            model_settings=settings,
        )

    def predict(
        self,
        artifact: LLMArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        agent = self._agent(artifact)
        predictions: list[LabelStudioTaskPrediction] = []
        for document in inputs:
            logging.getLogger(__name__).info("Extracting document %s with Gemma...", document.id)
            try:
                if (
                    len(artifact.prompt) + len(document.text)
                    > artifact.config.max_prompt_characters
                ):
                    raise ValueError(
                        "Prompt plus document exceeds the configured character budget."
                    )
                output = agent.run_sync(
                    "DOCUMENTO A ANOTAR (solo este texto; los ejemplos son de referencia):\n"
                    + document.text,
                    deps=document,
                ).output
                grounded = ground_quotes(output, document, artifact.task_spec)
                prediction = prediction_for(document, grounded.results, artifact.config.model)
                if grounded.warnings:
                    prediction["meta"] = {"extraction_warnings": [*grounded.warnings]}
            except (APIError, ModelHTTPError, UnexpectedModelBehavior, ValueError) as error:
                # Keep every document in the denominator; the CLI exits nonzero after reporting.
                cause: BaseException = error
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                detail = f"{error}: {cause}" if cause is not error else str(error)
                prediction = prediction_for(document, [], artifact.config.model)
                prediction["meta"] = {"extraction_error": detail}
                logging.getLogger(__name__).warning("Document %s failed: %s", document.id, detail)
            predictions.append(prediction)
            logging.getLogger(__name__).info(
                "Document %s: %s result items%s",
                document.id,
                len(prediction["predictions"][0]["result"]),
                " (failed; see saved error)"
                if prediction.get("meta", {}).get("extraction_error")
                else "",
            )
        return predictions
