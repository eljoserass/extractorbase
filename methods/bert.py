"""A small CPU token-classification baseline; BIO is an internal adapter only."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

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
class BertConfig:
    checkpoint: str = "distilbert/distilbert-base-multilingual-cased"
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 5e-4
    max_length: int = 256
    stride: int = 64
    seed: int = 42
    freeze_encoder: bool = True
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("BERT epochs, batch size, and learning rate must be positive.")
        if not 0 <= self.stride < self.max_length - 2 or self.cpu_threads < 1:
            raise ValueError("Invalid BERT stride, max length, or CPU thread count.")


@dataclass
class BertArtifact:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    task_spec: TaskSpec
    config: BertConfig
    fit_notes: list[str] = field(default_factory=list)


class BertManifest(BaseModel):
    method: str = "bert"
    task_spec: TaskSpec
    config: BertConfig
    fit_notes: list[str]


class TokenDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, windows: list[dict[str, list[int]]]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.windows[index]


def align_labels(
    example: TrainingExample,
    offsets: Sequence[tuple[int, int]],
    label_to_id: dict[str, int],
) -> tuple[list[int], list[str]]:
    """Use longest-first projection for overlapping training spans; never change gold."""
    labels = [-100 if start == end else 0 for start, end in offsets]
    occupied: set[int] = set()
    notes: list[str] = []
    entities = [item for item in example.gold.result if item["type"] == "labels"]
    entities.sort(
        key=lambda item: (
            -(item["value"]["end"] - item["value"]["start"]),
            item["value"]["start"],
            item["id"],
        )
    )
    visible = [(start, end) for start, end in offsets if start < end]
    if not visible:
        return labels, notes
    for entity in entities:
        value = entity["value"]
        indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if start < end and start < value["end"] and end > value["start"]
        ]
        if not indices:
            continue
        if value["start"] < visible[0][0] or value["end"] > visible[-1][1]:
            for i in indices:
                if i not in occupied:
                    labels[i] = -100  # Don't train window-edge fragments as negative examples.
            continue
        if occupied.intersection(indices):
            notes.append(
                f"Task {example.input.id}: BIO cannot retain overlapping entity {entity['id']}."
            )
            continue
        label = value["labels"][0]
        if len(value["labels"]) > 1:
            notes.append(f"Task {example.input.id}: BIO uses the first label of {entity['id']}.")
        if offsets[indices[0]][0] != value["start"] or offsets[indices[-1]][1] != value["end"]:
            notes.append(
                f"Task {example.input.id}: token boundaries project entity {entity['id']}."
            )
        for position, index in enumerate(indices):
            labels[index] = label_to_id[f"{'B' if position == 0 else 'I'}-{label}"]
        occupied.update(indices)
    return labels, notes


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    label: str
    confidence: float


def decode_window(
    offsets: Sequence[tuple[int, int]],
    tags: Sequence[str],
    confidences: Sequence[float],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    start, end, label = 0, 0, ""
    scores: list[float] = []

    def close() -> None:
        if label:
            candidates.append(Candidate(start, end, label, sum(scores) / len(scores)))

    for (token_start, token_end), tag, confidence in zip(offsets, tags, confidences, strict=True):
        if token_start == token_end or tag == "O":
            close()
            label, scores = "", []
            continue
        prefix, token_label = tag.split("-", 1)
        if prefix == "B" or token_label != label:
            close()
            start, label, scores = token_start, token_label, []
        end = token_end
        scores.append(confidence)
    close()
    return candidates


class BertMethod(Method[BertArtifact]):
    def __init__(self, config: BertConfig | None = None) -> None:
        self.config = config or BertConfig()

    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> BertArtifact:
        if not train_examples:
            raise ValueError("BERT fitting requires training examples.")
        config = self.config
        torch.set_num_threads(config.cpu_threads)
        set_seed(config.seed)
        tags = [
            "O",
            *[f"{prefix}-{label}" for label in task_spec.entity_labels for prefix in ("B", "I")],
        ]
        label_to_id = {label: index for index, label in enumerate(tags)}
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=True)
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError("BERT needs a fast tokenizer to preserve character offsets.")
        if config.max_length > tokenizer.model_max_length:
            raise ValueError("BERT max length exceeds this tokenizer's supported context.")
        model = AutoModelForTokenClassification.from_pretrained(
            config.checkpoint,
            num_labels=len(tags),
            id2label=dict(enumerate(tags)),
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
        if config.freeze_encoder:
            for parameter in model.base_model.parameters():
                parameter.requires_grad = False
        windows: list[dict[str, list[int]]] = []
        notes: set[str] = set()
        for example in train_examples:
            encoded = tokenizer(
                example.input.text,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=config.max_length,
                stride=config.stride,
            )
            for ids, mask, offsets in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["offset_mapping"],
                strict=True,
            ):
                labels, window_notes = align_labels(example, offsets, label_to_id)
                windows.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
                notes.update(window_notes)
        with TemporaryDirectory(prefix="extractorbase-bert-") as output:
            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=TokenDataset(windows),
                data_collator=DataCollatorForTokenClassification(tokenizer),
                args=TrainingArguments(
                    output_dir=output,
                    num_train_epochs=config.epochs,
                    per_device_train_batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    use_cpu=True,
                    seed=config.seed,
                    save_strategy="no",
                    report_to="none",
                    logging_strategy="epoch",
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                    optim="adamw_torch",
                ),
            )
            trainer.train()
        model.eval()
        return BertArtifact(model, tokenizer, task_spec, config, sorted(notes))

    def dump(self, artifact: BertArtifact, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        artifact.model.save_pretrained(directory / "model", safe_serialization=True)
        artifact.tokenizer.save_pretrained(directory / "model")
        manifest = BertManifest(
            task_spec=artifact.task_spec, config=artifact.config, fit_notes=artifact.fit_notes
        )
        (directory / "method.json").write_text(manifest.model_dump_json(indent=2))

    def load(self, directory: Path) -> BertArtifact:
        manifest = BertManifest.model_validate_json((directory / "method.json").read_text())
        model = AutoModelForTokenClassification.from_pretrained(
            directory / "model", local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(directory / "model", local_files_only=True)
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError("Saved BERT tokenizer is not a fast tokenizer.")
        model.eval()
        return BertArtifact(
            model, tokenizer, manifest.task_spec, manifest.config, manifest.fit_notes
        )

    def predict(
        self,
        artifact: BertArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        torch.set_num_threads(artifact.config.cpu_threads)
        artifact.model.cpu().eval()
        predictions: list[LabelStudioTaskPrediction] = []
        for document in inputs:
            encoded = artifact.tokenizer(
                document.text,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=artifact.config.max_length,
                stride=artifact.config.stride,
            )
            candidates: list[Candidate] = []
            for ids, mask, offsets in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["offset_mapping"],
                strict=True,
            ):
                with torch.inference_mode():
                    logits = artifact.model(
                        input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask])
                    ).logits[0]
                    probabilities, indices = logits.softmax(dim=-1).max(dim=-1)
                mapping = artifact.model.config.id2label
                if mapping is None:
                    raise ValueError("BERT artifact has no saved label mapping.")
                id_to_label = {int(index): label for index, label in mapping.items()}
                tags = [id_to_label[index] for index in indices.tolist()]
                candidates.extend(decode_window(offsets, tags, probabilities.tolist()))
            retained: list[Candidate] = []
            for candidate in sorted(
                candidates, key=lambda c: (-c.confidence, c.start, c.end, c.label)
            ):
                if not any(
                    candidate.start < other.end and candidate.end > other.start
                    for other in retained
                ):
                    retained.append(candidate)
            results: list[ResultItem] = [
                {
                    "id": f"e{index}",
                    "type": "labels",
                    "from_name": artifact.task_spec.from_name,
                    "to_name": artifact.task_spec.to_name,
                    "value": {
                        "start": item.start,
                        "end": item.end,
                        "text": document.text[item.start : item.end],
                        "labels": [item.label],
                    },
                }
                for index, item in enumerate(
                    sorted(retained, key=lambda c: (c.start, c.end, c.label))
                )
            ]
            predictions.append(prediction_for(document, results, "bert"))
        return predictions
