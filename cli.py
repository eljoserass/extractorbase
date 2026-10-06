"""CLI orchestration. Models never receive held-out gold; the runner owns evaluation."""

import hashlib
import importlib.metadata
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import cast

import typer
from pydantic import BaseModel, Field, TypeAdapter
from rich.console import Console
from rich.table import Table

from data import (
    DocumentId,
    GoldPolicy,
    LabelStudioTask,
    LabelStudioTaskPrediction,
    TaskSpec,
    annotation_issues,
    inspect_dataset,
    load_labelstudio,
    make_example,
    make_input,
    select_gold,
)
from evaluator import MetricsReport, score
from methods.base import Method

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()
DEFAULT_SCHEMA = Path(__file__).parent / "schemas" / "reumalago.xml"


class Stage(StrEnum):
    INSPECT = "inspect"
    FIT = "fit"
    PREDICT = "predict"
    RUN = "run"


class MethodName(StrEnum):
    DUMMY = "dummy"
    BERT = "bert"
    LLM = "llm"


class FitRecord(BaseModel):
    method: MethodName
    train_ids: list[DocumentId]
    annotation_ids: list[int]
    train_text_hashes: list[str]
    sources: dict[str, str]
    schema_sha256: str
    gold_policy: GoldPolicy
    reviewer_ids: list[int]
    created_at: str
    packages: dict[str, str]
    approved: bool = False


class MethodHeader(BaseModel):
    method: MethodName


class PredictionRecord(BaseModel):
    stage: Stage
    method: MethodName
    artifact: str
    evaluation_ids: list[DocumentId]
    annotation_ids: list[int | None]
    sources: dict[str, str]
    gold_policy: GoldPolicy
    reviewer_ids: list[int]
    created_at: str
    packages: dict[str, str]
    train_ids: list[DocumentId] = Field(default_factory=list)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("nervaluate", "pydantic-ai-slim", "typer", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select_slice(tasks: list[LabelStudioTask], offset: int, count: int) -> list[LabelStudioTask]:
    if offset < 0 or count < 1 or offset + count > len(tasks):
        raise ValueError(f"Requested [{offset}:{offset + count}], but data has {len(tasks)} tasks.")
    return tasks[offset : offset + count]


def build_method(
    name: MethodName,
    epochs: int,
    checkpoint: str,
    freeze_encoder: bool,
    endpoint: str,
    model: str,
    seed: int,
    thinking: bool,
) -> Method[object]:
    # This is the one dispatch boundary. Each implementation keeps a concrete artifact type.
    if name == MethodName.DUMMY:
        from methods.dummy import DummyMethod

        return cast(Method[object], DummyMethod())
    if name == MethodName.BERT:
        from methods.bert import BertConfig, BertMethod

        return cast(
            Method[object],
            BertMethod(
                BertConfig(
                    checkpoint=checkpoint,
                    epochs=epochs,
                    freeze_encoder=freeze_encoder,
                    learning_rate=5e-4 if freeze_encoder else 5e-5,
                    seed=seed,
                )
            ),
        )
    from methods.llm import LLMConfig, LLMMethod

    logger = logging.getLogger("methods.llm")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    return cast(
        Method[object], LLMMethod(LLMConfig(base_url=endpoint, model=model, thinking=thinking))
    )


def print_results(
    predictions: Sequence[LabelStudioTaskPrediction],
    report: MetricsReport | None,
    title: str,
) -> None:
    console.print(title, style="bold")
    table = Table("Document", "Entities", "Relations", "First extracted entities / error")
    for prediction in predictions:
        results = prediction["predictions"][0]["result"]
        entities = [item for item in results if item["type"] == "labels"]
        relations = [item for item in results if item["type"] == "relation"]
        preview = "; ".join(
            f"{','.join(item['value']['labels'])}: {item['value']['text']}" for item in entities[:8]
        )
        error = prediction.get("meta", {}).get("extraction_error")
        table.add_row(
            str(prediction["id"]),
            str(len(entities)),
            str(len(relations)),
            str(error) if error else preview,
        )
    console.print(table)
    for prediction in predictions:
        warnings = prediction.get("meta", {}).get("extraction_warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                console.print(f"Task {prediction['id']}: {warning}", style="yellow")
    if report is not None:
        metrics = report.strict_micro
        console.print(
            f"Strict entity micro: precision={metrics.precision:.4f} "
            f"recall={metrics.recall:.4f} F1={metrics.f1:.4f} support={metrics.support}"
        )
        labels = Table("Label", "Precision", "Recall", "F1", "Support")
        for label, values in report.per_label.items():
            labels.add_row(
                label,
                f"{values.precision:.3f}",
                f"{values.recall:.3f}",
                f"{values.f1:.3f}",
                str(values.support),
            )
        console.print(labels)
        console.print("Relation metrics: not computed in this milestone.")
    else:
        console.print("No gold annotations supplied; predictions saved without metrics.")


def warn_selected(
    tasks: list[LabelStudioTask], spec: TaskSpec, policy: GoldPolicy, reviewer_ids: tuple[int, ...]
) -> None:
    for task in tasks:
        selection = select_gold(task, policy, reviewer_ids)
        for warning in selection.warnings:
            console.print(f"Task {task.id}: {warning}", style="yellow")
        for issue in annotation_issues(task, selection.annotation, spec):
            console.print(f"Task {task.id}: {issue.message}", style="yellow")


def approve(directory: Path, record: FitRecord, yes: bool) -> None:
    if not yes:
        typer.confirm("Approve this fitted artifact for prediction?", abort=True)
    record.approved = True
    (directory / "fit.json").write_text(record.model_dump_json(indent=2))


@app.command()
def main(
    data: list[Path] = typer.Option(..., "--data", exists=True, dir_okay=False),
    stage: Stage = typer.Option(Stage.RUN, "--stage"),
    method: MethodName | None = typer.Option(None, "--method"),
    dump: Path | None = typer.Option(None, "--dump"),
    load: Path | None = typer.Option(None, "--load", exists=True, file_okay=False),
    output: Path | None = typer.Option(None, "--output"),
    offset: int | None = typer.Option(None, "--offset"),
    count: int = typer.Option(5, "--count", min=1),
    predict_count: int = typer.Option(5, "--predict-count", min=1),
    task_spec: Path = typer.Option(DEFAULT_SCHEMA, "--task-spec", exists=True, dir_okay=False),
    instructions: Path | None = typer.Option(None, "--instructions", exists=True, dir_okay=False),
    gold_policy: GoldPolicy = typer.Option(GoldPolicy.CLINICIAN, "--gold-policy"),
    reviewer_id: list[int] | None = typer.Option(None, "--reviewer-id"),
    epochs: int = typer.Option(3, "--epochs", min=1),
    checkpoint: str = typer.Option("distilbert/distilbert-base-multilingual-cased", "--checkpoint"),
    freeze_encoder: bool = typer.Option(True, "--freeze-encoder/--full-finetune"),
    endpoint: str = typer.Option("http://localhost:1234/v1", "--endpoint"),
    model: str = typer.Option("google/gemma-4-e2b", "--model"),
    thinking: bool = typer.Option(False, "--thinking/--no-thinking"),
    seed: int = typer.Option(42, "--seed"),
    yes: bool = typer.Option(False, "--yes", help="Approve noninteractively for scripted runs."),
) -> None:
    """Inspect, fit, predict, or run fit → dump → review → reload → predict."""
    try:
        tasks = load_labelstudio(data)
        spec = TaskSpec.from_xml(task_spec, instructions.read_text() if instructions else None)
        reviewers = tuple(reviewer_id or [3, 4])
        sources = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in data
        }
        if stage == Stage.INSPECT:
            inspection = inspect_dataset(tasks, spec, gold_policy, reviewers)
            typer.echo(json.dumps(asdict(inspection), ensure_ascii=False, indent=2))
            if output:
                write_json(output / "inspection.json", asdict(inspection))
            return
        fit_record: FitRecord | None = None
        if stage in (Stage.FIT, Stage.RUN):
            if method is None or dump is None:
                raise ValueError("Fitting requires --method and --dump.")
            if dump.exists() and any(dump.iterdir()):
                raise ValueError(f"Artifact directory {dump} is not empty; choose a new --dump.")
            fit_offset = offset if offset is not None else 0
            selected = select_slice(tasks, fit_offset, count)
            warn_selected(selected, spec, gold_policy, reviewers)
            examples = [make_example(task, spec, gold_policy, reviewers) for task in selected]
            runner = build_method(
                method, epochs, checkpoint, freeze_encoder, endpoint, model, seed, thinking
            )
            console.print(f"Fitting {method} on document IDs {[e.input.id for e in examples]}.")
            artifact = runner.fit(examples, spec, scorer=score)
            for note in getattr(artifact, "fit_notes", []):
                console.print(note, style="yellow")
            runner.dump(artifact, dump)
            fit_record = FitRecord(
                method=method,
                train_ids=[e.input.id for e in examples],
                annotation_ids=[e.gold.id for e in examples],
                train_text_hashes=[text_hash(e.input.text) for e in examples],
                sources=sources,
                schema_sha256=hashlib.sha256(task_spec.read_bytes()).hexdigest(),
                gold_policy=gold_policy,
                reviewer_ids=list(reviewers),
                created_at=datetime.now(timezone.utc).isoformat(),
                packages=package_versions(),
            )
            (dump / "fit.json").write_text(fit_record.model_dump_json(indent=2))
            # Preview the persisted extractor, not just its in-memory training instance.
            artifact = runner.load(dump)
            preview = runner.predict(artifact, [e.input for e in examples])
            report = score(preview, examples)
            write_json(dump / "training_predictions.json", preview)
            write_json(dump / "training_metrics.json", asdict(report))
            print_results(preview, report, "TRAIN preview — resubstitution, not a held-out score")
            console.print(
                f"Artifact saved to {dump}. Complete predictions are in training_predictions.json."
            )
            if report.failed_documents:
                raise ValueError(
                    "Training preview has inference failures; artifact remains unapproved."
                )
            approve(dump, fit_record, yes)
            if stage == Stage.FIT:
                return
            load = dump
            evaluation_offset = fit_offset + count
            evaluation_count = predict_count
        else:
            evaluation_offset = offset if offset is not None else 5
            evaluation_count = count
        if load is None:
            raise ValueError("Prediction requires --load.")
        header = MethodHeader.model_validate_json((load / "method.json").read_text())
        if method is not None and method != header.method:
            raise ValueError("--method does not match the saved artifact.")
        method = header.method
        runner = build_method(
            method, epochs, checkpoint, freeze_encoder, endpoint, model, seed, thinking
        )
        artifact = runner.load(load)
        # All concrete artifacts carry the task definition restored from their manifest.
        spec = getattr(artifact, "task_spec")
        if not isinstance(spec, TaskSpec):
            raise ValueError("Artifact is missing its task specification.")
        fit_record = FitRecord.model_validate_json((load / "fit.json").read_text())
        if not fit_record.approved:
            preview_report = TypeAdapter(MetricsReport).validate_json(
                (load / "training_metrics.json").read_text()
            )
            if preview_report.failed_documents:
                raise ValueError(
                    "Artifact has a failed training preview; fit again before approval."
                )
            saved_preview = TypeAdapter(list[LabelStudioTaskPrediction]).validate_json(
                (load / "training_predictions.json").read_text()
            )
            print_results(saved_preview, preview_report, "SAVED TRAIN preview — resubstitution")
            approve(load, fit_record, yes)
        selected = select_slice(tasks, evaluation_offset, evaluation_count)
        if any(text_hash(task.data.text) in fit_record.train_text_hashes for task in selected):
            raise ValueError("Prediction slice contains fitting documents; select held-out inputs.")
        labeled = [task for task in selected if task.annotations]
        if labeled and len(labeled) != len(selected):
            raise ValueError("Prediction slice mixes labeled and unlabeled tasks; select one kind.")
        warn_selected(labeled, spec, gold_policy, reviewers)
        gold = [make_example(task, spec, gold_policy, reviewers) for task in labeled]
        inputs = [make_input(task) for task in selected]
        console.print(
            f"Predicting with reloaded {method} on IDs {[document.id for document in inputs]}."
        )
        predictions = runner.predict(artifact, inputs)
        metrics = score(predictions, gold) if len(gold) == len(selected) else None
        destination = output or load.parent / "evaluation"
        write_json(destination / "predictions.json", predictions)
        write_json(destination / "metrics.json", asdict(metrics) if metrics else None)
        record = PredictionRecord(
            stage=stage,
            method=method,
            artifact=str(load.resolve()),
            train_ids=fit_record.train_ids,
            evaluation_ids=[task.id for task in selected],
            annotation_ids=[example.gold.id for example in gold]
            if gold
            else [None] * len(selected),
            sources=sources,
            gold_policy=gold_policy,
            reviewer_ids=list(reviewers),
            created_at=datetime.now(timezone.utc).isoformat(),
            packages=package_versions(),
        )
        (destination / "config.json").write_text(record.model_dump_json(indent=2))
        print_results(predictions, metrics, "HELD-OUT predictions")
        summary = (
            f"Method: {method}\nTrain IDs: {fit_record.train_ids}\n"
            f"Evaluation IDs: {record.evaluation_ids}\nGold policy: {gold_policy}\n"
        )
        if metrics:
            summary += f"Strict entity micro F1: {metrics.strict_micro.f1:.6f}\n"
        (destination / "run_summary.txt").write_text(summary)
        console.print(f"Results saved to {destination}.")
        if any(p.get("meta", {}).get("extraction_error") for p in predictions):
            raise ValueError(
                "Inference failures are recorded in predictions and included in the scores."
            )
    except (ValueError, OSError, ImportError) as error:
        console.print(f"Error: {error}", style="red")
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
