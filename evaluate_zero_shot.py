"""Evaluate an instruction-only local LLM, saving each document for safe resumption."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from pydantic import TypeAdapter

from data import (
    GoldPolicy,
    LabelStudioTaskPrediction,
    TaskSpec,
    load_labelstudio,
    make_example,
)
from evaluator import score
from methods.llm import LLMConfig, LLMMethod


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, action="append", required=True)
    result.add_argument("--task-spec", type=Path, required=True)
    result.add_argument("--model", required=True, help="Same model ID/path as the local server.")
    result.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    result.add_argument("--backend", choices=("mlx", "lmstudio"), default="mlx")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--offset", type=int, default=10)
    result.add_argument("--count", type=int, help="Default: all notes after offset.")
    result.add_argument("--instructions", type=Path, help="Optional label glosses/guidelines.")
    result.add_argument("--timeout", type=float, default=300.0)
    result.add_argument("--max-output-tokens", type=int, default=4096)
    result.add_argument(
        "--memory-report", type=Path, help="Live memory snapshot from serve_mlx.py."
    )
    result.add_argument(
        "--max-documents", type=int, help="Stop after N new requests; rerun without it to finish."
    )
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def performance(receipts: list[dict[str, Any]], target: int) -> dict[str, Any]:
    durations = [float(item["seconds"]) for item in receipts]
    ordered = sorted(durations)
    known = [
        item
        for item in receipts
        if item["prediction"].get("meta", {}).get("inference_usage", {}).get("input_tokens")
        is not None
    ]
    usages = [item["prediction"]["meta"]["inference_usage"] for item in known]
    input_tokens = sum(item["input_tokens"] for item in usages)
    output_tokens = sum(item["output_tokens"] for item in usages)
    known_seconds = sum(float(item["seconds"]) for item in known)
    requests = [item["requests"] for item in usages if item.get("requests") is not None]
    return {
        "completed": len(receipts),
        "target": target,
        "prediction_seconds": sum(durations),
        "mean_seconds_per_document": mean(durations) if durations else None,
        "median_seconds_per_document": median(durations) if durations else None,
        "p95_seconds_per_document": ordered[math.ceil(len(ordered) * 0.95) - 1]
        if ordered
        else None,
        "max_seconds_per_document": max(durations) if durations else None,
        "estimated_remaining_seconds": mean(durations) * (target - len(receipts))
        if durations
        else None,
        "documents_with_token_usage": len(known),
        "documents_without_token_usage": len(receipts) - len(known),
        "reported_input_tokens": input_tokens if known else None,
        "reported_output_tokens": output_tokens if known else None,
        "reported_total_tokens": input_tokens + output_tokens if known else None,
        "reported_api_requests": sum(requests) if requests else None,
        "reported_repair_requests": sum(max(0, value - 1) for value in requests)
        if requests
        else None,
        "output_tokens_per_second_end_to_end": output_tokens / known_seconds
        if known_seconds
        else None,
    }


def run(args: argparse.Namespace) -> int:
    tasks = load_labelstudio(args.data)
    count = args.count if args.count is not None else len(tasks) - args.offset
    if args.offset < 0 or count < 1 or args.offset + count > len(tasks):
        raise ValueError(f"Invalid slice [{args.offset}:{args.offset + count}] of {len(tasks)}.")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("--max-documents must be positive.")
    instructions = args.instructions.read_text(encoding="utf-8") if args.instructions else None
    spec = TaskSpec.from_xml(args.task_spec, instructions)
    selected = tasks[args.offset : args.offset + count]
    # Validate all gold before starting a long run. It never enters the prompt or requests.
    gold = [make_example(task, spec, GoldPolicy.CLINICIAN, (3, 4)) for task in selected]
    method = LLMMethod(
        LLMConfig(
            model=args.model,
            base_url=args.endpoint,
            backend=args.backend,
            thinking=False,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )
    )
    entity_spec = replace(spec, relation_labels=())
    artifact = method.fit([], entity_spec)
    prompt = (
        "Extrae exclusivamente entidades de textos clinicos en espanol. "
        "Esta tarea es zero-shot: no hay ejemplos anotados. "
        "Usa solo las etiquetas permitidas y copia cada cita literalmente del documento. "
        "No traduzcas, normalices ni expandas abreviaturas. No calcules offsets. "
        "Cada entidad necesita un id unico, label, text y occurrence: indice desde cero "
        "de esa cita exacta entre todas sus apariciones de izquierda a derecha, "
        "incluidas apariciones dentro de palabras. Si aparece una vez, occurrence=0. "
        "Respeta mayusculas, tildes y espacios. Las entidades pueden solaparse. "
        "Separa las menciones de pruebas e indices de sus valores cuando corresponda. "
        "No inventes menciones ni etiquetas. Devuelve entities=[] si no encuentras ninguna. "
        "No extraigas relaciones ni negacion; devuelve siempre relations=[].\n"
        f"Etiquetas permitidas: {json.dumps(spec.entity_labels, ensure_ascii=False)}\n"
    )
    if instructions:
        prompt += instructions + "\n"
    if len(prompt) > method.config.max_prompt_characters:
        raise ValueError("Instructions exceed the configured prompt character budget.")
    artifact = replace(artifact, prompt=prompt)
    source_root = Path(__file__).parent
    manifest = {
        "protocol": "zero-shot-entities-only-v2",
        "train_ids": [],
        "demonstration_count": 0,
        "config": asdict(method.config),
        "sources": {str(path.resolve()): sha256(path) for path in args.data},
        "schema_sha256": sha256(args.task_spec),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "entity_labels": list(spec.entity_labels),
        "gold_policy": "clinician",
        "reviewer_ids": [3, 4],
        "evaluation_ids": [example.input.id for example in gold],
        "annotation_ids": [example.gold.id for example in gold],
        "memory_report": str(args.memory_report.resolve()) if args.memory_report else None,
        "implementation_sha256": {
            name: sha256(source_root / name)
            for name in (
                "evaluate_zero_shot.py",
                "serve_mlx.py",
                "data.py",
                "evaluator.py",
                "methods/llm.py",
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("nervaluate", "pydantic", "pydantic-ai-slim")
        },
    }
    output = args.output
    manifest_path = output / "manifest.json"
    artifact_path = output / "artifact"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Saved run differs in data, recipe or code; choose a new --output.")
        if method.load(artifact_path) != artifact:
            raise ValueError(
                "Saved artifact differs from the frozen recipe; choose a new --output."
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output is nonempty without a run manifest; choose a new --output.")
        method.dump(artifact, artifact_path)
        write_json(manifest_path, manifest)
    # Use the persisted recipe for requests, just as the main harness does.
    artifact = method.load(artifact_path)
    adapter = TypeAdapter(LabelStudioTaskPrediction)
    predictions: list[LabelStudioTaskPrediction] = []
    receipts: list[dict[str, Any]] = []
    new_requests = 0
    for index, example in enumerate(gold):
        cache = output / "documents" / f"{index:06d}.json"
        if cache.exists():
            receipt = json.loads(cache.read_text())
            prediction = adapter.validate_python(receipt["prediction"])
        else:
            if args.max_documents is not None and new_requests >= args.max_documents:
                break
            started = perf_counter()
            batch = method.predict(artifact, [example.input])
            if len(batch) != 1:
                raise ValueError("Expected one prediction for the requested document.")
            prediction = adapter.validate_python(batch[0])
            receipt = {"prediction": prediction, "seconds": perf_counter() - started}
            new_requests += 1
        if prediction["id"] != example.input.id or prediction["data"]["text"] != example.input.text:
            raise ValueError("Prediction does not match the requested document.")
        if len(prediction["predictions"]) != 1:
            raise ValueError("Expected exactly one annotation per document.")
        if not cache.exists():
            if args.memory_report and args.memory_report.exists():
                receipt["server_memory"] = json.loads(args.memory_report.read_text())
            write_json(cache, receipt)
        predictions.append(prediction)
        receipts.append(receipt)
        timings = performance(receipts, len(gold))
        if args.memory_report and args.memory_report.exists():
            timings["server_memory"] = json.loads(args.memory_report.read_text())
        memories = [item["server_memory"] for item in receipts if "server_memory" in item]
        if memories:
            timings["server_peak_mlx_active_bytes"] = max(
                item["mlx_peak_active_bytes"] for item in memories
            )
            timings["server_peak_rss_bytes"] = max(
                item["process_peak_rss_bytes"] for item in memories
            )
        write_json(output / "performance.json", timings)
        failed = sum(bool(p.get("meta", {}).get("extraction_error")) for p in predictions)
        write_json(
            output / "progress.json",
            {"completed": len(predictions), "target": len(gold), "failed_documents": failed},
        )
        logging.info("Saved %s/%s documents; failures: %s", len(predictions), len(gold), failed)
    if len(predictions) != len(gold):
        print(f"Saved {len(predictions)}/{len(gold)}. Rerun without --max-documents to finish.")
        return int(any(p.get("meta", {}).get("extraction_error") for p in predictions))
    report = score(predictions, gold)
    write_json(output / "predictions.json", predictions)
    write_json(output / "metrics.json", asdict(report))
    write_json(output / "config.json", manifest)
    summary = {
        "documents": report.documents,
        "failed_documents": report.failed_documents,
        "strict_micro": asdict(report.strict_micro),
        "overlap_micro": asdict(report.overlap_micro) if report.overlap_micro else None,
        "strict_macro": asdict(report.strict_macro) if report.strict_macro else None,
        "overlap_macro": asdict(report.overlap_macro) if report.overlap_macro else None,
        "performance": timings,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(report.failed_documents > 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(run(parser().parse_args()))
    except KeyboardInterrupt:
        logging.info(
            "Interrupted. Completed documents are saved; rerun the same command to resume."
        )
        raise SystemExit(130) from None
    except ValueError as error:
        logging.error("%s", error)
        raise SystemExit(1) from None
