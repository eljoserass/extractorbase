"""Fixed train-10/test-789 RuleChef pilot, with a fresh-process artifact reload."""

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    GoldPolicy,
    TaskSpec,
    load_labelstudio,
    make_example,
    prediction_for,
)
from evaluator import score  # noqa: E402
from methods.rulechef import RuleChefConfig, RuleChefMethod  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss(who: int) -> int:
    value = resource.getrusage(who).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["run", "fit", "test"], default="run")
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--refinement-iterations", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = ROOT / "data/common_holdout_789_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_path = ROOT / manifest["data"]
    schema_path = ROOT / manifest["task_spec"]
    assert sha256(data_path) == manifest["source_sha256"]
    assert sha256(schema_path) == manifest["schema_sha256"]
    spec = TaskSpec.from_xml(
        schema_path,
        instructions=(ROOT / "data/zero_shot_label_glosses.md").read_text(),
    )
    tasks = load_labelstudio([data_path])
    by_id = {task.id: task for task in tasks}
    train_ids = manifest["reserved_ids"]
    test_ids = manifest["evaluation_ids"]
    assert train_ids == list(range(1, 11)) and len(test_ids) == 789
    assert not set(train_ids) & set(test_ids)
    method = RuleChefMethod(
        RuleChefConfig(
            backend=args.backend,
            model=args.model,
            endpoint=args.endpoint,
            refinement_iterations=args.refinement_iterations,
            trace_directory=str(output / "synthesis"),
        )
    )
    artifact_dir = output / "artifact"
    if args.stage in {"run", "fit"}:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose an empty output directory for a new fit.")
        output.mkdir(parents=True, exist_ok=True)
        source = output / "source"
        source.mkdir()
        shutil.copy2(ROOT / "methods/rulechef.py", source / "rulechef.py")
        shutil.copy2(Path(__file__), source / "evaluate_rulechef.py")
        train = [make_example(by_id[key], spec, GoldPolicy.CLINICIAN, (3, 4)) for key in train_ids]
        write_json(
            output / "config.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "method": "rulechef",
                "protocol": "10 train, no dev, 789 held-out; no test-driven changes",
                "train_ids": train_ids,
                "test_ids": test_ids,
                "data_sha256": sha256(data_path),
                "schema_sha256": sha256(schema_path),
                "implementation_sha256": sha256(ROOT / "methods/rulechef.py"),
                "runner_sha256": sha256(Path(__file__)),
                "config": asdict(method.config),
                "gold_policy": "clinician",
                "reviewer_ids": [3, 4],
            },
        )
        started = time.perf_counter()
        artifact = method.fit(train, spec, scorer=score)
        fit_seconds = time.perf_counter() - started
        method.dump(artifact, artifact_dir)
        predictions = method.predict(artifact, [example.input for example in train])
        restored = method.load(artifact_dir)
        assert predictions == method.predict(restored, [example.input for example in train])
        calls = artifact.provenance["model_calls"]
        usage: dict[str, int] = {}
        for call in calls:
            for key, value in call.get("usage", {}).items():
                usage[key] = usage.get(key, 0) + value
        write_json(output / "training_predictions.json", predictions)
        write_json(output / "training_metrics.json", asdict(score(predictions, train)))
        write_json(
            output / "fit_resources.json",
            {
                "fit_seconds": fit_seconds,
                "model_calls": len(calls),
                "tokens": usage,
                "local_process_peak_rss_bytes": peak_rss(resource.RUSAGE_SELF),
                "child_process_peak_rss_bytes": peak_rss(resource.RUSAGE_CHILDREN),
                "provider_model_memory_bytes": None,
                "artifact_sha256": sha256(artifact_dir / "artifact.json"),
                "saved_rule_count": len(artifact.rules),
                "reload_identical": True,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "fit",
                    "rules": len(artifact.rules),
                    "seconds": fit_seconds,
                    "tokens": usage,
                }
            ),
            flush=True,
        )
        if args.stage == "fit":
            return
        # The test worker never constructs a synthesis client or calls fit.
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--stage",
                "test",
            ],
            check=True,
            cwd=ROOT,
        )
        return

    if (output / "metrics.json").exists():
        raise ValueError("Test results already exist; refusing to overwrite them.")
    artifact_hash = sha256(artifact_dir / "artifact.json")
    started = time.perf_counter()
    artifact = method.load(artifact_dir)
    load_seconds = time.perf_counter() - started
    assert artifact.provenance["train_ids"] == train_ids
    train_hashes = set(artifact.provenance["train_text_sha256"])
    test = [make_example(by_id[key], spec, GoldPolicy.CLINICIAN, (3, 4)) for key in test_ids]
    assert not train_hashes & {hashlib.sha256(e.input.text.encode()).hexdigest() for e in test}
    predictions, failures = [], []
    started = time.perf_counter()
    for example in test:
        try:
            predictions.extend(method.predict(artifact, [example.input]))
        except Exception as error:
            failure = {"id": example.input.id, "error": str(error)}
            failures.append(failure)
            prediction = prediction_for(example.input, [], "rulechef-0.2.2-regex")
            prediction["meta"] = {"extraction_error": str(error)}
            predictions.append(prediction)
    prediction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = asdict(score(predictions, test))
    evaluation_seconds = time.perf_counter() - started
    assert sha256(artifact_dir / "artifact.json") == artifact_hash
    write_json(output / "predictions.json", predictions)
    write_json(output / "metrics.json", metrics)
    write_json(output / "failures.json", failures)
    write_json(
        output / "test_resources.json",
        {
            "documents": len(test),
            "failed_documents": len(failures),
            "fit_called": False,
            "load_seconds": load_seconds,
            "prediction_seconds": prediction_seconds,
            "evaluation_seconds": evaluation_seconds,
            "runtime_llm_calls": 0,
            "runtime_tokens": 0,
            "local_process_peak_rss_bytes": peak_rss(resource.RUSAGE_SELF),
            "artifact_sha256_before": artifact_hash,
            "artifact_sha256_after": sha256(artifact_dir / "artifact.json"),
            "test_implementation_sha256": sha256(ROOT / "methods/rulechef.py"),
            "annotation_ids": [example.gold.id for example in test],
        },
    )
    lines = [
        "# RuleChef: train 10 / test 789",
        "",
        "Fit on IDs 1–10 only; training refinement uses exact offsets. No dev set. "
        "Frozen JSON artifact loaded in a fresh process; regex inference, no model calls.",
        "",
        "| Metric | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("strict_micro", "strict_macro", "overlap_micro", "overlap_macro"):
        metric = metrics[key]
        lines.append(
            f"| {key} | {metric['precision']:.2%} | {metric['recall']:.2%} | {metric['f1']:.2%} |"
        )
    lines += [
        "",
        f"Gold entities: {metrics['strict_micro']['support']}. "
        f"Failed documents: {len(failures)}. Inference: {prediction_seconds:.3f} seconds.",
        "",
        "Entities only; relations are not evaluated. "
        "This exploratory test cohort was used in previous method comparisons.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "stage": "test",
                "documents": len(test),
                "failures": len(failures),
                "prediction_seconds": prediction_seconds,
                **{
                    key: metrics[key]
                    for key in ("strict_micro", "strict_macro", "overlap_micro", "overlap_macro")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
