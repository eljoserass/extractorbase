"""Checkpointed RuleChef ablation with frozen, label-aware learning/test splits."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    GoldPolicy,
    TaskSpec,
    TrainingExample,
    load_labelstudio,
    make_example,
    prediction_for,
)
from evaluator import score  # noqa: E402
from experiments.rulechef_splits import make_splits  # noqa: E402
from methods.rulechef import RuleChefArtifact, RuleChefConfig, RuleChefMethod  # noqa: E402

# (maximum patch iterations, dev acceptance, agentic coordination)
PROFILES = {
    "oneshot": (0, False, False),
    "train_refine": (3, False, False),
    "holdout": (3, True, False),
    "agentic3": (3, True, True),
    "agentic8": (8, True, True),
}
SOURCES = (
    "data.py",
    "evaluator.py",
    "methods/__init__.py",
    "methods/base.py",
    "methods/rulechef.py",
    "methods/rulechef_client.py",
    "methods/rulechef_learning.py",
    "experiments/rulechef_splits.py",
    "experiments/rulechef_ablation.py",
)
CLINICAL_GUIDANCE = (
    "Spanish rheumatology notes. Test/index names and their numeric values are separate "
    "entities. Follow the label descriptions and the annotated boundaries. "
    "Consider lexical variants and spelling errors; choose regex, spaCy or both as useful. "
    "Do not infer entities absent from the text."
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss(who: int = resource.RUSAGE_SELF) -> int:
    value = resource.getrusage(who).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_inputs(manifest: dict) -> tuple[TaskSpec, dict]:
    for name in ("data", "schema", "glosses"):
        if digest(Path(manifest[name + "_path"])) != manifest[name + "_sha256"]:
            raise ValueError(f"{name} changed after the experiment was frozen.")
    spec = TaskSpec.from_xml(
        Path(manifest["schema_path"]), instructions=Path(manifest["glosses_path"]).read_text()
    )
    tasks = load_labelstudio([Path(manifest["data_path"])])
    examples = [make_example(task, spec, GoldPolicy.CLINICIAN, (3, 4)) for task in tasks]
    return spec, {example.input.id: example for example in examples}


def prepare(args: argparse.Namespace, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare requires an empty output directory. Use run to resume.")
    paths = {
        "data": ROOT / "data/grupo1_2_799.json",
        "schema": ROOT / "data/label_studio_config.xml",
        "glosses": ROOT / "data/zero_shot_label_glosses.md",
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{name + "_path": str(path) for name, path in paths.items()},
        **{name + "_sha256": digest(path) for name, path in paths.items()},
        "mode": args.split_mode,
        "learning_budget": args.train_size,
        "dev_fraction": args.dev_fraction,
        "rounds": args.rounds,
        "profiles": args.profiles,
        "seed": args.seed,
        "exploratory": True,
        "grouping": "normalized text duplicates; patient IDs unavailable",
        "initial_rules_reused_within_round": True,
        "synthesis": asdict(
            RuleChefConfig(
                backend=args.backend,
                model=args.model,
                endpoint=args.endpoint,
                reasoning_effort=args.reasoning_effort,
                synthesis_strategy=args.synthesis_strategy,
                rule_formats=tuple(args.rule_formats),
                spacy_model=args.spacy_model,
                learning_instructions=CLINICAL_GUIDANCE,
                refinement_iterations=0,
                call_timeout_seconds=args.call_timeout,
            )
        ),
        "source_sha256": {name: digest(ROOT / name) for name in SOURCES},
    }
    _, by_id = load_inputs(manifest)
    manifest["splits"] = make_splits(
        list(by_id.values()),
        rounds=args.rounds,
        train_size=args.train_size,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        mode=args.split_mode,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"state": "prepared", "completed": 0})
    print(f"Prepared {args.rounds} rounds × {len(args.profiles)} profiles: {output}")


def configuration(
    manifest: dict,
    trace: Path,
    *,
    iterations: int = 0,
    agentic: bool = False,
) -> RuleChefConfig:
    config = dict(manifest["synthesis"])
    config.update(trace_directory=str(trace), refinement_iterations=iterations, agentic=agentic)
    config["rule_formats"] = tuple(config["rule_formats"])
    return RuleChefConfig(**config)


def initial_artifact(
    manifest: dict, split: dict, by_id: dict, spec: TaskSpec, root: Path, use_dev: bool
) -> tuple[RuleChefArtifact, list[TrainingExample]]:
    name = "initial_train" if use_dev else "initial_full"
    directory = root / name
    artifact_path = directory / "artifact" / "artifact.json"
    ids = split["train_ids"] if use_dev else split["learning_ids"]
    examples = [by_id[key] for key in ids]
    if artifact_path.exists() and (directory / "resources.json").exists():
        loader = RuleChefMethod()
        return loader.load(artifact_path.parent), examples
    # A failed attempt keeps its traces. A retry gets a separate directory.
    attempt = len(list(directory.glob("attempt_*"))) + 1
    trace = directory / f"attempt_{attempt:03d}"
    method = RuleChefMethod(configuration(manifest, trace))
    started = time.perf_counter()
    artifact = method.fit(examples, spec, scorer=score)
    method.dump(artifact, artifact_path.parent)
    write_json(
        directory / "resources.json",
        {
            "seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss(),
            "peak_child_process_rss_bytes": peak_rss(resource.RUSAGE_CHILDREN),
        },
    )
    return artifact, examples


def fit_cell(manifest: dict, output: Path, round_number: int, profile: str) -> None:
    spec, by_id = load_inputs(manifest)
    split = manifest["splits"][round_number - 1]
    root = output / f"round_{round_number}"
    destination = root / profile
    iterations, use_dev, agentic = PROFILES[profile]
    started = time.perf_counter()
    artifact, train = initial_artifact(manifest, split, by_id, spec, root, use_dev)
    initial_call_count = len(artifact.provenance["model_calls"])
    dev = [by_id[key] for key in split["dev_ids"]] if use_dev else []
    if iterations:
        attempt = len(list(destination.glob("attempt_*"))) + 1
        method = RuleChefMethod(
            configuration(
                manifest,
                destination / f"attempt_{attempt:03d}",
                iterations=iterations,
                agentic=agentic,
            )
        )
        artifact = method.refine(artifact, train, dev_examples=dev, scorer=score)
    else:
        method = RuleChefMethod(configuration(manifest, destination / "unused_synthesis"))
    artifact.provenance["experiment"] = {
        "round": round_number,
        "profile": profile,
        "initial_call_count": initial_call_count,
        "initial_rules_shared": True,
    }
    method.dump(artifact, destination / "artifact")
    # Exercise a load without a client and compare the complete LS prediction.
    restored = RuleChefMethod()
    loaded = restored.load(destination / "artifact")
    original = method.predict(artifact, [example.input for example in train])
    reloaded = restored.predict(loaded, [example.input for example in train])
    if original != reloaded:
        raise RuntimeError("Artifact reload changed training predictions.")
    write_json(
        destination / "fit_resources.json",
        {
            "cell_wall_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss(),
            "peak_child_process_rss_bytes": peak_rss(resource.RUSAGE_CHILDREN),
            "artifact_reload_equal": True,
        },
    )


def test_cell(manifest: dict, output: Path, round_number: int, profile: str) -> None:
    _, by_id = load_inputs(manifest)
    split = manifest["splits"][round_number - 1]
    destination = output / f"round_{round_number}" / profile
    method = RuleChefMethod()
    started = time.perf_counter()
    artifact = method.load(destination / "artifact")
    load_seconds = time.perf_counter() - started
    test = [by_id[key] for key in split["test_ids"]]
    train_texts = {
        " ".join(by_id[key].input.text.split()).casefold() for key in split["learning_ids"]
    }
    if any(" ".join(example.input.text.split()).casefold() in train_texts for example in test):
        raise ValueError("A test text duplicates a learning text.")
    predictions, failures = [], []
    started = time.perf_counter()
    for example in test:
        try:
            predictions.extend(method.predict(artifact, [example.input]))
        except Exception as error:
            failures.append({"id": example.input.id, "error": str(error)})
            predictions.append(prediction_for(example.input, [], "rulechef-failed"))
    seconds = time.perf_counter() - started
    metrics = replace(score(predictions, test), failed_documents=len(failures))
    calls = artifact.provenance["model_calls"]
    token_usage: dict[str, int] = {}
    for call in calls:
        for key, value in call.get("usage", {}).items():
            if isinstance(value, int):
                token_usage[key] = token_usage.get(key, 0) + value
    result = {
        "round": round_number,
        "profile": profile,
        "train_notes": len(artifact.provenance["train_ids"]),
        "dev_notes": len(artifact.provenance.get("dev_ids", [])),
        "test_notes": len(test),
        "metrics": asdict(metrics),
        "rules": len(artifact.rules),
        "rule_formats": artifact.provenance["rule_formats"],
        "calls": len(calls),
        "initial_calls_shared": artifact.provenance["experiment"]["initial_call_count"],
        "tokens": token_usage,
        "learning_model_seconds": sum(c.get("seconds", 0) for c in calls),
        "test_seconds": seconds,
        "artifact_load_seconds": load_seconds,
        "test_peak_process_rss_bytes": peak_rss(),
        "cost_usd": None,
        "cost_note": "Not inferred from subscription tokens or assumed API prices.",
        "memory_note": "Process RSS; remote model/server memory is not measured.",
    }
    write_json(destination / "predictions.json", predictions)
    write_json(destination / "failures.json", failures)
    write_json(destination / "result.json", result)
    print(
        json.dumps(
            {
                "profile": profile,
                "round": round_number,
                "strict_micro_f1": metrics.strict_micro.f1,
                "strict_macro_f1": metrics.strict_macro.f1 if metrics.strict_macro else None,
                "calls": len(calls),
            }
        ),
        flush=True,
    )


def report(output: Path, manifest: dict) -> None:
    rows = []
    for index in range(1, manifest["rounds"] + 1):
        for profile in manifest["profiles"]:
            path = output / f"round_{index}" / profile / "result.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    summary = []
    lines = [
        "# RuleChef ablation — exploratory",
        "",
        "Learning budgets and test sets are frozen in manifest.json. Few-shot rounds overlap; "
        "their standard deviation is descriptive, not a confidence interval. "
        "Fmt/Sem are TAB-specific and are not relabeled as clinical metrics.",
        "",
        "| Configuration | Rounds | Strict micro F1 | Strict macro F1 "
        "| Overlap micro F1 | Rules | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in manifest["profiles"]:
        selected = [row for row in rows if row["profile"] == profile]
        if not selected:
            lines.append(f"| {profile} | 0/{manifest['rounds']} | pending | | | | |")
            continue
        entry = {"profile": profile, "rounds_completed": len(selected)}
        values = []
        for key in ("strict_micro", "strict_macro", "overlap_micro"):
            scores = [row["metrics"][key]["f1"] for row in selected]
            mean = statistics.mean(scores)
            deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
            entry[key] = {"mean": mean, "standard_deviation": deviation}
            values.append(f"{mean * 100:.2f} ± {deviation * 100:.2f}")
        rules = statistics.mean(row["rules"] for row in selected)
        calls = statistics.mean(row["calls"] for row in selected)
        entry.update(rules_mean=rules, calls_mean=calls)
        summary.append(entry)
        lines.append(
            f"| {profile} | {len(selected)}/{manifest['rounds']} | "
            + " | ".join(values)
            + f" | {rules:.1f} | {calls:.1f} |"
        )
    lines.extend(
        [
            "",
            "Calls include the shared initial synthesis for each configuration. "
            "Do not sum table rows to estimate actual unique calls. Token usage, timing, "
            "RSS, failures and per-label scores are saved beside each result.",
        ]
    )
    write_json(output / "summary.json", {"profiles": summary, "results": rows})
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_matrix(output: Path, manifest: dict) -> None:
    source = output / "source"
    for name, expected in manifest["source_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    command = [
        sys.executable,
        str(source / "experiments/rulechef_ablation.py"),
        "--output",
        str(output),
    ]
    completed = 0
    for index in range(1, manifest["rounds"] + 1):
        for profile in manifest["profiles"]:
            destination = output / f"round_{index}" / profile
            if (destination / "result.json").exists():
                completed += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for stage in ("fit", "test"):
                if stage == "fit" and (destination / "fit_resources.json").exists():
                    continue
                write_json(
                    output / "status.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "round": index,
                        "profile": profile,
                        "stage": stage,
                        "completed": completed,
                    },
                )
                with (destination / f"{stage}.log").open("a") as log:
                    process = subprocess.run(
                        [
                            *command,
                            "--stage",
                            stage,
                            "--round-number",
                            str(index),
                            "--profile",
                            profile,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=source,
                    )
                if process.returncode:
                    write_json(
                        output / "status.json",
                        {
                            "state": "failed",
                            "round": index,
                            "profile": profile,
                            "stage": stage,
                            "completed": completed,
                            "log": str(destination / f"{stage}.log"),
                        },
                    )
                    report(output, manifest)
                    raise RuntimeError(f"Cell failed; inspect {destination / (stage + '.log')}")
            completed += 1
            report(output, manifest)
    write_json(output / "status.json", {"state": "complete", "completed": completed})
    report(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "run", "report", "fit", "test"], default="prepare"
    )
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--synthesis-strategy", choices=["bulk", "per_class"], default="per_class")
    parser.add_argument(
        "--rule-formats", nargs="+", choices=["regex", "spacy"], default=["regex", "spacy"]
    )
    parser.add_argument("--spacy-model", default="blank:es")
    parser.add_argument(
        "--train-size", type=int, default=20, help="Total learning budget before inner dev split"
    )
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--split-mode", choices=["fewshot", "fivefold"], default="fewshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--call-timeout", type=int, default=1200)
    parser.add_argument("--round-number", type=int, default=1)
    parser.add_argument("--profile", choices=list(PROFILES), default="oneshot")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        prepare(args, output)
        return
    manifest = json.loads((output / "manifest.json").read_text())
    if args.stage == "run":
        run_matrix(output, manifest)
    elif args.stage == "report":
        report(output, manifest)
    elif args.stage == "fit":
        fit_cell(manifest, output, args.round_number, args.profile)
    else:
        test_cell(manifest, output, args.round_number, args.profile)


if __name__ == "__main__":
    main()
