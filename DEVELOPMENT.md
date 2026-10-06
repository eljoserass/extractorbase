# Development record

## Stage 1 — typed data, evaluation, and lifecycle (6 October 2026)

Preserved the repository's top-level layout. Added the `uv` project and lockfile,
Python 3.12 selection, optional CPU BERT dependencies, and local artifact/cache ignores.
Copied the supplied Label Studio XML into `schemas/reumalago.xml`.

`data.py` now defines typed documents, annotations, entity/relation result items, task
specification, and inspection reports. Loading handles null entity metadata and missing
relation labels as present in the actual exports. Predictions remain Label Studio JSON.
Gold policy is explicit: prefer annotations created by reviewers 3/4, with a reported
fallback to originals updated by a reviewer. Later original updates and inferred review
status are reported, not hidden. Raw exports are never rewritten.

`methods/base.py` defines typed `fit`, `dump`, `load`, and text-only `predict`, with an
artifact type per method and optional development data/scorer. The dummy implementation
exercises persistence without a service. `evaluator.py` wraps installed nervaluate 1.2.1;
its dictionary loader, result dataclasses, and inclusive offset convention were inspected.
The primary score is strict entity micro precision/recall/F1 with per-label scores.
Relation results are retained but relation metrics remain explicitly uncomputed.

Validation: 14 synthetic tests passed; mypy passed on the four core implementation
modules. Both real exports loaded: grupo1 has 400 tasks, 6,929 selected entities and
2,170 relations; grupo2 has 400 tasks, 8,143 entities and 2,870 relations. Inspection
reports 17/27 issues respectively, including the one invalid zero-length span in grupo1.
No dataset or generated artifact is included in the commit.

Stage 1 was committed and pushed as `a254216` (`feat: add typed Label Studio data
and evaluation core`).

## Stage 2 — methods and reviewable CLI (6 October 2026)

Implemented the multilingual DistilBERT token classifier with a frozen encoder by
default, optional full fine-tuning, overlapping tokenizer windows, BIO projection
notes, and local safetensor/tokenizer persistence. The trainer owns loss and AdamW;
the shared evaluator is independent of that optimization loop. Reloading requires
no model download.

Implemented the Pydantic AI adapter for Gemma through LM Studio. Fitting saves a
Spanish prompt and the selected demonstrations. Inference uses validated exact
quotes and occurrence indices, then converts them to Label Studio character spans.
It validates entity labels, relation labels and endpoints, with one semantic repair
retry. Failed requests retain the document, save the error, and prevent approval.
HTTP retries are disabled so network failures are bounded. Thinking is off by default
using `reasoning_effort="none"`; `--thinking` leaves the server's setting active.
The installed server advertises Gemma's on/off reasoning capability; a live structured
probe returned zero reasoning tokens with the parameter.

The Typer CLI supports inspect/fit/predict/run, sequential slice flags, artifact
paths, task XML, optional instructions, gold selection, and method configuration.
Fitting reloads the saved extractor for a training preview, prints entity/relation
counts and strict scores, then asks for approval. `--yes` permits explicit scripted
approval. Prediction checks training-text hashes, receives text-only inputs, writes
Label Studio predictions and records gold provenance, package versions and scores.
Unlabeled inference writes null metrics. Approval cannot bypass an inference failure.

Validation before this push: 23 tests passed, including a tiny local BERT train/save/
reload test, long-document windows, LLM conversion/persistence/failure behavior and
the interactive CLI approval lifecycle. Ruff and mypy passed. Real dummy and BERT
runs completed on grupo1 IDs 1–5 for fitting and 6–10 for evaluation. The initial
three-epoch frozen BERT run had strict F1 0 on both slices; this validates execution,
not useful extraction quality. Live full Gemma verification is still in progress.

Stage 2 was committed and pushed as `f5c033d` (`feat: add local BERT and Gemma
methods with reviewable CLI`).

## Stage 3 — scoring edge cases and user documentation (6 October 2026)

Inspection of nervaluate's installed strict strategy exposed greedy matching:
an overlapping error can consume the gold entity needed by a later exact
prediction. The in-memory adapter now places available exact matches first,
respecting duplicate counts. Nervaluate still computes all reported metrics.
Regression cases cover prediction-order independence, nested labels and duplicate
predictions. Per-label results print in stable alphabetical order.

Added a regression test proving that a failed training preview cannot be approved
through a later predict command, including with `--yes`. A declined successful
artifact now shows both saved predictions and scores when approval is retried.
Documented uv setup, CLI flags, typed interfaces, method behavior, artifact layout,
gold provenance, metric definitions, limitations and the lifecycle Mermaid graph
in the README, preserving the original product notes and user edit.

Validation: 27 tests pass; Ruff, formatting and mypy pass. A separate-process BERT
reload produced predictions identical to the first run. The live no-thinking
Gemma run produced valid results for task 1, but four training documents exhausted
the single output-validation retry. The CLI retained all five documents, saved
the failures, and refused approval as intended. Response diagnosis and completion
of the Gemma smoke test remain for the next stage.

Stage 3 was committed and pushed as `432c693` (`fix: preserve strict entity
matches and document the local workflow`).

## Stage 4 — grounded LLM output and visible diagnostics (6 October 2026)

Live response capture showed the exact problem: Gemma expanded `EVA med` into
the absent quote `EVA médico`, sometimes used one-based occurrence indices, and
sometimes emitted relations with nonexistent endpoint IDs. More detailed repair
feedback still failed to correct every mention. This was a method output-quality
issue, not a need for evaluator feedback or access to held-out labels.

The final adapter abstains on absent quotes, invalid indices for repeated quotes,
unknown labels, duplicate entity IDs and invalid relations. A unique quote can be
unambiguously corrected to occurrence zero. It preserves valid mentions, emits
only literal character spans, and prints/saves every correction or omission as
`meta.extraction_warnings` before user approval. Schema/JSON errors still receive
one bounded repair retry; network/unrepaired errors retain empty predictions and
block approval. Full gold and all document denominators remain unchanged.

Strengthened the literal-copy prompt, separated the current document from the
demonstration markers, and retained the underlying cause in saved inference
errors. Validated loaded manifests with method-specific literal types, including
the dummy artifact, replacing its untyped JSON unpacking. Prediction now checks
labeled/unlabeled consistency and validates/reports gold issues before calling
the method, while passing only stripped inputs into inference.

Validation: 31 tests pass, including abstention on ungrounded/ambiguous mentions,
invalid labels/IDs/relations, unique-quote correction, and a CLI test showing
warnings before approval while saving valid predictions. Ruff and mypy pass.
The final live Gemma run has completed the first four training previews without
document errors; the remaining preview and held-out evaluation are in progress.
