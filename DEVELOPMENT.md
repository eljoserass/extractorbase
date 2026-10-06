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
