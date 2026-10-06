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
