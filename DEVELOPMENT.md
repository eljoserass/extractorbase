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

Stage 4 was committed and pushed as `092f235` (`fix: ground Gemma mentions and
report extraction warnings before approval`).

## Stage 5 — completed smoke tests and memoir (6 October 2026)

The final live Gemma command completed with exit code zero:

```bash
uv run --extra bert python cli.py --stage run --method llm \
  --data data/grupo1.json --dump runs/gemma_first5_final/artifact \
  --output runs/gemma_first5_final/evaluation --yes
```

The saved records confirm train IDs 1–5 and clinician annotation IDs
4585–4589, evaluation IDs 6–10, training support 165 and evaluation support 51.
Training resubstitution precision/recall/F1: 0.903226 / 0.678788 / 0.775087.
Held-out strict entity precision/recall/F1: 0.232558 / 0.196078 / 0.212766.
There are zero failed documents, 31 training grounding warnings and 35 evaluation
warnings. Per-document evaluation entity counts are 7, 4, 26, 6 and 0; relation
counts are 1, 2, 0, 2 and 0. These are weak baseline results, not an accuracy claim.
No held-out gold was passed to the method or used in the grounding changes.

The real dummy and three-epoch frozen BERT runs also completed on the same slices,
both with strict F1 0 and no document failures. BERT emitted 48 held-out entities,
none matching gold exactly. A separate-process BERT load reproduced its original
predictions exactly. A real PTY dummy fit printed its training preview and stopped
at the approval question; entering `y` saved approval, and a later prediction
process completed without another question. Synthetic tests cover declining and
failure cases.

An additional fresh-process Gemma prediction (`--stage predict --offset 5
--count 1 --load runs/gemma_first5_final/artifact`) completed with exit code zero
and reproduced the first held-out task prediction, including warnings, exactly.

Final checks: 31 tests passed, mypy passed for all nine implementation modules,
Ruff and formatting passed, and `uv pip check` reported all 66 installed packages
compatible. The working dependencies and CPU checkpoint are installed locally.
Git contains no dataset, prediction or model-weight files. Appended the requested
memoir at the end of README, with the measured scores, known limitations and
progressive push record. The final push contains this documentation only.

Stage 5 was committed and pushed as `08a5a76` (`docs: record verified real-data
runs and development memoir`).

## CLI follow-up — help aliases and stdout diagnosis (6 October 2026)

The user reported that successful commands and `--help` appeared blank while
missing-option errors were visible. `--help` reproduced normally in both captured
subprocess output and a PTY. Process descriptors showed a nested interactive bash
with stdout directed into a pipe and stderr still directed to its terminal; this
matches the reported symptom. Documented a stderr diagnostic and how to restore
stdout in an interactive terminal. No user terminal or descriptor was modified.

The existing `runs/gemma/artifact` contains all fitting/preview files, has zero
preview failures, and remains unapproved. Repeating the user's exact `run`
invocation printed the nonempty-directory error with exit code 1. The user can
resume from the saved preview using `--stage predict --load runs/gemma/artifact`
and review/approve it, or choose a fresh `--dump` for another fit.

Added the missing `-h` alias through Typer's context settings. Two tests assert
that `-h` and `--help` print options on stdout and exit successfully without
requiring a dataset. All 33 tests pass, plus Ruff and mypy. The pre-existing local
blank-line edit in cli.py is preserved and excluded from the commit.

The CLI help follow-up was committed and pushed as `a745da3` (`fix: support short
CLI help and document stdout diagnosis`).

## CLI documentation — offsets and command reference (6 October 2026)

Created `docs/cli.md` and linked it from the README. The guide documents setup,
stages, required flags, every option, methods, approval, saved files, gold
selection and troubleshooting. Offset examples use array positions rather than
assuming Label Studio IDs match their position. A selection diagram shows how
`run` predicts immediately after the fitting slice; standalone prediction's fixed
default offset 5 is called out, including fits of different sizes and new files.

Examples cover inspection, Gemma's 5+5 run, BERT fit/reload, resuming the existing
Gemma artifact, unlabeled inputs and a dummy smoke test. All option names and
defaults were checked against the current CLI. This change is documentation only;
the existing user changes and generated artifacts are untouched.

Added `docs/install.md` at the user's follow-up request. It covers cloning,
installing uv, locked dependency setup with/without the BERT extra, Python,
dataset placement, LM Studio, smoke tests, development checks and troubleshooting.
Installation commands were checked against the current uv documentation; server
setup was checked against LM Studio's official documentation. Both guides link to
one another and are linked from the README.

Validation: the guide covers all 25 CLI parameter options and both help aliases;
relative Markdown links, code fences and every Bash example in the guides and
README pass checks. `uv lock --check` succeeds, and a locked BERT sync dry-run
would make no changes. The documented dummy 5+5 run succeeds with `grupo1.json`
using a temporary destination: training IDs 1–5, prediction IDs 6–10, approval
recorded, prediction-start message and zero F1 printed. Existing artifacts were
not overwritten.

The user also reported that the prediction-start line was invisible. Read-only
inspection confirmed the shell's stdout still pointed to a pipe while stderr
pointed to its terminal, and `runs/gemma/artifact/fit.json` remained unapproved.
Reproducing the exact command with an explicit `n` printed the saved preview and
approval prompt, then aborted before prediction, as intended. The guide now
explains that this line follows approval and how to restore stdout in the user's
terminal. No CLI behavior was changed for this shell issue.

The installation/CLI guides were committed and pushed as `b710eeb` (`docs: add
installation and CLI guides with offset examples`).

## Method guides — background and implementation rationale (6 October 2026)

Added `docs/methods/README.md` as an overview of the typed method contract,
different learning signals, runner/evaluator responsibilities, common character
spans and method-specific artifacts. Separate `bert.md`, `llm.md` and `dummy.md`
guides explain each implementation from fitting through saved-state reload and
prediction, with commands, code links, tradeoffs and links to existing tests.

BERT covers the contextual encoder and classifier, BIO targets, loss/optimizer
responsibilities, frozen versus full fine-tuning, overflow windows, gold
projection, ignored edge fragments and confidence-based span merging. LLM covers
demonstration construction, native structured output, quote/occurrence grounding,
warnings versus document failures, retry boundaries, server dependence and
stored prompts. Dummy explains why a successful empty prediction is useful for
testing the lifecycle and differs from an inference failure.

All configuration defaults were read from the current implementations, including
the CLI-only learning-rate change for full fine-tuning, the BERT stride meaning,
and LLM's character rather than token budget. Background references use original
BERT/DistilBERT and few-shot prompting papers and official Hugging Face,
Pydantic AI and LM Studio docs.
Mermaid diagrams show the shared lifecycle, BERT training signal and LLM response
validation. Linked the guides from README, CLI and installation docs, and
extended the memoir at the end of README. This pass changes documentation only.

Validation: all relative links and code fences pass checks, Bash examples pass
`bash -n`, Python blocks compile, JSON examples parse, and all documented method
CLI flags exist. Executing the LLM grounding snippet and its JSON example gives
the documented PCR/value spans and relation without a server call. Both dummy
guide commands pass on `grupo1.json` with temporary destinations: interactive
approval, fitting IDs 1–5, prediction IDs 6–10 and identical predictions after a
fresh-process reload. The nine focused dummy/BERT/LLM tests pass in 5.24 seconds,
including the tiny local BERT fit; no pretrained checkpoint download or live LLM
request was needed. `git diff --check` passes. The user's CLI edits remain
unstaged and existing run artifacts are untouched.
