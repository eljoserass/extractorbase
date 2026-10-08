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

## Privacy correction — keep the task schema local (6 October 2026)

The user clarified that the Label Studio XML is private. It had been copied from
local data into `schemas/reumalago.xml` and tracked in the first implementation
commit. This was an incorrect assumption about what could be published.

Removed the XML from the Git index while preserving its local bytes, and ignored
the entire `schemas/` directory alongside the already ignored `data/` and `runs/`.
Installation/CLI documentation now requires a locally supplied XML at the default
path or through `--task-spec`. CLI tests generate their own minimal XML fixture
and pass its path explicitly, so Git does not need the private schema for tests.
All 33 tests pass and Ruff passes on the modified tests.

The published repository has one branch (`master`), no tags and no pull-request
refs. Cleaned that branch in an isolated clone with git-filter-repo's sensitive-data
removal mode, excluding `schemas/` and `data/` from its history. Verified that the
schema blob and paths are absent from every ref in the clean clone, and that
filtering preserved the correction commit's tree. All 33 tests also passed in
that clone without any private XML or datasets. Updated GitHub with an explicit
force-with-lease push and confirmed the remote tip. The local XML's checksum and
the user's unstaged CLI diff are unchanged. Earlier commit IDs in this development
record predate the history rewrite.

GitHub documents that cached views and other clones can retain removed data;
complete server-side cache removal requires GitHub Support. See its
[sensitive-data removal instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
After the push, a read-only request to the old raw-file URL still returned bytes
matching the private schema's checksum. Prepared a Support-request draft locally
with the first changed commit, affected-ref count and removal details. It was
not sent: the account owner must submit the request to GitHub Support for the
server-side cache/object purge. No private schema contents were added to the draft.

## Mac MLX LM — adapter and dependencies (6 October 2026)

The user requested a cloneable Mac setup using an existing local MLX LM model.
The official server accepts OpenAI chat requests but ignores `response_format`
and `reasoning_effort`. Merely changing the endpoint would therefore omit the
output schema from the model's instructions.

Added `--llm-backend lmstudio|mlx` and saved the backend in the LLM artifact.
LM Studio retains native JSON schema output. MLX uses Pydantic AI's prompted
schema output, the same typed parsing, retry and quote grounding, and its
`chat_template_kwargs.enable_thinking` setting. Older artifacts default to
LM Studio. Endpoint, model and backend are restored together on reload; changing
prediction flags does not silently replace a saved recipe. Logging now names
the selected model rather than always saying Gemma.

Added a locked `mlx` extra for Apple Silicon macOS. MLX libraries are skipped on
Linux/Windows, and macOS PyTorch resolves from PyPI rather than the CPU-only
index used elsewhere. This keeps the LLM installation independent of BERT.

Validation: all 40 tests pass with the BERT extra, including 21 focused LLM/CLI
tests covering actual HTTP serialization,
prompted versus native schema output, both thinking settings, malformed-output
repair, old artifact loading and a synthetic CLI fit → reload → held-out predict
without private files or a model server. Ruff and the five-source mypy check pass.
The transport tests use OpenAI's existing `httpx2` dependency, so a base checkout
does not need an unrelated optional package. Apple Silicon dependency selection
passes a macOS 14 target dry run; actual Metal/model generation is not tested
on this Linux machine. The installation guide is documented in the next increment.

## Mac MLX LM — clone-to-run guide (6 October 2026)

The implementation increment was committed and pushed as `ee16ec4`.
Updated README, installation, CLI and LLM guides with a short Apple Silicon
setup: clone, `uv sync --locked --extra mlx`, copy private exports and XML into
ignored `data/`, start or reuse the local server, fit a ten-demonstration prompt,
then reload it to predict held-out notes. Commands use the supplied existing
model through a portable home-directory path and port 8080; model selection
remains configurable. The guide explains which saved settings are restored,
how to supply the private schema, and how to copy an existing BERT artifact
for inference without further training. Extended the memoir at the end of README.

The delegated installation check used a fresh copy with no `data/`, `schemas/`
or `runs/`. Locked dependency setup, CLI help and `uv pip check` passed.
The base suite passed 38 tests with the optional BERT module skipped, and a
synthetic dummy fit → dump → reload → held-out predict succeeded with a generated
schema. A macOS 14 / arm64 dependency dry run selected mlx-lm 0.32.0, mlx 0.32.3
and mlx-metal 0.32.3 without PyTorch. Root verified all 40 tests with BERT,
including the mocked MLX CLI lifecycle. Real MLX/Metal generation needs the Mac.

Documentation checks passed for 37 shell examples (`bash -n`), Python snippets,
code fences, relative links and the documented `--llm-backend` help flag.

Private dataset analysis and preparation are stored only in ignored local
directories. Neither public increment contains private exports, task schema or
model artifacts. The user's existing CLI whitespace remains unstaged.

## Full inference evaluation — scoring and visibility (7 October 2026)

The requested Mac run reuses the saved BERT artifact without training or
altering its encoder, classifier, tokenizer or decoding. Checked the Mac fit
records before choosing the test set: it excludes every BERT fitting note and
LLM demonstration. Private split manifests store source hashes, ordered IDs
and selected clinician annotations; no private files enter public commits.

Extended nervaluate scoring with full-credit, same-label character overlap,
per-label scores and macro averages. A small strategy subclass removes its
default one-percent overlap threshold and prevents wrong labels from
consuming gold matches. Matching is one-to-one: exact predictions first, then
emitted order and closest same-label boundaries. Macro uses gold-supported
labels, saved in the report. The paper does not fully specify its ordering or
macro universe; this single holdout and the existing five-note model also
differ from its cross-validation and fine-tuning.

Added explicit BERT loading/progress messages and loading, prediction and
scoring timings. Predictions and old artifacts remain compatible. New scoring
cases cover positive overlap, wrong labels, duplicate predictions, exclusive
ends, macro averaging, failed-document denominators and empty entities. The
evaluation guide explains the shared holdout and SSH/tmux workflow.

Validation before transfer: all 46 tests pass, including the existing tiny-model
fit/save/reload and CLI lifecycle tests. Ruff and mypy pass for the modified
source. The mocked asynchronous HTTP tests required running outside the
network-restricted sandbox; BERT and CLI checks passed inside it. Verified that
the common heldout texts do not duplicate reserved fitting/demo texts. The Mac
checkout was clean and at the previous published commit before transfer.

## Full inference evaluation — Mac run and guide (7 October 2026)

The implementation increment was pushed as `1bb6c5c`. Transferred that exact
revision through a Git bundle, verified its prerequisite commit and fast-forwarded
the clean Mac checkout without requiring GitHub credentials there. All 46 tests
also passed on the Mac. Started prediction-only inference inside tmux, with
`caffeinate -i` preventing idle sleep while the process ran. No packages or model
weights were downloaded, and fitting was never called.

The full frozen holdout completed without inference failures. Verified every
ordered prediction ID and selected doctor annotation against the private split
manifest. Independently reproduced exact micro and per-label scores with
per-document span/label multisets. All artifact files, including the weights,
have identical before/after checksums. Copied the result files and provenance
back into ignored local runs; the private report contains scores, timings and
limitations. It evaluates the existing classifier, rather than retraining BETO
or recreating the paper's cross-validation. No LLM inference was launched.

Updated CLI and method guides, the README metric description and final memoir
to describe overlap/macro scoring, timing and saved-weight inference. Checked
documentation links, code fences and Bash examples. Preserved the user's README
notes, installation example and CLI whitespace edits outside these commits.

The user then requested a small local replay to diagnose the poor model
output, preserving the no-training constraint. Replayed three heldout notes
with the original local artifact and compared three training notes with their
saved predictions. Every encoder tensor matches the cached pretrained weights;
the classifier loads with no missing keys, and its weights differ from seed-42
initialization. Tokenizer outputs and label mappings match their originals.
Feeding gold-derived tags through the decoder recovers nearly all entities;
the actual classifier remains weak. This isolates classifier quality as the
main observed issue without changing weights or masking the bad predictions.
Documented the difference between supervised BERT fitting and LLM prompting,
what encoder freezing does, and why base BETO alone does not supply a medical
classifier. All diagnostic data and outputs remain in ignored runs.

## Manim explanation — animated model and data flow (8 October 2026)

Added a standalone Spanish Manim lesson at `animations/bert_explained.py`,
with PEP 723 dependencies pinned to Manim 0.21.0. It runs without the harness,
private exports, private schema, model weights, external assets or LaTeX.
Its examples, vector coordinates, attention weights and probability changes
are synthetic. Rendering draws a simulation; it never trains or executes an
extractor. The paper protocol comes from the supplied manuscript, especially
sections 3.6 and 3.8, with missing fold IDs/seed and implementation details
explicitly distinguished from what it reports.

The user's visual direction led to a continuous explanation rather than a
slide sequence: text becomes subwords and BIO targets, tokens become vector
coordinates, particles show weighted attention and backward gradients, and
probability bars change as the illustrative loss falls. Twelve encoder layers
feed a shared classifier. The scene distinguishes the frozen-encoder default
from full fine-tuning, then physically moves 750 note points through five
independent 600/150 splits and pools their heldout predictions. Changing span
boundaries demonstrates strict versus full-credit overlap. A final LLM flow
distinguishes the paper's zero-shot prompt from our saved demonstrations.

The command supports quality, frame rate, timing scale and output directory;
it writes an MP4, a reading guide and chapter timestamps. Progress goes to
stderr. All rendered output stays under ignored `runs/`. Native dependencies
were downloaded/extracted into `/tmp` and Manim installed in an isolated
environment, leaving the system installation and BERT/LLM environment intact.

Validation for this increment: Ruff and compilation pass; complete low-quality
renders cover every scene, and chapter frames were inspected for alignment,
readability and object cleanup. The documented `uv run --script` entry point
and its isolated dependency environment were checked with cached packages.
The final 1080p render and the installation guide follow in the next increment.
No private data, schema, model or rendered result is part of this commit.

## Manim explanation — rendering guide and final verification (8 October 2026)

The animation implementation was committed and pushed as `d04bd34`. Added
`docs/animation.md` with Mac/Linux native prerequisites, isolated uv execution,
quality/fps/pace options, generated files and the distinction between the
paper's transformer cross-validation and its rules/zero-shot LLM evaluations.
Linked it from the README and appended this work to the final memoir.

The visual polish replaces caption morphs with crossfades, adds reading time
around tokenizer/attention explanations, and ties the animated loss value to
the current gold-tag probability. Thus `loss = -log(p_gold)` remains true during
the transitions as well as at their endpoints. Text-based decimal counters
avoid requiring LaTeX. The source remains fully synthetic and standalone.
Moved the entity-type legend away from the emerging BIO tags and realigned its
character-span bracket when the words separate into subwords.

Rendered the complete final lesson through its PEP 723 uv entry point, using
cached dependencies in this network-restricted environment: H.264 MP4,
1920×1080, 30 fps, 7,739 frames and about 4 minutes 18 seconds.
Verified all twelve ordered chapter timestamps, inspected the final BIO,
attention, training, fold, pooling and scoring frames, and decoded the entire
video with FFmpeg without errors. Ruff and Python compilation pass; relative
documentation links and Bash examples were checked. Generated videos, frames,
reading notes and timestamps remain in ignored `runs/animations/`.

The user's existing README note, installation example and CLI whitespace
changes are preserved and excluded from these commits. No extractor code,
dataset, private schema or model weights changed, and no fitting was run.
