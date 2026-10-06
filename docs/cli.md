# CLI guide

Run commands from the repository root. The CLI reads Label Studio JSON exports,
fits or loads an extractor, prints predictions and scores, and saves results locally.
For cloning, uv installation, dataset setup and LM Studio, see the
[installation guide](install.md).

## Setup and help

```bash
# Dummy and LLM dependencies
uv sync --locked

# Include BERT dependencies when using that method
uv sync --locked --extra bert

uv run python cli.py --help
uv run python cli.py -h
```

In `uv run --extra bert python cli.py --method bert ...`, `--extra bert` makes
PyTorch, Transformers and Accelerate available through uv. `python cli.py` runs
the application; `--method bert` selects its BERT extractor. LLM and dummy commands
can use `uv run python cli.py ...`.

The private XML is supplied locally, not through Git. Put it at
`schemas/reumalago.xml` or select its local path with `--task-spec`; see
[schema setup](install.md#4-add-the-label-studio-exports). `data/` and `schemas/`
are ignored so exports and schema definitions stay local.

For LLM inference, start LM Studio's server and load `google/gemma-4-e2b`.
The default endpoint is `http://localhost:1234/v1`. BERT's first fit downloads the
multilingual DistilBERT checkpoint; later predictions load the saved weights.

## Stages

Select a stage with `--stage`. The default is `run`.

| Stage | What happens | Required flags besides `--data` |
| --- | --- | --- |
| `inspect` | Report annotation provenance, selected gold, label counts and validation issues for the entire loaded dataset. | None |
| `fit` | Fit, save and reload the extractor; predict the training texts; print the preview and scores; ask for approval; stop. | `--method`, `--dump` |
| `predict` | Load an artifact, ask for approval if still needed, predict the selected inputs, and score them when gold is supplied. | `--load` |
| `run` | Complete the fit/preview/approval flow, reload the artifact, then predict and evaluate the following slice. | `--method`, `--dump` |

The methods are `dummy`, `bert` and `llm`. Dummy predicts empty results without a
model or server. BERT trains a token classifier. LLM fitting saves a prompt and
the selected examples as demonstrations; it does not train Gemma's weights.
The [method guides](methods/README.md) explain the background, implementation
choices and limitations of each extractor.

Training-preview scores are resubstitution on fitting examples. Held-out scores
are computed separately after prediction. Predictions receive document IDs and
text; the runner keeps evaluation gold outside the extractor.

## Selecting samples: offset and count

**`--offset` is the zero-based position of the first task to select.** It counts
whole documents in the JSON array. Label Studio task IDs can be arbitrary; they
do not determine a task's position in that array.

**`--count` is how many tasks to select.** The slice is
`tasks[offset : offset + count]`, with its end excluded:

| Flags | Array indices | Positions in the file |
| --- | --- | --- |
| `--offset 0 --count 5` | 0–4 | First five documents |
| `--offset 5 --count 5` | 5–9 | Documents 6–10 |
| `--offset 10 --count 3` | 10–12 | Documents 11–13 |

For `fit`, these flags select training examples. For standalone `predict`, they
select prediction inputs. For `run`, they select training examples, and
`--predict-count` controls the size of the following prediction slice:

```text
run --offset 2 --count 3 --predict-count 4

Array index:       0  1 | 2  3  4 | 5  6  7  8 | 9 ...
Selection:        skip |   fit   |   predict   |
File positions:        |  3–5    |     6–9     |
```

| Stage | Default start | Size flag and default |
| --- | --- | --- |
| `inspect` | Entire dataset; slice flags do not limit inspection | Entire dataset |
| `fit` | Offset 0 | `--count 5` |
| `predict` | Offset 5 | `--count 5` |
| `run`: fitting | Offset 0 | `--count 5` |
| `run`: prediction | Fitting offset + fitting count | `--predict-count 5` |

Standalone `predict` always defaults to offset **5**. If you fitted ten examples,
set `--offset 10` yourself. If you load a separate file of new inputs, usually set
`--offset 0`. The CLI rejects prediction inputs whose text was used for fitting.

Tasks remain in export order. Repeating `--data` concatenates files in the order
given; offsets then refer to that combined list. Duplicate task IDs are rejected.
`--seed` controls BERT training, while sample selection stays sequential.
Offsets must be nonnegative, counts must be positive, and the whole requested
slice must exist. The CLI does not silently shorten an out-of-range slice.

## Examples

### Inspect annotation selection

```bash
uv run python cli.py --stage inspect \
  --data data/grupo1.json --data data/grupo2.json \
  --output runs/inspection
```

This prints an inspection report and saves `runs/inspection/inspection.json`.
It does not fit or call a model.

### Run Gemma on five training and five held-out examples

Choose an empty or new artifact directory for every fit:

```bash
uv run python cli.py --stage run --method llm \
  --data data/grupo1.json --offset 0 --count 5 --predict-count 5 \
  --endpoint http://localhost:1234/v1 --model google/gemma-4-e2b \
  --dump runs/gemma_example/artifact --output runs/gemma_example/evaluation
```

This fits positions 1–5, prints their preview, asks for approval, and then predicts
positions 6–10. `--no-thinking` is the default. Use `--thinking` during fitting
to preserve LM Studio's reasoning setting in the artifact.

### Fit BERT, then predict in another process

```bash
uv run --extra bert python cli.py --stage fit --method bert \
  --data data/grupo1.json --offset 0 --count 5 \
  --dump runs/bert_example/artifact

uv run --extra bert python cli.py --stage predict \
  --data data/grupo1.json --offset 5 --count 5 \
  --load runs/bert_example/artifact --output runs/bert_example/evaluation
```

The default fits the classifier head for three epochs with a frozen encoder.
Add `--full-finetune` to update the encoder too, or `--epochs N` to change the
training duration. Fitting settings are saved with the artifact.

### Continue from an existing Gemma artifact

If fitting already created `runs/gemma/artifact`, load that artifact to continue:

```bash
uv run python cli.py --stage predict \
  --data data/grupo1.json --offset 5 --count 5 \
  --load runs/gemma/artifact --output runs/gemma/evaluation
```

An unapproved artifact with a successful training preview shows the saved results
and asks for approval. An approved artifact proceeds without asking again. Its
method, schema, prompt and endpoint/model settings are restored from disk.
Passing new fitting settings to `predict` does not replace those saved settings.
If supplied, `--method` must match the saved artifact's method.

### Predict new, unlabeled text

A Label Studio input file may contain tasks like this:

```json
[
  {"id": "new-1", "data": {"text": "Texto clinico nuevo."}}
]
```

For that one-task file:

```bash
uv run python cli.py --stage predict \
  --data data/new_tasks.json --offset 0 --count 1 \
  --load runs/gemma/artifact --output runs/gemma_new
```

Predictions are saved as usual; `metrics.json` contains `null` because no gold
annotations were supplied. A prediction slice must be fully labeled or fully
unlabeled.

### Check the lifecycle without a model

```bash
uv run python cli.py --stage run --method dummy \
  --data data/grupo1.json --offset 0 --count 5 --predict-count 5 \
  --dump runs/dummy_example/artifact --output runs/dummy_example/evaluation \
  --yes
```

`--yes` explicitly approves a scripted run. Normal interactive examples omit it.

## Approval, warnings and scores

After fitting, the CLI saves and reloads the artifact before generating its
training preview. It prints entity/relation counts, example extractions, strict
global and per-label scores, and any extraction warnings. Then it asks:

```text
Approve this fitted artifact for prediction? [y/N]:
```

Answering `y` records approval in `fit.json`. Answering `n`, or accepting the
default no, aborts and leaves the artifact unapproved. A later `predict` can show
the saved preview and ask again. A preview with inference failures cannot be
approved, including with `--yes`; fit into a new directory after resolving them.

The `Predicting with reloaded ... on IDs ...` message appears after approval and
input validation, immediately before inference. An unapproved artifact first
prints its saved preview and waits for your answer. If stdout is hidden, that
preview and question are hidden too; see the troubleshooting steps below.

For Gemma, grounding warnings report omitted unsupported quotes/labels/relations
or corrected unique-quote occurrence indices. Valid mentions remain in the
results. API or unrepaired response failures save an empty result with
`meta.extraction_error`; those documents remain in scoring and produce a nonzero
exit. Warnings are saved as `meta.extraction_warnings`.

The primary score is strict entity micro precision/recall/F1: start, end and label
must match. Per-label scores are printed too. Relation predictions are retained,
but relation metrics are currently uncomputed. See the [README](../README.md)
for method limitations and dataset findings.

## Flag reference

### Data, stages and files

| Flag | Default | Use |
| --- | --- | --- |
| `--data PATH` | Required | Label Studio JSON export; repeat for multiple files. |
| `--stage STAGE` | `run` | `inspect`, `fit`, `predict` or `run`. |
| `--method METHOD` | Required for fitting | `dummy`, `bert` or `llm`; inferred from the artifact during prediction. |
| `--offset N` | 0 for fit/run; 5 for predict | First task position in the loaded dataset. |
| `--count N` | 5 | Fitting size, or standalone prediction size. |
| `--predict-count N` | 5 | Prediction size after fitting in `run`; unused in other stages. |
| `--dump PATH` | Required for fitting | Artifact destination; must be empty or new. |
| `--load PATH` | Required for predict | Existing artifact directory. `run` reloads its own `--dump`. |
| `--output PATH` | Artifact parent's `evaluation/` during prediction | Prediction/metrics destination; `inspect` optionally writes its report here. |
| `--task-spec PATH` | `schemas/reumalago.xml` | Label Studio XML vocabulary and control names used for fitting or inspection. Prediction uses the saved task definition. |
| `--instructions PATH` | None | Optional text/Markdown instructions saved with the fitted task definition. |
| `--yes` | Off | Approve without an interactive question after a successful preview. |
| `-h`, `--help` | — | Print help without requiring data. |

Prediction writes `predictions.json`, `metrics.json`, `config.json` and
`run_summary.txt` into its output directory. Reusing that output directory replaces
those result files. Choose separate output directories to preserve different runs.

### Gold annotation selection

| Flag | Default | Use |
| --- | --- | --- |
| `--gold-policy POLICY` | `clinician` | `clinician`, `latest` or `ground-truth`. |
| `--reviewer-id N` | Reviewers 3 and 4 | Repeat to supply your own set of reviewer IDs. |

`clinician` prefers annotations created by the configured reviewers. If none
exists, it falls back to an original updated by a reviewer and warns. `latest`
selects the latest update; `ground-truth` selects among explicitly flagged
annotations. Cancelled annotations are excluded. No matching annotation or an
ambiguous timestamp tie produces an error.

For this dataset, original automated annotations carry the ground-truth flag;
the default clinician policy is intended to select the doctor's revision.
Inspect and review the reported provenance before changing the policy.

```bash
uv run python cli.py --stage inspect --data data/grupo1.json \
  --gold-policy clinician --reviewer-id 3 --reviewer-id 4
```

### BERT fitting

| Flag | Default | Use |
| --- | --- | --- |
| `--checkpoint NAME` | `distilbert/distilbert-base-multilingual-cased` | Hugging Face checkpoint or local checkpoint directory. |
| `--epochs N` | 3 | Positive number of training epochs. |
| `--freeze-encoder` / `--full-finetune` | Frozen encoder | Train the classifier head only, or the whole model. |
| `--seed N` | 42 | BERT training randomness; it does not shuffle the selected tasks. |

### LLM fitting

| Flag | Default | Use |
| --- | --- | --- |
| `--endpoint URL` | `http://localhost:1234/v1` | LM Studio's OpenAI-compatible endpoint. |
| `--model NAME` | `google/gemma-4-e2b` | Model identifier served by LM Studio. |
| `--thinking` / `--no-thinking` | Thinking disabled | Preserve the server's reasoning setting, or disable reasoning per request. |

The saved artifact controls these settings when reloaded. Its LLM model must
remain available in the configured server.

Typer also provides `--show-completion` and `--install-completion` for shell
completion. These are separate from extractor stages.

## Saved files

```text
runs/example/
├── artifact/
│   ├── method.json                 # Method settings and task definition
│   ├── fit.json                    # IDs, hashes, provenance and approval
│   ├── training_predictions.json   # Training preview in Label Studio format
│   ├── training_metrics.json
│   ├── model/                      # BERT weights and tokenizer
│   ├── prompt.txt                  # LLM prompt
│   └── examples.json               # LLM demonstrations
└── evaluation/
    ├── predictions.json
    ├── metrics.json
    ├── config.json
    └── run_summary.txt
```

Method-specific files appear only for the relevant method. `inspect` optionally
saves `inspection.json`. Input data and generated artifacts stay local; `data/`
and `runs/` are ignored by Git.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Missing `--data` | Supply an export, or use `-h`/`--help` to see usage. |
| Artifact directory is not empty | Use `--load` with `--stage predict` to reuse it, or choose a fresh `--dump` for fitting. |
| Requested slice exceeds the dataset | Adjust offset/count; standalone prediction defaults to offset 5. |
| Prediction contains fitting documents | Select different inputs; matching training text is rejected even under a new task ID. |
| Slice mixes labeled and unlabeled tasks | Select one kind of input for that prediction call. |
| Training preview has inference failures | Review `training_predictions.json`, fix the reported cause, and fit again into a new directory. |
| LLM API/model error | Check LM Studio's server, model identifier and the endpoint saved in `method.json`. |
| Prediction-start message does not appear | An unapproved artifact first needs confirmation; make sure its saved preview and prompt are visible. |
| Help/results are blank but errors are visible | Check stdout redirection in the shell, as below. |

Check the terminal's two output streams:

```bash
printf 'stdout\n'
printf 'stderr\n' >&2
```

If only stderr appears, restore stdout in that interactive terminal:

If the CLI is waiting at an invisible prompt, press Ctrl+C first. Then run:

```bash
exec 1>/dev/tty
uv run python cli.py --help
```

For a single diagnostic command, `uv run python cli.py --help 1>&2` sends help
to stderr. This output-stream issue affects ordinary shell commands too.
