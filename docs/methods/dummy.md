# Dummy method: check the lifecycle with empty predictions

Dummy returns a successful empty extraction for every document. It has no model,
training loop or API call. Its purpose is to exercise the harness's data loading,
artifact handling, approval, prediction and evaluation with predictable behavior.
The implementation is [methods/dummy.py](../../methods/dummy.py).

See the [shared method contract](README.md) for the interfaces and the
[CLI guide](../cli.md) for stages and flags.

## Background: an empty extraction baseline

An extractor can legitimately find no entity in a document. The common output
format must represent that case, and evaluation must still count the document.
Dummy makes that behavior explicit: it emits the ordinary prediction envelope
with an empty `result` list.

On annotated documents, every gold entity is missed. The current evaluator
reports zero precision, recall and F1, with support equal to the number of gold
label/span targets. It also reports zeros when there are no entities at all;
that is the evaluator's chosen empty-case convention. On unlabeled inputs, the
CLI saves predictions without computing metrics.

A successful dummy extraction does not set `meta.extraction_error`. This differs
from a failed LLM request, which also has an empty result but carries an error.
Dummy therefore checks empty prediction handling without pretending a request
failed.

## What each operation does

### `fit`: retain the task definition

`fit` returns `DummyArtifact(task_spec)`. It does not inspect the training examples,
use their labels, or call the optional `dev_examples` or `scorer`. The artifact
stores the same task definition expected by the runner when reloading any method.

The Python method accepts an empty example sequence. The CLI still requires a
positive fitting count and valid selected gold examples, because its fitting
flow is common to all three methods.

### `dump` and `load`: exercise typed persistence

`dump` creates the destination directory and writes `method.json` using the
Pydantic `DummyManifest`:

```text
artifact/
└── method.json     # method="dummy" and the serialized TaskSpec
```

`load` validates that manifest and returns a new artifact with the restored task
definition. Keeping persistence even for an empty predictor lets the same
fit/dump/load/predict sequence be tested before weights or a server are involved.

There are no weights, tokenizer or prompt files. The CLI adds its own `fit.json`
and training-preview files alongside the method's manifest.

### `predict`: one result per input

For each `DocumentInput`, `predict` calls:

```python
prediction_for(document, [], "dummy")
```

The helper creates this shape, preserving input order, ID and text:

```json
{
  "id": "example-1",
  "data": {"text": "AR estable."},
  "predictions": [
    {"model_version": "dummy", "result": []}
  ]
}
```

The method does not omit documents just because they have no entities. The
artifact is accepted to satisfy the common interface, although an empty result
does not need to consult the schema's labels.

## Why keep this method?

| Choice | Rationale |
| --- | --- |
| Empty predictions instead of a fabricated entity | Gives an unambiguous outcome independent of text and labels. |
| Same abstract base class as BERT and LLM | Checks the common orchestration through an ordinary method implementation. |
| Typed manifest and restored `TaskSpec` | Exercises the artifact contract and method dispatch after reload. |
| Standard Label Studio envelope | Tests serialization and evaluation without a special output path. |
| No method configuration | There are no algorithm settings to tune. |

Dummy is useful for setup checks and regression tests. A passing run establishes
that the lifecycle works for a simple method; BERT and LLM still need their own
training/inference checks. Its zero F1 does not measure the quality of either
model, and approval does not give dummy extraction capability.

## Try the complete flow

Choose a new artifact directory and run from the repository root:

```bash
uv run python cli.py --stage run --method dummy \
  --data data/grupo1.json --offset 0 --count 5 --predict-count 5 \
  --dump runs/dummy_guide/artifact --output runs/dummy_guide/evaluation
```

This fits positions 1–5, saves and reloads the task definition, predicts empty
training results, prints their scores, and asks for approval. After answering `y`,
it predicts positions 6–10 and saves their empty results and evaluation.

For a scripted smoke test, append `--yes`. To exercise reloading in a separate
process afterward:

```bash
uv run python cli.py --stage predict \
  --data data/grupo1.json --offset 5 --count 5 \
  --load runs/dummy_guide/artifact --output runs/dummy_guide/reloaded
```

The second command uses the saved method name and task definition. Because the
artifact was already approved, it proceeds without another confirmation.

[tests/test_method.py](../../tests/test_method.py) checks dummy manifest reload.
[tests/test_cli.py](../../tests/test_cli.py) uses dummy to verify fitting the first
slice, predicting the next, approval/decline, rejecting reused training text and
handling unlabeled inputs.
