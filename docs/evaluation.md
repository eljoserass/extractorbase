# Evaluating an existing artifact

Use `--stage predict` to evaluate saved weights without fitting again. `--stage
run` performs fitting first. A base BERT encoder does not supply the project's
entity classifier: the artifact must already contain that fitted classifier.

## Freeze the held-out notes

Read `fit.json` in every artifact you want to compare. Reserve every document
used for BERT fitting or LLM demonstrations before evaluating either method.
The CLI rejects texts found in the selected artifact's fitting record;
reserving the union across methods is the runner's responsibility.

If those examples occupy the first `N` records, use `--offset N` and `--count`
equal to the remaining count. Offsets are JSON array positions, not task IDs.
Save the ordered test IDs, selected annotation IDs, source checksum and
exclusions in a local split manifest under ignored `data/`. Keep the split
fixed after seeing scores.

After setting these variables for your local data:

```bash
uv run --extra bert python cli.py --stage predict \
  --data "$EVALUATION_DATA" --task-spec "$TASK_SCHEMA" \
  --offset "$RESERVED_COUNT" --count "$TEST_COUNT" \
  --load "$BERT_ARTIFACT" --output runs/bert_heldout/evaluation
```

If the environment also hosts MLX LM, use `--extra mlx --extra bert` to retain
both sets of dependencies. BERT needs no server. The current adapter runs on
CPU and logs weight loading and completed document counts. Prediction uses
inference mode and does not update weights.

## Read the scores

| `metrics.json` field | Meaning |
|---|---|
| `strict_micro` | Identical label, start and exclusive end; counts pooled across documents. |
| `overlap_micro` | Same label and any positive character overlap; full credit per match. |
| `strict_macro`, `overlap_macro` | Unweighted averages of per-label precision, recall and F1 over labels with gold support. Macro F1 averages label F1 values. |
| `per_label`, `per_label_overlap` | Both scoring rules for all observed gold or prediction labels. |
| `macro_labels` | Exact labels included in macro averaging. |
| `documents`, `failed_documents` | All evaluated notes and failures; failed notes remain in the denominator. |

Scoring uses nervaluate's one-to-one greedy matching. Exact predictions come
first. Remaining overlap predictions retain their emitted order and match the
remaining same-label gold span with the closest boundaries. Label Studio's
exclusive ends are converted to inclusive ends; adjacent spans do not overlap.
A small strategy adapter accepts even one overlapping character and prevents
wrong labels from consuming gold spans. Duplicate predictions cannot claim
one gold entity twice. Gold and prediction spans are not rewritten.

These overlap scores give full credit; nervaluate's separate `partial` metric
gives half credit and is not reported here. The supplied paper's exact greedy
ordering and macro label policy are not fully specified. Those choices, cohort
differences and the saved model's training remain comparison limitations.
A single holdout is not the paper's pooled five-fold evaluation.

Relation metrics remain unavailable. The BERT adapter predicts entities only,
matching the type of task evaluated for the paper's transformer baseline. It
does not predict relation links or negation attributes.

`config.json` records fitting/test IDs, selected gold annotation IDs, source
hashes, package versions, and loading, prediction and scoring seconds.
`predictions.json` saves all predictions. `run_summary.txt` contains scores
and timings. Older artifacts remain loadable. Keep artifacts, predictions,
split manifests and clinical data in ignored `data/`, `schemas/` and `runs/`.

## Keep a remote run alive

On the remote machine:

```bash
tmux new -s bert-eval
# Run the prediction command above inside this session.
```

Detach with **Ctrl+B, then D**. To return after reconnecting:

```bash
tmux attach -t bert-eval
```

An already approved artifact needs no confirmation for prediction. An
unapproved artifact still requires approval; do not hide its prompt behind
log redirection. `tmux` survives SSH disconnections, but not a reboot.
