# Full zero-shot evaluation on a Mac with MLX

The saved Gemma result is a five-note test with five demonstrations. The
789-note evaluation belongs to the saved DistilBERT classifier. There is no
full Gemma or Qwen evaluation saved in this checkout.

`evaluate_zero_shot.py` provides an instruction-only, entities-only run through
the existing LLM adapter. It uses zero demonstrations and does not train model
weights. It freezes the prompt, schema, selected clinician gold, data hashes,
model settings and harness code. Each completed document is saved immediately;
the same command resumes from those files without repeating completed requests.

## Start the server

Use a checkout containing `evaluate_zero_shot.py`, and copy your private
`data/grupo1_2_799.json` and `data/label_studio_config.xml` onto the Mac. From
the repository root in terminal 1:

```bash
uv sync --locked --extra mlx
EXTRACTOR_MODEL="$HOME/Models/Qwen3.8-27B-bf16"
uv run --extra mlx python serve_mlx.py \
  --memory-report runs/mlx_zero_shot_789_server/memory.json \
  --model "$EXTRACTOR_MODEL" --host 127.0.0.1 --port 8081
```

Replace that model path with the existing MLX model you want to evaluate. If
the server is already running with that model, reuse it. Server startup and
API routes are described in the [official MLX LM server guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).

## Check three requests, then finish the same frozen run

In terminal 2, from the repository root:

```bash
EXTRACTOR_MODEL="$HOME/Models/Qwen3.8-27B-bf16"
curl --max-time 10 http://127.0.0.1:8081/v1/models

EXTRACTOR_ARGS=(
  --data data/grupo1_2_799.json
  --task-spec data/label_studio_config.xml
  --model "$EXTRACTOR_MODEL"
  --endpoint http://127.0.0.1:8081/v1
  --memory-report runs/mlx_zero_shot_789_server/memory.json
  --offset 10 --count 789
  --output runs/mlx_zero_shot_789
)

uv run --extra mlx python evaluate_zero_shot.py \
  "${EXTRACTOR_ARGS[@]}" --max-documents 3
```

Inspect `runs/mlx_zero_shot_789/progress.json` for failures and the first three
saved receipts in `documents/` for predictions, warnings and seconds per note.
This is a request/format check, not a prompt-tuning step on the test data.
The complete test selection and prompt are already frozen. Final metrics are
written only after all selected documents finish.

Keep terminal 1 running. Finish in terminal 2:

```bash
caffeinate -i uv run --extra mlx python evaluate_zero_shot.py \
  "${EXTRACTOR_ARGS[@]}"
```

For an SSH session, run the server and evaluation inside `tmux` so they survive
disconnection. `caffeinate` prevents idle sleep. If interrupted, rerun the same
command: the three probe notes and any later completed notes are reused.

## Results and experiment settings

The common BERT test contains **789** notes, not 729: the prepared cohort has
799 notes after excluding invalid task 400, and IDs 1–10 are reserved. Using
`--offset 10 --count 789` preserves that comparison set. To run all 799 prepared
notes, choose a separate output directory and use `--offset 0 --count 799`;
that is a different evaluation cohort.

Output files:

- `artifact/`: saved instructions, task definition, model settings and empty demonstrations.
- `manifest.json`: frozen recipe, zero demonstration count, test IDs and provenance.
- `documents/`: one atomic receipt per completed document, with prediction and latency.
- `progress.json`: completed, target and failed-document counts.
- `performance.json`: cumulative times, median/p95 latency, provider-reported
  input/output tokens (including repair requests), coverage and end-to-end throughput.
- `summary.json`: final accuracy and performance summary.
- `predictions.json`, `metrics.json`, `config.json`: complete results after the full run.

Metrics include strict/overlap entity micro and macro scores and per-label
results. Failed requests remain as empty predictions in the denominator and
produce a nonzero exit status. Resuming preserves failed requests too; it does
not retry them selectively. To evaluate a changed model, prompt or setting,
choose a new output directory. The runner rejects changed data, code, recipe or
saved prompt rather than combining different experiments. The server's external
model weights are not hashed; keep the same weights throughout a resumed run.

The server wrapper samples its own MLX active/cache/peak memory and process
peak RSS every five seconds. `memory.json` holds the latest snapshot and
`memory.jsonl` its history. These describe the server lifetime including model
loading, rather than memory attributed to each individual note. Receipts retain
the memory snapshot observed after each request. Memory is not inferred from
the separate evaluator process. See the [MLX memory API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html).

Token totals use the provider's usage reports, including successful repair
requests. If usage is unavailable, it remains null and the performance report
counts notes without usage. Exhausted output or transport errors may lack usage,
so totals cover only reported tokens. Tokens/second is end-to-end throughput
over notes with usage, including request and parsing time; it is not isolated
model decode speed. Resuming does not double-count tokens or times from cached
notes. Logs and progress contain no extracted clinical text, while saved
prediction receipts do and remain in ignored `runs/`.

The default backend is `mlx`, temperature is zero, thinking is requested off,
output is limited to 4096 tokens, and each HTTP request has a 300-second timeout.
These are independent from the server's model context limit. Use `--timeout`
or `--max-output-tokens` when creating a different run. MLX receives the output
schema in the prompt, and the existing adapter validates it and allows one
output-repair retry. The model's chat template determines whether its thinking
control is honored.

`--instructions PATH` adds label glosses or annotation guidelines, but no labeled
demonstrations. For LM Studio, use `--backend lmstudio` and its endpoint/model.
The artifact belongs to this runner; use this script for resumption.

This measures a new zero-shot baseline on our current schema and cohort. It
does not reproduce the paper's 750-note cohort, Qwen3.5 model, 46-label prompt
or sequential quote-matching convention. See [evaluation.md](evaluation.md)
for span matching and macro policies.
