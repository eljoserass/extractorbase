# RuleChef

RuleChef uses an LLM to synthesize and refine extraction rules during fitting.
Inference executes the saved rules locally, with no model call or credentials.
The adapter implements `Method[RuleChefArtifact]` and supports regex and Spanish
spaCy patterns. It does not execute generated Python rules or predict relations.

## Install and run

```bash
uv sync --locked --extra rulechef --extra spacy
```

The spaCy extra is needed for the default mixed-format ablation, including
`blank:es`; regex-only use needs just the RuleChef extra.

The existing CLI selector supports BERT, LLM and dummy. RuleChef currently uses
its Python interface or the experiment runners below.

For the five-round, label-aware ablation, with an authenticated Codex CLI:

```bash
uv run --extra rulechef --extra spacy python experiments/rulechef_ablation.py \
  --output runs/rulechef_ablation_new --stage prepare --model gpt-6.1-sol
uv run --extra rulechef --extra spacy python runs/rulechef_ablation_new/source/experiments/rulechef_ablation.py \
  --output runs/rulechef_ablation_new --stage run
```

The default protocol repeats a 20-note learning budget five times. Initial
prompting and train refinement use all 20; holdout variants use 16 train and
4 dev. Each round evaluates the final artifacts on its other 779 notes. These
are repeated few-shot rounds, not disjoint five-fold tests. The runner also
supports ordinary external folds with `--split-mode fivefold`.

The [experiment guide](rulechef-experiments.md) describes the configurations,
prompts, data separation, background execution, telemetry and restart limits.
It also explains the distinction between upstream methodology and our clinical
adaptation. The experiment inputs under `data/` are supplied privately.

To repeat the older fixed train-10/test-789 protocol:

```bash
uv run --extra rulechef python experiments/evaluate_rulechef.py \
  --output runs/rulechef_train10_test789_new
```

## Python interface

This example assumes already constructed `TrainingExample`, `TaskSpec` and
`DocumentInput` objects:

```python
from pathlib import Path

from methods.rulechef import RuleChefConfig, RuleChefMethod

method = RuleChefMethod(RuleChefConfig(
    model="gpt-6.1-sol",
    rule_formats=("regex", "spacy"),
    spacy_model="blank:es",
    synthesis_strategy="per_class",
    refinement_iterations=3,
    trace_directory="runs/example/synthesis",
))
artifact = method.fit(train_examples, task_spec, dev_examples=dev_examples)
method.dump(artifact, Path("runs/example/artifact"))

local = RuleChefMethod()
artifact = local.load(Path("runs/example/artifact"))
predictions = local.predict(artifact, inputs)
```

`TaskSpec.instructions` and `RuleChefConfig.learning_instructions` guide the
extractor. Domain-specific clinical instructions belong in the experiment or
project definition. `refine` takes a copy of an initial artifact for paired
experiments; the product's general user-feedback interface remains future work.

## Learning and runtime

RuleChef supplies the synthesis and patch prompts, parsing, rule execution,
merging, error sampling and optional agentic coordinator. Our adapter adds:

- Exact character-offset evaluation and per-rule validation statistics.
- Explicit dev separation before synthesis, without silently reverting to train.
- Spanish runtime selection and guidance about regex, token matching and FUZZY.
- Provider transport, provenance and Label Studio prediction conversion.

The local executor also resolves a capture inside a zero-width lookaround using
the regex match's capture span. Upstream 0.2.2 otherwise emits an empty full-match
span for that case. Ordinary native results and explicit offsets are preserved;
invalid outputs still fail validation. No installed dependency is patched.

`blank:es` provides a Spanish tokenizer without trained weights and supports
lexical/FUZZY patterns. `es_core_news_sm` or `es_core_news_lg` must be installed
explicitly when trained lemmas, POS or dependencies are needed. Generic spaCy
NER is disabled. A mixed-format fit may still generate only regex rules.

New regex-only artifacts avoid loading a spaCy model at inference. Older
artifact formats remain loadable. spaCy artifacts record their pipeline and
reject a mismatching installed model version. Native RuleChef conflict
resolution is retained: entities with identical start/end compete across labels.

## Model access and isolation

The default backend translates completions into Codex CLI calls. A model must
be explicit or resolvable from the user's Codex configuration; there is no
silent fallback to another model. The Linux backend requires bubblewrap and
isolates every call with a filesystem allowlist, fresh authentication-only
state and disabled tools. Event auditing rejects attempted tool execution.

For a compatible hosted or local endpoint, use
`RuleChefConfig(backend="openai", model=..., endpoint=...)`. The SDK uses normal
environment authentication; a local server without authentication can use a
placeholder key. This transport sends text requests and does not need Codex,
OpenCode or bubblewrap. Local MLX compatibility still needs a live Mac test.

Traces contain training text and belong in ignored run directories. Artifacts
record learning configuration, IDs/hashes, rules and usage; credentials are not
serialized. Inference receives text-only inputs and uses no synthesis client.
Codex completion-token limits are not enforced; metadata records that limit.

## Results, evidence and validation

The [archived pilots](rulechef-pilots.md) retain the measured ten-note results.
The [current ablation results](../results/rulechef-ablation.md) publish global
metrics and resource measurements, with pending configurations marked explicitly.
The [code audit](rulechef-paper-audit.md) explains the supplied human rules,
tokenizer/FUZZY behavior and observed gaps. Results from an unfinished ablation
must be labeled partial; reports under `runs/` update as cells finish.

Synthetic tests cover offsets, FUZZY without weights, serialization, data
separation, acceptance, native coordinator responses and the experiment runner.
`experiments/rulechef_probe.py` remains an offline upstream integration probe
using fake completions, rather than a clinical benchmark.

Upstream: [paper](https://arxiv.org/html/2607.01293v1),
[source](https://github.com/KRLabsOrg/rulechef),
[Apache-2.0 license](https://github.com/KRLabsOrg/rulechef/blob/0af49211cff3ed4e85419b43bab0184a11f8a846/LICENSE).
