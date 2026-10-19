# RuleChef: integration and pilot

## Running the implemented method

The adapter is in `methods/rulechef.py`. It implements `fit`, `dump`, `load`
and `predict`; it does not modify the installed RuleChef source, the shared
evaluator or the existing CLI. Install its optional dependency with:

```bash
uv sync --extra rulechef
```

With the Codex CLI authenticated, run the fixed comparison protocol:

```bash
uv run --extra rulechef python experiments/evaluate_rulechef.py \
  --output runs/rulechef_train10_test789_new
```

This pilot runner requires the existing local files under `data/`: the prepared
dataset, holdout manifest, Label Studio schema and label glosses. These private
inputs and the generated `runs/` artifacts are excluded from Git. The method
itself accepts ordinary `TrainingExample` / `TaskSpec` inputs and its synthetic
tests require none of those private files.

The runner uses IDs 1–10 for learning and exactly the 789 test IDs and clinician
annotation policy in `data/common_holdout_789_manifest.json`. It provides the
schema and existing label glosses to the learner, but no existing extractor's
rules or predictions. There is no dev set: all refinement feedback is training
resubstitution. This is a pilot using upstream RuleChef's bulk regex synthesis
and up to three patch rounds, rather than a reproduction of every paper setting.
Rule selection optimizes exact micro F1; the final unchanged evaluator reports
strict/overlap micro/macro and per-label scores. No relation extraction is added.

Two small learner overrides live inside the adapter: dataset evaluation and
per-rule quality stamping request upstream's existing `mode="exact"`. There
are no global monkeypatches. The native executor's conflict policy is retained:
entities with identical start/end compete even when they have different labels.

The default synthesis backend is a completion bridge to `codex exec`; it uses
the model and reasoning effort from the user's Codex configuration. Each request
runs outside the repository, independently, with shell/apps/web/delegation
disabled. Event auditing rejects completions that used tools. Raw requests,
responses, events, token usage and wall times are saved under `synthesis/` when
a trace directory is configured. They contain training text and belong in the
ignored run directory. The CLI does not expose the same completion-token limit
as the OpenAI client; metadata explicitly records that the upstream requested
limit was not enforced. Provider model memory and monetary cost are not measured.

An OpenAI-compatible client may instead be injected into `RuleChefMethod`, or
configured with `RuleChefConfig(backend="openai", model=..., endpoint=...)`.
It uses the OpenAI client's normal environment authentication. The Codex backend
uses the existing CLI login. No credentials are serialized into the artifact.

The initial pilot did **not** use the bubblewrap filesystem allowlist employed
in the HardSpaCy agent experiment. A temporary working directory and Codex's
`read-only` mode do not hide the host filesystem. The CLI also reused the host
Codex state/auth directory, despite starting fresh ephemeral threads. All four
event logs contain only an agent response and no tool calls; explicit RuleChef
requests contain no full held-out note texts, prior extractor or test-file paths.
This is evidence against test access through model tools, but not an equivalent
OS isolation guarantee or a complete audit of automatically assembled CLI
context. `isolation_audit.json` records the retrospective checks and their limits.

The runner freezes `artifact/artifact.json` before spawning a fresh test process.
That worker only loads rules and executes regexes; it creates no model client.
Outputs include `metrics.json`, `predictions.json`, `failures.json`, `report.md`,
training scores, fit/test resource measurements and reproducibility hashes.
Fit resources include Codex-reported tokens, Python peak RSS and child-process
peak RSS; remote model memory is unavailable. Inference tokens/model calls are
zero. Existing results are never overwritten. An interrupted completed fit can
be evaluated using the same output directory and `--stage test`.

Synthetic contract tests in `tests/test_rulechef.py` cover exact-position
refinement, per-rule precision, Unicode/repeated mentions, serialization without
a model client, and rejection of malformed predicted offsets.

## Completed pilot: 19 October 2026

Results are saved in `runs/rulechef_train10_test789_2026-10-19/`. Training
examples, test IDs, selected clinician annotation IDs and scoring were checked
against the HardSpaCy comparison; test IDs/annotations also match the Qwen run.
All 789 notes were scored, with no failed documents and 14,834 gold entities.

| Method | Strict micro F1 | Strict macro F1 | Overlap micro F1 | Overlap macro F1 |
|---|---:|---:|---:|---:|
| RuleChef / Codex | 86.71% | 80.30% | 90.66% | 87.66% |
| HardSpaCy | 79.88% | 61.58% | 81.29% | 62.76% |
| Qwen zero-shot | 71.09% | 76.17% | 75.90% | 84.53% |

Synthesis used the configured `gpt-6.1-sol` model with `xhigh` reasoning.
The initial 52 regex rules cover all 42 allowed labels. Their training exact
micro F1 is 95.39% (207 TP, 11 FP, 9 FN). All three repair candidates were
rejected: training F1 was 93.24%, 92.99% and 93.24%, respectively. The upstream
repair loop repeats the same prompt after each rejection in this configuration;
the frozen artifact therefore contains the original 52 rules. No human rule
edits or test-driven changes were made. `learning_trajectory.json` records this
training-only audit, including agreement with the project's exact scorer.

Fit took 1,305.95 seconds (21m46s), across four model calls: 179,540 input tokens
and 63,075 output tokens, including 50,016 reasoning tokens. No tool calls were
used by the synthesis model. Fresh-process test inference took 1.130 seconds on
the Linux CPU with zero model calls/tokens; process peak RSS was 184.9 MiB.
Artifact loading took 0.00056 seconds and scoring 0.311 seconds; these timings
exclude process startup/imports and data loading. Provider model memory and
monetary cost are unavailable. `fit_resources.json` and `test_resources.json`
contain the measurements and artifact hashes.

The previous agent produced 16,278 output tokens, including 5,936 reasoning
tokens; RuleChef produced 63,075 output tokens, including 50,016 reasoning
tokens. Both used `gpt-6.1-sol` / `xhigh`. RuleChef used four independent CLI
requests with no reported input cache hits. Its initial synthesis took 519.8s;
the three rejected repairs added 785.9s. Repair prompts were each about 102,380
characters and identical across rounds. The CLI bridge did not enforce the
upstream requested completion-token limits. These execution settings and
unproductive repeated repair attempts account for the expensive pilot; 21m46s
is not an inherent requirement for applying or learning rules with RuleChef.

An independent strict multiset check gives 12,264 TP, 1,190 FP and 2,570 FN.
The largest missed-entity counts are numeric values (365), FR (208), erosive
disease (181), ankylosing spondylitis (176), and azathioprine (164); these are
strict errors and include boundary errors as well as entirely missed mentions.
Per-label/per-document errors are in `error_summary.json`. `example_11.json`
contains the first test note, gold and predictions, chosen without cherry-picking
(10 exact TP, 1 FP, 2 FN). The report and complete predictions are alongside it.
This is one exploratory run, with different synthesis/inference models across
methods; it is not cross-validation or a reproduction of the paper's benchmark.

## Initial source inspection

Inspected on 19 October 2026: PyPI `rulechef==0.2.2` and upstream commit
`0af49211cff3ed4e85419b43bab0184a11f8a846`. The package is installed in an
isolated environment at `/tmp/extractorbase-rulechef-env`; the repository is
cloned at `/tmp/extractorbase-rulechef-research`. That initial inspection did
not modify project dependencies or existing extractors; the implemented adapter
now adds the optional `rulechef==0.2.2` dependency.

Sources: [paper](https://arxiv.org/html/2607.01293v1),
[repository](https://github.com/KRLabsOrg/rulechef),
[license](https://github.com/KRLabsOrg/rulechef/blob/0af49211cff3ed4e85419b43bab0184a11f8a846/LICENSE).
The paper was downloaded and read, including its synthesis/refinement prompts.
The inspected executor, learner and pipeline files match the installed package.
The upstream engine additionally has a traffic-export helper absent from the
PyPI build despite reporting the same version number.

## What it learns

RuleChef uses an LLM during fitting to propose regex, Python functions or spaCy
patterns. It packages each rule with its output template, priority and measured
quality. Learning can run per observed class, use positive examples and
counterexamples, incorporate grex pattern suggestions, and generate patches
from clustered failures. A simple or LLM coordinator can guide iterations;
ranking/pruning removes rules that do not help. Prediction can run rules alone.

The paper evaluates primarily regex rules, with Kimi models and English TAB /
Banking77 data. Its format-type and semantic-type scores are distinct; its
78.7% format-type F1 is not a result on our clinical labels. More refinement
does not consistently improve every class group in its ablations.

## Adapter design

`methods/rulechef.py` implements `Method[RuleChefArtifact]`; the original design
below includes future development-set support, which the initial pilot rejects
explicitly rather than silently mixing into synthesis:

| Operation | Project behavior |
|---|---|
| `fit(train_examples, task_spec, dev_examples=..., scorer=...)` | Convert selected gold to RuleChef NER entities, define all allowed labels/glosses, synthesize candidates, refine with an explicit development protocol, and select rules with the intended score. No test documents enter the learning dataset. |
| `dump` | Export rules and their validation statistics, task/schema metadata, package/source version, learning settings, provenance and usage. Export runtime state separately from the dataset store, which also persists examples. |
| `load` | Restore the frozen rules and task definition. An executor alone can be reconstructed without an LLM client or key. |
| `predict` | Apply rules to `DocumentInput.text`, validate labels/character coordinates, and convert entities to Label Studio results. Keep `llm_fallback=False`. Score with our unchanged evaluator. |

Start with entities-only regex rules for a direct test of the paper's main
approach. Python and Spanish spaCy execution can be additional experiments.
Do not reuse the ten-note artifact's test results as feedback for patching.
Development data must be reserved explicitly; the historical 789-note cohort
is already exploratory, so a new patient-grouped test would support a stronger
generalization claim.

## Findings that affect implementation

These were checked against code and synthetic examples, rather than inferred
from the README:

1. `learn_rules` defaults to `holdout_fraction=0.0`. With ten examples and a
   20% requested holdout, the split returns no dev set because it requires at
   least five dev examples. Many small NER strata can also leave no dev set.
2. Initial synthesis runs on `chef.dataset` before the refinement train/dev
   split. Passing one combined train/dev dataset does not keep development
   examples out of the initial synthesis prompt. Explicit separation needs
   orchestration or a small upstream adaptation.
3. `RuleLearner._evaluate_rules` hardcodes `mode="text"`. Setting
   `Task.matching_mode="exact"` does not change that path. A rule extracting
   the first `PCR` in `PCR PCR`, when gold marks the second, scores 1.0 in the
   refinement evaluator and 0.0 with exact offsets. Align acceptance/ranking
   with our scorer, not just the final report.
4. The executor drops an entity when another rule has already emitted the
   same start/end, regardless of type. Our gold can retain multiple labels for
   one span. Decide and record the adapter's conflict policy before comparing.
5. spaCy execution lazily loads/downloads `en_core_web_sm`. `rulechef[spacy]`
   installs the library, not a Spanish pipeline. Spanish token rules need a
   deliberately selected tokenizer/model; dependency patterns need the
   corresponding trained components.
6. Several synthesis calls use Chat Completions parameters directly, while
   other paths use `LLMCallManager`. Test every used path when adapting token
   parameters, JSON output and thinking controls for a local provider.

These findings do not prevent an adapter. They mean a four-method wrapper alone
does not establish a comparable learning/evaluation protocol.

## Model access and credentials

| Generator | Credentials and integration |
|---|---|
| Hosted OpenAI-compatible API | Provider API key, endpoint and model. Credentials belong to the fitting configuration, not the exported rules. |
| Local MLX / LM Studio | Local compatible endpoint and model. A nonempty placeholder key satisfies the OpenAI client when the server does not authenticate. Validate token/JSON/thinking behavior. |
| Codex | A custom client bridge can translate RuleChef's `chat.completions.create` calls into Codex CLI/SDK tasks and return their final JSON in the expected response shape. This is not a built-in RuleChef backend. |

The installed Codex CLI reports `Logged in using ChatGPT`. Official
[authentication](https://learn.chatgpt.com/docs/auth) documentation distinguishes
ChatGPT subscription access from API-key access. Official
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
supports `codex exec`, structured final output, event logs and usage reporting;
the [SDK](https://learn.chatgpt.com/docs/codex-sdk) is another supported
programmatic integration. A bridge can use that supported interface and its
configured authentication; the normal OpenAI API still uses Platform keys.
The bridge is now implemented inside the method; the pilot records all live calls.

## License

The library is Apache-2.0. Its license permits commercial use, modification and
distribution, subject to its redistribution conditions, including retaining the
license and applicable notices and marking distributed modifications. Model,
provider and dependency terms are separate. The paper is CC BY 4.0.

## Reproducible offline probe

```bash
uv run --no-project --python .venv/bin/python experiments/rulechef_probe.py \
  --output runs/rulechef_probe_new
```

The script declares `rulechef==0.2.2` using PEP 723, so it leaves the project's
environment and lockfile alone. It uses synthetic PCR text and a fake client
returning two predefined rules; it does not assess LLM synthesis quality.
It exercises the synthesis/parser plumbing, exports JSON, reloads the rules
into `RuleExecutor` without a client, verifies Unicode and repeated-quote
offsets, and checks the holdout/position/conflict behaviors above. Real API calls,
clinical data reads and inference model calls are all zero.

The checked result is in
`runs/rulechef_research_2026-10-19/verified_probe/probe.json`. This probe remains
synthetic and is separate from the clinical pilot.
