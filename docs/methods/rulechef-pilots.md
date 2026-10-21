# RuleChef: archived ten-note pilot results

These are the frozen 19–20 October 2026 experiments, before the five-round
20-note ablation. For the current implementation and commands, use
[the method guide](rulechef.md); for the ablation protocol, use
[the experiment guide](rulechef-experiments.md).

All numbers below are exploratory, measured on the same 789-note cohort.
Private predictions, prompts and artifacts remain under the ignored `runs/`.

## Completed mixed-format pilots: 20 October 2026

The two sequential runs are in
`runs/rulechef_mixed_spacy_sm_train10_test789_2026-10-20/` and
`runs/rulechef_mixed_spacy_lg_train10_test789_2026-10-20/`. Both allow regex and
spaCy and use the same ten training notes, 789 test notes, clinician annotations
and unchanged scorer. Neither had failed test documents. Each made one
bubblewrap-isolated `gpt-6.1-sol` / `xhigh` call with no repair rounds. Their
requests were identical except for the declared pipeline name.

| Allowed profile | Strict micro F1 | Strict macro F1 | Overlap micro F1 | Overlap macro F1 | Actual rules |
|---|---:|---:|---:|---:|---|
| Previous regex-only | 86.71% | 80.30% | 90.66% | 87.66% | 52 regex |
| Previous spaCy-only / small | 80.51% | 79.52% | 81.55% | 83.13% | 64 spaCy |
| Regex + spaCy / small | 87.21% | 80.68% | 90.43% | 87.65% | 47 regex |
| Regex + spaCy / large | 87.10% | 81.81% | 90.59% | 88.17% | 56 regex |

Both mixed-format calls proposed regex rules only; every proposal was accepted.
There were no discarded spaCy proposals. The trained pipelines were loaded but
processed zero documents. Running each frozen artifact without loading a
trained model produced identical full JSON predictions on all 789 notes.
These runs therefore compare independently generated regex rulesets, not the
effect of consuming small versus large linguistic predictions. The earlier
spaCy-only restriction was an experiment setting, not a library limitation.

Small took 474.25s to fit, using 30,562 input and 25,875 output tokens. Large
took 635.71s, using 30,563 input and 29,306 output tokens. Fresh-process rule
inference took 2.080s / 1.363s, excluding 4.078s / 3.287s model loading. Test
peak process RSS was 421.1 / 1,891.5 MiB, respectively. Both had zero inference
LLM calls; model tools were unused during synthesis, and no completion recovery
was needed. Reasoning-token details and independent strict TP/FP/FN checks are
in `runs/rulechef_mixed_comparison_2026-10-20/comparison.json`; its `report.md`
summarizes the experiments. This remains one exploratory generation per profile,
without cross-validation or test-driven rule changes.

## Completed Spanish pilot: 20 October 2026

Results are saved in `runs/rulechef_spacy_es_train10_test789_2026-10-20/`.
The same ten training notes and 789 test notes/clinician annotations were used.
All test notes were scored, with no failures and 14,834 gold entities.

| Method | Strict micro F1 | Strict macro F1 | Overlap micro F1 | Overlap macro F1 |
|---|---:|---:|---:|---:|
| RuleChef Spanish spaCy rules | 80.51% | 79.52% | 81.55% | 83.13% |
| Previous RuleChef regex rules | 86.71% | 80.30% | 90.66% | 87.66% |

One bubblewrap-isolated `gpt-6.1-sol` / `xhigh` synthesis produced 64 accepted
spaCy rules in 565.45 seconds, with no repair rounds. It used 29,909 input
tokens (29,696 cached) and 25,197 output tokens, including 15,635 reasoning
tokens. A CLI startup diagnostic was initially misclassified as tool execution;
the raw trace contains no model tool calls. The actual saved JSON completion
was replayed through native fit after checking exact request equality, without
editing the proposals or making another LLM call. Original logs, hashes and
recovery details are retained in the run. Future calls disable code mode and
audit diagnostics separately from tool actions.

Fresh-process inference took 27.148 seconds and peaked at 447.5 MiB RSS, with
zero LLM calls. All 64 generated rules use lexical/token patterns; none uses
LEMMA, POS, TAG, MORPH or DEP. An inference-only diagnostic using the frozen
rules with `spacy.blank("es")` produced exactly identical predictions on all
789 notes in 9.738 seconds. The trained linguistic components therefore did
not contribute to this particular ruleset's decisions.

The default Spanish token boundaries do not align with 2,623 gold entities;
all were strict false negatives, out of 4,388 total. Token-only patterns cannot
recover those exact spans. A clinical tokenizer or mixed regex/spaCy rules are
possible follow-up experiments; neither was tuned using this test feedback.
This comparison changes both the rule format and linguistic pipeline, so it
does not isolate Spanish versus English model quality. The complete report,
resource measurements and diagnostics are in the run directory.

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
Per-label/per-document errors, examples, gold and predictions remain private.
This is one exploratory run, with different synthesis/inference models across
methods; it is not cross-validation or a reproduction of the paper's benchmark.

## Isolation of the initial regex pilot

The 19 October pilot used a temporary working directory and read-only Codex
execution, without the later bubblewrap filesystem allowlist. Its four event
logs contained no tool calls. That is not the same isolation guarantee as the
subsequent pilots and current runner, which hide the host project and use a
fresh authentication-only Codex state. The retrospective audit is preserved
with that run; the original artifacts have not been rewritten.
