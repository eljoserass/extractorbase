"""Explicit dev acceptance around RuleChef's native proposals and coordinator.

RuleChef 0.2.2 cannot receive an external dev dataset in evaluate_and_refine.
Its public pipeline synthesizes before splitting and may silently skip small
holdouts. This small orchestration layer keeps dev out of every model prompt.
"""

from copy import deepcopy
from typing import Any

from rulechef.core import Dataset, Rule
from rulechef.evaluation import EvalResult


def accepts(previous: EvalResult, candidate: EvalResult) -> bool:
    """Use RuleChef's acceptance policy; retain a separate best-F1 snapshot."""
    return (
        candidate.micro_f1 >= previous.micro_f1 - 0.005
        or candidate.micro_precision > previous.micro_precision
    )


def refine_on_dev(
    learner: Any,
    rules: list[Rule],
    train: Dataset,
    dev: Dataset,
    iterations: int,
    coordinator: Any = None,
) -> tuple[list[Rule], list[dict[str, Any]]]:
    current = deepcopy(rules)
    best = deepcopy(current)
    best_score = learner._evaluate_rules(best, dev).micro_f1
    history: list[dict[str, Any]] = []

    def consider(candidate: list[Rule], iteration: int, operation: str) -> None:
        nonlocal current, best, best_score
        previous = learner._evaluate_rules(current, dev)
        result = learner._evaluate_rules(candidate, dev)
        accepted = accepts(previous, result)
        history.append(
            {
                "iteration": iteration,
                "operation": operation,
                "accepted": accepted,
                "previous_dev_f1": previous.micro_f1,
                "candidate_dev_f1": result.micro_f1,
                "candidate_dev_precision": result.micro_precision,
                "rules": len(candidate),
            }
        )
        if accepted:
            current = candidate
            if result.micro_f1 > best_score:
                best, best_score = deepcopy(current), result.micro_f1
        print(f"{operation} {iteration}: dev F1={result.micro_f1:.4f}, {accepted=}", flush=True)

    for index in range(iterations):
        train_result = learner._evaluate_rules(current, train)
        dev_result = learner._evaluate_rules(current, dev)
        if dev_result.exact_match >= 0.9 or not train_result.failures:
            break
        if coordinator:
            if coordinator.enable_critic and index % coordinator.critic_interval == 0:
                learner._run_critic(current, train, coordinator, train_result)
            if (index + 1) % coordinator.audit_interval == 0:
                candidate = learner._run_mid_refinement_audit(
                    deepcopy(current), train, coordinator, train_result, index + 1
                )
                consider(candidate, index + 1, "audit")
                train_result = learner._evaluate_rules(current, train)
            guidance, should_continue = coordinator.guide_refinement(
                train_result, index, iterations
            )
            if not should_continue:
                history.append({"iteration": index + 1, "operation": "coordinator_stop"})
                break
        else:
            guidance = ""
        patch, deleted = learner.synthesize_patch_ruleset(
            current,
            train_result.failures,
            max_rules=min(learner.max_rules, 8),
            dataset=train,
            guidance=guidance,
            class_metrics=train_result.per_class,
            fp_examples=train_result.fp_examples,
        )
        candidate = learner._merge_patch(deepcopy(current), patch, deleted)
        consider(candidate, index + 1, "patch")

    if coordinator and coordinator.prune_after_learn:
        # The native auditor sees train examples/metrics only; dev gates its changes.
        current = deepcopy(best)
        candidate = learner._run_mid_refinement_audit(
            deepcopy(best), train, coordinator, learner._evaluate_rules(best, train), iterations
        )
        consider(candidate, iterations, "final_audit")
        # Leave-one-out pruning needs no model call and never exposes dev examples.
        for rule in list(best):
            candidate = [item for item in best if item.id != rule.id]
            result = learner._evaluate_rules(candidate, dev)
            if result.micro_f1 >= best_score:
                best, best_score = candidate, result.micro_f1

    learner._stamp_validated_stats(best, dev)
    return best, history
