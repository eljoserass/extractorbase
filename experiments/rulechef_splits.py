"""Deterministic, label-aware few-shot rounds and ordinary grouped five-folds."""

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from data import TrainingExample


@dataclass
class NoteGroup:
    ids: list[int | str]
    labels: Counter[str]


def label_counts(examples: Sequence[TrainingExample]) -> tuple[Counter[str], Counter[str]]:
    documents: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    for example in examples:
        labels = [
            label
            for item in example.gold.result
            if item["type"] == "labels"
            for label in item["value"]["labels"]
        ]
        documents.update(set(labels))
        mentions.update(labels)
    return documents, mentions


def groups_for(examples: Sequence[TrainingExample]) -> list[NoteGroup]:
    grouped: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        normalized = " ".join(example.input.text.split()).casefold()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        grouped[digest].append(example)
    return [
        NoteGroup([example.input.id for example in items], label_counts(items)[0])
        for items in grouped.values()
    ]


def representative_sample(groups: list[NoteGroup], count: int, seed: int) -> list[NoteGroup]:
    """Cover rare labels, then fill deficits to the corpus document proportions.

    A minimum target of one deliberately boosts rare labels in a small sample.
    Duplicate groups stay together. No claim of optimal set cover is made.
    """
    if count <= 0:
        return []
    size = sum(len(group.ids) for group in groups)
    if count >= size:
        return list(groups)
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group.labels)
    targets = {label: max(1.0, total * count / size) for label, total in totals.items()}
    available = list(groups)
    random.Random(seed).shuffle(available)
    selected: list[NoteGroup] = []
    observed: Counter[str] = Counter()
    remaining = count
    while remaining:
        candidates = [group for group in available if len(group.ids) <= remaining]
        if not candidates:
            raise ValueError("Cannot meet the requested note count without splitting duplicates.")

        def gain(group: NoteGroup) -> tuple[float, float]:
            coverage = sum(
                1 / totals[label] ** 0.5 for label in group.labels if not observed[label]
            )
            deficit = sum(
                min(amount, max(0.0, targets[label] - observed[label])) / targets[label]
                for label, amount in group.labels.items()
            )
            return coverage / len(group.ids), deficit / len(group.ids)

        chosen = max(candidates, key=gain)
        selected.append(chosen)
        observed.update(chosen.labels)
        remaining -= len(chosen.ids)
        available.remove(chosen)
    return selected


def split_learning(
    groups: list[NoteGroup],
    dev_count: int,
    seed: int,
) -> tuple[list[int | str], list[int | str]]:
    """Prefer dev groups whose labels still have support in the synthesis split."""
    total: Counter[str] = Counter()
    for group in groups:
        total.update(group.labels)
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    chosen: list[NoteGroup] = []
    dev_labels: Counter[str] = Counter()
    remaining = dev_count
    while remaining:
        candidates = [g for g in shuffled if len(g.ids) <= remaining]
        if not candidates:
            raise ValueError("Cannot allocate dev without splitting a duplicate group.")

        def quality(group: NoteGroup) -> tuple[int, int]:
            lost = sum(
                total[label] - dev_labels[label] - amount <= 0
                for label, amount in group.labels.items()
            )
            added = sum(not dev_labels[label] for label in group.labels)
            return -lost, added

        group = max(candidates, key=quality)
        chosen.append(group)
        dev_labels.update(group.labels)
        remaining -= len(group.ids)
        shuffled.remove(group)
    return (
        [key for group in groups if group not in chosen for key in group.ids],
        [key for group in chosen for key in group.ids],
    )


def classic_folds(groups: list[NoteGroup], rounds: int, seed: int) -> list[list[NoteGroup]]:
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group.labels)
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda g: min((totals[label] for label in g.labels), default=float("inf")))
    folds: list[list[NoteGroup]] = [[] for _ in range(rounds)]
    counts: list[Counter[str]] = [Counter() for _ in range(rounds)]
    sizes = [0] * rounds
    target_size = sum(len(g.ids) for g in groups) / rounds
    for group in ordered:

        def deficit(index: int) -> tuple[bool, float, int]:
            gain = sum(
                max(0, totals[label] / rounds - counts[index][label]) / totals[label]
                for label in group.labels
            )
            return sizes[index] < target_size, gain, -sizes[index]

        index = max(range(rounds), key=deficit)
        folds[index].append(group)
        counts[index].update(group.labels)
        sizes[index] += len(group.ids)
    return folds


def make_splits(
    examples: Sequence[TrainingExample],
    *,
    rounds: int = 5,
    train_size: int = 20,
    dev_fraction: float = 0.2,
    seed: int = 42,
    mode: str = "fewshot",
) -> list[dict]:
    if rounds < 2 or train_size < 2 or not 0 < dev_fraction < 1:
        raise ValueError("Need multiple rounds, at least two learning notes and a dev fraction.")
    if mode not in {"fewshot", "fivefold"}:
        raise ValueError("mode must be fewshot or fivefold.")
    groups = groups_for(examples)
    all_ids = [example.input.id for example in examples]
    by_id = {example.input.id: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError("Duplicate document IDs.")
    total_docs, total_mentions = label_counts(examples)
    folds = classic_folds(groups, rounds, seed) if mode == "fivefold" else []
    result = []
    for index in range(rounds):
        if mode == "fewshot":
            if train_size >= len(examples):
                raise ValueError("The learning budget must leave test notes.")
            learning = representative_sample(groups, train_size, seed + index)
        else:
            learning = [group for fold in folds if fold is not folds[index] for group in fold]
        learning_ids = [key for group in learning for key in group.ids]
        dev_count = max(1, round(len(learning_ids) * dev_fraction))
        train_ids, dev_ids = split_learning(learning, dev_count, seed + 1000 + index)
        test_ids = [key for key in all_ids if key not in set(learning_ids)]
        partitions = {
            "learning": learning_ids,
            "train": train_ids,
            "dev": dev_ids,
            "test": test_ids,
        }
        distributions = {}
        for name, ids in partitions.items():
            documents, mentions = label_counts([by_id[key] for key in ids])
            distributions[name] = {
                "notes": len(ids),
                "labels_covered": len(documents),
                "missing_labels": sorted(total_docs.keys() - documents.keys()),
                "documents_per_label": dict(documents),
                "mentions_per_label": dict(mentions),
            }
        result.append(
            {
                "round": index + 1,
                "seed": seed + index,
                **{name + "_ids": ids for name, ids in partitions.items()},
                "distributions": distributions,
                "corpus_documents_per_label": dict(total_docs),
                "corpus_mentions_per_label": dict(total_mentions),
            }
        )
    return result
