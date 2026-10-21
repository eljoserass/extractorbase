"""Rare label coverage, repeatability and duplicate separation without private data."""

from data import DocumentInput, GoldAnnotation, TrainingExample
from experiments.rulechef_splits import make_splits


def corpus():
    examples = []
    for index in range(40):
        labels = ["common"] + (["less_common"] if index % 4 == 0 else [])
        if index == 3:
            labels.append("rare")
        text = f"note {index}"
        results = [
            {
                "id": label,
                "type": "labels",
                "from_name": "labels",
                "to_name": "text",
                "value": {"start": 0, "end": 4, "text": "note", "labels": [label]},
            }
            for label in labels
        ]
        examples.append(
            TrainingExample(DocumentInput(index, text), GoldAnnotation(id=index, result=results))
        )
    # A normalized duplicate has its own ID and must stay with its original.
    examples.append(TrainingExample(DocumentInput(40, " NOTE   0 "), examples[0].gold))
    return examples


def test_fewshot_rounds_cover_rare_labels_and_keep_groups_together():
    examples = corpus()
    splits = make_splits(examples, train_size=10)
    assert splits == make_splits(examples, train_size=10)
    assert len(splits) == 5
    for split in splits:
        learning, test = set(split["learning_ids"]), set(split["test_ids"])
        train, dev = set(split["train_ids"]), set(split["dev_ids"])
        assert len(learning) == 10 and len(test) == 31
        assert not learning & test and not train & dev
        assert train | dev == learning
        assert split["distributions"]["learning"]["labels_covered"] == 3
        assert 3 in train
        for partition in (train, dev, test):
            assert (0 in partition) == (40 in partition)


def test_fivefold_test_partitions_are_disjoint_and_exhaustive():
    examples = corpus()
    splits = make_splits(examples, mode="fivefold")
    seen = set()
    for split in splits:
        test = set(split["test_ids"])
        assert not seen & test
        seen.update(test)
        assert not test & set(split["learning_ids"])
    assert seen == {example.input.id for example in examples}
