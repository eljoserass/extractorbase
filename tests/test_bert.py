import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BertConfig as ModelConfig
from transformers import BertForTokenClassification, BertTokenizerFast

from data import DocumentInput, TaskSpec, TrainingExample
from methods.bert import BertConfig, BertMethod, align_labels
from tests.conftest import entity


def test_bio_projection_reports_overlap_and_preserves_gold(example: TrainingExample) -> None:
    example.gold.result.append(entity(0, 1, "T:FR", "short"))
    labels, notes = align_labels(
        example,
        [(0, 0), (0, 2), (3, 5), (0, 0)],
        {"O": 0, "B-D:AR": 1, "I-D:AR": 2, "B-T:FR": 3, "I-T:FR": 4},
    )
    assert labels == [-100, 1, 0, -100]
    assert len(example.gold.result) == 2 and any("overlapping" in note for note in notes)


def test_bert_fit_save_reload_and_long_document(
    tmp_path: Path, example: TrainingExample, spec: TaskSpec
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    vocab = checkpoint / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nAR\nFR\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.save_pretrained(checkpoint)
    BertForTokenClassification(
        ModelConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    ).save_pretrained(checkpoint)
    method = BertMethod(BertConfig(checkpoint=str(checkpoint), epochs=1, max_length=16, stride=4))
    artifact = method.fit([example], spec)
    directory = tmp_path / "artifact"
    method.dump(artifact, directory)
    before = method.predict(artifact, [example.input])
    fresh = BertMethod()
    restored = fresh.load(directory)
    assert fresh.predict(restored, [example.input]) == before
    assert (directory / "model" / "model.safetensors").exists()
    assert json.loads((directory / "method.json").read_text())["config"]["freeze_encoder"]
    long_document = DocumentInput(2, "AR FR " * 40)
    assert fresh.predict(restored, [long_document])[0]["id"] == 2
