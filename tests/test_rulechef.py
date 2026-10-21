"""Synthetic contract and position regressions; no credentials or clinical data."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("rulechef")

from rulechef.core import Dataset, Example, Rule, Task, TaskType  # noqa: E402

from data import DocumentInput, GoldAnnotation, TaskSpec, TrainingExample  # noqa: E402
from evaluator import score  # noqa: E402
from methods.rulechef import RuleChefConfig, RuleChefMethod  # noqa: E402

RULE = {
    "name": "pcr",
    "description": "PCR",
    "format": "regex",
    "content": r"\bPCR\b",
    "output_key": "entities",
    "priority": 5,
    "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "T:PCR"},
}


class FakeClient:
    model = "synthetic"

    def __init__(self, rules=None):
        self.calls = []
        self.rules = rules if rules is not None else [RULE]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"rules": self.rules}),
                    )
                )
            ]
        )


def test_fit_unicode_repeated_mentions_and_reload_without_client(tmp_path):
    client = FakeClient()
    method = RuleChefMethod(RuleChefConfig(refinement_iterations=1), client=client)
    spec = TaskSpec(("T:PCR",), (), from_name="clinical", to_name="note")
    example = TrainingExample(
        DocumentInput(1, "PCR"),
        GoldAnnotation(
            id=1,
            result=[
                {
                    "id": "gold",
                    "type": "labels",
                    "from_name": "clinical",
                    "to_name": "note",
                    "value": {"start": 0, "end": 3, "text": "PCR", "labels": ["T:PCR"]},
                }
            ],
        ),
    )
    artifact = method.fit([example], spec, scorer=score)
    assert len(client.calls) == 1
    assert artifact.provenance["training_metrics"]["strict_micro"]["f1"] == 1.0
    assert artifact.rules[0].validated_precision == 1.0
    document = DocumentInput("new", "ñ: PCR PCR")
    prediction = method.predict(artifact, [document])
    entities = prediction[0]["predictions"][0]["result"]
    assert [(e["value"]["start"], e["value"]["end"]) for e in entities] == [(3, 6), (7, 10)]
    assert all(e["from_name"] == "clinical" and e["to_name"] == "note" for e in entities)
    method.dump(artifact, tmp_path)
    restored = RuleChefMethod()
    assert restored.client is None
    assert restored.predict(restored.load(tmp_path), [document]) == prediction
    assert len(client.calls) == 1


def test_refinement_and_rule_priority_do_not_credit_wrong_occurrence():
    from methods.rulechef import ExactRuleLearner

    learner = ExactRuleLearner(FakeClient(), use_grex=False)
    task = Task("positions", "PCR", {"text": "str"}, {"entities": "list"}, TaskType.NER)
    dataset = Dataset(
        name="positions",
        task=task,
        examples=[
            Example(
                id="1",
                input={"text": "PCR PCR"},
                source="synthetic",
                expected_output={
                    "entities": [{"text": "PCR", "start": 4, "end": 7, "type": "T:PCR"}]
                },
            )
        ],
    )
    wrong = Rule.from_dict({**RULE, "content": "^PCR"})
    assert learner._evaluate_rules([wrong], dataset).micro_f1 == 0.0
    learner._stamp_validated_stats([wrong], dataset)
    assert wrong.validated_precision == 0.0


def test_invalid_predictions_fail_instead_of_silently_moving_offsets():
    from methods.rulechef import RuleChefArtifact

    method = RuleChefMethod()
    artifact = RuleChefArtifact(
        TaskSpec(("T:PCR",), ()),
        [
            Rule.from_dict(
                {
                    **RULE,
                    "output_template": {**RULE["output_template"], "start": 999},
                }
            )
        ],
    )
    with pytest.raises(ValueError, match="Invalid RuleChef entity"):
        method.predict(artifact, [DocumentInput(1, "PCR")])


@pytest.mark.parametrize("pattern", [r"(?=(PCR))", r"(?<=(PCR))"])
def test_zero_width_captures_keep_their_real_unicode_and_occurrence_offsets(tmp_path, pattern):
    from methods.rulechef import RuleChefArtifact

    rule = Rule.from_dict(
        {
            **RULE,
            "content": pattern,
            "output_template": {**RULE["output_template"], "text": "$1"},
        }
    )
    artifact = RuleChefArtifact(TaskSpec(("T:PCR",), ()), [rule])
    method = RuleChefMethod()
    document = DocumentInput(1, "ñ PCR PCR")
    predictions = method.predict(artifact, [document])
    entities = predictions[0]["predictions"][0]["result"]
    assert [(e["value"]["start"], e["value"]["end"]) for e in entities] == [(2, 5), (6, 9)]
    method.dump(artifact, tmp_path)
    restored = RuleChefMethod()
    assert restored.predict(restored.load(tmp_path), [document]) == predictions


def test_spanish_spacy_token_values_unicode_and_artifact_reload(tmp_path):
    pytest.importorskip("es_core_news_sm")
    rules = [
        {**RULE, "format": "spacy", "content": json.dumps([{"LOWER": "pcr"}])},
        {
            **RULE,
            "name": "value",
            "format": "spacy",
            "content": json.dumps([{"LOWER": "pcr"}, {"LIKE_NUM": True}]),
            "output_template": {
                "text": "$2.text",
                "start": "$2.start",
                "end": "$2.end",
                "type": "V:VALOR",
            },
        },
    ]
    method = RuleChefMethod(
        RuleChefConfig(
            rule_formats=("spacy",),
            spacy_model="es_core_news_sm",
            refinement_iterations=0,
        ),
        client=FakeClient(rules),
    )
    spec = TaskSpec(("T:PCR", "V:VALOR"), ())
    text = "PCR 1"
    example = TrainingExample(
        DocumentInput(1, text),
        GoldAnnotation(
            id=1,
            result=[
                {
                    "id": label,
                    "type": "labels",
                    "from_name": "labels",
                    "to_name": "text",
                    "value": {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "labels": [label],
                    },
                }
                for start, end, label in [(0, 3, "T:PCR"), (4, 5, "V:VALOR")]
            ],
        ),
    )
    artifact = method.fit([example], spec, scorer=score)
    assert artifact.provenance["training_metrics"]["strict_micro"]["f1"] == 1
    assert artifact.provenance["spacy_pipeline"]["language"] == "es"
    assert artifact.provenance["rule_formats"] == {"spacy": 2}
    doc = DocumentInput("new", "ñ: PCR 12 y PCR 4")
    predictions = method.predict(artifact, [doc])
    actual = {
        (e["value"]["start"], e["value"]["end"], e["value"]["labels"][0])
        for e in predictions[0]["predictions"][0]["result"]
    }
    assert actual == {(3, 6, "T:PCR"), (7, 9, "V:VALOR"), (12, 15, "T:PCR"), (16, 17, "V:VALOR")}
    assert method.executor._nlp._parse.cache_info().hits > 0
    method.dump(artifact, tmp_path)
    restored = RuleChefMethod()
    assert restored.predict(restored.load(tmp_path), [doc]) == predictions
    saved = tmp_path / "artifact.json"
    state = json.loads(saved.read_text())
    state["provenance"]["spacy_pipeline"]["version"] = "different-version"
    saved.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="version differs"):
        restored.load(tmp_path)


def test_spanish_trained_lemmas_are_available():
    pytest.importorskip("es_core_news_sm")
    from methods.rulechef import RuleChefArtifact

    method = RuleChefMethod(RuleChefConfig(rule_formats=("spacy",), spacy_model="es_core_news_sm"))
    rule = Rule.from_dict({**RULE, "format": "spacy", "content": json.dumps([{"LEMMA": "tomar"}])})
    artifact = RuleChefArtifact(TaskSpec(("T:PCR",), ()), [rule], spacy_model="es_core_news_sm")
    result = method.predict(artifact, [DocumentInput(1, "El paciente toma metotrexato.")])
    assert [e["value"]["text"] for e in result[0]["predictions"][0]["result"]] == ["toma"]


def test_english_pipeline_is_rejected(monkeypatch):
    spacy = pytest.importorskip("spacy")
    english = spacy.blank("en")
    monkeypatch.setattr(spacy, "load", lambda _: english)
    with pytest.raises(ValueError, match="requires an es pipeline"):
        RuleChefMethod(RuleChefConfig(rule_formats=("spacy",), spacy_model="wrong-language"))


def test_bubblewrap_hides_project_and_host_files_with_fresh_state(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from methods.rulechef import CodexClient

    bwrap = shutil.which("bwrap")
    if not bwrap:
        pytest.skip("bubblewrap is unavailable")
    state = tmp_path / "host-state"
    state.mkdir()
    (state / "auth.json").write_text('{"test": "fake credential"}')
    (state / "history.jsonl").write_text("Previous session must stay hidden")
    host_secret = tmp_path / "host-secret.txt"
    host_secret.write_text("This host file must not be readable")
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(state))
    monkeypatch.setattr(shutil, "which", lambda name: bwrap if name == "bwrap" else "/usr/bin/true")
    prefix, audit, environment = CodexClient._bubblewrap(inputs, outputs, tmp_path / "fresh-state")
    code = (
        f"from pathlib import Path; assert not Path({str(host_secret)!r}).exists(); "
        f"assert not Path({str(state / 'history.jsonl')!r}).exists()"
    )
    subprocess.run([*prefix, "/usr/bin/python3", "-c", code], env=environment, check=True)
    assert audit["original_project_visible"] is False
    assert audit["fresh_codex_state"] is True


@pytest.mark.parametrize("tool_used", [False, True])
def test_cli_diagnostics_are_distinguished_from_tool_execution(tmp_path, monkeypatch, tool_used):
    import subprocess
    from pathlib import Path

    from methods.rulechef import CodexClient

    reply = json.dumps({"analysis": "synthetic", "rules": [], "deleted_rules": []})

    def complete(command, **kwargs):
        response_path = Path(command[command.index("--output-last-message") + 1])
        response_path.write_text(reply)
        events = [
            {"type": "item.completed", "item": {"type": "error", "message": "startup notice"}},
            {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}},
        ]
        if tool_used:
            events.append({"type": "item.completed", "item": {"type": "command_execution"}})
        for event in events:
            kwargs["stdout"].write(json.dumps(event) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", complete)
    client = CodexClient(RuleChefConfig(codex_isolation="none", trace_directory=str(tmp_path)))
    request = {"model": "synthetic", "messages": [{"role": "user", "content": "synthetic"}]}
    if tool_used:
        with pytest.raises(RuntimeError, match="attempted tools"):
            client.create(**request)
    else:
        assert client.create(**request).choices[0].message.content == reply
    assert client.calls[0]["diagnostics"] == ["startup notice"]
    assert bool(client.calls[0]["tool_calls"]) is tool_used


def synthetic_example(key, text, spans):
    return TrainingExample(
        DocumentInput(key, text),
        GoldAnnotation(
            id=key,
            result=[
                {
                    "id": str(index),
                    "type": "labels",
                    "from_name": "labels",
                    "to_name": "text",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "labels": [label],
                    },
                }
                for index, (start, end, label) in enumerate(spans)
            ],
        ),
    )


def test_external_dev_rejects_train_memorization_without_entering_prompts(tmp_path):
    client = FakeClient()
    replies = iter(
        [
            {"rules": [RULE]},
            {"rules": [{**RULE, "content": r"\b(?:PCR|ESR|NOISE)\b"}]},
        ]
    )

    def create(**kwargs):
        client.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies))))]
        )

    client.chat.completions.create = create
    train = synthetic_example(1, "PCR ESR", [(0, 3, "T:PCR"), (4, 7, "T:PCR")])
    dev = synthetic_example(2, "PCR XYZ NOISE NOISE", [(0, 3, "T:PCR"), (4, 7, "T:PCR")])
    method = RuleChefMethod(RuleChefConfig(refinement_iterations=1), client=client)
    artifact = method.fit([train], TaskSpec(("T:PCR",), ()), dev_examples=[dev])
    assert len(client.calls) == 2
    assert all("XYZ" not in json.dumps(call) for call in client.calls)
    assert artifact.rules[0].content == RULE["content"]
    assert artifact.provenance["refinement_history"][0]["accepted"] is False
    assert artifact.provenance["dev_ids"] == [2]


def test_train_and_dev_ids_and_duplicate_texts_are_rejected():
    train = synthetic_example(1, "PCR", [(0, 3, "T:PCR")])
    method = RuleChefMethod(client=FakeClient())
    with pytest.raises(ValueError, match="IDs overlap"):
        method.fit([train], TaskSpec(("T:PCR",), ()), dev_examples=[train])
    duplicate = synthetic_example(2, " pcr ", [(1, 4, "T:PCR")])
    with pytest.raises(ValueError, match="duplicate texts"):
        method.fit([train], TaskSpec(("T:PCR",), ()), dev_examples=[duplicate])


def test_fuzzy_without_weights_and_regex_only_artifact_dependencies(tmp_path):
    pytest.importorskip("spacy")
    rule = {**RULE, "format": "spacy", "content": json.dumps([{"LOWER": {"FUZZY1": "urbason"}}])}
    method = RuleChefMethod(
        RuleChefConfig(
            rule_formats=("regex", "spacy"),
            spacy_model="blank:es",
            refinement_iterations=0,
        ),
        client=FakeClient([rule]),
    )
    example = synthetic_example(1, "Urbasón", [(0, 7, "T:PCR")])
    artifact = method.fit([example], TaskSpec(("T:PCR",), ()), scorer=score)
    assert artifact.provenance["training_metrics"]["strict_micro"]["f1"] == 1.0
    assert method.executor._nlp.language.pipe_names == []
    method.dump(artifact, tmp_path)
    loader = RuleChefMethod()
    loaded = loader.load(tmp_path)
    assert loader.predict(loaded, [example.input]) == method.predict(artifact, [example.input])
    regex_method = RuleChefMethod(method.config, client=FakeClient())
    regex_artifact = regex_method.fit(
        [synthetic_example(2, "PCR", [(0, 3, "T:PCR")])], TaskSpec(("T:PCR",), ())
    )
    regex_method.dump(regex_artifact, tmp_path / "regex")
    loader.load(tmp_path / "regex")
    assert loader.executor._nlp is None


@pytest.mark.parametrize(
    "reply",
    [
        {"guidance": "Try context", "focus_classes": ["T:PCR"], "should_continue": True},
        {"analysis": "Keep", "actions": []},
        {"analysis": "Critique", "rule_feedback": {}, "task_guidance": "Use context"},
    ],
)
def test_codex_bridge_preserves_native_coordinator_responses(tmp_path, monkeypatch, reply):
    import subprocess
    from pathlib import Path

    from methods.rulechef import CodexClient

    def complete(command, **kwargs):
        assert "--output-schema" not in command
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(reply))
        kwargs["stdout"].write(json.dumps({"type": "turn.completed", "usage": {}}) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", complete)
    client = CodexClient(RuleChefConfig(codex_isolation="none", trace_directory=str(tmp_path)))
    response = client.create(model="synthetic", messages=[{"role": "user", "content": "native"}])
    assert json.loads(response.choices[0].message.content) == reply


def test_missing_sdk_model_fails_explicitly():
    client = FakeClient()
    client.model = None
    method = RuleChefMethod(client=client)
    example = synthetic_example(1, "PCR", [(0, 3, "T:PCR")])
    with pytest.raises(ValueError, match="explicit synthesis model"):
        method.fit([example], TaskSpec(("T:PCR",), ()))


@pytest.mark.parametrize("iterations", [3, 8])
def test_native_agentic_prompts_and_patch_budget(iterations):
    client = FakeClient()

    def create(**kwargs):
        client.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "You are the Refinement Coordinator" in prompt:
            result = {"guidance": "Use context", "should_continue": True, "focus_classes": []}
        elif "You are an expert Rule Critic" in prompt:
            result = {"analysis": "Keep", "rule_feedback": {}, "task_guidance": ""}
        elif "You are a Rule Auditor" in prompt:
            result = {"analysis": "Keep", "actions": []}
        elif "You are updating an existing" in prompt:
            result = {"rules": [], "deleted_rules": []}
        else:
            result = {"rules": [RULE]}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))]
        )

    client.chat.completions.create = create
    train = synthetic_example(1, "PCR ESR", [(0, 3, "T:PCR"), (4, 7, "T:PCR")])
    dev = synthetic_example(2, "PCR XYZ", [(0, 3, "T:PCR"), (4, 7, "T:PCR")])
    method = RuleChefMethod(
        RuleChefConfig(
            refinement_iterations=iterations,
            agentic=True,
        ),
        client=client,
    )
    artifact = method.fit([train], TaskSpec(("T:PCR",), ()), dev_examples=[dev])
    prompts = [call["messages"][0]["content"] for call in client.calls]
    assert sum("You are the Refinement Coordinator" in p for p in prompts) == iterations
    assert sum("You are updating an existing" in p for p in prompts) == iterations
    assert any("You are an expert Rule Critic" in p for p in prompts)
    assert all("XYZ" not in p for p in prompts)
    assert (
        len([h for h in artifact.provenance["refinement_history"] if h["operation"] == "patch"])
        == iterations
    )
