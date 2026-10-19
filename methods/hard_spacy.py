"""An observed-term spaCy matcher with local, character-preserving value rules."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer

from data import (
    DocumentInput,
    EntityResult,
    LabelStudioTaskPrediction,
    ResultItem,
    TaskSpec,
    TrainingExample,
    prediction_for,
)
from methods.base import Method, Scorer

MODEL_VERSION = "hard-spacy-2"
FORMAT_VERSION = 1
Span = tuple[int, int, str]
NUMBER = re.compile(r"[\s:=(]*([0-9]+(?:[.,][0-9]+)?(?:\s*/\s*[0-9]+)?)")


def _language() -> Language:
    nlp = spacy.blank("es")
    # The notes concatenate counts, indices and punctuation without whitespace.
    # Splitting these boundaries preserves all original character positions.
    infix = re.compile(
        r"[^\w\s]|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])"
        # Keep the observed antibody prefix aCCP intact while splitting puñoEVA.
        r"|(?<=[a-záéíóúñ])(?=[A-ZÁÉÍÓÚÑ]{2,})(?<!\ba)"
    )
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        rules={},
        prefix_search=re.compile(r"^[^\w\s]").search,
        suffix_search=re.compile(r"[^\w\s]$").search,
        infix_finditer=infix.finditer,
        token_match=None,
    )
    return nlp


@dataclass
class HardSpacyArtifact:
    task_spec: TaskSpec
    lexicon: dict[str, list[str]]
    polarity_terms: dict[str, list[str]]
    relation_sources: list[str]
    supported_labels: list[str]
    slash_labels: list[str]
    model_version: str = MODEL_VERSION
    nlp: Language = field(init=False, repr=False)
    matcher: Matcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nlp = _language()
        self.matcher = Matcher(self.nlp.vocab)
        for label, terms in sorted(self.lexicon.items()):
            patterns = []
            for term in terms:
                tokens = [token for token in self.nlp.make_doc(term) if not token.is_space]
                pattern: list[dict[str, str | bool]] = []
                for token in tokens:
                    if pattern:
                        pattern.append({"IS_SPACE": True, "OP": "*"})
                    pattern.append({"LOWER": token.lower_})
                if pattern:
                    patterns.append(pattern)
            if patterns:
                self.matcher.add(label, patterns)


class HardSpacyMethod(Method[HardSpacyArtifact]):
    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        task_spec: TaskSpec,
        *,
        dev_examples: Sequence[TrainingExample] | None = None,
        scorer: Scorer | None = None,
    ) -> HardSpacyArtifact:
        if dev_examples is not None:
            raise ValueError("This one-off method uses training examples only.")
        terms: dict[str, Counter[str]] = defaultdict(Counter)
        polarities: dict[str, set[str]] = defaultdict(set)
        supported: set[str] = set()
        relation_sources: set[str] = set()
        slash_labels: set[str] = set()
        for example in train_examples:
            entities = {
                item["id"]: item for item in example.gold.result if item["type"] == "labels"
            }
            for item in entities.values():
                value = item["value"]
                # Read spans from the original input, rather than a possibly stale text field.
                term = example.input.text[value["start"] : value["end"]]
                term = " ".join(term.casefold().split())
                for label in value["labels"]:
                    if label not in task_spec.entity_labels:
                        raise ValueError(f"Unknown entity label: {label}")
                    supported.add(label)
                    if label in {"V:+", "V:-"}:
                        if any(char.isalpha() for char in term):
                            polarities[label].add(term)
                    elif not label.startswith("V:") and any(c.isalpha() for c in term):
                        if term.startswith("/"):
                            slash_labels.add(label)
                        # Slash is a separator; a contextual rule can retain it when needed.
                        terms[term.lstrip("/")][label] += 1
            for relation in example.gold.result:
                if relation["type"] == "relation" and "valor" in relation.get("labels", []):
                    source = entities.get(relation["from_id"])
                    target = entities.get(relation["to_id"])
                    if (
                        source
                        and target
                        and any(label.startswith("V:") for label in target["value"]["labels"])
                    ):
                        relation_sources.update(source["value"]["labels"])
        lexicon: dict[str, list[str]] = defaultdict(list)
        for term, counts in sorted(terms.items()):
            # A family-context fragment is not a diagnosis name. Whitespace and
            # numeric-only annotation accidents were already excluded above.
            if term == "madre con":
                continue
            # Majority vote resolves conflicting annotations of the same term;
            # ties prefer an index over a laboratory test for composite headings.
            label = min(
                counts,
                key=lambda name: (-counts[name], not name.startswith("AC:"), name),
            )
            lexicon[label].append(term)
        return HardSpacyArtifact(
            task_spec,
            dict(lexicon),
            {label: sorted(values) for label, values in sorted(polarities.items())},
            sorted(relation_sources),
            sorted(supported),
            sorted(slash_labels),
        )

    def dump(self, artifact: HardSpacyArtifact, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": FORMAT_VERSION,
            "task_spec": asdict(artifact.task_spec),
            "lexicon": artifact.lexicon,
            "polarity_terms": artifact.polarity_terms,
            "relation_sources": artifact.relation_sources,
            "supported_labels": artifact.supported_labels,
            "slash_labels": artifact.slash_labels,
            "model_version": artifact.model_version,
        }
        (directory / "artifact.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self, directory: Path) -> HardSpacyArtifact:
        state = json.loads((Path(directory) / "artifact.json").read_text(encoding="utf-8"))
        if state.pop("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported HardSpacy artifact format.")
        spec = state.pop("task_spec")
        spec["entity_labels"] = tuple(spec["entity_labels"])
        spec["relation_labels"] = tuple(spec["relation_labels"])
        return HardSpacyArtifact(task_spec=TaskSpec(**spec), **state)

    @staticmethod
    def _mentions(artifact: HardSpacyArtifact, text: str) -> list[Span]:
        doc = artifact.nlp.make_doc(text)
        candidates = {
            (
                doc[start].idx,
                doc[end - 1].idx + len(doc[end - 1]),
                doc.vocab.strings[key],
            )
            for key, start, end in artifact.matcher(doc)
        }
        # The gold uses /ACPA after an isolated diagnostic +, but ACPA when
        # the slash separates two tests (FR+/ACPA-).
        candidates = {
            (start - 1, end, label)
            if label in artifact.slash_labels
            and text[max(0, start - 2) : start] == "+/"
            and (start < 3 or not text[start - 3].isalnum())
            else (start, end, label)
            for start, end, label in candidates
        }
        if "sero" in artifact.lexicon.get("T:FR", []):
            for match in re.finditer(r"\b(sero)negativa\b", text, re.IGNORECASE):
                if any(
                    label == "D:AR" and end <= match.start() and text[end : match.start()].isspace()
                    for _, end, label in candidates
                ):
                    candidates.add((match.start(1), match.end(1), "T:FR"))
        # Retain overlaps across labels, except a lab abbreviation already forming
        # part of a composite activity index (e.g. PCR within ASDAS PCR).
        return sorted(
            span
            for span in candidates
            if not any(
                other != span
                and other[0] <= span[0] < span[1] <= other[1]
                and (
                    other[2] == span[2]
                    or (other[2].startswith("AC:EVA_") and span[2].startswith("AC:EVA_"))
                    or (other[2].startswith("AC:") and span[2].startswith("T:"))
                )
                for other in candidates
            )
        )

    @staticmethod
    def _value(artifact: HardSpacyArtifact, text: str, source: Span) -> Span | None:
        _, end, label = source
        if not label.startswith(("AC:", "T:")):
            return None
        tail = text[end:]
        number = NUMBER.match(tail)
        if number:
            value_label = "V:PORCENTAJE" if "/" in number[1] else "V:VALOR"
            if value_label in artifact.supported_labels:
                return end + number.start(1), end + number.end(1), value_label
        if label.startswith("T:"):
            sign = re.match(r"[\s(]*([+-])", tail)
            if sign:
                value_label = "V:+" if sign[1] == "+" else "V:-"
                if value_label in artifact.supported_labels:
                    return end + sign.start(1), end + sign.end(1), value_label
            for value_label, terms in artifact.polarity_terms.items():
                for term in terms:
                    word = re.match(r"[\s(]*(" + re.escape(term) + r")(?!\w)", tail, re.IGNORECASE)
                    if word:
                        return end + word.start(1), end + word.end(1), value_label
        return None

    def predict(
        self,
        artifact: HardSpacyArtifact,
        inputs: Sequence[DocumentInput],
    ) -> list[LabelStudioTaskPrediction]:
        predictions = []
        for document in inputs:
            sources = self._mentions(artifact, document.text)
            spans = set(sources)
            links: set[tuple[Span, Span]] = set()
            for source in sources:
                value = self._value(artifact, document.text, source)
                if value:
                    spans.add(value)
                    if (
                        "valor" in artifact.task_spec.relation_labels
                        and source[2] in artifact.relation_sources
                    ):
                        links.add((source, value))
            # IDs depend only on predicted spans, never gold IDs or input IDs.
            ids = {span: f"hs_{index:04d}" for index, span in enumerate(sorted(spans), 1)}
            results: list[ResultItem] = []
            for (start, end, label), entity_id in ids.items():
                entity: EntityResult = {
                    "id": entity_id,
                    "type": "labels",
                    "from_name": artifact.task_spec.from_name,
                    "to_name": artifact.task_spec.to_name,
                    "value": {
                        "start": start,
                        "end": end,
                        "text": document.text[start:end],
                        "labels": [label],
                    },
                }
                results.append(entity)
            for source, value in sorted(links):
                results.append(
                    {
                        "type": "relation",
                        "from_id": ids[source],
                        "to_id": ids[value],
                        "labels": ["valor"],
                        "direction": "right",
                    }
                )
            predictions.append(prediction_for(document, results, artifact.model_version))
        return predictions
