"""
DummyLM-backed tests for the segmentation stage (segmentation_dspy.py),
including citation tracking, plus one end-to-end check that its output
composes with the *existing, unmodified* SyntaxAnalysis. Greek analogue of
arsgrammatica's test_segmentation.py.
"""

import re

import dspy
from dspy.utils.dummies import DummyLM

from grammatike import analyze, validate
from grammatike.models import CitedText
from grammatike.segmentation_dspy import segment_sources
from fixtures.gold_examples import GOLD_EXAMPLES


def _shift_ids(obj, offset):
    """Recursively shift every 't<N>' id-shaped string in obj by offset.
    Used below to reuse the existing unit_verb_root_ten_thuran_anoixen gold
    fixture as if it were a later sentence in a longer input, without
    hand-retyping every id. Citation strings like "Lysias 1.1" never match
    t<N>, so this is safe to run over a whole canned answer unconditionally."""
    if isinstance(obj, dict):
        return {k: _shift_ids(v, offset) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shift_ids(v, offset) for v in obj]
    if isinstance(obj, str) and re.fullmatch(r"t\d+", obj):
        return f"t{int(obj[1:]) + offset}"
    return obj


# ---------------------------------------------------------------------------
# Single citation unit, two sentences: global sequential ids across
# sentence boundaries, under the sources: List[CitedText] contract, with
# every token still carrying its citation.
# ---------------------------------------------------------------------------

SOURCES_ONE_UNIT = [
    CitedText(
        citation="Lysias 1.1",
        text="ἡ ναῦς ἀπόλλυται. τήν θύραν ἀνέῳξεν.",
    ),
]

CANNED_ONE_UNIT = {
    "reasoning": "Two sentences from one citation unit; ids run continuously across both.",
    "sentences": [
        {"tokens": [
            {"id": "t0", "text": "ἡ", "citation": "Lysias 1.1"},
            {"id": "t1", "text": "ναῦς", "citation": "Lysias 1.1"},
            {"id": "t2", "text": "ἀπόλλυται", "citation": "Lysias 1.1"},
            {"id": "t3", "text": ".", "citation": "Lysias 1.1"},
        ]},
        {"tokens": [
            {"id": "t4", "text": "τήν", "citation": "Lysias 1.1"},
            {"id": "t5", "text": "θύραν", "citation": "Lysias 1.1"},
            {"id": "t6", "text": "ἀνέῳξεν", "citation": "Lysias 1.1"},
            {"id": "t7", "text": ".", "citation": "Lysias 1.1"},
        ]},
    ],
}


def test_segmentation_round_trips_with_globally_unique_sequential_ids():
    dspy.configure(lm=DummyLM([CANNED_ONE_UNIT]))
    sentences = segment_sources(SOURCES_ONE_UNIT)

    assert len(sentences) == 2
    assert [t.id for t in sentences[0].tokens] == [f"t{i}" for i in range(0, 4)]
    assert [t.id for t in sentences[1].tokens] == [f"t{i}" for i in range(4, 8)]

    all_ids = [t.id for s in sentences for t in s.tokens]
    assert len(all_ids) == len(set(all_ids)), "token ids must be unique across the whole input"

    assert all(t.citation == "Lysias 1.1" for s in sentences for t in s.tokens)


def test_segmented_sentence_feeds_unmodified_syntax_analysis():
    """The point of keeping segmentation as a separate stage: its output
    (a Sentence's tokens) must work as SyntaxAnalysis's input with zero
    changes to SyntaxAnalysis -- even though its tokens now carry a
    citation field SyntaxAnalysis has never heard of."""
    dspy.configure(lm=DummyLM([CANNED_ONE_UNIT]))
    sentences = segment_sources(SOURCES_ONE_UNIT)
    thuran_sentence = sentences[1]

    thuran_gold = next(
        e for e in GOLD_EXAMPLES if e.slug == "unit_verb_root_ten_thuran_anoixen"
    )
    shifted_answer = _shift_ids(thuran_gold.canned_answer, offset=4)

    dspy.configure(lm=DummyLM([shifted_answer]))
    result = analyze(passage=thuran_gold.passage, tokens=thuran_sentence.tokens)

    problems = validate(thuran_sentence.tokens, result)
    assert not problems, problems
    assert result.tokengraph[0].id == "t4"  # τήν, at its shifted global id


# ---------------------------------------------------------------------------
# The actual point of citation tracking: one sentence spanning two citation
# units, each token still correctly attributed.
# ---------------------------------------------------------------------------

SOURCES_SPANNING_TWO_UNITS = [
    CitedText(citation="Iliad 1.1", text="μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"),
    CitedText(citation="Iliad 1.2", text="οὐλομένην, ἣ μυρί' Ἀχαιοῖς ἄλγε' ἔθηκε"),
]

CANNED_SPANNING_TWO_UNITS = {
    "reasoning": (
        "One sentence continues from Iliad 1.1 into 1.2; ids stay global "
        "and each token keeps the citation of the source unit it came from."
    ),
    "sentences": [
        {"tokens": [
            {"id": "t0", "text": "μῆνιν", "citation": "Iliad 1.1"},
            {"id": "t1", "text": "ἄειδε", "citation": "Iliad 1.1"},
            {"id": "t2", "text": "θεὰ", "citation": "Iliad 1.1"},
            {"id": "t3", "text": "Πηληϊάδεω", "citation": "Iliad 1.1"},
            {"id": "t4", "text": "Ἀχιλῆος", "citation": "Iliad 1.1"},
            {"id": "t5", "text": "οὐλομένην", "citation": "Iliad 1.2"},
            {"id": "t6", "text": ",", "citation": "Iliad 1.2"},
            {"id": "t7", "text": "ἣ", "citation": "Iliad 1.2"},
            {"id": "t8", "text": "μυρί'", "citation": "Iliad 1.2"},
            {"id": "t9", "text": "Ἀχαιοῖς", "citation": "Iliad 1.2"},
            {"id": "t10", "text": "ἄλγε'", "citation": "Iliad 1.2"},
            {"id": "t11", "text": "ἔθηκε", "citation": "Iliad 1.2"},
        ]},
    ],
}


def test_one_sentence_spanning_two_citation_units_keeps_each_token_attributed():
    dspy.configure(lm=DummyLM([CANNED_SPANNING_TWO_UNITS]))
    sentences = segment_sources(SOURCES_SPANNING_TWO_UNITS)

    # It's one sentence, not two -- the whole point of the example.
    assert len(sentences) == 1
    tokens = sentences[0].tokens

    # ids stay globally sequential and unique even though this sentence
    # crosses a citation-unit boundary.
    assert [t.id for t in tokens] == [f"t{i}" for i in range(12)]
    assert len({t.id for t in tokens}) == 12

    # Each half of the sentence is attributed to its own source unit.
    assert [t.citation for t in tokens[:5]] == ["Iliad 1.1"] * 5
    assert [t.citation for t in tokens[5:]] == ["Iliad 1.2"] * 7

    # The exact crossover point: "Ἀχιλῆος" (end of 1.1) into "οὐλομένην"
    # (start of 1.2), still one continuous sentence with no gap or reset
    # in ids.
    assert tokens[4].text == "Ἀχιλῆος" and tokens[4].citation == "Iliad 1.1"
    assert tokens[5].text == "οὐλομένην" and tokens[5].citation == "Iliad 1.2"
    assert tokens[5].id == "t5"
