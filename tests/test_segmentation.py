"""
Tests for the segmentation stage (grammatike/segmentation.py), including
citation tracking, plus one end-to-end check that its output composes with
the *existing, unmodified* SyntaxAnalysis. Greek analogue of
arsgrammatica's test_segmentation.py.

segment_sources() itself is deterministic (see segmentation.py's own module
docstring) -- these tests call it directly, no DummyLM involved. DummyLM
still shows up in test_segmented_sentence_feeds_unmodified_syntax_analysis,
but only for the SyntaxAnalysis half of that test, which is still LM-driven.
"""

import re

import dspy
from dspy.utils.dummies import DummyLM

from grammatike import analyze, validate
from grammatike.models import CitedText
from grammatike.segmentation import segment_sources
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


def test_segmentation_round_trips_with_globally_unique_sequential_ids():
    sentences = segment_sources(SOURCES_ONE_UNIT)

    assert len(sentences) == 2
    assert [t.id for t in sentences[0].tokens] == [f"t{i}" for i in range(0, 4)]
    assert [t.id for t in sentences[1].tokens] == [f"t{i}" for i in range(4, 8)]
    assert [t.text for t in sentences[0].tokens] == ["ἡ", "ναῦς", "ἀπόλλυται", "."]
    assert [t.text for t in sentences[1].tokens] == ["τήν", "θύραν", "ἀνέῳξεν", "."]

    all_ids = [t.id for s in sentences for t in s.tokens]
    assert len(all_ids) == len(set(all_ids)), "token ids must be unique across the whole input"

    assert all(t.citation == "Lysias 1.1" for s in sentences for t in s.tokens)


def test_segmentation_is_deterministic_across_repeated_calls():
    """Running segment_sources() again on the same input must produce the
    exact same ids for the exact same tokens -- segmentation.py's own
    module docstring promises this now that there's no LM involved."""
    first = segment_sources(SOURCES_ONE_UNIT)
    second = segment_sources(SOURCES_ONE_UNIT)
    flatten = lambda sentences: [(t.id, t.text, t.citation) for s in sentences for t in s.tokens]
    assert flatten(first) == flatten(second)


def test_segmented_sentence_feeds_unmodified_syntax_analysis():
    """The point of keeping segmentation as a separate stage: its output
    (a Sentence's tokens) must work as SyntaxAnalysis's input with zero
    changes to SyntaxAnalysis -- even though its tokens now carry a
    citation field SyntaxAnalysis has never heard of."""
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


def test_one_sentence_spanning_two_citation_units_keeps_each_token_attributed():
    sentences = segment_sources(SOURCES_SPANNING_TWO_UNITS)

    # It's one sentence, not two -- neither source unit ends in a period or
    # interrogative, so nothing ever triggers a split.
    assert len(sentences) == 1
    tokens = sentences[0].tokens

    assert [t.text for t in tokens] == [
        "μῆνιν", "ἄειδε", "θεὰ", "Πηληϊάδεω", "Ἀχιλῆος",
        "οὐλομένην", ",", "ἣ", "μυρί'", "Ἀχαιοῖς", "ἄλγε'", "ἔθηκε",
    ]

    # ids stay globally sequential and unique even though this sentence
    # crosses a citation-unit boundary.
    assert [t.id for t in tokens] == [f"t{i}" for i in range(12)]
    assert len({t.id for t in tokens}) == 12

    # Each half of the sentence is attributed to its own source unit.
    assert [t.citation for t in tokens[:5]] == ["Iliad 1.1"] * 5
    assert [t.citation for t in tokens[5:]] == ["Iliad 1.2"] * 7

    # The exact crossover point: "Ἀχιλῆος" (end of 1.1) into "οὐλομένην"
    # (start of 1.2), still one continuous sentence with no gap or reset
    # in ids. Also confirms the elision apostrophes in μυρί'/ἄλγε' stay
    # fused to their words rather than becoming their own punctuation
    # tokens (see segmentation.py's _ELISION_MARKS).
    assert tokens[4].text == "Ἀχιλῆος" and tokens[4].citation == "Iliad 1.1"
    assert tokens[5].text == "οὐλομένην" and tokens[5].citation == "Iliad 1.2"
    assert tokens[5].id == "t5"
