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


def test_crasis_splits_kago_into_two_tokens():
    """syntax_model.md's 'Two special notes': κἀγώ, crasis for καὶ ἐγώ,
    must be entered as two lexical tokens κ and ἀγώ -- not lemmatized here
    (Token at this stage has no lemma field), just split, the same shape
    as the existing -περ enclitic split but via segmentation.py's own
    _CRASIS_COMPOUNDS table."""
    sentences = segment_sources([
        CitedText(citation="ex.9", text="ταῦτα εἶδε κἀγώ."),
    ])
    assert len(sentences) == 1
    tokens = sentences[0].tokens
    assert [t.text for t in tokens] == ["ταῦτα", "εἶδε", "κ", "ἀγώ", "."]
    assert [t.id for t in tokens] == [f"t{i}" for i in range(5)]
    assert all(t.citation == "ex.9" for t in tokens)


def test_crasis_of_ho_autos_splits_in_both_its_accent_states():
    """ὡυτός (crasis of ὁ αὐτός, 'the same') is a whole recurring paradigm,
    not a one-off idiom like κἀγώ -- and because _CRASIS_COMPOUNDS matches
    literally, its oxytone forms need BOTH accent-states registered: the
    acute spelling used sentence-finally/before punctuation ('ὡυτός') and
    the grave spelling used mid-clause ('ὡυτὸς'), per the regular Greek
    acute-to-grave shift. Both must split the same way."""
    acute = segment_sources([
        CitedText(citation="ex.13", text="ἀνὴρ ἦν ὡυτός."),
    ])
    assert [t.text for t in acute[0].tokens] == ["ἀνὴρ", "ἦν", "ὡ", "υτός", "."]

    grave = segment_sources([
        CitedText(citation="ex.14", text="ὡυτὸς ἀνὴρ ἦλθεν."),
    ])
    assert [t.text for t in grave[0].tokens] == ["ὡ", "υτὸς", "ἀνὴρ", "ἦλθεν", "."]


def test_crasis_of_ho_autos_oblique_and_neuter_and_plural_forms():
    """Spot-checks the rest of the seeded ὁ αὐτός paradigm beyond the
    nominative singular: a circumflex-accented oblique case (ὡυτοῦ, which
    -- unlike ὡυτός -- never shifts to grave, so it needs only the one
    spelling), the neuter τὠυτό (which keeps the article's own leading
    τ), and the masculine plural ὡυτοί."""
    genitive = segment_sources([
        CitedText(citation="ex.15", text="τοῦτο ὡυτοῦ ἐστιν."),
    ])
    assert [t.text for t in genitive[0].tokens] == ["τοῦτο", "ὡ", "υτοῦ", "ἐστιν", "."]

    neuter = segment_sources([
        CitedText(citation="ex.16", text="τὠυτό ἐστιν."),
    ])
    assert [t.text for t in neuter[0].tokens] == ["τὠ", "υτό", "ἐστιν", "."]

    plural = segment_sources([
        CitedText(citation="ex.17", text="ὡυτοί εἰσιν ἄνδρες."),
    ])
    assert [t.text for t in plural[0].tokens] == ["ὡ", "υτοί", "εἰσιν", "ἄνδρες", "."]


def test_ho_ti_pronoun_merges_but_hoti_conjunction_does_not():
    """The other of syntax_model.md's 'Two special notes': the two
    space-delimited words 'ὅ' and 'τι' (neuter nom./acc. singular of
    ὅστις) must merge into a SINGLE lexical token 'ὅ τι', to keep it
    distinct from the one-word conjunction ὅτι, which stays exactly as
    written -- one word, one token, untouched by the merge pre-pass."""
    sentences = segment_sources([
        CitedText(citation="ex.10", text="οὐκ οἶδα ὅ τι λέγεις."),
        CitedText(citation="ex.11", text="εὖ οἶδα ὅτι ἀληθῆ λέγεις."),
    ])
    assert len(sentences) == 2

    ho_ti_tokens = sentences[0].tokens
    assert [t.text for t in ho_ti_tokens] == ["οὐκ", "οἶδα", "ὅ τι", "λέγεις", "."]
    assert [t.id for t in ho_ti_tokens] == [f"t{i}" for i in range(5)]

    hoti_tokens = sentences[1].tokens
    assert [t.text for t in hoti_tokens] == ["εὖ", "οἶδα", "ὅτι", "ἀληθῆ", "λέγεις", "."]
    assert [t.id for t in hoti_tokens] == [f"t{i}" for i in range(5, 11)]

    # ids stay globally sequential and unique across both sentences, exactly
    # as the existing citation-spanning test above already checks.
    all_ids = [t.id for s in sentences for t in s.tokens]
    assert len(all_ids) == len(set(all_ids))


def test_ho_ti_merges_correctly_with_trailing_punctuation_attached():
    """'ὅ τι' followed directly by internal punctuation (a comma here, not
    the final period) must still merge into one token, with the comma
    split off separately as its own token -- exercising _strip_punctuation
    being shared correctly between _merge_multiword_tokens() and
    _split_word()."""
    sentences = segment_sources([
        CitedText(citation="ex.12", text="ὅ τι, βούλει, ποίει."),
    ])
    assert len(sentences) == 1
    tokens = sentences[0].tokens
    assert [t.text for t in tokens] == [
        "ὅ τι", ",", "βούλει", ",", "ποίει", ".",
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
