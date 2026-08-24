"""
Tests for pipeline.py's analyze_passage() -- the convenience wrapper around
analyze_sources(). Greek analogue of arsgrammatica's test_analyze_passage.py.

analyze_passage() wraps `passage` as one CitedText and returns exactly what
analyze_sources() returns, (sentences, results), one entry per sentence
segmentation finds -- no restriction to a single sentence.

Segmentation itself is deterministic (segmentation.py, no LM call at all --
see that module's own docstring), so DummyLM here only needs to supply
canned answers for the SyntaxAnalysis call(s) analyze_passage() makes per
sentence it actually finds -- not for segmentation, unlike an earlier
version of this test file from when segmentation was itself LM-driven.

Two things worth checking that nothing else in the suite exercises:
  - the single-sentence case still returns one-element (sentences, results)
    lists, with the right tokens/citation/verbalunits;
  - the multi-sentence case analyzes every sentence it finds, in order,
    with each sentence's own SyntaxAnalysis result lining up positionally.
"""

import dspy
from dspy.utils.dummies import DummyLM

from grammatike import analyze_passage

_ONE_SENTENCE_ANALYSIS = {
    "reasoning": "ἀνέῳξεν is the main verb; θύραν is its direct object.",
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "τήν", "tokentype": "lexical",
         "relatedtoken1": "t1", "relationship1": "article"},
        {"id": "t1", "token": "θύραν", "tokentype": "lexical",
         "relatedtoken1": "t2", "relationship1": "direct object"},
        {"id": "t2", "token": "ἀνέῳξεν", "tokentype": "lexical", "verbalunitid": "t2",
         "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": ".", "tokentype": "punctuation"},
    ],
}

_HOUTOS_ANALYSIS = {
    "reasoning": "No finite verb in this fragment; οὕτως is a bare adverb, no verbal units.",
    "verbalunits": [],
    "tokengraph": [
        {"id": "t0", "token": "οὕτως", "tokentype": "lexical"},
        {"id": "t1", "token": ".", "tokentype": "punctuation"},
    ],
}

_CHAIRE_ANALYSIS = {
    "reasoning": "χαῖρε is an imperative verbal expression.",
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t2", "token": "χαῖρε", "tokentype": "lexical", "verbalunitid": "t2",
         "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": ".", "tokentype": "punctuation"},
    ],
}


def test_analyze_passage_single_sentence_returns_one_sentence_and_result():
    dspy.configure(lm=DummyLM([_ONE_SENTENCE_ANALYSIS]))

    sentences, results = analyze_passage("τήν θύραν ἀνέῳξεν.", citation="ex.1")

    assert len(sentences) == 1
    assert len(results) == 1
    tokens = sentences[0].tokens
    assert [t.id for t in tokens] == ["t0", "t1", "t2", "t3"]
    assert [t.text for t in tokens] == ["τήν", "θύραν", "ἀνέῳξεν", "."]
    assert all(t.citation == "ex.1" for t in tokens)
    assert results[0].verbalunits[0].id == "t2"


def test_analyze_passage_multi_sentence_analyzes_every_sentence_in_order():
    dspy.configure(lm=DummyLM([_HOUTOS_ANALYSIS, _CHAIRE_ANALYSIS]))

    sentences, results = analyze_passage("οὕτως. χαῖρε.", citation="ex.2")

    assert len(sentences) == 2
    assert len(results) == 2

    assert [t.id for t in sentences[0].tokens] == ["t0", "t1"]
    assert [t.id for t in sentences[1].tokens] == ["t2", "t3"]

    # results line up positionally with sentences: results[0] is "οὕτως."'s
    # analysis (no verbal unit), results[1] is "χαῖρε."'s (one, at t2).
    assert results[0].verbalunits == []
    assert results[1].verbalunits[0].id == "t2"
