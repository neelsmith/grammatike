"""
Live counterpart to test_segmentation_examples.py: runs the same four
scenarios against the actual configured LM (via the `real_lm` fixture in
conftest.py) instead of DummyLM. Greek analogue of arsgrammatica's
test_segmentation_live.py.

This is the half of the picture DummyLM tests structurally cannot cover --
whether the LM itself actually performs the context-dependent segmentation
correctly, not just whether the code can represent a correct answer. Costs
real API calls, so it's marked `live` and skipped by default; run it with:

    pytest -m live tests/test_segmentation_live.py

I have not been able to run this against the configured model myself --
verify it once against the real thing before trusting these four gaps are
actually closed, not just well-specified.
"""

import pytest

from grammatike.models import CitedText
from grammatike.segmentation_dspy import segment_sources

pytestmark = pytest.mark.live


def _texts(sentence):
    return [t.text for t in sentence.tokens]


def test_live_peri_is_not_split(real_lm):
    sentences = segment_sources([CitedText(citation="ex.1", text="περὶ τούτου λέγει.")])
    assert _texts(sentences[0]) == ["περὶ", "τούτου", "λέγει", "."]


def test_live_eiper_splits_into_ei_plus_per(real_lm):
    sentences = segment_sources(
        [CitedText(citation="ex.2", text="εἴπερ ἀληθῆ λέγει, πείσομαι.")]
    )
    assert _texts(sentences[0]) == ["εἴ", "περ", "ἀληθῆ", "λέγει", ",", "πείσομαι", "."]


def test_live_milesian_numeral_stays_its_own_token(real_lm):
    sentences = segment_sources([CitedText(citation="ex.3", text="τῇ γʹ ἡμέρᾳ ἦλθεν.")])
    assert _texts(sentences[0]) == ["τῇ", "γʹ", "ἡμέρᾳ", "ἦλθεν", "."]


def test_live_raised_dot_is_not_a_sentence_boundary(real_lm):
    sentences = segment_sources(
        [CitedText(citation="ex.5", text="ἐγὼ μὲν ἔμεινα· σὺ δὲ ἀπῆλθες.")]
    )
    assert len(sentences) == 1
    assert _texts(sentences[0]) == [
        "ἐγὼ", "μὲν", "ἔμεινα", "·", "σὺ", "δὲ", "ἀπῆλθες", ".",
    ]


def test_live_greek_question_mark_ends_the_sentence(real_lm):
    sentences = segment_sources([CitedText(citation="ex.6", text="τίς ἦλθεν;")])
    assert len(sentences) == 1
    assert _texts(sentences[0]) == ["τίς", "ἦλθεν", ";"]
