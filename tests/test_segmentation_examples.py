"""
Runs every fixture in fixtures/segmentation_examples.py through the
deterministic segment_sources() and checks the result matches the expected
sentences exactly: same number of sentences, same token ids/text/citation
in order. Greek analogue of arsgrammatica's test_segmentation_examples.py.

Unlike test_gold_examples.py (which proves the code handles a
correctly-shaped LM answer, not that a real LM would produce it),
segment_sources() has no LM in the loop at all any more -- this is simply
checking the deterministic function against its own correct answer, and
there is no live-LM counterpart left to point to (test_segmentation_live.py
was deleted along with the LM-driven implementation it used to exercise).
"""

import pytest

from conftest import run_segmentation_example
from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES


def _flatten(expected_sentences):
    """(id, text, citation) triples across every sentence in the expected
    answer, in order -- the shape we compare actual output against."""
    return [
        (tok["id"], tok["text"], tok.get("citation"))
        for sentence in expected_sentences["sentences"]
        for tok in sentence["tokens"]
    ]


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda e: e.slug)
def test_segmentation_example_matches_expected(example):
    sentences = run_segmentation_example(example)

    expected_sentence_lengths = [len(s["tokens"]) for s in example.expected_sentences["sentences"]]
    actual_sentence_lengths = [len(s.tokens) for s in sentences]
    assert actual_sentence_lengths == expected_sentence_lengths, example.slug

    expected = _flatten(example.expected_sentences)
    actual = [(t.id, t.text, t.citation) for s in sentences for t in s.tokens]
    assert actual == expected, example.slug


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda e: e.slug)
def test_segmentation_example_ids_are_globally_unique(example):
    sentences = run_segmentation_example(example)
    all_ids = [t.id for s in sentences for t in s.tokens]
    assert len(all_ids) == len(set(all_ids)), example.slug
