"""
Runs every fixture in fixtures/segmentation_examples.py through
segment_sources() (DummyLM-backed) and checks the result matches the gold
sentences exactly: same number of sentences, same token ids/text/citation
in order. Greek analogue of arsgrammatica's test_segmentation_examples.py.

This is a structural check, same caveat as test_gold_examples.py: it proves
the code handles a correctly-shaped answer, not that a real LM would
produce it. See test_segmentation_live.py for the live-LM half of this.
"""

import pytest

from conftest import run_segmentation_example
from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES


def _flatten(canned_sentences):
    """(id, text, citation) triples across every sentence in a canned
    answer, in order -- the shape we compare actual output against."""
    return [
        (tok["id"], tok["text"], tok.get("citation"))
        for sentence in canned_sentences["sentences"]
        for tok in sentence["tokens"]
    ]


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda e: e.slug)
def test_segmentation_example_matches_gold(example):
    sentences = run_segmentation_example(example)

    expected_sentence_lengths = [len(s["tokens"]) for s in example.canned_sentences["sentences"]]
    actual_sentence_lengths = [len(s.tokens) for s in sentences]
    assert actual_sentence_lengths == expected_sentence_lengths, example.slug

    expected = _flatten(example.canned_sentences)
    actual = [(t.id, t.text, t.citation) for s in sentences for t in s.tokens]
    assert actual == expected, example.slug


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda e: e.slug)
def test_segmentation_example_ids_are_globally_unique(example):
    sentences = run_segmentation_example(example)
    all_ids = [t.id for s in sentences for t in s.tokens]
    assert len(all_ids) == len(set(all_ids)), example.slug
