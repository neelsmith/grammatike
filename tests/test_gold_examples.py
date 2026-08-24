"""
Runs every fixture in fixtures/gold_examples.py through the real pipeline
(analyze() backed by DummyLM, not a live model) and checks that:

- validate() finds no referential-integrity problems, and
- tokengraph_to_mermaid() renders every non-punctuation token as a node,
  with no warnings.

This is a structural check, not an accuracy check: it confirms the code
handles each gold answer correctly, not that a real LM would produce that
answer. Add new fixtures to fixtures/gold_examples.py, not here -- these
tests are parametrized over GOLD_EXAMPLES and need no changes to cover a
new sentence. Greek analogue of arsgrammatica's test_gold_examples.py.
"""

import pytest

from grammatike import validate, tokengraph_to_mermaid
from conftest import run_gold_example
from fixtures.gold_examples import GOLD_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_gold_example_validates(example):
    tokens, result = run_gold_example(example)
    problems = validate(tokens, result)
    assert not problems, f"{example.slug}: {problems}"


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_gold_example_renders_mermaid(example):
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_mermaid(result.tokengraph)
    assert not warnings, f"{example.slug}: {warnings}"
    for tok in result.tokengraph:
        if tok.tokentype != "punctuation":
            assert f'{tok.id}["' in diagram, f"{example.slug}: missing node for {tok.id}"


# --- Spot-checks specific to one gold example -------------------------------
# The parametrized tests above are deliberately generic (they have to hold
# for every fixture); a few hand-picked assertions about *this* sentence's
# expected edges are worth keeping too.

def test_ten_thuran_specific_edges():
    tokens, result = run_gold_example(_example("unit_verb_root_ten_thuran_anoixen"))
    diagram, _warnings = tokengraph_to_mermaid(result.tokengraph)

    # The trailing period must not become a node.
    assert 't3["' not in diagram

    assert "t2 -->|article| t1" not in diagram  # sanity: no reversed edge
    assert "t1 -->|direct object| t2" in diagram
    assert "t0 -->|article| t1" in diagram
