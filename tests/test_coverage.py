"""
Meta-tests: check that every documented enum value in models.py -- relation
labels, verbal-expression classifications, and token types -- has at least
one gold example exercising it. syntax_model.md specifies all of these, not
just relation labels, so "coverage" means all of them. Greek analogue of
arsgrammatica's test_coverage.py.

These are expected to fail as syntax_model.md and models.py grow faster
than fixtures/gold_examples.py -- that's the point. Each failure names
exactly which value(s) still need an example.
"""

import typing

import pytest

from grammatike.models import RelationLabel, VerbalExpression, TokenAnalysis
from fixtures.gold_examples import GOLD_EXAMPLES


def _literal_values(model, field_name):
    """Allowed values for a Literal-typed pydantic field."""
    return set(typing.get_args(model.model_fields[field_name].annotation))


def _relationship_labels_seen():
    return {
        tok.get(field)
        for example in GOLD_EXAMPLES
        for tok in example.canned_answer["tokengraph"]
        for field in ("relationship1", "relationship2")
    } - {None}


def _verbalunit_field_seen(field_name):
    return {
        vu.get(field_name)
        for example in GOLD_EXAMPLES
        for vu in example.canned_answer["verbalunits"]
    } - {None}


def _tokengraph_field_seen(field_name):
    return {
        tok.get(field_name)
        for example in GOLD_EXAMPLES
        for tok in example.canned_answer["tokengraph"]
    } - {None}


# Each entry is one thing syntax_model.md specifies a fixed set of values
# for: (readable name, values actually seen in GOLD_EXAMPLES, values allowed
# by the pydantic model). Add a row here whenever models.py gains a new
# Literal-typed field -- nothing else about this file needs to change.
COVERAGE_DIMENSIONS = [
    (
        "relationship label",
        _relationship_labels_seen(),
        set(typing.get_args(RelationLabel)),
    ),
    (
        "verbal expression syntactic_type",
        _verbalunit_field_seen("syntactic_type"),
        _literal_values(VerbalExpression, "syntactic_type"),
    ),
    (
        "verbal expression semantic_type",
        _verbalunit_field_seen("semantic_type"),
        _literal_values(VerbalExpression, "semantic_type"),
    ),
    (
        "token tokentype",
        _tokengraph_field_seen("tokentype"),
        _literal_values(TokenAnalysis, "tokentype"),
    ),
]


@pytest.mark.parametrize(
    "name, seen, allowed",
    COVERAGE_DIMENSIONS,
    ids=[dimension[0] for dimension in COVERAGE_DIMENSIONS],
)
def test_dimension_fully_covered(name, seen, allowed):
    missing = allowed - seen
    assert not missing, f"no gold example exercises {name}: {sorted(missing)}"
