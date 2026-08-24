"""
Offline, LM-free tests for grammatike/gepa_metric.py. Greek analogue of
arsgrammatica's test_gepa_metric.py.

These don't touch dspy.GEPA or the network at all -- just the metric
function itself, run against gold examples from fixtures/gold_examples.py
and hand-built "predictions" (some perfect, some deliberately wrong), to
confirm the score and feedback behave sensibly before ever spending a real
LM call on an actual GEPA run.
"""

import dspy
import pytest

from grammatike.gepa_metric import syntax_metric
from grammatike.models import TokenAnalysis, VerbalExpression
from conftest import tokens_from_canned_answer
from fixtures.gold_examples import GOLD_EXAMPLES


def _gold_example(slug):
    example = next(e for e in GOLD_EXAMPLES if e.slug == slug)
    tokens = tokens_from_canned_answer(example.canned_answer)
    verbalunits = [VerbalExpression(**vu) for vu in example.canned_answer["verbalunits"]]
    tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
    return dspy.Example(
        passage=example.passage, tokens=tokens, verbalunits=verbalunits, tokengraph=tokengraph
    ).with_inputs("passage", "tokens")


def _pred_from(gold):
    """A dspy.Prediction that's a perfect copy of `gold`'s outputs -- the
    starting point for tests that then mutate one field to introduce a
    specific, known error."""
    return dspy.Prediction(
        reasoning="(irrelevant to the metric)",
        verbalunits=[vu.model_copy() for vu in gold.verbalunits],
        tokengraph=[tok.model_copy() for tok in gold.tokengraph],
    )


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_perfect_prediction_scores_one(example):
    gold = _gold_example(example.slug)
    pred = _pred_from(gold)
    result = syntax_metric(gold, pred)
    assert result.score == pytest.approx(1.0), result.feedback
    assert "Perfect match" in result.feedback


def test_missing_relation_is_penalized_and_named():
    gold = _gold_example("unit_verb_root_ten_thuran_anoixen")
    pred = _pred_from(gold)
    # τήν (t0) should relate to θύραν (t1) as its article; drop it.
    for tok in pred.tokengraph:
        if tok.id == "t0":
            tok.relatedtoken1 = None
            tok.relationship1 = None
    result = syntax_metric(gold, pred)
    assert result.score < 1.0
    assert "t0" in result.feedback
    assert "article" in result.feedback


def test_extra_hallucinated_relation_is_penalized_and_named():
    gold = _gold_example("unit_verb_root_ten_thuran_anoixen")
    pred = _pred_from(gold)
    # The trailing period (t3) has no relation in the gold answer; hallucinate one.
    for tok in pred.tokengraph:
        if tok.id == "t3":
            tok.relatedtoken1 = "t2"
            tok.relationship1 = "adverbial"
    result = syntax_metric(gold, pred)
    assert result.score < 1.0
    assert "t3" in result.feedback
    assert "unexpected relation" in result.feedback


def test_relatedtoken_slot_swap_is_not_penalized():
    """relatedtoken1/relationship1 vs. relatedtoken2/relationship2 is an
    overflow slot (see models.py's RelationLabel comment) -- a prediction
    that puts the same two relations in the opposite slots from the gold
    answer must score identically to the gold answer itself."""
    gold = _gold_example("relative_pronoun_ho_aner_hon_eidon")
    pred = _pred_from(gold)
    for tok in pred.tokengraph:
        if tok.id == "t2":  # ὅν
            tok.relatedtoken1, tok.relatedtoken2 = tok.relatedtoken2, tok.relatedtoken1
            tok.relationship1, tok.relationship2 = tok.relationship2, tok.relationship1
    result = syntax_metric(gold, pred)
    assert result.score == pytest.approx(1.0), result.feedback


def test_wrong_syntactic_type_is_penalized_and_named():
    gold = _gold_example("circumstantial_fits_clause_ego_hapanta_epideixo")
    pred = _pred_from(gold)
    for vu in pred.verbalunits:
        if vu.id == "t8":  # παραλείπων, should be "circumstantial"
            vu.syntactic_type = "independent"
    result = syntax_metric(gold, pred)
    assert result.score < 1.0
    assert "t8" in result.feedback
    assert "syntactic_type" in result.feedback


def test_missing_verbal_expression_is_penalized_and_named():
    gold = _gold_example("indirect_statement_infinitive_ephaske_lychnon")
    pred = _pred_from(gold)
    pred.verbalunits = [vu for vu in pred.verbalunits if vu.id != "t3"]
    result = syntax_metric(gold, pred)
    assert result.score < 1.0
    assert "t3" in result.feedback
    assert "missing from verbalunits" in result.feedback


def test_completely_empty_prediction_scores_zero():
    gold = _gold_example("unit_verb_root_ten_thuran_anoixen")
    pred = dspy.Prediction(reasoning="(nothing)", verbalunits=[], tokengraph=[])
    result = syntax_metric(gold, pred)
    assert result.score == pytest.approx(0.0), result.feedback


def test_result_exposes_the_three_unblended_dimension_scores():
    """score is a 0.2/0.5/0.3 blend of field_score/relation_score/vu_score
    -- callers that want to know WHERE a prediction fell down (not just by
    how much) should be able to read those three sub-scores directly off
    the returned Prediction, without re-deriving them or parsing the
    feedback string."""
    gold = _gold_example("unit_verb_root_ten_thuran_anoixen")
    pred = _pred_from(gold)
    for tok in pred.tokengraph:
        if tok.id == "t0":
            tok.relatedtoken1 = None
            tok.relationship1 = None
    result = syntax_metric(gold, pred)
    assert result.field_score == pytest.approx(1.0)
    assert result.relation_score < 1.0
    assert result.vu_score == pytest.approx(1.0)
    assert result.score == pytest.approx(
        0.2 * result.field_score + 0.5 * result.relation_score + 0.3 * result.vu_score
    )
