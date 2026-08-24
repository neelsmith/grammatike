"""
Tests for validate() itself -- proving it actually catches bad output, as
opposed to test_gold_examples.py, which checks that well-formed gold
answers pass cleanly. Greek analogue of arsgrammatica's test_validate.py.
Reuses a gold example as a convenient base to mutate rather than defining
its own fixture.
"""

import dspy
from dspy.utils.dummies import DummyLM

from grammatike import analyze, validate
from conftest import tokens_from_canned_answer
from fixtures.gold_examples import GOLD_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


def test_bad_answer_is_caught():
    """A response that refers to a nonexistent token id should be flagged
    by validate(), not silently accepted."""
    base = _example("unit_verb_root_ten_thuran_anoixen")

    # Copy before mutating -- GOLD_EXAMPLES is shared across the whole
    # suite, so this must not touch the original dict or list in place.
    bad_answer = dict(base.canned_answer)
    bad_answer["tokengraph"] = list(base.canned_answer["tokengraph"])
    bad_answer["tokengraph"][0] = {
        **base.canned_answer["tokengraph"][0],
        "relatedtoken1": "t99",  # does not exist
    }

    dspy.configure(lm=DummyLM([bad_answer]))
    tokens = tokens_from_canned_answer(base.canned_answer)
    result = analyze(passage=base.passage, tokens=tokens)

    problems = validate(tokens, result)
    assert problems, "expected validate() to catch the bogus id 't99', but it found nothing"


# ---------------------------------------------------------------------------
# Implied/elided tokens (tokentype='implied eimi'/'implied repetition'; see
# models.py's TokenAnalysis and IMPLIED_TOKENTYPES)
# ---------------------------------------------------------------------------
#
# These run validate() directly against hand-built tokens/result objects
# (rather than through analyze()+DummyLM like test_bad_answer_is_caught
# above) since the point here is validate()'s own acceptance/rejection
# logic for the implied-token fields specifically, not the whole pipeline --
# the gold-example-backed tests (test_gold_examples.py's
# test_gold_example_validates) already cover the well-formed case for both
# implied_* fixtures end to end.

from grammatike.models import Token, TokenAnalysis, VerbalExpression


def _result(tokengraph, verbalunits=()):
    return dspy.Prediction(tokengraph=list(tokengraph), verbalunits=list(verbalunits))


def _well_formed_implied_case():
    """"ταῦτα [ἐστι] καλά." -- a minimal well-formed implied-token case:
    t0 is the real token, t0_implied is the new implied one anchoring it."""
    tokens = [Token(id="t0", text="ταῦτα")]
    tokengraph = [
        TokenAnalysis(id="t0", token="ταῦτα", tokentype="lexical",
                      relatedtoken1="t0_implied", relationship1="subject"),
        TokenAnalysis(id="t0_implied", token=None, tokentype="implied eimi",
                      verbalunitid="t0_implied", relatedtoken1="root", relationship1="unit verb"),
    ]
    verbalunits = [VerbalExpression(id="t0_implied", syntactic_type="independent", semantic_type="linking verb")]
    return tokens, tokengraph, verbalunits


def test_well_formed_implied_token_is_accepted():
    """The baseline well-formed case must NOT be flagged -- a new id, not
    in `tokens`, with token=None, is exactly what an implied entry is
    supposed to look like."""
    tokens, tokengraph, verbalunits = _well_formed_implied_case()
    problems = validate(tokens, _result(tokengraph, verbalunits))
    assert not problems, problems


def test_implied_token_reusing_a_real_id_is_caught():
    """An implied entry (tokentype='implied eimi' or 'implied repetition')
    must use a NEW id -- reusing one already in `tokens` (as if it were
    describing that real token) is malformed, not a legitimate implied
    token."""
    tokens, tokengraph, verbalunits = _well_formed_implied_case()
    tokengraph[1] = TokenAnalysis(
        id="t0", token=None, tokentype="implied eimi",  # reuses t0's own id
        verbalunitid="t0", relatedtoken1="root", relationship1="unit verb",
    )
    tokengraph[0].relatedtoken1 = "t0"
    verbalunits[0].id = "t0"
    problems = validate(tokens, _result(tokengraph, verbalunits))
    assert problems, "expected validate() to catch an implied entry reusing a real token's id"
    assert any("reuses an id" in p for p in problems), problems


def test_implied_token_with_real_text_is_caught():
    """tokentype='implied eimi'/'implied repetition' with a non-None
    `token` value contradicts itself -- an implied token is defined by
    having NO surface text."""
    tokens, tokengraph, verbalunits = _well_formed_implied_case()
    tokengraph[1].token = "ἐστι"  # should have stayed None
    problems = validate(tokens, _result(tokengraph, verbalunits))
    assert problems, "expected validate() to catch an implied entry with real token text"
    assert any("non-None token value" in p for p in problems), problems


def test_non_implied_token_with_none_text_is_caught():
    """The inverse case: a token that ISN'T marked with one of the implied
    tokentypes must not have token=None -- only 'implied eimi'/'implied
    repetition' may omit real surface text."""
    tokens, tokengraph, verbalunits = _well_formed_implied_case()
    tokengraph[0].token = None  # t0 is tokentype='lexical', not implied
    problems = validate(tokens, _result(tokengraph, verbalunits))
    assert problems, "expected validate() to catch a non-implied token with token=None"
    assert any("may omit surface text" in p for p in problems), problems


def test_root_id_reserved_is_caught():
    """A real token literally named 'root' must be flagged -- 'root' is
    reserved as the sentinel relatedtoken1 value for independent verbs."""
    tokens = [Token(id="root", text="oops")]
    tokengraph = [TokenAnalysis(id="root", token="oops", tokentype="lexical")]
    problems = validate(tokens, _result(tokengraph))
    assert any("reserved" in p for p in problems), problems
