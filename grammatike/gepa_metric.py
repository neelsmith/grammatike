"""
grammatike: GEPA optimization metric for Greek syntax analysis.

Greek analogue of arsgrammatica's gepa_metric.py. Scores greek_syntax_dspy's
SyntaxAnalysis signature.

GEPA (see dspy.GEPA) is a "reflective" prompt optimizer: it uses an LM to
read the metric's *feedback* text -- not just its numeric score -- and
propose better instructions. That makes the feedback string here at least
as important as the score itself: it should read like a teacher's marginal
comments on a student's parse, naming the actual token, the actual
relation, and the actual expected-vs-got values, in the same vocabulary
syntax_model.md and SyntaxAnalysis's docstring use.

Scoring is structural, not exact-dict-equality, in one deliberate way:
relatedtoken1/relationship1 vs. relatedtoken2/relationship2 is documented
as an overflow slot (see syntax_model.md and models.py's RelationLabel
comment) for a token that needs to record two relations at once (e.g. a
relative pronoun that is also a clause's subject, or its own function
inside its own relative clause -- syntax_model.md's worked example is the
relative pronoun ὃν in "ὁ τῆς πόλεως νόμος, ὃν σὺ περὶ ἐλάττονος τῶν
ἡδονῶν ἐποιήσω", which relates both to its antecedent νόμος
(relationship1 = "relative pronoun") and to ἐποιήσω (relationship2 =
"direct object")). A predicted answer that puts the same two relations in
the opposite slots from the gold answer is exactly as correct as the gold
answer -- so relations are compared as an unordered set of (token id,
relationship label, related token id) triples per tokengraph, not as a
positional relatedtoken1/relatedtoken2 comparison.

Everything else this metric checks (tokentype, lemma, verbalunitid, and
verbal-expression syntactic_type/semantic_type) is a plain per-field
comparison. None of this file hardcodes any actual RelationLabel or
tokentype value -- the label/type vocabulary lives entirely in models.py,
so this metric works unchanged for any set of Greek labels defined there.

This module has no dependency on tests/fixtures/gold_examples.py or dspy's
GEPA machinery itself -- optimize_gepa.py wires this metric, GOLD_EXAMPLES,
and dspy.GEPA together. Keeping the metric here, dependency-free, makes it
importable and unit-testable (see tests/test_gepa_metric.py) without ever
touching the network or the GOLD_EXAMPLES fixtures module.
"""

from typing import Any, Optional

import dspy


def _get(obj: Any, field: str) -> Any:
    """Read `field` off `obj`, whether `obj` is a pydantic model instance
    (VerbalExpression/TokenAnalysis, from a real prediction) or a plain
    dict (a canned_answer entry, if a caller passes those directly)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _relation_triples(tokengraph_by_id: dict) -> set:
    """Every (token id, relationship label, related token id) triple in a
    tokengraph, collapsing the relatedtoken1/relationship1 vs.
    relatedtoken2/relationship2 slots into one unordered set (see module
    docstring for why)."""
    triples = set()
    for tid, tok in tokengraph_by_id.items():
        for related_field, label_field in (
            ("relatedtoken1", "relationship1"),
            ("relatedtoken2", "relationship2"),
        ):
            related = _get(tok, related_field)
            label = _get(tok, label_field)
            if related is not None and label is not None:
                triples.add((tid, label, related))
    return triples


def _safe_ratio(correct: int, total: int) -> float:
    """1.0 for a 0/0 check (nothing to get wrong -> nothing wrong), not 0.0
    -- e.g. a sentence with no lemma-bearing tokens shouldn't be penalized
    on the lemma dimension."""
    return correct / total if total else 1.0


def syntax_metric(
    gold: "dspy.Example",
    pred: "dspy.Prediction",
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
    program_trace: Optional[Any] = None,
) -> "dspy.Prediction":
    """Score a SyntaxAnalysis prediction against a gold answer built from a
    GoldExample (see optimize_gepa.py's build_trainset()).

    `gold` must have `.tokengraph` (list of TokenAnalysis) and
    `.verbalunits` (list of VerbalExpression) fields -- the same shape
    SyntaxAnalysis itself outputs, so the exact same dict-or-model access
    pattern works for both the gold example and a live prediction.

    Returns a dspy.Prediction(score=..., feedback=...) -- GEPA's expected
    "ScoreWithFeedback" shape (see dspy.teleprompt.gepa.gepa_utils). `score`
    is a single float in [0, 1], a weighted blend of three dimensions
    (relations, verbal-expression classification, and basic per-token
    fields); `feedback` is a human-readable list of every mismatch found,
    for GEPA's reflection LM to read. There's only one predictor in
    `analyze` (SyntaxAnalysis is a single dspy.ChainOfThought), so
    `pred_name`/`pred_trace` are accepted for protocol compatibility but
    not used to change scoring -- GEPA will use the same feedback whether
    it's asking at the program level or the (only) predictor level.

    The returned Prediction also carries the three unblended dimension
    scores as their own fields -- `field_score`, `relation_score`,
    `vu_score` -- alongside `score` and `feedback`. GEPA itself only reads
    `score`/`feedback`, so this is purely additive (existing callers that
    only look at those two are unaffected), but it's useful for any caller
    that wants to know WHERE a prediction fell down rather than just by how
    much -- e.g. a model-bakeoff script uses these to tell "gets the
    surface tokenization right but can't chase multi-hop relations" apart
    from "just generally worse," which a single blended number can't
    distinguish.

    The 0.2/0.5/0.3 weighting (fields/relations/verbal-expressions) is a
    judgment call, not something syntax_model.md specifies -- relations are
    weighted highest since they're the heart of the scheme, but this is an
    easy knob to retune once real GEPA runs are observed.
    """
    problems = []

    gold_tg = {_get(t, "id"): t for t in gold.tokengraph}
    pred_tokengraph = list(getattr(pred, "tokengraph", None) or [])
    pred_tg = {}
    for tok in pred_tokengraph:
        tid = _get(tok, "id")
        if tid in pred_tg:
            problems.append(
                f"tokengraph has more than one entry for id {tid!r} "
                "(only the first is scored)"
            )
            continue
        pred_tg[tid] = tok

    gold_ids = set(gold_tg)
    pred_ids = set(pred_tg)

    field_total = 0
    field_correct = 0

    for tid in gold_ids:
        g = gold_tg[tid]
        gtext = _get(g, "token")
        if tid not in pred_tg:
            problems.append(f"token {tid} ({gtext!r}) is missing from tokengraph entirely")
            field_total += 2
            continue
        p = pred_tg[tid]
        for field in ("tokentype", "verbalunitid"):
            gval = _get(g, field)
            pval = _get(p, field)
            field_total += 1
            if gval == pval:
                field_correct += 1
            else:
                problems.append(f"token {tid} ({gtext!r}): expected {field}={gval!r}, got {pval!r}")
        gval = _get(g, "lemma")
        if gval is not None:
            pval = _get(p, "lemma")
            field_total += 1
            if gval == pval:
                field_correct += 1
            else:
                problems.append(f"token {tid} ({gtext!r}): expected lemma={gval!r}, got {pval!r}")

    extra_tok_ids = pred_ids - gold_ids
    if extra_tok_ids:
        problems.append(
            f"tokengraph has entries for id(s) not in the input tokens: {sorted(extra_tok_ids)}"
        )

    gold_rels = _relation_triples(gold_tg)
    pred_rels = _relation_triples(pred_tg)
    missing_rels = gold_rels - pred_rels
    extra_rels = pred_rels - gold_rels

    for tid, label, related in sorted(missing_rels):
        gtext = _get(gold_tg.get(tid), "token")
        problems.append(f"token {tid} ({gtext!r}) is missing the relation {label!r} -> {related}")
    for tid, label, related in sorted(extra_rels):
        text = _get(pred_tg.get(tid), "token")
        problems.append(
            f"token {tid} ({text!r}) has an unexpected relation {label!r} -> {related} "
            "not in the gold answer"
        )

    relation_denominator = len(gold_rels) + len(extra_rels)
    relation_correct = len(gold_rels) - len(missing_rels)
    relation_score = _safe_ratio(relation_correct, relation_denominator)

    gold_vu = {_get(v, "id"): v for v in gold.verbalunits}
    pred_vu = {_get(v, "id"): v for v in (getattr(pred, "verbalunits", None) or [])}

    vu_total = 0
    vu_correct = 0
    for vid, gv in gold_vu.items():
        gtext = _get(gold_tg.get(vid), "token") or vid
        if vid not in pred_vu:
            problems.append(f"verbal expression at {vid} ({gtext!r}) is missing from verbalunits entirely")
            vu_total += 2
            continue
        pv = pred_vu[vid]
        for field in ("syntactic_type", "semantic_type"):
            vu_total += 1
            if _get(gv, field) == _get(pv, field):
                vu_correct += 1
            else:
                problems.append(
                    f"verbal expression at {vid} ({gtext!r}): expected {field}={_get(gv, field)!r}, "
                    f"got {_get(pv, field)!r}"
                )

    extra_vu_ids = set(pred_vu) - set(gold_vu)
    if extra_vu_ids:
        problems.append(
            "verbalunits has unexpected extra entries not anchored on a gold verbal "
            f"expression: {sorted(extra_vu_ids)}"
        )

    field_score = _safe_ratio(field_correct, field_total)
    vu_score = _safe_ratio(vu_correct, vu_total)

    score = 0.2 * field_score + 0.5 * relation_score + 0.3 * vu_score

    if not problems:
        feedback = (
            "Perfect match with the gold analysis: every token's fields, relations, "
            "and verbal-expression classification are correct."
        )
    else:
        feedback = (
            f"Score {score:.2f} (fields {field_score:.2f}, relations {relation_score:.2f}, "
            f"verbal expressions {vu_score:.2f}). Problems found:\n- " + "\n- ".join(problems)
        )

    return dspy.Prediction(
        score=score,
        feedback=feedback,
        field_score=field_score,
        relation_score=relation_score,
        vu_score=vu_score,
    )
