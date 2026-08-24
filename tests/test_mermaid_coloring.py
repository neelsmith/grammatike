"""
Tests for mermaid.py's color_by_verbal_unit support (default True) --
separate from test_gold_examples.py's generic renders-cleanly checks, since
these specifically exercise the classDef/class coloring output, not just
node/edge rendering. Greek analogue of arsgrammatica's
test_mermaid_coloring.py.
"""

import re

import pytest

from grammatike import tokengraph_to_mermaid, validate
from conftest import run_gold_example
from fixtures.gold_examples import GOLD_EXAMPLES


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_coloring_adds_no_new_warnings(example):
    """Coloring is a purely additive rendering step -- it must never
    introduce a warning that plain rendering (color_by_verbal_unit=False)
    doesn't already have, for any well-formed gold fixture."""
    tokens, result = run_gold_example(example)
    _plain_diagram, plain_warnings = tokengraph_to_mermaid(
        result.tokengraph, color_by_verbal_unit=False
    )
    _colored_diagram, colored_warnings = tokengraph_to_mermaid(
        result.tokengraph, color_by_verbal_unit=True
    )
    assert colored_warnings == plain_warnings, example.slug


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_every_colored_node_gets_exactly_one_class(example):
    """Covers both verbal-unit classes (vuN) and the dedicated `implied`
    class an implied/elided token always gets instead (see
    tokengraph_to_mermaid()'s own docstring) -- every node should still end
    up in exactly one class either way."""
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_mermaid(result.tokengraph)

    class_of = {}
    for line in diagram.splitlines():
        m = re.match(r"\s*class ([\w,]+) (vu\d+|implied);", line)
        if m:
            ids = m.group(1).split(",")
            class_name = m.group(2)
            for tid in ids:
                assert tid not in class_of, f"{example.slug}: {tid} assigned to more than one class"
                class_of[tid] = class_name

    classdefs = set(re.findall(r"classDef (vu\d+|implied) ", diagram))
    assert set(class_of.values()) <= classdefs, example.slug


def test_disabling_coloring_reproduces_the_old_plain_diagram():
    example = next(e for e in GOLD_EXAMPLES if e.slug == "unit_verb_root_ten_thuran_anoixen")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_mermaid(result.tokengraph, color_by_verbal_unit=False)
    assert "classDef" not in diagram
    assert "class " not in diagram
    assert diagram.startswith("graph BT")


def test_orientation_and_coloring_compose():
    example = next(e for e in GOLD_EXAMPLES if e.slug == "unit_verb_root_ten_thuran_anoixen")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_mermaid(result.tokengraph, orientation="LR")
    assert diagram.startswith("graph LR")
    assert "classDef vu0" in diagram


def test_implied_token_gets_its_own_dedicated_class_and_label():
    """An implied token (here, the elided infinitive of εἰμί in "ταύτην
    τὴν ὕβριν ... ἡγοῦνται") always gets the special `implied` class --
    colored with verbal_units._IMPLIED_TOKEN_COLOR, NOT whatever
    `_VERBAL_UNIT_PALETTE` color its own verbal unit (which it anchors)
    would otherwise get -- and its node label is "elided eimi"
    (mermaid.py's own _IMPLIED_TOKEN_LABELS, keyed by its tokentype
    "implied eimi"), since it has no surface text of its own. This is the
    ONE place an implied token is shown at all -- tokengraph_to_html()
    omits it entirely (see test_rendering.py's
    test_implied_tokens_are_omitted_from_html_entirely)."""
    example = next(e for e in GOLD_EXAMPLES if e.slug == "implied_eimi_tauten_ten_hybrin")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_mermaid(result.tokengraph)
    assert not warnings

    implied_ids = [tok.id for tok in result.tokengraph if tok.tokentype == "implied eimi"]
    assert implied_ids, "fixture should contain an implied eimi token"

    assert 'classDef implied fill:#ffc107,stroke:#7a5200,color:#000000;' in diagram
    implied_class_lines = [
        line for line in diagram.splitlines() if line.strip().endswith("implied;")
    ]
    assert len(implied_class_lines) == 1
    assert set(implied_class_lines[0].split()[1].split(",")) == set(implied_ids)

    for tid in implied_ids:
        assert f'{tid}["elided eimi"]' in diagram
    # No vuN classDef should also claim an implied token.
    for line in diagram.splitlines():
        if re.match(r"\s*class ([\w,]+) vu\d+;", line):
            ids = line.split()[1].split(",")
            for tid in implied_ids:
                assert tid not in ids


def test_multiple_verbal_units_get_three_distinct_colors():
    """circumstantial_fits_clause_ego_hapanta_epideixo has three verbal
    units (ἐπιδείξω's main clause, and the two circumstantial participles
    παραλείπων/λέγων) -- confirms multiple simultaneous colors actually
    show up in one diagram, not just single-unit sentences."""
    example = next(
        e for e in GOLD_EXAMPLES if e.slug == "circumstantial_fits_clause_ego_hapanta_epideixo"
    )
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_mermaid(result.tokengraph)
    assert not warnings
    assert "classDef vu0" in diagram
    assert "classDef vu1" in diagram
    assert "classDef vu2" in diagram
    assert "classDef vu3" not in diagram


def test_sentence_connector_gets_its_own_dedicated_class():
    """A token whose relationship1 is specifically 'sentence connector'
    (e.g. γάρ tying this sentence back to the previous one) gets the
    dedicated `sentenceconnector` classDef -- neon-yellow fill, strong
    black border -- instead of its own verbal unit's vuN color, same
    convention as the `implied` class for implied/elided tokens."""
    from grammatike.models import TokenAnalysis

    tg = [
        TokenAnalysis(
            id="t0", token="γάρ", tokentype="lexical",
            relatedtoken1="t1", relationship1="sentence connector",
        ),
        TokenAnalysis(
            id="t1", token="εἰμί", tokentype="lexical", verbalunitid="t1",
            relatedtoken1="root", relationship1="unit verb",
        ),
        TokenAnalysis(id="t2", token=".", tokentype="punctuation"),
    ]
    diagram, warnings = tokengraph_to_mermaid(tg)
    assert not warnings

    assert (
        "classDef sentenceconnector fill:#ffff00,stroke:#000000,"
        "stroke-width:4px,color:#000000;" in diagram
    )
    connector_class_lines = [
        line for line in diagram.splitlines() if line.strip().endswith("sentenceconnector;")
    ]
    assert len(connector_class_lines) == 1
    assert connector_class_lines[0].split()[1] == "t0"

    # t0 must not ALSO be claimed by any vuN class.
    for line in diagram.splitlines():
        if re.match(r"\s*class ([\w,]+) vu\d+;", line):
            ids = line.split()[1].split(",")
            assert "t0" not in ids


def test_unrelated_token_gets_no_class():
    """οὖν and ὦ in aside_proton_men_oun_dei have no relation at all --
    must not be assigned any vuN/implied class."""
    example = next(e for e in GOLD_EXAMPLES if e.slug == "aside_proton_men_oun_dei")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_mermaid(result.tokengraph)
    assert not warnings
    assert "classDef vu0" in diagram
    assert "classDef vu1" in diagram
    for line in diagram.splitlines():
        if line.strip().startswith("class "):
            ids = line.split()[1].split(",")
            assert "t2" not in ids  # οὖν
            assert "t4" not in ids  # ὦ
