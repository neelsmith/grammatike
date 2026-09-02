"""
Tests for dot.py's tokengraph_to_dot() -- the Graphviz-DOT counterpart to
mermaid.py's tokengraph_to_mermaid(), covering the same concerns
test_gold_examples.py and test_mermaid_coloring.py cover for the Mermaid
renderer, adapted to DOT's own syntax:

- coloring is `fillcolor`/`color`/`fontcolor` node attributes instead of
  Mermaid's `classDef`/`class` statements (DOT has no reusable named
  class);
- ranking is a `{rank=same; id1; id2; ...}` subgraph statement instead of
  a heuristic Mermaid never had for grammatike in the first place --
  `rank_by_depth` is the whole reason dot.py exists alongside mermaid.py
  (see its own module docstring): `rank=same` is a hard layout constraint.
  Grouped here by verbal_units.compute_subordination_depths() -- see
  dot.py's own docstring, "Two adaptations", for why (arsgrammatica's own
  dot.py groups by a compute_aat_depths() grammatike's verbal_units.py has
  no counterpart for), and why that substitution means, unlike
  arsgrammatica's own ranking tests, `rank_by_depth` here CAN add a
  warning.

Greek analogue of arsgrammatica's test_dot.py, using grammatike's own gold
fixtures (fixtures/gold_examples.py) in place of arsgrammatica's Latin
ones, and grammatike's 3-tuple read_analyses()-shaped tokengraph directly
(these tests build/run tokengraphs, not files, so read_analyses() itself
doesn't come up).
"""

import re

import pytest

from grammatike import tokengraph_to_dot
from grammatike.dot import compute_graph_depths, max_graph_depth
from grammatike.models import TokenAnalysis
from grammatike.verbal_units import compute_subordination_depths

from conftest import run_gold_example
from fixtures.gold_examples import GOLD_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


# ---------------------------------------------------------------------------
# Generic rendering -- mirrors test_gold_examples.py's
# test_gold_example_renders_mermaid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_gold_example_renders_dot(example):
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph)
    assert not warnings, f"{example.slug}: {warnings}"
    assert diagram.startswith("digraph tokengraph {")
    assert diagram.rstrip().endswith("}")
    for tok in result.tokengraph:
        if tok.tokentype == "punctuation":
            continue
        assert re.search(rf"^\s*{re.escape(tok.id)} \[", diagram, re.MULTILINE), (
            f"{example.slug}: {tok.id} missing its own node line"
        )


def test_ten_thuran_punctuation_excluded_from_nodes():
    tokens, result = run_gold_example(_example("unit_verb_root_ten_thuran_anoixen"))
    diagram, _warnings = tokengraph_to_dot(result.tokengraph)

    # The trailing period (t3) must not become a node.
    assert "t3 [" not in diagram


# ---------------------------------------------------------------------------
# Coloring -- mirrors test_mermaid_coloring.py.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_coloring_adds_no_new_warnings(example):
    tokens, result = run_gold_example(example)
    _plain, plain_warnings = tokengraph_to_dot(result.tokengraph, color_by_verbal_unit=False)
    _colored, colored_warnings = tokengraph_to_dot(result.tokengraph, color_by_verbal_unit=True)
    assert colored_warnings == plain_warnings, example.slug


def test_disabling_coloring_reproduces_a_plain_diagram():
    tokens, result = run_gold_example(_example("unit_verb_root_ten_thuran_anoixen"))
    diagram, _warnings = tokengraph_to_dot(result.tokengraph, color_by_verbal_unit=False)
    assert "fillcolor" not in diagram
    assert diagram.startswith("digraph tokengraph {\n    rankdir=BT;")


def test_orientation_and_coloring_compose():
    tokens, result = run_gold_example(_example("unit_verb_root_ten_thuran_anoixen"))
    diagram, _warnings = tokengraph_to_dot(result.tokengraph, orientation="LR")
    assert "rankdir=LR;" in diagram
    assert "fillcolor" in diagram


def test_implied_token_gets_its_own_dedicated_color_and_label():
    """An implied token (here, the elided infinitive of εἰμί in "ταύτην τὴν
    ὕβριν ... ἡγοῦνται") always gets the dedicated amber
    (verbal_units._IMPLIED_TOKEN_COLOR), NOT whatever color its own verbal
    unit (which it anchors) would otherwise get -- same convention as
    tokengraph_to_mermaid(), just as inline fillcolor/color/fontcolor
    attributes instead of a class. Its label is "elided eimi"
    (mermaid.token_label()'s own placeholder), and it gets `style=rounded`
    (combined with `filled` here) instead of the plain box every other node
    uses."""
    example = _example("implied_eimi_tauten_ten_hybrin")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph)
    assert not warnings

    implied_ids = [tok.id for tok in result.tokengraph if tok.tokentype == "implied eimi"]
    assert implied_ids, "fixture should contain an implied eimi token"

    for tid in implied_ids:
        m = re.search(rf'^\s*{re.escape(tid)} \[(.*)\];$', diagram, re.MULTILINE)
        assert m, f"{tid} missing its own node line"
        attrs = m.group(1)
        assert 'label="elided eimi"' in attrs
        assert 'fillcolor="#ffc107"' in attrs
        assert 'color="#7a5200"' in attrs
        assert 'fontcolor="#000000"' in attrs
        assert "rounded" in attrs


def test_sentence_connector_gets_its_own_dedicated_color():
    """A token whose relationship1 is specifically 'sentence connector'
    (e.g. γάρ tying this sentence back to the previous one) gets the
    dedicated neon-yellow/strong-border color instead of its own verbal
    unit's color -- the DOT counterpart to mermaid.py's own
    `sentenceconnector` classDef (test_mermaid_coloring.py's
    test_sentence_connector_gets_its_own_dedicated_class). No arsgrammatica
    counterpart to adapt from -- see dot.py's own docstring, "Two
    adaptations" (2)."""
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
    diagram, warnings = tokengraph_to_dot(tg)
    assert not warnings

    m = re.search(r"^\s*t0 \[(.*)\];$", diagram, re.MULTILINE)
    assert m
    attrs = m.group(1)
    assert 'fillcolor="#ffff00"' in attrs
    assert 'color="#000000"' in attrs
    assert 'fontcolor="#000000"' in attrs
    assert 'penwidth="4"' in attrs

    # t1 (an ordinary verbal-unit anchor, not a connector) must not also
    # get the connector color or the strong border.
    m2 = re.search(r"^\s*t1 \[(.*)\];$", diagram, re.MULTILINE)
    assert m2
    assert 'fillcolor="#ffff00"' not in m2.group(1)
    assert "penwidth" not in m2.group(1)


def test_circumstantial_example_gets_three_distinct_colors():
    """circumstantial_fits_clause_ego_hapanta_epideixo has three verbal
    units (ἐπιδείξω's main clause, and the two circumstantial participles
    παραλείπων/λέγων) -- confirms multiple simultaneous colors actually
    show up in one diagram, not just single-unit sentences."""
    example = _example("circumstantial_fits_clause_ego_hapanta_epideixo")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph)
    assert not warnings

    fillcolors = set(re.findall(r'fillcolor="(#[0-9a-fA-F]+)"', diagram))
    assert len(fillcolors) >= 3


def test_unrelated_token_gets_no_fillcolor():
    """οὖν in aside_proton_men_oun_dei has no relation at all -- must not
    get a fillcolor attribute at all (same fixture and token
    test_mermaid_coloring.py's own test_unrelated_token_gets_no_class
    checks for the Mermaid renderer)."""
    example = _example("aside_proton_men_oun_dei")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph)
    assert not warnings

    m = re.search(r"^\s*t2 \[(.*)\];$", diagram, re.MULTILINE)
    assert m
    assert "fillcolor" not in m.group(1)


# ---------------------------------------------------------------------------
# Ranking (`rank_by_depth`) -- grouped by
# verbal_units.compute_subordination_depths(), grammatike's own substitute
# for arsgrammatica's compute_aat_depths() -- see dot.py's own docstring,
# "Two adaptations" (1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_ranking_adds_no_new_warnings(example):
    """Every gold fixture validates cleanly (test_gold_examples.py's own
    test_gold_example_validates), so compute_subordination_depths() always
    resolves every anchor's depth for these -- ranking must add no warning
    beyond whatever plain rendering already has. The cyclic-anchor test
    below is where rank_by_depth's OWN warning path actually gets
    exercised, since no gold fixture has an unresolvable cycle."""
    tokens, result = run_gold_example(example)
    _plain, plain_warnings = tokengraph_to_dot(result.tokengraph, rank_by_depth=False)
    _ranked, ranked_warnings = tokengraph_to_dot(result.tokengraph, rank_by_depth=True)
    assert ranked_warnings == plain_warnings, example.slug


def test_two_circumstantial_participles_get_ranked_together():
    """circumstantial_fits_clause_ego_hapanta_epideixo: παραλείπων (t8) and
    λέγων (t11) are both circumstantial participles agreeing with ἐγώ,
    which is itself ἐπιδείξω's subject -- so both anchors resolve to
    subordination depth 1 (one clause removed from ἐπιδείξω's own depth
    0), and should get one `rank=same` statement forcing them onto the
    same rank, in tokengraph order (t8 before t11)."""
    example = _example("circumstantial_fits_clause_ego_hapanta_epideixo")
    tokens, result = run_gold_example(example)

    depths, depth_warnings = compute_subordination_depths(result.tokengraph)
    assert not depth_warnings
    assert depths["t8"] == depths["t11"] == 1
    assert depths["t2"] == 0

    diagram, warnings = tokengraph_to_dot(result.tokengraph)
    assert not warnings
    assert "    {rank=same; t8; t11;}" in diagram.splitlines()
    rank_lines = [line for line in diagram.splitlines() if "rank=same" in line]
    assert len(rank_lines) == 1  # t2 is alone at depth 0 -- no rank=same for it


def test_disabling_ranking_produces_no_rank_statements():
    example = _example("circumstantial_fits_clause_ego_hapanta_epideixo")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_dot(result.tokengraph, rank_by_depth=False)
    assert "rank=same" not in diagram


def test_ranking_and_coloring_compose():
    example = _example("circumstantial_fits_clause_ego_hapanta_epideixo")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_dot(result.tokengraph)
    assert "rank=same" in diagram
    assert "fillcolor" in diagram


def test_cyclic_anchors_are_excluded_from_ranking_with_a_warning():
    """Two anchors in a direct mutual 'unit verb' relation cycle leave
    compute_subordination_depths() unable to resolve either one's depth
    (with a warning) -- unlike arsgrammatica's own compute_aat_depths(),
    which has no unresolved state at all (see that project's own
    test_cyclic_anchors_still_get_ranked_with_no_warning). Here, both
    anchors are simply excluded from ranking, and the resolution warning
    is folded into tokengraph_to_dot()'s own returned warnings -- see
    dot.py's docstring, 'Two adaptations' (1)."""
    tokengraph = [
        TokenAnalysis(
            id="t0", token="a", tokentype="lexical", verbalunitid="t0",
            relatedtoken1="t1", relationship1="unit verb",
        ),
        TokenAnalysis(
            id="t1", token="b", tokentype="lexical", verbalunitid="t1",
            relatedtoken1="t0", relationship1="unit verb",
        ),
    ]
    depths, direct_warnings = compute_subordination_depths(tokengraph)
    assert depths == {"t0": None, "t1": None}
    assert direct_warnings  # compute_subordination_depths() still warns

    diagram, warnings = tokengraph_to_dot(tokengraph, color_by_verbal_unit=False)
    assert warnings  # folded into tokengraph_to_dot()'s own warnings
    assert any("cycle detected" in w for w in warnings)
    assert "rank=same" not in diagram
    # The nodes and their (mutual) edges are still drawn -- only ranking is
    # affected.
    assert "t0 [" in diagram
    assert "t1 [" in diagram
    assert "t0 -> t1" in diagram
    assert "t1 -> t0" in diagram


# ---------------------------------------------------------------------------
# Depth filtering (`depth` parameter) -- a SEPARATE feature from
# rank_by_depth above, and a THIRD depth notion, distinct from both
# compute_subordination_depths() (rank_by_depth's, above) and
# verbal_units.compute_subordination_depths() as used by
# rendering.tokengraph_to_depth_html() (the CLAUSE-level notion behind that
# function's own indented-HTML `depth` parameter -- confusingly the SAME
# underlying function as rank_by_depth uses here, but consumed
# differently): compute_graph_depths() -- a plain graph distance, in
# edges, from a token back to the nearest root anchor, following the same
# relatedtoken1/relatedtoken2 edges drawn as `->` lines. A whole clause's
# subject, object, and other ordinary dependents are each their own hop of
# graph depth (unlike subordination depth, which gives them all the SAME
# depth as their governing verb) -- so depth=0 shows ONLY root verbal-unit
# anchors, not "the whole root clause." See tokengraph_to_dot()'s own
# docstring for the full rationale, including why a dropped node can leave
# a KEPT node's edge pointing at an excluded one.
# ---------------------------------------------------------------------------


def test_depth_zero_shows_only_root_anchors():
    """"ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη": ἧκεν (t6) is the only root
    anchor (relatedtoken1='root'); everything else -- including its own
    subject/sentence-connector/subordinate-clause dependents -- is at
    least one edge away. depth=0 must show ONLY ἧκεν, not the rest of its
    clause -- the exact distinction from tokengraph_to_depth_html()'s
    block-level `depth` this parameter deliberately does NOT share."""
    example = _example("dependent_verb_epeide_de_en_hos_hekeen")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph, depth=0)
    assert warnings == []
    assert re.search(r"^    t6 \[", diagram, re.MULTILINE)  # ἧκεν, the root anchor
    for other_id in ("t0", "t1", "t2", "t3", "t4", "t7"):
        assert f"{other_id} [" not in diagram, other_id


def test_depth_one_adds_direct_dependents_of_the_root():
    """depth=1 on the same fixture adds ἧκεν's own direct dependents --
    ἐπειδή (subordinating conjunction), δέ (sentence connector), ἐκείνη
    (subject) -- but not ἦν/πρός/ἡμέραν, which are two or more edges from
    ἧκεν."""
    example = _example("dependent_verb_epeide_de_en_hos_hekeen")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph, depth=1)
    assert warnings == []
    for kept_id in ("t0", "t1", "t6", "t7"):
        assert re.search(rf"^    {kept_id} \[", diagram, re.MULTILINE), kept_id
    for dropped_id in ("t2", "t3", "t4"):
        assert f"{dropped_id} [" not in diagram, dropped_id


def test_depth_at_or_beyond_passage_max_matches_depth_none():
    """A `depth` at or beyond the passage's own max_graph_depth() must
    render identically to leaving `depth` unset."""
    example = _example("dependent_verb_epeide_de_en_hos_hekeen")
    tokens, result = run_gold_example(example)

    maxd = max_graph_depth(result.tokengraph)
    diagram_max, warnings_max = tokengraph_to_dot(result.tokengraph, depth=maxd)
    diagram_none, warnings_none = tokengraph_to_dot(result.tokengraph, depth=None)
    assert diagram_max == diagram_none
    assert warnings_max == warnings_none


def test_depth_negative_raises():
    example = _example("dependent_verb_epeide_de_en_hos_hekeen")
    tokens, result = run_gold_example(example)
    with pytest.raises(ValueError, match="depth must be >= 0"):
        tokengraph_to_dot(result.tokengraph, depth=-1)


def test_depth_and_coloring_compose():
    example = _example("dependent_verb_epeide_de_en_hos_hekeen")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_dot(result.tokengraph, depth=1)
    assert "fillcolor" in diagram  # kept nodes still get colored


def test_depth_and_ranking_compose():
    """A rank_by_depth `{rank=same; ...}` statement must never name an
    anchor `depth` filtering has excluded. circumstantial_fits_clause_ego_
    hapanta_epideixo's two circumstantial-participle anchors (t8, t11,
    both graph depth 2) are both still within a depth=2 cutoff, so the
    rank=same statement grouping them survives; their own dependents
    (t7/t10/t12, graph depth 3) are excluded."""
    example = _example("circumstantial_fits_clause_ego_hapanta_epideixo")
    tokens, result = run_gold_example(example)
    diagram, _warnings = tokengraph_to_dot(result.tokengraph, depth=2)
    node_ids = set(re.findall(r"^    (\S+) \[", diagram, re.MULTILINE))
    assert "t8" in node_ids and "t11" in node_ids
    rank_lines = [line for line in diagram.splitlines() if "rank=same" in line]
    assert rank_lines  # the grouping must have survived the depth cutoff
    ranked_ids = {tid for line in rank_lines for tid in re.findall(r"t\d+\w*", line)}
    assert ranked_ids <= node_ids


def test_depth_relative_pronoun_double_duty_deepens_correctly():
    """ὅν (t2) in "ὁ ἀνήρ ὅν εἶδον ἀπῆλθεν" is BOTH an anaphoric pointer
    (relatedtoken1 -> its antecedent ἀνήρ, 'relative pronoun') and its own
    dependent clause's direct object (relatedtoken2 -> εἶδον, 'direct
    object'), the verb it also introduces via εἶδον's OWN 'unit verb'
    relation pointing back at ὅν -- a genuine two-way link. Depth must
    follow relatedtoken1 (the antecedent chain), not average or take the
    shallower of both edges: ἀπῆλθεν (root, depth 0) <- ἀνήρ (subject,
    depth 1) <- ὅν (relative pronoun, depth 2) <- εἶδον (unit verb, depth
    3), each link its own hop."""
    example = _example("relative_pronoun_ho_aner_hon_eidon")
    tokens, result = run_gold_example(example)

    depths = compute_graph_depths(result.tokengraph)
    assert depths["t4"] == 0  # ἀπῆλθεν, root
    assert depths["t1"] == 1  # ἀνήρ, subject of ἀπῆλθεν
    assert depths["t2"] == 2  # ὅν, relative pronoun -> ἀνήρ
    assert depths["t3"] == 3  # εἶδον, unit verb -> ὅν
    assert depths["t0"] == 2  # ὁ, article of ἀνήρ (depth 1) -> depth 2

    diagram0, warnings0 = tokengraph_to_dot(result.tokengraph, depth=0, color_by_verbal_unit=False)
    assert warnings0 == []
    assert re.search(r"^    t4 \[", diagram0, re.MULTILINE)
    for other_id in ("t0", "t1", "t2", "t3"):
        assert f"{other_id} [" not in diagram0, other_id


def test_depth_dangling_edge_from_kept_node_is_skipped_with_a_warning():
    """Same relative-pronoun fixture, at depth=2: ὅν (t2, depth 2) is kept,
    but its relatedtoken2 edge to εἶδον (t3, depth 3) -- excluded at this
    cutoff -- must be skipped with a warning rather than producing a
    `t2 -> t3` line pointing at a token with no node."""
    example = _example("relative_pronoun_ho_aner_hon_eidon")
    tokens, result = run_gold_example(example)
    diagram, warnings = tokengraph_to_dot(result.tokengraph, depth=2, color_by_verbal_unit=False)

    assert "t2 [" in diagram
    assert "t3 [" not in diagram
    assert "-> t3" not in diagram  # the dangling edge itself must not appear
    assert any(
        "t2 -[direct object]-> t3" in w and "excluded by the depth cutoff" in w
        for w in warnings
    )


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_depth_filtering_never_leaves_a_dangling_edge(example):
    """Property check across every gold example, at every depth level from
    0 up to that passage's own max_graph_depth(): no edge may be left
    whose source or target has no node line of its own."""
    tokens, result = run_gold_example(example)
    tokengraph = result.tokengraph
    maxd = max_graph_depth(tokengraph)
    if maxd is None:
        return

    for cap in range(0, maxd + 1):
        diagram, _warnings = tokengraph_to_dot(tokengraph, depth=cap)
        node_ids = set(re.findall(r"^    (\S+) \[", diagram, re.MULTILINE))
        for source, target in re.findall(r"^    (\S+) -> (\S+) \[", diagram, re.MULTILINE):
            assert source in node_ids, f"{example.slug} depth={cap}: edge source {source} has no node"
            assert target in node_ids, f"{example.slug} depth={cap}: edge target {target} has no node"
