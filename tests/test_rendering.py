"""
Tests for grammatike/rendering.py's tokengraph_to_text(), tokengraph_to_html(),
and tokengraph_to_depth_html(). Greek analogue of arsgrammatica's
test_rendering.py.

Split into parts: targeted unit tests for each join rule (built from small
hand-built TokenAnalysis lists), a round-trip check against every gold
example's own passage for tokengraph_to_text(), a section of
tokengraph_to_html() tests covering verbal-unit span-wrapping, HTML
escaping, and cross-checking colors against tokengraph_to_mermaid(), and a
tokengraph_to_depth_html() section covering subordination-depth blocks.

Greek has no "praenomen"/"abbreviation" tokentype (see rendering.py's own
docstring) -- the corresponding Latin tests are replaced here with the
Greek-specific "numeral" carve-out and the one-sided "connecting word"
carve-out.
"""

import html
import re

import pytest

from grammatike.mermaid import tokengraph_to_mermaid
from grammatike.models import IMPLIED_TOKENTYPES, TokenAnalysis
from grammatike.rendering import (
    tokengraph_to_text,
    tokengraph_to_html,
    tokengraph_to_depth_html,
)
from grammatike.verbal_units import _VERBAL_UNIT_PALETTE, max_subordination_depth
from conftest import run_gold_example
from fixtures.gold_examples import GOLD_EXAMPLES


def _tok(id, token, tokentype, **kw):
    return TokenAnalysis(id=id, token=token, tokentype=tokentype, **kw)


def test_plain_lexical_tokens_get_single_spaces():
    tg = [_tok("t0", "ἀνήρ", "lexical"), _tok("t1", "ἀγαθός", "lexical")]
    assert tokengraph_to_text(tg) == "ἀνήρ ἀγαθός"


def test_enclitic_attaches_directly_no_space():
    tg = [
        _tok("t0", "εἴ", "lexical"),
        _tok("t1", "περ", "enclitic"),
        _tok("t2", "οὕτως", "lexical"),
        _tok("t3", "ἔχει", "lexical"),
        _tok("t4", ".", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "εἴπερ οὕτως ἔχει."


def test_period_comma_semicolon_hyphen_are_left_joining():
    tg = [
        _tok("t0", "foo", "lexical"),
        _tok("t1", ",", "punctuation"),
        _tok("t2", "bar", "lexical"),
        _tok("t3", ";", "punctuation"),
        _tok("t4", "baz", "lexical"),
        _tok("t5", "-", "punctuation"),
        _tok("t6", "qux", "lexical"),
        _tok("t7", ".", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "foo, bar; baz- qux."


def test_greek_question_mark_and_raised_dot_are_left_joining():
    tg = [
        _tok("t0", "τίς", "lexical"),
        _tok("t1", "ἦλθεν", "lexical"),
        _tok("t2", ";", "punctuation"),  # U+037E, Greek question mark
        _tok("t3", "ἔμεινα", "lexical"),
        _tok("t4", "·", "punctuation"),  # raised dot / ano teleia
        _tok("t5", "ἀπῆλθον", "lexical"),
    ]
    assert tokengraph_to_text(tg) == "τίς ἦλθεν; ἔμεινα· ἀπῆλθον"


def test_opening_bracket_is_right_joining_closing_is_left_joining():
    tg = [
        _tok("t0", "δεῖ", "lexical"),
        _tok("t1", "(", "punctuation"),
        _tok("t2", "γάρ", "lexical"),
        _tok("t3", ")", "punctuation"),
        _tok("t4", ".", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "δεῖ (γάρ)."


def test_square_and_curly_brackets_too():
    tg = [
        _tok("t0", "δεῖ", "lexical"),
        _tok("t1", "[", "punctuation"),
        _tok("t2", "γάρ", "lexical"),
        _tok("t3", "]", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "δεῖ [γάρ]"

    tg2 = [
        _tok("t0", "δεῖ", "lexical"),
        _tok("t1", "{", "punctuation"),
        _tok("t2", "γάρ", "lexical"),
        _tok("t3", "}", "punctuation"),
    ]
    assert tokengraph_to_text(tg2) == "δεῖ {γάρ}"


def test_double_quote_pair_first_right_second_left():
    tg = [
        _tok("t0", '"', "punctuation"),
        _tok("t1", "ἵνα", "lexical"),
        _tok("t2", "σύ", "lexical"),
        _tok("t3", '"', "punctuation"),
        _tok("t4", "ἔφη", "lexical"),
        _tok("t5", ".", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == '"ἵνα σύ" ἔφη.'


def test_two_separate_double_quote_spans_alternate_correctly():
    """A third and fourth occurrence of the same quote character must
    resume the open/close alternation (open again, then close), not stay
    "closed" forever."""
    tg = [
        _tok("t0", '"', "punctuation"),
        _tok("t1", "a", "lexical"),
        _tok("t2", '"', "punctuation"),
        _tok("t3", "b", "lexical"),
        _tok("t4", '"', "punctuation"),
        _tok("t5", "c", "lexical"),
        _tok("t6", '"', "punctuation"),
    ]
    assert tokengraph_to_text(tg) == '"a" b "c"'


def test_single_and_double_quotes_are_tracked_independently():
    tg = [
        _tok("t0", '"', "punctuation"),
        _tok("t1", "a", "lexical"),
        _tok("t2", "'", "punctuation"),
        _tok("t3", "b", "lexical"),
        _tok("t4", "'", "punctuation"),
        _tok("t5", "c", "lexical"),
        _tok("t6", '"', "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "\"a 'b' c\""


def test_numeral_gets_normal_spacing():
    """Unlike Latin's "praenomen"/"abbreviation" tokentypes -- which
    don't exist in Greek's scheme (see models.py) -- "numeral" (a number
    written numerically, e.g. Milesian notation) gets the same ordinary
    spacing as any other non-enclitic, non-punctuation token."""
    tg = [
        _tok("t0", "εἶδον", "lexical"),
        _tok("t1", "γʹ", "numeral"),
        _tok("t2", "ἄνδρας", "lexical"),
        _tok("t3", ".", "punctuation"),
    ]
    assert tokengraph_to_text(tg) == "εἶδον γʹ ἄνδρας."


def test_empty_tokengraph_returns_empty_string():
    assert tokengraph_to_text([]) == ""


def test_single_token():
    assert tokengraph_to_text([_tok("t0", "χαῖρε", "lexical")]) == "χαῖρε"


def test_right_joining_token_first_gets_no_leading_space():
    tg = [_tok("t0", "(", "punctuation"), _tok("t1", "γάρ", "lexical")]
    assert tokengraph_to_text(tg) == "(γάρ"


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_round_trips_gold_example_passages(example):
    """Every existing gold fixture's tokengraph should reconstruct its
    exact original passage string via ordinary punctuation/enclitic
    spacing rules alone."""
    tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
    assert tokengraph_to_text(tokengraph) == example.passage


# ---------------------------------------------------------------------------
# tokengraph_to_html()
# ---------------------------------------------------------------------------

_SPAN_RE = re.compile(
    r'<span style="background-color: (#[0-9a-fA-F]{6}); color: (#[0-9a-fA-F]{6});">'
    r"(.*?)</span>"
)


def test_no_verbal_units_leaves_plain_spacing_and_no_spans():
    """With no verbalunitid/relation fields set at all, every token is
    unassigned -- so tokengraph_to_html() should produce exactly the same
    string as tokengraph_to_text() (escaped, but nothing here needs
    escaping), with no <span> tags at all."""
    tg = [_tok("t0", "ἀνήρ", "lexical"), _tok("t1", "ἀγαθός", "lexical")]
    assert tokengraph_to_html(tg) == "ἀνήρ ἀγαθός"
    assert "<span" not in tokengraph_to_html(tg)


def test_single_verbal_unit_wraps_its_lexical_tokens():
    tg = [
        _tok("t0", "τήν", "lexical", relatedtoken1="t1", relationship1="article"),
        _tok("t1", "θύραν", "lexical", relatedtoken1="t2", relationship1="direct object"),
        _tok("t2", "ἀνέῳξεν", "lexical", verbalunitid="t2"),
        _tok("t3", ".", "punctuation"),
    ]
    fill, _stroke, text_color = _VERBAL_UNIT_PALETTE[0]
    span = lambda word: (
        f'<span style="background-color: {fill}; color: {text_color};">{word}</span>'
    )
    assert tokengraph_to_html(tg) == f"{span('τήν')} {span('θύραν')} {span('ἀνέῳξεν')}."


def test_multiple_verbal_units_get_distinct_first_appearance_colors():
    """Two independent verbal units, anchored by t1 and t4, each colored
    per _VERBAL_UNIT_PALETTE's order of first non-punctuation appearance."""
    tg = [
        _tok("t0", "παῖς", "lexical", relatedtoken1="t1", relationship1="subject"),
        _tok("t1", "βλέπει", "lexical", verbalunitid="t1"),
        _tok("t2", ",", "punctuation"),
        _tok("t3", "κόρη", "lexical", relatedtoken1="t4", relationship1="subject"),
        _tok("t4", "ἔρχεται", "lexical", verbalunitid="t4"),
        _tok("t5", ".", "punctuation"),
    ]
    fill0, _s0, text0 = _VERBAL_UNIT_PALETTE[0]
    fill1, _s1, text1 = _VERBAL_UNIT_PALETTE[1]
    span0 = lambda word: f'<span style="background-color: {fill0}; color: {text0};">{word}</span>'
    span1 = lambda word: f'<span style="background-color: {fill1}; color: {text1};">{word}</span>'
    expected = (
        f"{span0('παῖς')} {span0('βλέπει')}, {span1('κόρη')} {span1('ἔρχεται')}."
    )
    assert tokengraph_to_html(tg) == expected


def test_only_lexical_and_numeral_tokens_get_wrapped_even_when_others_are_assigned():
    """Punctuation and a non-connecting enclitic can both be assigned a
    verbal unit by assign_verbal_units() (it assigns every token id), but
    neither is a connecting word (their own relationship1 is "subject" or
    "adverbial") -- so only tokentype == "lexical" or "numeral" gets a
    <span> here. This numeral token ("γʹ") is given relationship1
    "adverbial" -- NOT a connecting-word relation -- specifically to
    isolate that tokengraph_to_html() wraps it because of its tokentype,
    not because of which relation it happens to carry (see
    test_enclitic_connecting_word_gets_wrapped_too for the corresponding
    relationship-keyed carve-out)."""
    tg = [
        _tok("t0", "εἶδον", "lexical", verbalunitid="t0"),
        _tok("t1", "τε", "enclitic", relatedtoken1="t0", relationship1="subject"),
        _tok("t2", "γʹ", "numeral", relatedtoken1="t0", relationship1="adverbial"),
        _tok("t3", ".", "punctuation", relatedtoken1="t0", relationship1="adverbial"),
    ]
    html_out = tokengraph_to_html(tg)
    matches = _SPAN_RE.findall(html_out)
    assert len(matches) == 2
    assert matches[0][2] == "εἶδον"
    assert matches[1][2] == "γʹ"
    # same verbal unit -> same fill color, for both wrapped tokens
    assert matches[0][0] == matches[1][0]


def test_numeral_token_gets_wrapped_too():
    """syntax_model.md's numeral-vs-lexical clarification makes clear a
    numeral is otherwise an ordinary participant in the clause, able to
    carry a real relation (here relatedtoken1 -> "ἄνδρας", relationship1 =
    "attributive") that assign_verbal_units() resolves like any other --
    so it must get the SAME colored span as the rest of its verbal unit,
    matching what tokengraph_to_mermaid() already does."""
    tg = [
        _tok("t0", "εἶδον", "lexical", verbalunitid="t0"),
        _tok("t1", "γʹ", "numeral", relatedtoken1="t2", relationship1="attributive"),
        _tok("t2", "ἄνδρας", "lexical", relatedtoken1="t0", relationship1="direct object"),
        _tok("t3", ".", "punctuation"),
    ]
    fill, _stroke, text_color = _VERBAL_UNIT_PALETTE[0]
    span = lambda word: (
        f'<span style="background-color: {fill}; color: {text_color};">{word}</span>'
    )
    assert tokengraph_to_html(tg) == (
        f"{span('εἶδον')} {span('γʹ')} {span('ἄνδρας')}."
    )


def test_enclitic_connecting_word_gets_wrapped_too():
    """The Greek-specific carve-out: an enclitic connecting word is not
    tokentype "lexical", but assign_verbal_units() still resolves it to
    one of the units it connects -- so it should get the SAME colored
    span as that unit's other tokens, not be left plain like other
    non-lexical, non-numeral tokentypes."""
    tg = [
        _tok("t0", "ἠθέλησεν", "lexical", verbalunitid="t0"),
        _tok("t1", "Ἑλένην", "lexical", relatedtoken1="t3", relationship1="direct object"),
        _tok(
            "t2", "τε", "enclitic",
            relatedtoken1="t0", relationship1="connecting word",
        ),
        _tok("t3", "ἤγαγεν", "lexical", verbalunitid="t3"),
        _tok("t4", ".", "punctuation"),
    ]
    fill0, _s0, text0 = _VERBAL_UNIT_PALETTE[0]
    fill1, _s1, text1 = _VERBAL_UNIT_PALETTE[1]
    span0 = lambda word: f'<span style="background-color: {fill0}; color: {text0};">{word}</span>'
    span1 = lambda word: f'<span style="background-color: {fill1}; color: {text1};">{word}</span>'
    # τε is glued directly onto Ἑλένην (no space, enclitic), but gets ITS
    # OWN color (unit t0, the first verb) distinct from Ἑλένην's (unit t3).
    assert tokengraph_to_html(tg) == f"{span0('ἠθέλησεν')} {span1('Ἑλένην')}{span0('τε')} {span1('ἤγαγεν')}."


def test_implied_tokens_are_omitted_from_html_entirely():
    """An implied/elided token (models.py's IMPLIED_TOKENTYPES) has no
    surface text of its own (tok.token is always None) -- tokengraph_to_html()
    omits it entirely, same as tokengraph_to_text() does, rather than
    inventing placeholder text for it. tokengraph_to_mermaid() is the one
    place these ARE shown (see test_mermaid_coloring.py) -- inserting
    placeholder text into reconstructed HTML prose would misrepresent what
    the passage actually says."""
    tg = [
        _tok("t0", "ταῦτα", "lexical", relatedtoken1="t0_implied", relationship1="subject"),
        _tok(
            "t0_implied", None, "implied eimi",
            verbalunitid="t0_implied", relatedtoken1="root", relationship1="unit verb",
        ),
        _tok("t1", "καλά", "lexical", relatedtoken1="t0_implied", relationship1="predicate"),
        _tok("t2", ".", "punctuation"),
    ]
    fill, _stroke, text_color = _VERBAL_UNIT_PALETTE[0]
    span = lambda word: f'<span style="background-color: {fill}; color: {text_color};">{word}</span>'
    html_out = tokengraph_to_html(tg)
    assert html_out == f"{span('ταῦτα')} {span('καλά')}."
    assert "implied" not in html_out


def test_lexical_token_with_no_verbal_unit_is_unwrapped():
    tg = [
        _tok("t0", "χαῖρε", "lexical", verbalunitid="t0"),
        _tok("t1", "φεῦ", "lexical"),  # unrelated interjection, no relation at all
    ]
    html_out = tokengraph_to_html(tg)
    matches = _SPAN_RE.findall(html_out)
    assert len(matches) == 1
    assert matches[0][2] == "χαῖρε"
    assert "φεῦ" in html_out
    assert "<span" not in html_out.split("φεῦ")[0].split("</span>")[-1]


def test_html_special_characters_are_escaped():
    tg = [
        _tok("t0", "σύ", "lexical", verbalunitid="t0"),
        _tok("t1", "<3", "lexical"),
        _tok("t2", "&", "punctuation"),
    ]
    html_out = tokengraph_to_html(tg)
    assert "<3" not in html_out
    assert "&lt;3" in html_out
    assert "&amp;" in html_out


def test_quote_pair_tokens_still_join_correctly_around_spans():
    tg = [
        _tok("t0", '"', "punctuation"),
        _tok("t1", "ταῦτα", "lexical", verbalunitid="t1"),
        _tok("t2", "ἐστίν", "lexical", relatedtoken1="t1", relationship1="predicate"),
        _tok("t3", '"', "punctuation"),
        _tok("t4", "ἔφη", "lexical"),
        _tok("t5", ".", "punctuation"),
    ]
    fill, _stroke, text_color = _VERBAL_UNIT_PALETTE[0]
    span = lambda word: (
        f'<span style="background-color: {fill}; color: {text_color};">{word}</span>'
    )
    assert tokengraph_to_html(tg) == f'&quot;{span("ταῦτα")} {span("ἐστίν")}&quot; ἔφη.'


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_html_colors_match_mermaid_colors_for_every_gold_example(example):
    """The whole point of this function: whatever fill color a token's
    verbal unit gets in the Mermaid diagram, the same token's <span> here
    must use that exact same fill (and the matching text color) -- checked
    against tokengraph_to_mermaid()'s own classDef/class output. Covers
    every wrapped token, not just lexical ones -- a numeral or an enclitic
    connecting word gets wrapped here too. An implied token is excluded
    from this comparison entirely: tokengraph_to_html() omits it, while
    the Mermaid diagram shows it in its own dedicated `implied` class."""
    tokens, result = run_gold_example(example)
    tokengraph = result.tokengraph

    diagram, _warnings = tokengraph_to_mermaid(tokengraph)
    fill_of_class = dict(
        re.findall(r"classDef (vu\d+) fill:(#[0-9a-fA-F]{6}),", diagram)
    )
    class_of_id = {}
    for ids, class_name in re.findall(r"class ([\w,]+) (vu\d+);", diagram):
        for tid in ids.split(","):
            class_of_id[tid] = class_name

    expected_fills = [
        fill_of_class[class_of_id[tok.id]]
        for tok in tokengraph
        if (
            tok.tokentype in ("lexical", "numeral")
            or tok.relationship1 in ("connecting word", "sentence connector")
            or tok.relationship2 in ("connecting word", "sentence connector")
        )
        and tok.id in class_of_id
    ]

    html_out = tokengraph_to_html(tokengraph)
    actual_fills = [m[0] for m in _SPAN_RE.findall(html_out)]

    assert actual_fills == expected_fills, example.slug


# ---------------------------------------------------------------------------
# tokengraph_to_depth_html()
# ---------------------------------------------------------------------------

_DIV_RE = re.compile(
    r'<div style="margin-left: ([0-9.]+)em; margin-bottom: 0\.35em;">(.*?)</div>'
)


def _tokengraph(slug):
    example = next(e for e in GOLD_EXAMPLES if e.slug == slug)
    return [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]


def _tauron_relative_clause_tokengraph():
    """"ταῦρον ὅν εἶδον ζῶντα ἤγαγον." ("The bull which I saw, alive, I
    led away.") -- a hand-built fixture (not from gold_examples.py, to
    keep this depth-shape test independent of that fixture set) producing
    exactly three blocks at depths 0/1/0: ταῦρον (0), ὅν εἶδον (1, the
    relative clause), ζῶντα ἤγαγον. (0, resuming the main clause)."""
    return [
        _tok("t0", "ταῦρον", "lexical", relatedtoken1="t4", relationship1="direct object"),
        _tok(
            "t1", "ὅν", "lexical",
            relatedtoken1="t0", relationship1="relative pronoun",
            relatedtoken2="t2", relationship2="direct object",
        ),
        _tok("t2", "εἶδον", "lexical", verbalunitid="t2", relatedtoken1="t1", relationship1="unit verb"),
        _tok("t3", "ζῶντα", "lexical", relatedtoken1="t0", relationship1="attributive"),
        _tok("t4", "ἤγαγον", "lexical", verbalunitid="t4", relatedtoken1="root", relationship1="unit verb"),
        _tok("t5", ".", "punctuation"),
    ]


def test_depth_html_tauron_example_produces_three_blocks_at_expected_depths():
    tokengraph = _tauron_relative_clause_tokengraph()
    html_out, warnings = tokengraph_to_depth_html(tokengraph)
    assert warnings == []

    blocks = _DIV_RE.findall(html_out)
    assert len(blocks) == 3

    margins = [float(m) for m, _content in blocks]
    assert margins == [0.0, 2.0, 0.0]

    assert "ταῦρον" in blocks[0][1]
    assert "ἤγαγον" not in blocks[0][1]

    for word in ("ὅν", "εἶδον"):
        assert word in blocks[1][1], word
    assert "ταῦρον" not in blocks[1][1]
    assert "ἤγαγον" not in blocks[1][1]

    for word in ("ζῶντα", "ἤγαγον"):
        assert word in blocks[2][1], word
    assert "εἶδον" not in blocks[2][1]


def test_depth_html_enclitic_never_starts_a_new_block():
    """τε in "ἠθέλησεν Ἑλένην τε ἤγαγεν" resolves (per assign_verbal_units())
    to ἠθέλησεν's verbal unit -- a DIFFERENT unit than Ἑλένην, which
    resolves to ἤγαγεν's -- since τε's own relation (connecting word)
    joins the two verbs, not the noun it's glued to. τε must stay in
    whichever block Ἑλένην opened -- even though, as a connecting word, it
    now gets its OWN color span there (ἠθέλησεν's color, not Ἑλένην's),
    immediately adjacent with no space."""
    tokengraph = [
        _tok("t0", "ἠθέλησεν", "lexical", verbalunitid="t0",
             relatedtoken1="root", relationship1="unit verb"),
        _tok("t1", "Ἑλένην", "lexical", relatedtoken1="t3", relationship1="direct object"),
        _tok("t2", "τε", "enclitic", relatedtoken1="t0", relationship1="connecting word"),
        _tok("t3", "ἤγαγεν", "lexical", verbalunitid="t3",
             relatedtoken1="root", relationship1="unit verb"),
        _tok("t4", ".", "punctuation"),
    ]
    html_out, warnings = tokengraph_to_depth_html(tokengraph)
    assert warnings == []

    blocks = _DIV_RE.findall(html_out)
    assert len(blocks) == 2

    helene_block = next(content for _m, content in blocks if "Ἑλένην" in content)
    assert "τε" in helene_block
    # Two adjacent, differently-colored spans -- Ἑλένην's (ἤγαγεν's unit)
    # immediately followed, with no space and no other markup between
    # them, by τε's own (ἠθέλησεν's unit).
    assert "Ἑλένην</span><span" in helene_block
    matches = _SPAN_RE.findall(helene_block)
    words = [m[2] for m in matches]
    fills = [m[0] for m in matches]
    helene_index = words.index("Ἑλένην")
    assert words[helene_index + 1] == "τε"
    assert fills[helene_index] != fills[helene_index + 1]


def test_depth_html_depth_cap_drops_deeper_blocks_entirely():
    """Same fixture as test_depth_html_tauron_example_produces_three_blocks_
    at_expected_depths (three blocks at depths 0/1/0): passing depth=0
    must drop the middle (depth-1) block ENTIRELY -- not render it empty
    or grayed out -- leaving only the two depth-0 blocks; passing depth=1
    (>= the passage's own max) must show all three, identical to leaving
    `depth` unset."""
    tokengraph = _tauron_relative_clause_tokengraph()

    html_out, warnings = tokengraph_to_depth_html(tokengraph, depth=0)
    assert warnings == []  # depth filtering must not itself produce warnings
    blocks = _DIV_RE.findall(html_out)
    assert len(blocks) == 2
    assert "ταῦρον" in blocks[0][1]
    assert "ἤγαγον" in blocks[1][1]
    assert "εἶδον" not in html_out  # the depth-1 block is gone, not just emptied

    html_out_1, _warnings = tokengraph_to_depth_html(tokengraph, depth=1)
    html_out_none, _warnings = tokengraph_to_depth_html(tokengraph, depth=None)
    assert html_out_1 == html_out_none
    assert len(_DIV_RE.findall(html_out_1)) == 3


def test_depth_html_negative_depth_raises():
    tokengraph = _tauron_relative_clause_tokengraph()
    with pytest.raises(ValueError, match="depth must be >= 0"):
        tokengraph_to_depth_html(tokengraph, depth=-1)


def test_max_subordination_depth_matches_the_deepest_resolved_anchor():
    """max_subordination_depth() is just the max of compute_subordination_
    depths()'s own resolved values -- checked against two fixtures whose
    depths are already pinned down elsewhere (test_verbal_units.py): one
    that bottoms out at depth 1 (the ταῦρον relative-clause fixture), and
    one that reaches depth 2 (ἐπεί ᾔδει ἡμαρτηκέναι nesting case)."""
    tokengraph = _tauron_relative_clause_tokengraph()
    assert max_subordination_depth(tokengraph) == 1

    tokengraph = _tokengraph("depth_two_epei_edei_hemartekenai")
    assert max_subordination_depth(tokengraph) == 2


def test_max_subordination_depth_none_when_no_verbal_expressions():
    assert max_subordination_depth([]) is None
    tg = [_tok("t0", "φεῦ", "lexical")]  # no verb at all -- no anchors
    assert max_subordination_depth(tg) is None


def test_depth_html_custom_indent_scales_margins():
    tokengraph = _tauron_relative_clause_tokengraph()
    html_out, _warnings = tokengraph_to_depth_html(tokengraph, indent_em=1.5)
    blocks = _DIV_RE.findall(html_out)
    margins = sorted({float(m) for m, _content in blocks})
    assert margins == [0.0, 1.5]


def test_depth_html_colors_match_tokengraph_to_html_for_the_same_passage():
    """Splitting into per-block <div>s must not change which color a given
    lexical token gets -- the whole point of sharing one precomputed
    assignment/colors mapping (via _tokens_to_html()) between this function
    and tokengraph_to_html()."""
    tokengraph = _tokengraph("aside_proton_men_oun_dei")
    whole_html = tokengraph_to_html(tokengraph)
    depth_html, _warnings = tokengraph_to_depth_html(tokengraph)

    whole_fills = [m[0] for m in _SPAN_RE.findall(whole_html)]
    depth_fills = [m[0] for m in _SPAN_RE.findall(depth_html)]
    assert depth_fills == whole_fills


def test_depth_html_leading_unassigned_token_opens_a_placeholder_depth_zero_block():
    tg = [
        _tok("t0", "φεῦ", "lexical"),
        _tok("t1", "παῖς", "lexical", relatedtoken1="t2", relationship1="subject"),
        _tok("t2", "ἔρχεται", "lexical", verbalunitid="t2", relatedtoken1="root", relationship1="unit verb"),
    ]
    html_out, warnings = tokengraph_to_depth_html(tg)
    assert warnings == []
    blocks = _DIV_RE.findall(html_out)
    assert len(blocks) == 2
    assert blocks[0][0] == "0.0"
    assert "φεῦ" in blocks[0][1]
    assert "παῖς" in blocks[1][1] and "ἔρχεται" in blocks[1][1]


def test_depth_html_trailing_unassigned_token_folds_into_open_block():
    tg = [
        _tok("t0", "παῖς", "lexical", relatedtoken1="t1", relationship1="subject"),
        _tok("t1", "ἔρχεται", "lexical", verbalunitid="t1", relatedtoken1="root", relationship1="unit verb"),
        _tok("t2", "φεῦ", "lexical"),
    ]
    html_out, warnings = tokengraph_to_depth_html(tg)
    assert warnings == []
    blocks = _DIV_RE.findall(html_out)
    assert len(blocks) == 1
    assert "φεῦ" in blocks[0][1]


def test_depth_html_empty_tokengraph_returns_empty_string():
    html_out, warnings = tokengraph_to_depth_html([])
    assert html_out == ""
    assert warnings == []


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_depth_html_never_warns_and_covers_every_token_for_every_gold_example(example):
    """Structural check across the whole fixture set, mirroring
    test_html_colors_match_mermaid_colors_for_every_gold_example above:
    every gold fixture should render with no warnings, and every token's
    surface text (once HTML-escaped) should appear somewhere in the
    output -- confirming no token silently gets dropped while blocks are
    assembled."""
    tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
    html_out, warnings = tokengraph_to_depth_html(tokengraph)
    assert warnings == [], f"{example.slug}: {warnings}"
    for tok in tokengraph:
        if tok.tokentype in IMPLIED_TOKENTYPES:
            # An implied/elided token has no surface text at all --
            # tokengraph_to_depth_html() deliberately renders nothing for
            # it, so there is no escaped text to look for here.
            continue
        assert html.escape(tok.token) in html_out, f"{example.slug}: missing {tok.id}"
