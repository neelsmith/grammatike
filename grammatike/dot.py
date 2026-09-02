"""
grammatike: render an analyzed Greek tokengraph as Graphviz DOT source -- a
second renderer alongside mermaid.py's tokengraph_to_mermaid(), sharing the
exact same node/edge selection, labeling, and coloring rules (token_label(),
verbal_units.assign_verbal_units()/assign_verbal_unit_colors(),
IMPLIED_TOKENTYPES, and grammatike's own sentence-connector special case --
see below), so the two diagrams of the same tokengraph agree in every way
except the output format and how same-depth alignment is achieved. Greek
analogue of arsgrammatica's dot.py, adapted at two points where grammatike's
own package doesn't have an exact Latin-side counterpart -- see "Two
adaptations" below.

That last difference -- how same-depth alignment is achieved -- is the whole
reason this module exists. tokengraph_to_mermaid() has no rank_by_depth
parameter in grammatike (unlike arsgrammatica's own mermaid.py), so there is
no existing same-depth-alignment nudge to compare against here; but the
underlying reason a second renderer is still worth having is the same one
arsgrammatica's dot.py documents: Graphviz's `dot` layout engine has an
actual primitive for forcing nodes onto the same rank -- a
`{rank=same; id1; id2; ...}` subgraph statement, the same horizontal level
for `rankdir=BT`/`TB`, or vertical column for `LR`/`RL` -- a hard
constraint, not a heuristic a layout engine could override. `rank_by_depth`
here builds exactly that.

This module only builds DOT source text -- like tokengraph_to_mermaid(), it
has no rendering dependency at all (no `graphviz` package, no `pydot`,
nothing beyond the standard library). Turning the text into a picture needs
a separate Graphviz installation (the `dot` command-line tool) wherever you
view it -- e.g. `dot -Tsvg analysis.dot > analysis.svg`, an online DOT
viewer, or Quarto's own fenced ```{dot}``` code-block support (which needs
Graphviz on the machine building the site) -- the same division of labor as
a Mermaid diagram needing a Mermaid-capable renderer (mermaid.js, marimo's
`mo.mermaid()`, ...) to actually draw it. Unlike Mermaid, marimo has no
built-in `mo.graphviz()`-equivalent as of this writing -- showing one of
these diagrams in a notebook cell needs one extra manual step: render to
SVG yourself (however you like -- `subprocess` to the `dot` binary, or the
`graphviz` PyPI package, which does the same subprocess call for you) and
wrap the result in `mo.Html(svg_text)`. See notes/dot_diagrams.md and
marimo/greek_syntaxer_dot.py.

tokengraph_to_dot()'s `depth` parameter is a SEPARATE feature from the
rank-alignment one above, and uses a THIRD depth notion, distinct from both
`rank_by_depth`'s (see "Two adaptations" below) and
verbal_units.compute_subordination_depths() (the CLAUSE-level notion behind
rendering.tokengraph_to_depth_html()'s own indented-HTML `depth` slider):
compute_graph_depths(), a plain graph distance -- the number of edges from a
token back to the nearest root/independent verbal anchor, following the
exact same relatedtoken1/relatedtoken2 edges this module draws as `->`
lines. A clause's ordinary dependents (subject, object, dative, adjectival
modifiers, ...) are each their own hop from their governing verb, so
`depth=0` shows ONLY root verbal-unit anchors -- not "the whole root
clause" the way tokengraph_to_depth_html()'s block-level depth would. See
tokengraph_to_dot()'s own docstring for the full rationale, including how a
dropped node's dangling edges are handled.

Two adaptations from arsgrammatica's own dot.py
-------------------------------------------------
1. `rank_by_depth`'s grouping. arsgrammatica's dot.py groups verbal-unit
   anchors by verbal_units.compute_aat_depths() -- a function that has no
   counterpart in grammatike's own verbal_units.py. `rank_by_depth` here
   uses verbal_units.compute_subordination_depths() instead -- grammatike's
   own existing CLAUSE-level depth function (the same one
   rendering.tokengraph_to_depth_html() already uses for its indented-HTML
   view), grouped by whatever depth value it assigns each anchor.
   Practically, this is the "how many verbal expressions is this one
   subordinate to" notion, not a distinct AAT-specific one -- close enough
   in spirit to what arsgrammatica's rank_by_depth achieves (same-depth
   verbal units forced onto the same rank) that reusing it, rather than
   inventing a new depth function with no other consumer, is the more
   maintainable choice. One real consequence of the substitution:
   compute_subordination_depths() CAN leave an anchor's depth unresolved
   (`None`, with a warning) -- a relation cycle, or a governing verbal
   expression that couldn't be found -- a state compute_aat_depths() never
   has. An anchor with an unresolved depth is simply excluded from every
   `rank=same` grouping here (nothing to align it WITH, or rather: nothing
   safe to align it with), and compute_subordination_depths()'s own warning
   about it is folded into this function's returned warnings the same way
   any other warning is -- so `rank_by_depth` CAN add a warning here, unlike
   arsgrammatica's own version (see compute_aat_depths()'s own docstring
   there, which has no unresolved state at all).
2. Sentence-connector coloring. A token whose `relationship1` is
   specifically "sentence connector" (e.g. γάρ tying this sentence back to
   the previous one) always gets its own dedicated color here, instead of
   whatever color its own verbal unit would otherwise get -- mermaid.py's
   own `sentenceconnector` node class, ported to DOT's inline-attribute
   idiom the same way the implied-token color already is (see
   `_node_attrs()` below). arsgrammatica's dot.py has no equivalent, since
   this is a grammatike/Greek-specific convention (see mermaid.py's own
   module docstring) with no Latin-side counterpart to adapt from -- kept
   here purely for internal consistency between grammatike's two renderers
   of the same tokengraph.
"""

from typing import Dict, List, Optional, Tuple

from .mermaid import token_label
from .models import IMPLIED_TOKENTYPES, TokenAnalysis
from .verbal_units import (
    _IMPLIED_TOKEN_COLOR,
    assign_verbal_units,
    assign_verbal_unit_colors,
    compute_subordination_depths,
)

# Characters that need escaping inside a DOT quoted string: backslash first
# (so escaping it doesn't double-escape the quotes/newlines added after),
# then the quote character itself. DOT has no HTML-entity-style escaping
# the way Mermaid labels do (see mermaid.py's _LABEL_ESCAPES) -- a quoted
# DOT string is just a C-style string literal.
_LABEL_ESCAPES = [
    ("\\", "\\\\"),
    ('"', '\\"'),
]


def _escape_label(text: str) -> str:
    for char, replacement in _LABEL_ESCAPES:
        text = text.replace(char, replacement)
    return text


# A token whose relationship1 is "sentence connector" gets this dedicated
# (fill, stroke, text) triple instead of its own verbal unit's color --
# same neon-yellow-fill/strong-black-border cue mermaid.py's
# `sentenceconnector` classDef uses (`fill:#ffff00,stroke:#000000,
# stroke-width:4px,color:#000000`). DOT has no direct equivalent of a
# stroke WIDTH as part of a plain (fill, stroke, text) color triple the way
# _VERBAL_UNIT_PALETTE/_IMPLIED_TOKEN_COLOR are shaped -- _node_attrs()
# below adds the matching `penwidth="4"` attribute separately, via its own
# `strong_border` flag, whenever this color is used.
_SENTENCE_CONNECTOR_COLOR = ("#ffff00", "#000000", "#000000")


def _node_attrs(
    tok: TokenAnalysis,
    color: Optional[Tuple[str, str, str]] = None,
    strong_border: bool = False,
) -> str:
    """The bracketed attribute list for one node line: always `label=`,
    plus `style=rounded` for an implied/elided token (models.py's
    IMPLIED_TOKENTYPES -- the same rounded-corner-box cue
    tokengraph_to_mermaid() draws with Mermaid's `(...)` node shape, instead
    of the plain `[...]`/`shape=box` rectangle every other node uses),
    regardless of whether `color` is given. `color`, when given, is an
    (fill, stroke, text) triple from _VERBAL_UNIT_PALETTE,
    _IMPLIED_TOKEN_COLOR, or _SENTENCE_CONNECTOR_COLOR -- adds
    `style=filled` (combined with `rounded` into one comma-separated
    `style` value when both apply) plus `fillcolor`/`color`/`fontcolor`.
    `color=None` (color_by_verbal_unit=False, or a token assigned to no
    verbal unit) leaves the node with the digraph's own default
    `node [shape=box]` styling, unfilled. `strong_border=True` (a sentence
    connector -- see this module's own docstring, "Two adaptations") adds
    `penwidth="4"`, the DOT counterpart to mermaid.py's `sentenceconnector`
    class's `stroke-width:4px`; meaningless (and never passed) without a
    `color` to draw a border around in the first place.
    """
    label = token_label(tok)
    styles = []
    if tok.tokentype in IMPLIED_TOKENTYPES:
        styles.append("rounded")

    attrs = [f'label="{_escape_label(label)}"']
    if color is not None:
        fill, stroke, text = color
        styles.append("filled")
        attrs.append(f'fillcolor="{fill}"')
        attrs.append(f'color="{stroke}"')
        attrs.append(f'fontcolor="{text}"')
        if strong_border:
            attrs.append('penwidth="4"')
    if styles:
        attrs.append(f'style="{",".join(styles)}"')

    return ", ".join(attrs)


def compute_graph_depths(tokengraph: List[TokenAnalysis]) -> Dict[str, int]:
    """Each non-punctuation token's *graph depth*: the number of edges
    separating it from the nearest root/independent verbal-unit anchor,
    following the exact same relatedtoken1/relatedtoken2 edges
    tokengraph_to_dot() itself draws as `->` lines (dependent -> governor)
    -- the depth notion behind tokengraph_to_dot()'s own `depth` parameter.
    See this module's own docstring for how this differs from
    verbal_units.compute_subordination_depths() (rendering.
    tokengraph_to_depth_html()'s clause-level notion, also what
    `rank_by_depth` uses here -- see "Two adaptations" above).

    A root anchor (relatedtoken1 == 'root') is depth 0. Every other token's
    depth is one more than its PARENT's -- relatedtoken1, falling back to
    relatedtoken2 only when relatedtoken1 itself doesn't resolve to a
    usable parent (None, or an id not in `tokengraph`) -- the SAME
    "relatedtoken1, fall back to relatedtoken2" preference
    verbal_units.compute_subordination_depths() already uses to chase a
    verbal expression's own governor. This matters for a token that plays
    two roles at once, most notably a relative pronoun: e.g. syntax_model.md's
    ὅν pointing at its antecedent ἀνήρ via relatedtoken1 ('relative
    pronoun') AND at the dependent verb it's ALSO the direct object of via
    relatedtoken2 ('direct object') -- that second edge points forward,
    toward a token that in turn points back at the pronoun itself (its own
    'unit verb' relation), a genuine two-way link the data model allows.
    Taking the shallower of BOTH edges (rather than preferring
    relatedtoken1) would let that forward edge "cheat" the pronoun's own
    depth down to whatever the dependent verb's -- itself only computable
    FROM the pronoun -- happens to resolve to first, collapsing what should
    be a deeper chain. Only ever falling back to relatedtoken2, never
    averaging or taking a minimum over both, avoids that: relatedtoken1
    alone already resolves to the antecedent here, so relatedtoken2 is
    simply never consulted for depth (it's still drawn as its own edge
    below, same as always -- this only affects which relation DEPTH
    follows).

    A token whose relatedtoken1 AND any fallback relatedtoken2 both fail to
    resolve (neither set, or pointing at ids not in `tokengraph`), or which
    is caught in a relation cycle even after preferring relatedtoken1,
    defaults to depth 0 -- the same "can't determine, default to root
    level" fallback verbal_units.compute_subordination_depths() and
    rendering.tokengraph_to_depth_html() both use for their own unresolved
    cases, rather than raising.

    Returns `{token id: depth}`, one entry per non-punctuation token in
    `tokengraph` (punctuation is never part of the diagram, so never
    included here either).
    """
    by_id = {tok.id: tok for tok in tokengraph}
    depths: Dict[str, int] = {}
    in_progress: set = set()

    def depth_of(tok_id: str) -> int:
        if tok_id in depths:
            return depths[tok_id]
        tok = by_id[tok_id]
        if tok.relatedtoken1 == "root":
            depths[tok_id] = 0
            return 0

        if tok_id in in_progress:
            # A relation cycle -- fall back to 0 rather than recursing
            # forever; NOT cached, so a non-cyclic call further up the
            # stack still computes (and caches) this token's real depth if
            # some other path reaches it.
            return 0
        in_progress.add(tok_id)

        parent_id = None
        if tok.relatedtoken1 is not None and tok.relatedtoken1 != "root" and tok.relatedtoken1 in by_id:
            parent_id = tok.relatedtoken1
        elif tok.relatedtoken2 is not None and tok.relatedtoken2 in by_id:
            parent_id = tok.relatedtoken2

        result = 1 + depth_of(parent_id) if parent_id is not None else 0

        in_progress.discard(tok_id)
        depths[tok_id] = result
        return result

    for tok in tokengraph:
        if tok.tokentype == "punctuation":
            continue
        depth_of(tok.id)

    return depths


def max_graph_depth(tokengraph: List[TokenAnalysis]) -> Optional[int]:
    """The highest value compute_graph_depths() assigns to any token in
    `tokengraph` -- the upper end of the meaningful range for
    tokengraph_to_dot()'s own `depth` parameter, the same role
    verbal_units.max_subordination_depth() plays for
    tokengraph_to_depth_html()'s unrelated depth notion. Returns None for
    an empty tokengraph, or one with only punctuation."""
    depths = compute_graph_depths(tokengraph)
    return max(depths.values()) if depths else None


def tokengraph_to_dot(
    tokengraph: List[TokenAnalysis],
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
    rank_by_depth: bool = True,
    depth: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Build a Graphviz DOT `digraph` from a tokengraph -- the same diagram
    tokengraph_to_mermaid() draws (same nodes, same edges, same coloring),
    as DOT source instead of Mermaid source. See this module's own
    docstring for why this exists alongside tokengraph_to_mermaid(), the
    two adaptations from arsgrammatica's own dot.py, and what actually
    rendering the result requires.

    `orientation` maps directly onto DOT's `rankdir` graph attribute -- `BT`
    (bottom-to-top, the default here, matching tokengraph_to_mermaid()'s
    own default), `TB`, `LR`, or `RL`. Not validated here, same as
    tokengraph_to_mermaid()'s `orientation` -- a typo just becomes an
    attribute value Graphviz itself will reject.

    `color_by_verbal_unit` (default True) colors every node by the verbal
    unit it belongs to, per verbal_units.assign_verbal_units() -- the exact
    same colors (and the same >8-verbal-units warning) as
    tokengraph_to_mermaid(), just written as `fillcolor`/`color`/
    `fontcolor` attributes directly on each node line instead of Mermaid's
    separate `classDef`/`class` statements (DOT has no equivalent of a
    named, reusable class -- inline per-node attributes are the idiomatic
    way to do this). An implied/elided token (IMPLIED_TOKENTYPES) always
    gets its own dedicated amber (verbal_units._IMPLIED_TOKEN_COLOR)
    instead of whatever color its own verbal unit would otherwise get, and
    a token whose relationship1 is "sentence connector" always gets its own
    dedicated neon-yellow/strong-border color (_SENTENCE_CONNECTOR_COLOR),
    both exactly mirroring tokengraph_to_mermaid()'s own precedence (see
    that function's docstring). Pass False for a plain, uncolored diagram.

    `rank_by_depth` (default True) is the reason this module exists
    alongside tokengraph_to_mermaid() -- see the module docstring. Every
    verbal-unit anchor node (any token with `verbalunitid` set to its own
    id, implied tokens included) at the same depth in
    verbal_units.compute_subordination_depths() -- grammatike's own
    substitute for arsgrammatica's compute_aat_depths(), see "Two
    adaptations" in this module's own docstring -- gets listed together in
    one `{rank=same; id1; id2; ...}` subgraph statement, which *forces*
    Graphviz's layout engine to place them on the same rank -- not a nudge,
    a hard constraint. A depth with only one anchor gets no `rank=same`
    statement (nothing to align it WITH); an anchor whose subordination
    depth came back unresolved (None -- a relation cycle, or no governing
    verbal expression found; see compute_subordination_depths()'s own
    docstring) is excluded from every grouping, and its own warning about
    why is folded into this function's returned warnings. Pass False to
    skip this and let Graphviz's own layout heuristics place every node.

    `depth`, if given, caps the diagram to nodes at or within that many
    edges of a root/independent verbal-unit anchor -- compute_graph_depths()
    above, a plain GRAPH distance along the same relatedtoken1/
    relatedtoken2 edges drawn as `->` lines below, NOT
    verbal_units.compute_subordination_depths() (`rank_by_depth` above, and
    the CLAUSE-level notion behind tokengraph_to_depth_html()'s own
    indented-HTML `depth` slider -- a whole clause's subject, object, and
    other ordinary dependents share ONE subordination depth with their
    verb, but each is its own hop of GRAPH depth). `depth=0` shows ONLY
    root anchors -- an independent verb with no dependents at all;
    `depth=1` adds every token one edge away from a root anchor (its
    subject, object, adverbials, ...); and so on. A token farther than
    `depth` is dropped entirely: omitted as a node, exactly as if it had
    never been in `tokengraph`. Omit `depth` (or pass `None`, the default)
    to show every node, same as before this parameter existed. A `depth` at
    or beyond max_graph_depth()'s own return value for this `tokengraph`
    shows everything too; a negative `depth` raises ValueError.

    Dropping a node can leave a KEPT node's edge pointing at a now-excluded
    one. Such an edge is skipped, with the same combined warning already
    used for an edge targeting punctuation or a genuinely absent id (see
    Returns below) -- `depth` filtering degrades visibly rather than
    emitting a dangling `->` line Graphviz would reject.

    Returns `(dot_source, warnings)` -- same shape and same warnings as
    tokengraph_to_mermaid(): an edge skipped because it targets a
    punctuation token, a token excluded by the `depth` cutoff, or an id not
    present in `tokengraph` (except the 'root' sentinel, skipped silently,
    same as there); if `color_by_verbal_unit` is True and the passage has
    more than 8 verbal units, one warning that colors are repeating.
    `depth` filtering itself never adds a warning (compute_graph_depths()
    has no unresolved state -- an unrelated or cyclic token just defaults
    to depth 0), but `rank_by_depth` CAN add one -- see above, and the "Two
    adaptations" section of this module's own docstring for why this
    differs from arsgrammatica's own version.
    """
    if depth is not None and depth < 0:
        raise ValueError(f"depth must be >= 0 (root nodes only), got {depth!r}")

    node_ids = {tok.id for tok in tokengraph if tok.tokentype != "punctuation"}

    warnings: List[str] = []
    if depth is not None:
        graph_depths = compute_graph_depths(tokengraph)
        depth_excluded_ids = {tok_id for tok_id, d in graph_depths.items() if d > depth}
        node_ids -= depth_excluded_ids

    colors_by_unit: Dict[str, Tuple[str, str, str]] = {}
    assignment: Dict[str, Optional[str]] = {}
    implied_ids: set = set()
    connector_ids: set = set()
    if color_by_verbal_unit:
        assignment = assign_verbal_units(tokengraph)
        colors_by_unit, color_warnings = assign_verbal_unit_colors(tokengraph, assignment=assignment)
        warnings.extend(color_warnings)
        implied_ids = {
            tok.id
            for tok in tokengraph
            if tok.id in node_ids and tok.tokentype in IMPLIED_TOKENTYPES
        }
        # See this module's own docstring, "Two adaptations" (2) --
        # grammatike-specific, no arsgrammatica counterpart. Excluded from
        # implied_ids (in the never-really-expected case a token is somehow
        # both) so the two colors never compete for the same node, matching
        # mermaid.py's own connector_ids precedent exactly.
        connector_ids = {
            tok.id
            for tok in tokengraph
            if tok.id in node_ids
            and tok.id not in implied_ids
            and tok.relationship1 == "sentence connector"
        }

    lines = ["digraph tokengraph {", f"    rankdir={orientation};", "    node [shape=box];", ""]
    for tok in tokengraph:
        if tok.id not in node_ids:
            continue
        color = None
        strong_border = False
        if color_by_verbal_unit:
            if tok.id in implied_ids:
                color = _IMPLIED_TOKEN_COLOR
            elif tok.id in connector_ids:
                color = _SENTENCE_CONNECTOR_COLOR
                strong_border = True
            else:
                unit_id = assignment.get(tok.id)
                color = colors_by_unit.get(unit_id) if unit_id is not None else None
        lines.append(f"    {tok.id} [{_node_attrs(tok, color, strong_border)}];")

    lines.append("")
    for tok in tokengraph:
        if tok.id not in node_ids:
            continue
        for related_field, label_field in (
            ("relatedtoken1", "relationship1"),
            ("relatedtoken2", "relationship2"),
        ):
            related_id = getattr(tok, related_field)
            label = getattr(tok, label_field)
            if related_id is None or label is None:
                continue
            if related_id == "root":
                # An independent verb's own unit-verb relation, per
                # syntax_model.md -- intentionally not a real node, so not
                # a warning-worthy gap. Just draw no edge for it.
                continue
            if related_id not in node_ids:
                warnings.append(
                    f"skipped edge {tok.id} -[{label}]-> {related_id}: "
                    f"target is punctuation, excluded by the depth cutoff, "
                    f"or not in tokengraph"
                )
                continue
            lines.append(f'    {tok.id} -> {related_id} [label="{_escape_label(label)}"];')

    if rank_by_depth:
        sub_depths, depth_warnings = compute_subordination_depths(tokengraph)
        warnings.extend(depth_warnings)

        # Same grouping tokengraph_to_mermaid() would build for a `~~~`
        # chain if it had one -- see this module's own docstring, "Two
        # adaptations" (1), for why sub_depths.get() being None here means
        # "unresolved" (a cycle, or no governing verbal expression found),
        # unlike arsgrammatica's own compute_aat_depths(), which has no such
        # state. Named sub_depths (not `depths`, and this loop's own
        # variable not `depth`) to avoid shadowing the `depth` PARAMETER
        # above -- a different depth notion entirely, see this function's
        # own docstring.
        depth_groups: dict = {}
        for tok in tokengraph:
            if tok.id not in node_ids:
                continue
            anchor_depth = sub_depths.get(tok.id)
            if anchor_depth is None:
                continue
            depth_groups.setdefault(anchor_depth, []).append(tok.id)

        rank_lines = [
            "    {rank=same; " + "; ".join(ids) + ";}"
            for anchor_depth in sorted(depth_groups)
            for ids in (depth_groups[anchor_depth],)
            if len(ids) > 1
        ]
        if rank_lines:
            lines.append("")
            lines.extend(rank_lines)

    lines.append("}")
    return "\n".join(lines), warnings


def save_dot(
    tokengraph: List[TokenAnalysis],
    path: str,
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
    rank_by_depth: bool = True,
    depth: Optional[int] = None,
) -> List[str]:
    """Write the diagram to `path` (e.g. 'analysis.dot') and return any
    warnings from tokengraph_to_dot()."""
    diagram, warnings = tokengraph_to_dot(
        tokengraph,
        orientation=orientation,
        color_by_verbal_unit=color_by_verbal_unit,
        rank_by_depth=rank_by_depth,
        depth=depth,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram + "\n")
    return warnings
