"""
grammatike: render an analyzed Greek tokengraph as a Mermaid flowchart.
Greek analogue of arsgrammatica's mermaid.py.

Render a `tokengraph` (a list of TokenAnalysis, as produced by
greek_syntax_dspy.analyze_passage) as a Mermaid flowchart.

- Every non-punctuation token becomes a node, labelled with the token's
  surface text.
- Every `relatedtoken1`/`relationship1` and `relatedtoken2`/`relationship2`
  pair on a token becomes a labelled edge from that token to the related
  token.
- By default (`color_by_verbal_unit=True`), every token is also colored by
  the verbal unit it belongs to (see verbal_units.py's assign_verbal_units())
  -- so the clauses a sentence breaks into are visually distinguishable at a
  glance, not just inferable from following edges by hand. The one
  exception: an implied/elided token (models.py's IMPLIED_TOKENTYPES --
  "implied eimi", "implied repetition") always gets a dedicated "caution"
  amber (verbal_units._IMPLIED_TOKEN_COLOR) instead, regardless of which
  unit it anchors -- flagging "a real word is missing here" rather than
  blending in as an ordinary member of that unit's color. Its label is
  "elided eimi" or "implied repetition" (see `_IMPLIED_TOKEN_LABELS`
  below) rather than its own surface text, since it has none (`token` is
  `None`). This is the ONE place these tokens are shown at all --
  rendering.py's tokengraph_to_html()/tokengraph_to_depth_html() omit them
  entirely, same as tokengraph_to_text() does, since there's no real word
  to display in reconstructed prose; the Mermaid diagram is where an
  implied token's presence (and what it stands in for, via its edges) is
  actually worth seeing.
- A token whose `relationship1` is specifically "sentence connector" (e.g.
  γάρ tying this sentence back to the previous one) similarly always gets
  its own dedicated `sentenceconnector` node class -- neon-yellow fill,
  strong black border -- instead of its own verbal unit's color, regardless
  of which unit it's assigned to. rendering.py's tokengraph_to_html()/
  tokengraph_to_depth_html() apply the equivalent inline style to the same
  tokens, so the convention matches across both renderings.

(These are the fields syntax_model.md calls `relation1`/`relationship1` and
`relation2`/`relationship2` -- in models.py the "relation" side is named
`relatedtoken*` to make clear it holds a token id, not the relation label.)

Punctuation tokens are dropped as nodes. Any edge that would point at a
dropped or unrecognized token id is skipped rather than emitted as a broken
reference, and reported back to the caller so silent gaps are visible --
except the special sentinel target 'root' (an independent verb's own
relatedtoken1, per syntax_model.md), which is skipped silently: it isn't a
real node and was never supposed to be one, so it's not a gap worth
reporting.
"""

from typing import List, Tuple

from .models import IMPLIED_TOKENTYPES, TokenAnalysis
from .verbal_units import _IMPLIED_TOKEN_COLOR, assign_verbal_units, assign_verbal_unit_colors

# Characters that need escaping inside a Mermaid quoted label.
_LABEL_ESCAPES = {
    '"': "&quot;",
    "<": "&lt;",
    ">": "&gt;",
}


def _escape_label(text: str) -> str:
    for char, replacement in _LABEL_ESCAPES.items():
        text = text.replace(char, replacement)
    return text


# The Mermaid node label for an implied/elided token (keyed by its own
# tokentype, since it has no surface text of its own to use instead) --
# "implied eimi" reads as "elided eimi" here specifically because that's
# the more immediately legible term for a reader scanning the diagram
# (mirroring arsgrammatica's Latin "implied sum" -> "elided sum"); "implied
# repetition" already reads fine as-is. A future IMPLIED_TOKENTYPES value
# not listed here falls back to its own tokentype string verbatim (see its
# use below), so this mapping is a display nicety, not something either
# tokentype strictly depends on.
_IMPLIED_TOKEN_LABELS = {
    "implied eimi": "elided eimi",
    "implied repetition": "implied repetition",
}


# The verbal-unit color palette and the first-appearance ordering rule now
# live in verbal_units.py (assign_verbal_unit_colors()), shared with
# rendering.py's tokengraph_to_html() -- so both consumers assign identical
# colors to the same verbal units instead of each maintaining its own copy.


def tokengraph_to_mermaid(
    tokengraph: List[TokenAnalysis],
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
) -> Tuple[str, List[str]]:
    """Build a Mermaid `graph` diagram from a tokengraph.

    `orientation` is Mermaid's own flowchart orientation code -- `BT`
    (bottom-to-top, the default here), `TB`, `LR`, or `RL` -- used verbatim
    in the diagram's opening line (`graph BT`, `graph LR`, etc.). See
    https://mermaid.js.org/syntax/flowchart.html for what each value looks
    like; this function doesn't validate it, so a typo just becomes invalid
    Mermaid syntax in the output rather than an error here.

    `color_by_verbal_unit` (default True) colors every node by the verbal
    unit it belongs to, per verbal_units.assign_verbal_units() -- so each
    clause is visually distinguishable. Verbal units are assigned colors
    from `_VERBAL_UNIT_PALETTE` in the order their tokens first appear in
    `tokengraph`; a token assigned to no verbal unit is left with Mermaid's
    default node styling. The one exception is an implied/elided token
    (models.py's IMPLIED_TOKENTYPES) -- it always gets its own dedicated
    `implied` class, colored with `verbal_units._IMPLIED_TOKEN_COLOR`,
    instead of whatever `_VERBAL_UNIT_PALETTE` color its own verbal unit
    would otherwise get (see this module's own docstring for why). Pass
    False to skip coloring and get a plain diagram, as before this
    parameter existed.

    Returns (diagram_text, warnings). `warnings` lists any edges that were
    skipped because they referenced a punctuation token or an id not present
    in `tokengraph` -- worth checking, since it usually means the id came
    from a validation problem upstream (see greek_syntax_dspy.validate) --
    plus, if `color_by_verbal_unit` is True and the passage has more than 8
    verbal units, one warning that colors are repeating rather than staying
    distinct (the palette has 8 slots; see _VERBAL_UNIT_PALETTE).
    """
    node_ids = {tok.id for tok in tokengraph if tok.tokentype != "punctuation"}

    lines = [f"graph {orientation}"]
    for tok in tokengraph:
        if tok.id not in node_ids:
            continue
        # An implied/elided token (see models.py's IMPLIED_TOKENTYPES) has
        # no surface text at all -- tok.token is None -- so it needs a
        # placeholder label rather than crashing _escape_label() on None.
        # _IMPLIED_TOKEN_LABELS supplies that ("elided eimi" for "implied
        # eimi", "implied repetition" verbatim); the node's color (below)
        # is what actually marks it as an implied token, not the label
        # text.
        label = (
            tok.token
            if tok.token is not None
            else _IMPLIED_TOKEN_LABELS.get(tok.tokentype, tok.tokentype)
        )
        lines.append(f'    {tok.id}["{_escape_label(label)}"]')

    warnings = []
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
                    f"target is punctuation or not in tokengraph"
                )
                continue
            lines.append(f'    {tok.id} -->|{_escape_label(label)}| {related_id}')

    if color_by_verbal_unit:
        assignment = assign_verbal_units(tokengraph)
        colors, color_warnings = assign_verbal_unit_colors(tokengraph, assignment=assignment)
        warnings.extend(color_warnings)

        # Implied tokens (models.py's IMPLIED_TOKENTYPES) always get a
        # dedicated "caution" amber (_IMPLIED_TOKEN_COLOR) instead of
        # whatever color their own verbal unit would otherwise get --
        # regardless of which unit they anchor -- so they're excluded from
        # every per-unit `member_ids` group below and given their own
        # classDef/class pair instead. See rendering.py's
        # tokengraph_to_html() docstring for the matching HTML behavior.
        implied_ids = [
            tok.id
            for tok in tokengraph
            if tok.id in node_ids and tok.tokentype in IMPLIED_TOKENTYPES
        ]

        # A token whose relationship1 is specifically "sentence connector"
        # (e.g. γάρ tying this sentence back to the previous one -- see
        # rendering.py's identical carve-out, and models.py's RelationLabel
        # docstring for the distinction from the more general "connecting
        # word") always gets its own dedicated `sentenceconnector` class --
        # a neon-yellow fill with a strong black border -- instead of
        # whatever color its own verbal unit would otherwise get,
        # regardless of which unit it's assigned to. Excluded from
        # implied_ids (in the never-really-expected case a token is somehow
        # both) so the two classes never compete for the same node.
        connector_ids = [
            tok.id
            for tok in tokengraph
            if tok.id in node_ids
            and tok.id not in implied_ids
            and tok.relationship1 == "sentence connector"
        ]

        if colors or implied_ids or connector_ids:
            lines.append("")
            class_names = {}
            for i, (unit_id, (fill, stroke, text)) in enumerate(colors.items()):
                class_name = f"vu{i}"
                class_names[unit_id] = class_name
                lines.append(
                    f"    classDef {class_name} fill:{fill},stroke:{stroke},color:{text};"
                )
            for unit_id in colors:
                member_ids = [
                    tok.id
                    for tok in tokengraph
                    if tok.id in node_ids
                    and assignment.get(tok.id) == unit_id
                    and tok.id not in implied_ids
                    and tok.id not in connector_ids
                ]
                if member_ids:
                    lines.append(f"    class {','.join(member_ids)} {class_names[unit_id]};")
            if implied_ids:
                fill, stroke, text = _IMPLIED_TOKEN_COLOR
                lines.append(
                    f"    classDef implied fill:{fill},stroke:{stroke},color:{text};"
                )
                lines.append(f"    class {','.join(implied_ids)} implied;")
            if connector_ids:
                lines.append(
                    "    classDef sentenceconnector fill:#ffff00,stroke:#000000,stroke-width:4px,color:#000000;"
                )
                lines.append(f"    class {','.join(connector_ids)} sentenceconnector;")

    return "\n".join(lines), warnings


def save_mermaid(
    tokengraph: List[TokenAnalysis],
    path: str,
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
) -> List[str]:
    """Write the diagram to `path` (e.g. 'analysis.mmd') and return any
    warnings from tokengraph_to_mermaid."""
    diagram, warnings = tokengraph_to_mermaid(
        tokengraph, orientation=orientation, color_by_verbal_unit=color_by_verbal_unit
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram + "\n")
    return warnings
