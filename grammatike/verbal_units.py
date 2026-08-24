"""
grammatike: computing verbal-unit membership, subordination depth, and
rendering colors for a Greek tokengraph. Greek analogue of arsgrammatica's
verbal_units.py.

Partitions a `tokengraph` into the verbal units its own relations already
imply, so every token can be labelled "this token belongs to verbal unit
X" (or to none) -- e.g. for coloring a Mermaid diagram by clause (see
mermaid.py), or for any future visualization that wants the same grouping.

The key idea: a verbal unit's anchor token (the finite verb, infinitive, or
participle that owns an entry in `verbalunits` -- see models.py's
VerbalExpression) already marks itself in the tokengraph via `verbalunitid`
(set to its own id). Every OTHER token can be assigned to whichever anchor
its own relatedtoken1/relatedtoken2 chain eventually leads to -- with a
couple of wrinkles, handled specially below.

The first wrinkle: "subordinating conjunction" and "relative pronoun" are
themselves cross-clause pointers. A subordinating conjunction's own
relatedtoken1 points at the OUTER clause's verb (the one it modifies, e.g.
"ἐπειδὴ" -> "ἧκεν" in "ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη"), and a
relative pronoun's own relatedtoken1 points at its antecedent, which can
likewise sit in an outer/different clause (e.g. "ὅν" -> "νόμος" in "ὁ τῆς
πόλεως νόμος, ὃν σὺ περὶ ἐλάττονος τῶν ἡδονῶν ἐποιήσω"). Followed naively,
both would pull the conjunction/pronoun itself into the OUTER clause it
modifies or refers back to -- backwards from how a reader would group the
sentence: "ἐπειδὴ" reads as part of the dependent clause it introduces
("ἐπειδὴ δὲ ἦν πρὸς ἡμέραν"), not part of the main clause ("ἧκεν ἐκείνη")
it happens to modify.

The fix: syntax_model.md's own "unit verb (dependent)" rule already
records the *reverse* link explicitly -- every dependent verb's own
relatedtoken1 points AT its subordinating conjunction or relative pronoun,
with relationship1 = "unit verb" (see greek_syntax_dspy.py's docstring).
That reverse link is the authoritative "this token introduces clause V"
signal, so it's checked first, for every token: if some verb V has a
"unit verb" relation pointing at this token, this token belongs to V's
verbal unit, full stop -- its own outgoing relations (the antecedent link,
the outer-clause-modifies link) are not used to override that. Only when
no such reverse link exists does a token fall back to following its own
relatedtoken1/relatedtoken2 chain forward.

A second, analogous wrinkle: a noun or pronoun in a true "genitive
absolute" relation to a verb (syntax_model.md's "verbal units with
circumstantial participles") is syntactically absolute -- it does not
function inside that verb's own clause, even though its own relatedtoken1
points straight at that verb. Grammatically it belongs instead to the
circumstantial participle it agrees with, which always relates back to
that same noun via "circumstantial participle". Example: in "προϊόντος δὲ
τοῦ χρόνου ἧκον μὲν ἀπροσδοκήτως ἐξ ἀγροῦ", the noun *χρόνου* relates to
the verb *ἧκον* as "genitive absolute", but the participle *προϊόντος*
relates to *χρόνου* as "circumstantial participle" -- so *χρόνου* (and
anything that in turn chains through it, e.g. an adjective or apposition)
belongs to *προϊόντος*'s own verbal unit, NOT to *ἧκον*'s. This is checked
the same way as the "unit verb" wrinkle above -- via a reverse index --
but only overrides the default forward chase when the noun's own outgoing
relation actually is "genitive absolute"; a noun a circumstantial
participle agrees with that otherwise fits normally into the surrounding
clause (e.g. "ἐγώ" as subject in "ἐγὼ ἅπαντα ἐπιδείξω τὰ ἐμαυτοῦ πράγματα,
οὐδὲν παραλείπων, ἀλλὰ λέγων τἀληθῆ", where *παραλείπων* and *λέγων* are
circumstantial participles agreeing with *ἐγώ*) keeps that normal relation
and is NOT redirected -- syntax_model.md's own distinction between the two
cases is exactly this outgoing-relation label.

A third construction that might look like it needs the same treatment,
but does NOT: attributive participles. This is a genuinely Greek
construction with no Latin precedent -- Latin's scheme never treats an
attributive participle as a verbal expression at all, while Greek's does
(see models.py's VerbalExpression docstring). An attributive participle's
own relatedtoken1 points at the noun/pronoun it agrees with, with
relationship1 "attributive participle" -- structurally identical to the
circumstantial-participle link above. But unlike a circumstantial
participle's noun, the noun an attributive participle agrees with has NO
"absolute" construction to worry about: syntax_model.md documents no
free-floating/unconnected case for attributive participles the way it
does for circumstantial ones (there is no "attributive absolute"). The
noun always keeps a real, ordinary syntactic role in the surrounding
clause. So no reverse index and no override are needed here at all -- the
noun's own default forward chase already lands in the right place.
Example: in "ὁ γὰρ ἀνὴρ ὁ ὑβρίζων εἰς σὲ ... τυγχάνει", the participle
*ὑβρίζων* relates to *ἀνήρ* as "attributive participle", and *ἀνήρ* is
simply the subject of *τυγχάνει* (an ordinary "subject" relation, exactly
like any other subject) -- so *ἀνήρ* resolves to *τυγχάνει*'s verbal unit
via the plain fallback chase, and *ὑβρίζων* (plus anything chaining
through it, e.g. "εἰς σέ") resolves to its own verbal unit as the anchor
it is. This module therefore needs no code path for attributive
participles beyond the generic ones already in place for every other
non-anchor token.
"""

from typing import Dict, List, Optional, Tuple

from .models import TokenAnalysis

_UNIT_VERB = "unit verb"
_CIRCUMSTANTIAL_PARTICIPLE = "circumstantial participle"
_GENITIVE_ABSOLUTE = "genitive absolute"
_CONNECTING_WORD = "connecting word"

# Categorical palette for coloring verbal units: 8 (fill, stroke, text)
# triples, in a fixed order chosen so adjacent slots stay distinguishable
# under color-vision deficiency as well as normal vision. Pastel-hued by
# request: each `fill` is a light, low-saturation tint; `stroke` is that
# same hue at full saturation (the vivid color a non-pastel categorical
# palette would use), giving each swatch a colored outline instead of a
# colored fill as its primary identity cue; `text` is black throughout --
# every one of these fills has strong contrast against black (all comfortably
# above the WCAG AA text threshold of 4.5:1, several above 10:1).
#
# Lives here (rather than in mermaid.py, where it originated) so every
# consumer that wants "the same verbal-unit colors as the mermaid graph" --
# currently mermaid.py's own node coloring and rendering.py's
# tokengraph_to_html() -- shares one definition and one ordering rule
# (assign_verbal_unit_colors(), below) instead of each re-deriving it and
# risking drift.
#
# Pushing this light necessarily fails the dataviz skill's OKLCH lightness
# ceiling (0.77 for a light surface) -- true pastel and that ceiling are
# mutually exclusive, since the ceiling exists specifically to keep marks
# from reading as washed-out. That gate was designed for un-labeled marks
# (points, bars) where color alone carries identity; every node/span this
# palette colors already carries its own visible text label, which is the
# mitigation the skill itself prescribes for exactly this trade-off. What
# was NOT relaxed: adjacent-pair separation. This ordering was tuned (see
# scripts/validate_palette.js in Claude's dataviz skill) so it still clears
# both the CVD separation target (worst adjacent ΔE 10.6, target ≥8) and
# the normal-vision floor (worst adjacent ΔE 18.1, floor ≥15) -- the checks
# that actually determine whether two colors can be told apart.
# Cycles (mod 8) if a sentence has more than 8 verbal units -- see
# assign_verbal_unit_colors(), which reports this as a warning rather than
# silently repeating colors.
_VERBAL_UNIT_PALETTE = [
    ("#82bbff", "#2a78d6", "#000000"),  # blue
    ("#ffa682", "#eb6834", "#000000"),  # orange
    ("#70ffcc", "#1baf7a", "#000000"),  # aqua
    ("#ffd170", "#eda100", "#000000"),  # yellow
    ("#ff94bc", "#e87ba4", "#000000"),  # magenta
    ("#7aff7a", "#008300", "#000000"),  # green
    ("#a494ff", "#4a3aa7", "#000000"),  # violet
    ("#ff9594", "#e34948", "#000000"),  # red
]

# A dedicated "caution" color for implied/elided tokens (models.py's
# IMPLIED_TOKENTYPES: "implied eimi", "implied repetition") -- a strong,
# saturated amber, deliberately NOT drawn from _VERBAL_UNIT_PALETTE above
# (whose pastel tints it would otherwise be confusable with, especially the
# "yellow" slot) and deliberately not pastel itself, so it reads as "this
# marks something MISSING from the surface text" rather than as just
# another clause's color. Every consumer that renders an implied token
# (currently rendering.py's tokengraph_to_html()/tokengraph_to_depth_html()
# and mermaid.py's tokengraph_to_mermaid()) uses this SAME color for it,
# regardless of which verbal unit the token itself anchors -- the warning
# is about the token's own kind, not about which clause it's in. Black
# text keeps strong contrast against the fill, same convention as every
# palette slot above.
_IMPLIED_TOKEN_COLOR = ("#ffc107", "#7a5200", "#000000")  # amber warning



def assign_verbal_units(tokengraph: List[TokenAnalysis]) -> Dict[str, Optional[str]]:
    """Return {token id: verbal unit id or None}, one entry per token in
    `tokengraph` (including punctuation and unrelated tokens, so every id
    is accounted for -- callers that only care about assigned tokens can
    filter out the None values themselves).

    A verbal unit's own anchor token is assigned to itself (its
    `verbalunitid`). Every other token is assigned to the verbal unit its
    relations resolve to, per this module's docstring; a token with no
    resolvable relation (e.g. a bare accusative of time, an enclitic left
    unrelated) gets None.

    A true genitive-absolute noun (its own outgoing relation is "genitive
    absolute", not some normal clause role) is redirected to the verbal
    unit of the circumstantial participle it agrees with, rather than to
    the verb its own relatedtoken1 points at -- see this module's
    docstring for the full "προϊόντος δὲ τοῦ χρόνου ... ἧκον" example.
    Anything that in turn chains through that noun (an adjective, an
    appositive) follows it into the participle's unit too, since this
    redirect happens once, at the noun itself, and every other resolution
    is unchanged.

    Attributive participles need NO analogous redirect: the noun/pronoun
    they agree with always keeps its ordinary syntactic role (there is no
    "attributive absolute" construction), so it resolves correctly via the
    plain fallback chase below without any special-casing -- see this
    module's docstring for why.
    """
    by_id = {tok.id: tok for tok in tokengraph}

    # Reverse index: for every token that some OTHER token points at via a
    # "unit verb" relation, record who points at it. Per syntax_model.md,
    # a "unit verb" target is always either the literal sentinel 'root'
    # (from an independent verb -- never a real token) or a subordinating
    # conjunction/relative (or interrogative) pronoun's id (from a
    # dependent verb) -- so a hit here always means "this token introduces
    # the pointing verb's clause."
    introduces_clause_for: Dict[str, str] = {}
    # Reverse index: for every token that some OTHER token points at via a
    # "circumstantial participle" relation, record who points at it (the
    # participle -- real or implied -- that agrees with it). Used below to
    # redirect a TRUE genitive-absolute noun to that participle's own
    # verbal unit instead of the verb it otherwise points at; a noun a
    # circumstantial participle agrees with that fits normally into the
    # clause (its own outgoing relation isn't "genitive absolute") is left
    # alone and keeps resolving normally, so this index is consulted but
    # not always used.
    #
    # There is deliberately NO analogous reverse index for "attributive
    # participle": the noun/pronoun an attributive participle agrees with
    # never needs redirecting (see this module's docstring), so building
    # one here would just be dead code.
    circumstantial_participle_for: Dict[str, str] = {}
    for tok in tokengraph:
        for related_field, label_field in (
            ("relatedtoken1", "relationship1"),
            ("relatedtoken2", "relationship2"),
        ):
            related = getattr(tok, related_field)
            label = getattr(tok, label_field)
            if related is None or related == "root":
                continue
            if label == _UNIT_VERB:
                introduces_clause_for[related] = tok.id
            elif label == _CIRCUMSTANTIAL_PARTICIPLE:
                circumstantial_participle_for[related] = tok.id

    resolved: Dict[str, Optional[str]] = {}
    in_progress: set = set()

    def resolve(tid: str) -> Optional[str]:
        if tid in resolved:
            return resolved[tid]
        tok = by_id.get(tid)
        if tok is None:
            return None

        if tok.verbalunitid is not None:
            resolved[tid] = tok.verbalunitid
            return tok.verbalunitid

        if tid in in_progress:
            # A cycle in the relation graph (malformed LM output) -- bail
            # out on this token rather than recursing forever.
            return None
        in_progress.add(tid)

        result = None

        clause_verb_id = introduces_clause_for.get(tid)
        if clause_verb_id is not None:
            result = resolve(clause_verb_id)

        if result is None:
            participle_id = circumstantial_participle_for.get(tid)
            is_genitive_absolute = (
                tok.relationship1 == _GENITIVE_ABSOLUTE
                or tok.relationship2 == _GENITIVE_ABSOLUTE
            )
            if participle_id is not None and is_genitive_absolute:
                result = resolve(participle_id)

        if result is None:
            for related_field in ("relatedtoken1", "relatedtoken2"):
                related = getattr(tok, related_field)
                if related is None or related == "root":
                    continue
                result = resolve(related)
                if result is not None:
                    break

        in_progress.discard(tid)
        resolved[tid] = result
        return result

    for tid in by_id:
        resolve(tid)

    return resolved


def assign_verbal_unit_colors(
    tokengraph: List[TokenAnalysis],
    assignment: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[str, Tuple[str, str, str]], List[str]]:
    """Assign each verbal unit found in `tokengraph` a stable (fill, stroke,
    text) triple from `_VERBAL_UNIT_PALETTE`, using the exact ordering rule
    `tokengraph_to_mermaid()` uses for its node coloring -- so any other
    caller wanting "the same colors as the mermaid graph" (currently
    rendering.py's `tokengraph_to_html()`) gets an identical mapping without
    re-deriving the rule itself.

    Order is by first appearance of each verbal unit among tokengraph's
    *non-punctuation* tokens, since those are the only tokens that become
    mermaid nodes at all -- a verbal unit whose earliest token happens to be
    punctuation (it can't be: punctuation tokens aren't assigned to a
    verbal unit's anchor, but could in principle inherit one from a
    relation) still gets ordered by its first non-punctuation member.

    Pass `assignment` (the result of `assign_verbal_units(tokengraph)`) if
    the caller already computed it, to avoid re-deriving it here; otherwise
    it's computed internally.

    Returns `({verbal unit id: (fill, stroke, text)}, warnings)` --
    `warnings` holds one entry, with the same wording
    `tokengraph_to_mermaid()` uses, if there are more distinct verbal units
    than palette slots (colors repeat past the 8th unit). A verbal unit id
    absent from the returned dict was never assigned to any non-punctuation
    token -- callers should treat that the same as "no verbal unit" (no
    coloring), same as `tokengraph_to_mermaid()` does.
    """
    if assignment is None:
        assignment = assign_verbal_units(tokengraph)

    non_punctuation_ids = {tok.id for tok in tokengraph if tok.tokentype != "punctuation"}

    unit_order: List[str] = []
    seen_units = set()
    for tok in tokengraph:
        if tok.id not in non_punctuation_ids:
            continue
        unit_id = assignment.get(tok.id)
        if unit_id is not None and unit_id not in seen_units:
            seen_units.add(unit_id)
            unit_order.append(unit_id)

    warnings: List[str] = []
    if len(unit_order) > len(_VERBAL_UNIT_PALETTE):
        warnings.append(
            f"{len(unit_order)} verbal units but only {len(_VERBAL_UNIT_PALETTE)} "
            "distinct colors -- colors repeat and may be ambiguous between units"
        )

    colors = {
        unit_id: _VERBAL_UNIT_PALETTE[i % len(_VERBAL_UNIT_PALETTE)]
        for i, unit_id in enumerate(unit_order)
    }
    return colors, warnings


def compute_subordination_depths(
    tokengraph: List[TokenAnalysis],
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """Compute each verbal expression's *depth of subordination*: the
    number of verbal expressions it is removed from an independent ("root")
    clause. An independent verb is depth 0; a verb it introduces (a
    dependent clause, a direct quote, an aside) is depth 1; a verbal
    expression THAT verb in turn introduces (e.g. an indirect statement
    inside a dependent clause) is depth 2; and so on.

    A "verbal expression" here is any token that anchors one -- i.e. any
    token with `verbalunitid` set to its own id (the same convention
    `assign_verbal_units()` relies on). For each anchor, this function
    finds its *parent* anchor -- the verbal expression it's subordinate to
    -- by following the anchor's own relatedtoken1 (falling back to
    relatedtoken2), through as many intermediate non-anchor tokens as
    necessary, until it lands on another anchor. This one chase handles
    every documented case uniformly, without needing to special-case by
    relationship label, because they all eventually resolve to another
    anchor via forward pointers already in the graph:

    - unit verb (independent): relatedtoken1 == 'root' -> no parent, depth 0.
    - unit verb (dependent): relatedtoken1 -> a subordinating conjunction or
      relative/interrogative pronoun (not itself an anchor) -> ITS
      relatedtoken1 -> the superior verb (a conjunction) or an antecedent
      noun (a relative pronoun), the latter requiring one more hop through
      the noun's own relation to reach the verb it depends on. Example:
      "ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη" -- ἦν's relatedtoken1 is
      ἐπειδὴ, whose own relatedtoken1 is ἧκεν.
    - direct quote / aside / indirect statement: relatedtoken1 -> the verb
      of the clause it interrupts, is framed by, or (for an indirect-
      statement infinitive or participle) governs it, directly (no
      intermediate token). Examples: πειρᾷς -> ἔφη (direct quote); δεῖ ->
      ἔστι (aside); ἀποσβεσθῆναι -> ἔφασκε (indirect statement, an
      infinitive); φεύγουσαν -> εἶδε (indirect statement, a participle
      after a verb of perception).
    - circumstantial participle: relatedtoken1 -> the noun/pronoun it
      agrees with (not itself an anchor) -> that noun's own relation,
      either its normal role in the surrounding clause (one more hop to a
      verb, e.g. παραλείπων/λέγων -> ἐγώ -> ἐπιδείξω via "subject") or, for
      a true genitive absolute, 'genitive absolute' pointing directly at
      the main verb (e.g. προϊόντος -> χρόνου -> ἧκον).
    - attributive participle: structurally identical to the circumstantial
      case above, just with a different relationship label and no
      "absolute" variant -- relatedtoken1 -> the noun/pronoun it agrees
      with (not itself an anchor) -> that noun's own ordinary relation,
      one more hop to the governing verb. Example: "ὁ γὰρ ἀνὴρ ὁ ὑβρίζων
      εἰς σὲ ... τυγχάνει" -- ὑβρίζων's relatedtoken1 is ἀνήρ, whose own
      relatedtoken1 (relationship1 "subject") is τυγχάνει. This is the one
      construction with no Latin precedent at all (Latin's scheme never
      treats an attributive participle as a verbal expression), but it
      needs no new code here: the same generic hop-through-the-noun chase
      that already handles circumstantial participles handles it too.

    Returns `({anchor id: depth or None}, warnings)`. A depth of `None`
    means the chase from that anchor never reached another anchor (a
    malformed or genuinely disconnected verbal expression) or a cycle was
    detected; `warnings` names which anchor(s) and why, mirroring
    `tokengraph_to_mermaid()`'s warnings-list convention rather than
    raising.
    """
    by_id = {tok.id: tok for tok in tokengraph}
    anchor_ids = {tok.id for tok in tokengraph if tok.verbalunitid == tok.id}

    warnings: List[str] = []

    def chase(token_id: str, visited: set) -> Optional[str]:
        """Follow relatedtoken1 (then relatedtoken2) forward from
        `token_id`, returning the first anchor id reached, or None if the
        chain dead-ends or cycles before reaching one. `token_id` itself
        counts as a hit if it's already an anchor (the direct-link cases:
        direct quote, aside, indirect statement)."""
        if token_id in visited:
            return None
        visited.add(token_id)
        if token_id in anchor_ids:
            return token_id
        tok = by_id.get(token_id)
        if tok is None:
            return None
        for field in ("relatedtoken1", "relatedtoken2"):
            target = getattr(tok, field)
            if target is None or target == "root":
                continue
            result = chase(target, visited)
            if result is not None:
                return result
        return None

    def parent_of(anchor_id: str) -> Optional[str]:
        tok = by_id[anchor_id]
        for field in ("relatedtoken1", "relatedtoken2"):
            target = getattr(tok, field)
            if target is None or target == "root":
                continue
            result = chase(target, visited=set())
            if result is not None and result != anchor_id:
                return result
        return None

    depths: Dict[str, Optional[int]] = {}
    in_progress: set = set()

    def depth_of(anchor_id: str) -> Optional[int]:
        if anchor_id in depths:
            return depths[anchor_id]
        tok = by_id[anchor_id]
        if tok.relatedtoken1 == "root":
            depths[anchor_id] = 0
            return 0

        if anchor_id in in_progress:
            warnings.append(
                f"cycle detected resolving the governing verbal expression "
                f"for {anchor_id!r} -- leaving its depth (and its parent's) "
                f"unresolved"
            )
            return None
        in_progress.add(anchor_id)

        parent = parent_of(anchor_id)
        if parent is None:
            warnings.append(
                f"could not find a governing verbal expression for "
                f"{anchor_id!r} -- leaving its depth unresolved"
            )
            result = None
        else:
            parent_depth = depth_of(parent)
            result = None if parent_depth is None else parent_depth + 1

        in_progress.discard(anchor_id)
        depths[anchor_id] = result
        return result

    for anchor_id in anchor_ids:
        depth_of(anchor_id)

    return depths, warnings


def max_subordination_depth(
    tokengraph: List[TokenAnalysis],
    depths: Optional[Dict[str, Optional[int]]] = None,
) -> Optional[int]:
    """Return the deepest level of subordination reached anywhere in
    `tokengraph` -- the highest value `compute_subordination_depths()`
    assigns to any verbal expression. Root/independent clauses are depth
    0, so this is also the upper end of the valid `depth` range for
    `rendering.tokengraph_to_depth_html()`'s own `depth` parameter (whose
    valid range is 0, root clauses only, through this function's return
    value, everything).

    Pass `depths` (the first element of `compute_subordination_depths()`'s
    return value) if the caller already computed it, to avoid re-deriving
    it here; otherwise it's computed internally (any resolution warnings
    are silently dropped in that case -- call
    `compute_subordination_depths()` directly first if the caller also
    needs those).

    Returns `None` if `tokengraph` has no verbal expressions at all (an
    empty passage, or one with none of the five constructions
    syntax_model.md counts as one), or if every anchor's own depth came
    back unresolved (see `compute_subordination_depths()`'s own
    warnings for why an anchor might be unresolved -- a relation cycle, or
    a governing verbal expression that couldn't be found). Otherwise
    returns the maximum of every RESOLVED anchor's depth, ignoring
    unresolved ones rather than letting a single bad anchor blank out the
    whole result.
    """
    if depths is None:
        depths, _warnings = compute_subordination_depths(tokengraph)

    resolved = [d for d in depths.values() if d is not None]
    if not resolved:
        return None
    return max(resolved)


def find_unanchored_coordinated_verbs(tokengraph: List[TokenAnalysis]) -> List[str]:
    """Heuristic sanity check, adapted from arsgrammatica's original, for a
    class of live-LM mistake that Greek's simpler coordination scheme makes
    harder to catch precisely -- see below for why this function's
    mechanism had to change, not just its relation-label names.

    Latin's version of this check exploited a specific feature of Latin's
    "coordinating conjunction" relation: when it joined two verbal
    expressions, BOTH conjuncts' ids were recorded, one on relatedtoken1
    and one on relatedtoken2 of the same conjunction token. That gave a
    precise, name-both-sides asymmetry check: if exactly one side was a
    recognized verbal-unit anchor and the other wasn't, something was
    almost certainly wrong.

    Greek's analytic scheme has no such two-sided relation.
    syntax_model.md's "connecting word" (the construction used when a
    coordinating conjunction or particle "continue[s] a series" of nouns,
    adjectives, adverbs, OR whole clauses) records only ONE id on
    relation1: "the id of the first item" in the series -- never the id of
    the item the connecting word itself is attached to. So a "connecting
    word" that continues a series of CLAUSES points at the FIRST clause's
    verb, and says nothing at all, anywhere in the graph, about which
    token is the second (or later) clause's own verb. If an LM drops that
    second verb's verbalunitid entirely -- the exact failure this check
    was designed to catch in Latin (see gold_examples.py's
    coordinating_conjunction_dedit_et_dixit_esse fixture for the original,
    Latin-side observation) -- there is no longer a second id anywhere to
    compare against the first. That specific, precise check is not
    portable; rebuilding it exactly would mean inventing data the schema
    doesn't record.

    What this function checks instead, as the closest sound analogue:
    "connecting word" (never "sentence connector" -- see below) relations
    whose relation1 target IS a recognized verbal-unit anchor mark that
    anchor as the first member of a coordinate clause series, which by
    definition needs at least one more member. Coordinate clauses are
    ordinarily syntactic peers, so this function looks for at least one
    OTHER verbal expression at the SAME subordination depth (via
    `compute_subordination_depths()`) as the first member -- a plausible
    sibling. If none exists anywhere in `tokengraph`, that's flagged: the
    series' other member(s) may be missing their own verbalunitid (an
    explicit verb that was never flagged as an anchor, or a missing
    'implied repetition' token for an elided one).

    "sentence connector" is deliberately excluded from this check:
    syntax_model.md defines it as pointing at "the verb of THIS sentence"
    -- i.e. the sentence it introduces, not a cross-clause pairing with
    whatever precedes it ("we only mark the conjunction as related to the
    explicit verb" -- no implied relation to a preceding sentence is ever
    recorded). A sentence connector therefore never asserts "there are two
    coordinate members" the way a series-continuing "connecting word"
    does, so there is nothing here for it to be checked against.

    This is a coarser, lower-precision check than Latin's: it can only
    say "this coordinate series looks like it's missing a member
    SOMEWHERE in the passage," not name the specific missing token the
    way Latin's version could (Latin named the unanchored id directly).
    It can also under-report: if `tokengraph` happens to contain some
    OTHER, unrelated verbal expression at the same depth, this check will
    treat that as a plausible sibling and stay silent even when the real
    partner is genuinely missing.
    # TODO: if a future revision of the tokengraph threads sentence/clause
    # boundaries through to this module, this check could be tightened to
    # look only within the same sentence rather than across the whole
    # passage.

    Returns a list of warning strings (empty if nothing looks suspicious),
    the same "degrade visibly, don't raise" convention every other
    warnings-returning function in this codebase uses. This is a
    heuristic, not a guarantee: a clean result here isn't a substitute for
    validate() or a human read of the analysis, and a flagged result
    deserves a look rather than an automatic "fix."
    """
    by_id = {tok.id: tok for tok in tokengraph}
    anchor_ids = {tok.id for tok in tokengraph if tok.verbalunitid == tok.id}
    depths, _depth_warnings = compute_subordination_depths(tokengraph)

    # Every anchor that is the recorded "first item" of at least one
    # "connecting word" relation -- i.e. every anchor that syntax_model.md
    # says is the head of a coordinate clause series.
    coordinated_first_items = {
        tok.relatedtoken1
        for tok in tokengraph
        if tok.relationship1 == _CONNECTING_WORD
        and tok.relatedtoken1 is not None
        and tok.relatedtoken1 in anchor_ids
    }

    warnings: List[str] = []
    for first_id in sorted(coordinated_first_items):
        first_depth = depths.get(first_id)
        if first_depth is None:
            # compute_subordination_depths() already warned about this
            # anchor's own depth; piling on here wouldn't add information.
            continue

        has_sibling = any(
            other_id != first_id and depths.get(other_id) == first_depth
            for other_id in anchor_ids
        )
        if not has_sibling:
            tok = by_id.get(first_id)
            text = tok.token if tok is not None else first_id
            warnings.append(
                f"{first_id} ({text!r}) is the first item of a coordinate "
                f"clause series (a 'connecting word' relation points at it) "
                f"at subordination depth {first_depth}, but no other verbal "
                "expression in the passage shares that depth -- the "
                "clause(s) it coordinates with may be missing their own "
                "verbalunitid/anchor (an explicit verb that wasn't flagged, "
                "or a missing 'implied repetition' token if the verb was "
                "elided)."
            )

    return warnings
