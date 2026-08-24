"""
Pydantic models describing the two structures from syntax_model.md:

1. A list of VerbalExpression entries (the "table of verbal expressions").
2. A list of TokenAnalysis entries (the "token-level table of dependencies").

These need to be real pydantic BaseModel subclasses (not plain classes with
bare `=` assignments) for DSPy to generate and validate structured output
against them when used inside `List[...]` input/output fields.

This module is the Greek analogue of arsgrammatica's `models.py`: same
architecture and field names (so downstream code and callers familiar with
arsgrammatica feel at home), but the enumerated literal values and examples
follow `syntax_model.md`'s Greek-specific analytic scheme rather than the
Latin one.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CitedText(BaseModel):
    """One citable unit of source text -- e.g. one line of poetry, one
    section of prose -- paired with its citation label. A sequence of
    these is segmentation.py's input: sentence boundaries do NOT need
    to respect CitedText boundaries (one sentence may span several units),
    but every resulting token still records which unit it came from via
    Token.citation."""

    citation: str = Field(description="Citation label for this unit, e.g. 'Lysias 1.1'.")
    text: str = Field(description="This unit's raw text, exactly as written.")


class Token(BaseModel):
    """A single pre-segmented token with a stable id.

    `citation` is optional so this model still works for citation-free
    callers -- e.g. a test fixture built directly from a canned tokengraph,
    with no CitedText source at all -- as well as for the citation-aware
    segmentation stage (segmentation.py), which is the only thing that
    actually populates it, knowing which CitedText source unit each token
    came from."""

    id: str = Field(description="Stable token id, globally unique and sequential across the whole input, e.g. 't0', 't1', ...")
    text: str = Field(description="The token's surface text, exactly as it appears in the source.")
    citation: Optional[str] = Field(
        default=None,
        description="Citation label of the source unit this token came from (e.g. 'Lysias 1.1'), if known.",
    )


class Sentence(BaseModel):
    """One sentence's worth of tokens, in reading order, as produced by the
    deterministic segmentation stage (segmentation.py). Token ids are
    global across the whole passage -- numbering continues across sentence
    boundaries rather than restarting at t0 for each sentence -- so a
    Sentence is a contiguous slice of the passage's id sequence, not an
    independently-numbered unit."""

    tokens: List[Token] = Field(
        description="This sentence's tokens, in reading order, using the passage's global token ids."
    )


class VerbalExpression(BaseModel):
    """One entry in the table of verbal expressions (syntax_model.md, 'Basic
    model' / verbal-expression sections). Three constructions count as a
    verbal expression: finite verbs, infinitives (only when part of indirect
    speech), and participles -- but only some participles: an *attributive*
    participle (e.g. ὁ ἀνὴρ ὁ ὑβρίζων εἰς σέ, "the man who is insulting
    you") and a *circumstantial* participle (e.g. χρόνου μεταξὺ
    διαγενομένου, "when some time had passed") each constitute a verbal
    expression of their own, as does a participle expressing indirect
    speech after a verb of perception or thinking (εἶδε τὴν βασίλειαν
    φεύγουσαν, "he saw the queen fleeing"). A *supplementary* participle
    (e.g. ὁ ἀνὴρ ἐχθρὸς ὢν ἡμῖν τυγχάνει, where ὤν supplements τυγχάνει) is
    NOT a verbal expression at all -- it does not get its own entry here.

    Each construction has its own set of allowed `syntactic_type` values,
    given explicitly by syntax_model.md rather than left to convention:
    a finite verb is 'independent', 'dependent', 'direct quote' (occurring
    in directly quoted speech, e.g. πειρᾷς in '"ἵνα σύ γε" ἔφη "πειρᾷς
    ἐνταῦθα τὴν παιδίσκην"'), or 'aside' (a verbal expression that
    interrupts the surrounding syntax, e.g. δεῖ in 'δεῖ γὰρ καὶ ταῦθ᾽
    ὑμῖν διηγήσασθαι'); an infinitive or participle anchoring an indirect
    statement is always 'indirect statement'; an attributive participle is
    'attributive'; a circumstantial participle is 'circumstantial'."""

    id: str = Field(
        description=(
            "The token id (from the input `tokens` list) of the finite verb, "
            "infinitive, or participle that anchors this verbal expression. "
            "For a multi-word compound form with a conjugated form of εἰμί "
            "(e.g. the perfect passive/middle system, ὁ νόμος γεγραμμένος "
            "ἐστίν), use the id of the conjugated form of εἰμί, not the "
            "participle. For an implied/elided verbal expression (see "
            "TokenAnalysis's 'implied eimi'/'implied repetition' tokentypes, "
            "IMPLIED_TOKENTYPES), use the new implied token's id instead -- "
            "an implied token always anchors its own verbal expression."
        )
    )
    syntactic_type: Literal[
        "independent",
        "dependent",
        "direct quote",
        "aside",
        "indirect statement",
        "attributive",
        "circumstantial",
    ] = Field(
        description=(
            "For a finite verb: 'independent' (main/principal), 'dependent' "
            "(subordinate/secondary, introduced by a subordinating "
            "conjunction or relative/interrogative pronoun), 'direct quote' "
            "(occurring in directly quoted speech), or 'aside' (interrupts "
            "the surrounding syntax). For an infinitive, or a participle "
            "after a verb of perception/thinking, anchoring an indirect "
            "statement: 'indirect statement'. For an attributive "
            "participle: 'attributive'. For a circumstantial participle "
            "(including a genitive absolute): 'circumstantial'."
        )
    )
    semantic_type: Literal[
        "transitive active", "transitive passive", "intransitive", "linking verb"
    ] = Field(description="The verb's semantic/voice type.")


# The relation labels documented in syntax_model.md ("Token-level table of
# dependencies"). Keep relationship1 and relationship2 restricted to the
# same set of labels.
#
# Most tokens use only relation1/relationship1. relation2/relationship2 is
# an overflow slot used when a token has a second, independent relation
# that relation1 can't also express -- the clearest case is a relative
# pronoun, which relates to its antecedent (relation1 = antecedent's id,
# relationship1 = "relative pronoun") AND to its own function inside the
# relative clause (relation2 = the id of the token it serves inside that
# clause -- e.g. the verb it's the object of -- relationship2 = that
# ordinary relation, e.g. "direct object"). syntax_model.md's own worked
# example: in "ὁ τῆς πόλεως νόμος, ὃν σὺ περὶ ἐλάττονος τῶν ἡδονῶν
# ἐποιήσω", the relative pronoun ὃν has relation1 -> "νόμος" (its
# antecedent) with relationship1 "relative pronoun", and relation2 ->
# "ἐποιήσω" with relationship2 "direct object".
#
# "root" is a reserved sentinel value for relation1 (never relationship1,
# and never an actual token id): the relation1 of an independent verb is
# literally the string "root", with relationship1 "unit verb".
#
# A short map from syntax_model.md section to label, for anyone extending
# this scheme later:
#   - verb of independent clause -> "unit verb" (relation1 = "root")
#   - verb of dependent clause -> "unit verb" (relation1 = subordinating
#     word or relative/interrogative pronoun)
#   - verb in direct quote -> "direct quote"
#   - verb in an aside -> "aside"
#   - infinitive/participle anchoring indirect statement -> "indirect
#     statement"
#   - participle of a compound εἰμί form -> "auxiliary" (relation1 = the
#     conjugated form of εἰμί)
#   - agent of a passive verb (ὑπό + genitive) -> the preposition ὑπό
#     itself takes "agent" (relation1 = the passive verb); the noun/pronoun
#     governed by ὑπό takes the ordinary "object of preposition" relation
#     to ὑπό
#   - circumstantial participle -> "circumstantial participle" (relation1 =
#     the noun/pronoun it agrees with); if that noun/pronoun is a genitive
#     otherwise unconnected to the rest of the sentence (a genitive
#     absolute), the noun/pronoun in turn takes "genitive absolute"
#     (relation1 = the governing verb)
#   - attributive participle -> "attributive participle" (relation1 = the
#     noun/pronoun it agrees with)
#   - sentence-level connecting word (coordinating conjunction/particle
#     joining this sentence to the previous one) -> "sentence connector"
#     (relation1 = the verb of this sentence); μέν/δέ get this label too
#     when the items they list are split across distinct sentences rather
#     than joined within one
#   - other connecting words joining a pair or series of nouns, adjectives,
#     adverbs, or whole clauses WITHIN a sentence -> "connecting word", in
#     one of three shapes: a single connecting word joining a pair
#     (relation1 = the first item, relation2 = the second item, both
#     relation2/relationship2 set on that one token); a paired correlative
#     like τε...καί or καὶ...καί, or μέν...δέ within one sentence (each
#     connector's relation1 = its OWN adjacent item, relation2 = the OTHER
#     connector's id); or a 3+-member series like οὔτε...οὔτε...οὔτε (each
#     connector's relation1 = its own adjacent item again, but relation2
#     chains the connectors themselves together -- the first one forward
#     to the second, every later one backward to the one before it)
#   - subordinating conjunction -> "subordinating conjunction" (relation1 =
#     the verb of its governing/superior clause)
#   - relative pronoun -> "relative pronoun" (relation1 = its antecedent;
#     relation2/relationship2 = its function inside the relative clause,
#     see above)
#   - subject / direct object / predicate -> "subject" / "direct object" /
#     "predicate" (relation1 = the verb; for a compound εἰμί form, the id
#     of the conjugated εἰμί)
#   - complementary infinitive (e.g. with βούλομαι, δεῖ, ἐθέλω) ->
#     "complementary infinitive" (relation1 = the governing verb); NOT
#     itself a separate verbal expression, unlike an indirect-statement
#     infinitive
#   - article -> "article" (relation1 = the noun it substantivizes, or --
#     for a repeated article introducing an attributive adjective or
#     participle -- the id of that adjective/participle)
#   - attributive adjective/participle-in-attributive-position/prepositional
#     phrase modifying a noun -> "attributive" (relation1 = the noun)
#   - demonstrative pronoun modifying a noun -> "demonstrative" (relation1 =
#     the noun; unlike an ordinary adjective, a demonstrative is NOT in
#     attributive position)
#   - substantive use of a pronoun/adjective -> whatever ordinary noun
#     relation fits its syntactic role (e.g. "subject"), same as any noun
#   - adverb -> "adverbial" (relation1 = the verb it modifies, or the noun
#     it modifies when in attributive position)
#   - prepositional phrase adverbial to a verb -> "adverbial" (relation1 =
#     the verb); prepositional phrase attributive to a noun -> "attributive"
#     (relation1 = the noun); either way the preposition's own object takes
#     "object of preposition" (relation1 = the preposition)
#   - genitive / dative / accusative / vocative relations not otherwise
#     covered above (i.e. not direct object, subject, predicate, or object
#     of a preposition) -> "genitive" / "dative" / "accusative" /
#     "vocative" (relation1 = the verb or noun it depends on; "vocative" is
#     always linked to a verb)
#   - apposition -> "apposition" (relation1 = the first/head noun)
#   - modal particle ἄν -> "modal particle" (relation1 = the verb of its
#     own verbal unit)
#   - exclamatory words -> "exclamation" (relation1 = the verb of their
#     verbal unit -- EXCEPT the exclamatory particle ὦ introducing a
#     vocative, which instead takes the vocative noun/pronoun itself as
#     relation1, not the verb)
RelationLabel = Literal[
    "unit verb",
    "direct quote",
    "aside",
    "indirect statement",
    "auxiliary",
    "agent",
    "object of preposition",
    "circumstantial participle",
    "genitive absolute",
    "attributive participle",
    "sentence connector",
    "connecting word",
    "subordinating conjunction",
    "relative pronoun",
    "subject",
    "direct object",
    "predicate",
    "complementary infinitive",
    "article",
    "attributive",
    "demonstrative",
    "adverbial",
    "genitive",
    "dative",
    "accusative",
    "vocative",
    "apposition",
    "modal particle",
    "exclamation",
]


class TokenAnalysis(BaseModel):
    """One entry per token in the dependency graph (syntax_model.md,
    'Token-level table of dependencies'). Not every token will have a
    relation -- leave the relatedtoken*/relationship* fields unset when none
    of the documented relations apply (e.g. an independent verb has no
    relation2, and a punctuation token typically has none at all).

    Most entries correspond 1:1 to an entry in the input `tokens` list. The
    exceptions are the two IMPLIED_TOKENTYPES values below: syntax_model.md's
    'understood or implied verbal expressions' section documents two
    DIFFERENT situations where a verbal expression exists grammatically but
    has no surface realization at all in the passage, and this codebase
    distinguishes them with two distinct tokentype values rather than one
    generic 'implied':

    - 'implied eimi': an elided form of εἰμί ("to be") in a predicate
      expression. Example: ταύτην τὴν ὕβριν ἅπαντες ἄνθρωποι δεινοτάτην
      ἡγοῦνται has an implied infinitive of εἰμί governing the predicate
      ταύτην τὴν ὕβριν ... δεινοτάτην in indirect statement after
      ἡγοῦνται.
    - 'implied repetition': a verb elided from a later verbal expression in
      a coordinated series because it repeats the verb of an earlier one.
      Example: ἐγὼ μὲν ἄνω διῃτώμην, αἱ δὲ γυναῖκες κάτω elides a second
      διῃτώμην after γυναῖκες.

    For either, add a NEW entry here -- with a NEW id, not present in
    `tokens` -- rather than skipping the construction entirely; see
    greek_syntax_dspy.SyntaxAnalysis's docstring for the full rules and the
    id-naming convention."""

    id: str = Field(
        description=(
            "For an ordinary entry, must match the id of the corresponding "
            "entry in the input `tokens` list. For an implied token "
            "(tokentype in IMPLIED_TOKENTYPES -- 'implied eimi' or "
            "'implied repetition'), a NEW id not used by any entry in "
            "`tokens` or elsewhere in this tokengraph -- see "
            "SyntaxAnalysis's docstring for the naming convention."
        )
    )
    token: Optional[str] = Field(
        default=None,
        description=(
            "The token's surface text; should match the `text` of the input "
            "token with this id. Leave as None ONLY for an implied token "
            "(tokentype 'implied eimi' or 'implied repetition') -- one with "
            "no surface realization in the passage at all; every other "
            "tokentype must have real text."
        ),
    )
    tokentype: Literal[
        "lexical", "enclitic", "punctuation", "numeral",
        "implied eimi", "implied repetition",
    ] = Field(
        description=(
            "'numeral' is a number written NUMERICALLY (e.g. in Milesian "
            "notation) rather than spelled out as a word; a number spelled "
            "out as an ordinary word (e.g. δύω for 'two') is 'lexical' "
            "instead, even though it's semantically a number -- e.g. in "
            "'Ἀτρεΐδα δὲ μάλιστα δύω', δύω is 'lexical', not 'numeral'. "
            "'enclitic' tokenization must consider context -- syntax_model.md's "
            "tokenization section documents this. "
            "'implied eimi' and 'implied repetition' each mark a token with "
            "NO surface realization at all (see this model's own docstring "
            "for the distinction) -- the only two tokentypes whose `token` "
            "field is None and whose `id` is not one of the input `tokens`' "
            "own ids."
        )
    )

    lemma: Optional[str] = Field(default=None, description="Dictionary headword, for lexical tokens. Omit for punctuation.")
    verbalunitid: Optional[str] = Field(
        default=None,
        description="If this token anchors a verbal expression in `verbalunits`, repeat its own id here; otherwise omit.",
    )

    relatedtoken1: Optional[str] = Field(
        default=None,
        description=(
            "Token id this token relates to (primary relation). For an "
            "INDEPENDENT verb's own 'unit verb' relation, use the special "
            "sentinel string 'root' instead of a token id -- 'root' is "
            "reserved and must never be assigned as an actual token's id."
        ),
    )
    relationship1: Optional[RelationLabel] = Field(default=None, description="The primary relation type, if any.")

    relatedtoken2: Optional[str] = Field(default=None, description="Token id this token relates to (secondary relation, used when relation1 is already occupied -- e.g. a relative pronoun's function inside its own clause).")
    relationship2: Optional[RelationLabel] = Field(default=None, description="The secondary relation type, if any.")


# The two tokentype values (see TokenAnalysis's own docstring) that mark a
# token with no surface realization at all -- an elided form of εἰμί, or an
# elided repeated verb. Every other module that needs to ask "is this an
# implied token" (validate(), rendering.py, serialization.py, conftest.py's
# tokens_from_canned_answer()) checks membership in this set rather than
# hardcoding either string itself, so adding a third implied-token category
# later only needs a change here.
IMPLIED_TOKENTYPES = frozenset({"implied eimi", "implied repetition"})
