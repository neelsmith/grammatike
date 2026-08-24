"""
Gold-annotated example sentences for grammatike's test suite. Greek
analogue of arsgrammatica's fixtures/gold_examples.py.

Each GoldExample pairs a Greek passage with a hand-written,
syntax_model.md-correct `canned_answer` -- the same dict shape
dspy.utils.dummies.DummyLM expects, and the same shape a dspy.Example's
outputs will eventually take if these feed a GEPA trainset later. `tags`
names the relation(s)/construction the example is meant to exercise;
test_coverage.py checks that every RelationLabel/tokentype/syntactic_type/
semantic_type in models.py has at least one tagged example.

Wherever possible, a fixture below transcribes one of syntax_model.md's own
worked examples verbatim (see each fixture's own comment for the exact
line(s) it comes from), rather than inventing a new sentence -- this keeps
the fixtures verifiably correct against the spec. A few fixtures ARE
necessarily constructed (no relation label exists for the construction
without a Greek illustration in syntax_model.md itself, or the doc's own
worked example turns out to create a graph-resolution problem for this
codebase's verbal_units.py -- see relative_pronoun_ho_aner_hon_eidon
below): each such fixture says so explicitly in its own comment, together
with the philological judgment call made to fill the gap.

Add new examples here, not in the test files -- test_gold_examples.py,
test_coverage.py, and test_validate.py all read GOLD_EXAMPLES rather than
defining their own fixtures.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GoldExample:
    slug: str
    passage: str
    tags: list[str]
    canned_answer: dict[str, Any]


# ---------------------------------------------------------------------------
# "τὴν θύραν ἀνέῳξεν."
#   t0 τήν  t1 θύραν  t2 ἀνέῳξεν  t3 .
#
# syntax_model.md's own worked example for the "root" sentinel/"unit verb"
# relation ("in τὴν θύραν ἀνέῳξεν, ἀνέῳξεν is an independent verb with
# relation1 value root, and relationship1 value unit verb") and for the
# *lexical* and *punctuation* tokentypes (Tokenization section: "the tokens
# τὴν, θύραν and ἀνέῳξεν"; "'.' in the sentence τὴν θύραν ἀνέῳξεν.").
# ---------------------------------------------------------------------------

_UNIT_VERB_TEN_THURAN_ANSWER = {
    "reasoning": (
        "ἀνέῳξεν is the independent main verb ('he opened', transitive "
        "active), with the sentinel relatedtoken1 'root'; θύραν is its "
        "direct object; τήν is the article accompanying θύραν."
    ),
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "τὴν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t1", "relationship1": "article"},
        {"id": "t1", "token": "θύραν", "tokentype": "lexical", "lemma": "θύρα",
         "relatedtoken1": "t2", "relationship1": "direct object"},
        {"id": "t2", "token": "ἀνέῳξεν", "tokentype": "lexical", "lemma": "ἀνοίγνυμι",
         "verbalunitid": "t2", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη."
#   t0 ἐπειδή  t1 δέ  t2 ἦν  t3 πρός  t4 ἡμέραν  t5 ,  t6 ἧκεν  t7 ἐκείνη  t8 .
#
# syntax_model.md's own worked example for *independent* vs *dependent*
# syntactic_type ("ἧκεν is classified as an independent verbal expression,
# and ἦν is classified as dependent, introduced by the subordinating
# conjunction ἐπειδή") and for the *subordinating conjunction* relation
# ("ἐπειδή has the id of ἧκεν as relation1, with subordinating conjunction
# as the value of relationship1"). δέ is postpositive, second word of the
# whole sentence -- treated as a *sentence connector* pointing at the
# sentence's own main verb ἧκεν, by direct analogy with syntax_model.md's
# own γάρ example in the same position (ταύτην γὰρ ἐμαυτῷ μόνην ἡγοῦμαι
# σωτηρίαν) -- not itself drawn from this sentence's own worked commentary.
# πρὸς ἡμέραν ("toward daybreak") is left as an ordinary adverbial
# prepositional phrase modifying ἦν.
# ---------------------------------------------------------------------------

_DEPENDENT_VERB_EPEIDE_DE_EN_ANSWER = {
    "reasoning": (
        "ἧκεν is the independent main verb (root, intransitive, 'she "
        "arrived'), with ἐκείνη as its subject. ἦν is the dependent verb "
        "of the ἐπειδή-clause (intransitive, existential 'it was'), "
        "linked to ἐπειδή as its unit verb; ἐπειδή in turn relates to "
        "ἧκεν as subordinating conjunction. δέ, second word of the "
        "sentence, is a sentence connector pointing at ἧκεν. πρός ἡμέραν "
        "is an adverbial prepositional phrase modifying ἦν."
    ),
    "verbalunits": [
        {"id": "t2", "syntactic_type": "dependent", "semantic_type": "intransitive"},
        {"id": "t6", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐπειδὴ", "tokentype": "lexical", "lemma": "ἐπειδή",
         "relatedtoken1": "t6", "relationship1": "subordinating conjunction"},
        {"id": "t1", "token": "δὲ", "tokentype": "lexical", "lemma": "δέ",
         "relatedtoken1": "t6", "relationship1": "sentence connector"},
        {"id": "t2", "token": "ἦν", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t2", "relatedtoken1": "t0", "relationship1": "unit verb"},
        {"id": "t3", "token": "πρὸς", "tokentype": "lexical", "lemma": "πρός",
         "relatedtoken1": "t2", "relationship1": "adverbial"},
        {"id": "t4", "token": "ἡμέραν", "tokentype": "lexical", "lemma": "ἡμέρα",
         "relatedtoken1": "t3", "relationship1": "object of preposition"},
        {"id": "t5", "token": ",", "tokentype": "punctuation"},
        {"id": "t6", "token": "ἧκεν", "tokentype": "lexical", "lemma": "ἥκω",
         "verbalunitid": "t6", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t7", "token": "ἐκείνη", "tokentype": "lexical", "lemma": "ἐκεῖνος",
         "relatedtoken1": "t6", "relationship1": "subject"},
        {"id": "t8", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# '"ἵνα σύ γε" ἔφη "πειρᾷς ἐνταῦθα τὴν παιδίσκην".'
#   t0 "  t1 ἵνα  t2 σύ  t3 γε  t4 "  t5 ἔφη  t6 "  t7 πειρᾷς  t8 ἐνταῦθα
#   t9 τήν  t10 παιδίσκην  t11 "  t12 .
#
# syntax_model.md's own worked example for the *direct quote* syntactic
# type: "the token ἔφη is a verbal unit classified syntactically as an
# `independent` clause, while the verb πειρᾷς occurs in directly quoted
# speech and is classifed as `direct quote`." A terminal period is added
# (the doc's own excerpt is a mid-passage fragment with no sentence-final
# punctuation) to make this a complete, analyzable sentence; quotation
# marks are rendered as plain straight double quotes, matching how
# arsgrammatica's own analogous direct-quote fixture handles unmarked
# source punctuation. γε is a genuine, real, space-separated word in this
# text (not fused onto σύ the way an enclitic like περ can be -- see
# enclitic_eiper_houtos_echei below) and is given tokentype "lexical"
# rather than "enclitic" so that tokengraph_to_text()'s round-trip spacing
# (which -- correctly, per rendering.py's own docstring -- always renders
# tokentype "enclitic" with NO leading space, matching only the
# fused-in-the-source case) doesn't collapse "σύ γε" into "σύγε". ἵνα here
# is the colloquial discourse use ("so", "then"), not the purpose
# conjunction, and syntax_model.md gives no relation for it, so it (like
# γε) is left unrelated.
# ---------------------------------------------------------------------------

_DIRECT_QUOTE_HINA_SU_GE_ANSWER = {
    "reasoning": (
        "ἔφη is the independent framing verb (root, intransitive -- a "
        "verb of saying introducing direct speech). πειρᾷς anchors a "
        "'direct quote' verbal expression (transitive active), relating "
        "back to ἔφη via 'direct quote'; σύ is its subject, ἐνταῦθα "
        "adverbial, and τήν παιδίσκην its direct object (with τήν as "
        "παιδίσκην's article). ἵνα (colloquial 'so, then') and γε "
        "(an intensifying particle here) are both left unrelated -- "
        "syntax_model.md gives no relation for either."
    ),
    "verbalunits": [
        {"id": "t5", "syntactic_type": "independent", "semantic_type": "intransitive"},
        {"id": "t7", "syntactic_type": "direct quote", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "\"", "tokentype": "punctuation"},
        {"id": "t1", "token": "ἵνα", "tokentype": "lexical", "lemma": "ἵνα"},
        {"id": "t2", "token": "σύ", "tokentype": "lexical", "lemma": "σύ",
         "relatedtoken1": "t7", "relationship1": "subject"},
        {"id": "t3", "token": "γε", "tokentype": "lexical", "lemma": "γε"},
        {"id": "t4", "token": "\"", "tokentype": "punctuation"},
        {"id": "t5", "token": "ἔφη", "tokentype": "lexical", "lemma": "φημί",
         "verbalunitid": "t5", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t6", "token": "\"", "tokentype": "punctuation"},
        {"id": "t7", "token": "πειρᾷς", "tokentype": "lexical", "lemma": "πειράω",
         "verbalunitid": "t7", "relatedtoken1": "t5", "relationship1": "direct quote"},
        {"id": "t8", "token": "ἐνταῦθα", "tokentype": "lexical", "lemma": "ἐνταῦθα",
         "relatedtoken1": "t7", "relationship1": "adverbial"},
        {"id": "t9", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t10", "relationship1": "article"},
        {"id": "t10", "token": "παιδίσκην", "tokentype": "lexical", "lemma": "παιδίσκη",
         "relatedtoken1": "t7", "relationship1": "direct object"},
        {"id": "t11", "token": "\"", "tokentype": "punctuation"},
        {"id": "t12", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "πρῶτον μὲν οὖν, ὦ ἄνδρες, (δεῖ γὰρ καὶ ταῦθ' ὑμῖν διηγήσασθαι) οἰκίδιον
# ἔστι μοι διπλοῦν."
#   t0 πρῶτον  t1 μέν  t2 οὖν  t3 ,  t4 ὦ  t5 ἄνδρες  t6 ,  t7 (  t8 δεῖ
#   t9 γάρ  t10 καί  t11 ταῦθ'  t12 ὑμῖν  t13 διηγήσασθαι  t14 )  t15 οἰκίδιον
#   t16 ἔστι  t17 μοι  t18 διπλοῦν  t19 .
#
# syntax_model.md's own worked example for the *aside* syntactic type and
# relation ("the verb ἔστι is classified as independent, and the phrase
# δεῖ γὰρ καὶ ταῦθ' ὑμῖν διηγήσασθαι is an aside anchored by the finite
# verbal expression δεῖ of type aside") AND for the *vocative* relation --
# reusing the same sentence rather than syntax_model.md's separate, purely
# illustrative vocative example, since ἄνδρες/ὦ appear in this sentence
# too and a relation is needed here regardless. διηγήσασθαι is δεῖ's
# *complementary infinitive*, per the general rule for δεῖ/βούλομαι/
# ἐθέλω-type verbs, NOT its own verbal expression. γάρ is treated as a
# sentence connector for the aside's own little clause (by the same logic
# as δέ in dependent_verb_epeide_de_en above); καί modifies the single
# word διηγήσασθαι ("also/even this") rather than joining two items, so
# per greek_syntax_dspy.py's own extrapolation it is adverbial, not a
# connecting word. μέν/οὖν are left largely unrelated as bare discourse
# particles (μέν given a sentence-connector reading to ἔστι, the
# sentence's main verb, οὖν left unrelated) -- syntax_model.md does not
# work through this sentence's opening particles token by token.
# ---------------------------------------------------------------------------

_ASIDE_PROTON_MEN_OUN_DEI_ANSWER = {
    "reasoning": (
        "ἔστι is the independent main verb (root, intransitive, "
        "existential 'there is'), with οἰκίδιον as its subject, μοι "
        "dative, διπλοῦν attributive to οἰκίδιον, πρῶτον adverbial, ἄνδρες "
        "vocative, and μέν a sentence connector. δεῖ anchors an 'aside' "
        "verbal expression (intransitive, impersonal 'it is necessary'), "
        "relating back to ἔστι via 'aside'; γάρ is its own sentence "
        "connector; καί is adverbial, emphasizing the single word "
        "διηγήσασθαι; ταῦθ' is διηγήσασθαι's direct object and ὑμῖν its "
        "dative; διηγήσασθαι is δεῖ's complementary infinitive, not its "
        "own verbal expression. οὖν is left unrelated."
    ),
    "verbalunits": [
        {"id": "t8", "syntactic_type": "aside", "semantic_type": "intransitive"},
        {"id": "t16", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "πρῶτον", "tokentype": "lexical", "lemma": "πρῶτος",
         "relatedtoken1": "t16", "relationship1": "adverbial"},
        {"id": "t1", "token": "μέν", "tokentype": "lexical", "lemma": "μέν",
         "relatedtoken1": "t16", "relationship1": "sentence connector"},
        {"id": "t2", "token": "οὖν", "tokentype": "lexical", "lemma": "οὖν"},
        {"id": "t3", "token": ",", "tokentype": "punctuation"},
        {"id": "t4", "token": "ὦ", "tokentype": "lexical", "lemma": "ὦ"},
        {"id": "t5", "token": "ἄνδρες", "tokentype": "lexical", "lemma": "ἀνήρ",
         "relatedtoken1": "t16", "relationship1": "vocative"},
        {"id": "t6", "token": ",", "tokentype": "punctuation"},
        {"id": "t7", "token": "(", "tokentype": "punctuation"},
        {"id": "t8", "token": "δεῖ", "tokentype": "lexical", "lemma": "δεῖ",
         "verbalunitid": "t8", "relatedtoken1": "t16", "relationship1": "aside"},
        {"id": "t9", "token": "γάρ", "tokentype": "lexical", "lemma": "γάρ",
         "relatedtoken1": "t8", "relationship1": "sentence connector"},
        {"id": "t10", "token": "καί", "tokentype": "lexical", "lemma": "καί",
         "relatedtoken1": "t13", "relationship1": "adverbial"},
        {"id": "t11", "token": "ταῦθ'", "tokentype": "lexical", "lemma": "οὗτος",
         "relatedtoken1": "t13", "relationship1": "direct object"},
        {"id": "t12", "token": "ὑμῖν", "tokentype": "lexical", "lemma": "σύ",
         "relatedtoken1": "t13", "relationship1": "dative"},
        {"id": "t13", "token": "διηγήσασθαι", "tokentype": "lexical", "lemma": "διηγέομαι",
         "relatedtoken1": "t8", "relationship1": "complementary infinitive"},
        {"id": "t14", "token": ")", "tokentype": "punctuation"},
        {"id": "t15", "token": "οἰκίδιον", "tokentype": "lexical", "lemma": "οἰκίδιον",
         "relatedtoken1": "t16", "relationship1": "subject"},
        {"id": "t16", "token": "ἔστι", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t16", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t17", "token": "μοι", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t16", "relationship1": "dative"},
        {"id": "t18", "token": "διπλοῦν", "tokentype": "lexical", "lemma": "διπλοῦς",
         "relatedtoken1": "t15", "relationship1": "attributive"},
        {"id": "t19", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἔφασκε τὸν λύχνον ἀποσβεσθῆναι."
#   t0 ἔφασκε  t1 τόν  t2 λύχνον  t3 ἀποσβεσθῆναι  t4 .
#
# syntax_model.md's own worked example for the infinitive *indirect
# statement* syntactic type and relation: "ἔφασκε is an independent verbal
# expression, and ἀποσβεσθῆναι is the verb of the indirect statement. The
# verbal unit will be anchored to the infinitive ἀποσβεσθῆναι of syntactic
# type indirect statement." λύχνον is the accusative subject of the
# infinitive, per the general subject rule extended to indirect statement.
# ἀποσβεσθῆναι (aorist passive infinitive, "to be put out") is classified
# transitive passive.
# ---------------------------------------------------------------------------

_INDIRECT_STATEMENT_INFINITIVE_EPHASKE_ANSWER = {
    "reasoning": (
        "ἔφασκε is the independent main verb (root, transitive active, "
        "'he claimed'). ἀποσβεσθῆναι anchors the indirect-statement "
        "verbal expression (transitive passive, 'that it was put out'), "
        "relating back to ἔφασκε via 'indirect statement'; λύχνον is its "
        "accusative subject, with τόν as λύχνον's article."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t3", "syntactic_type": "indirect statement", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἔφασκε", "tokentype": "lexical", "lemma": "φάσκω",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "τόν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t2", "token": "λύχνον", "tokentype": "lexical", "lemma": "λύχνος",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t3", "token": "ἀποσβεσθῆναι", "tokentype": "lexical", "lemma": "ἀποσβέννυμι",
         "verbalunitid": "t3", "relatedtoken1": "t0", "relationship1": "indirect statement"},
        {"id": "t4", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "εἶδε δὲ τὴν βασίλειαν φεύγουσαν."
#   t0 εἶδε  t1 δέ  t2 τήν  t3 βασίλειαν  t4 φεύγουσαν  t5 .
#
# syntax_model.md's own worked example for a *participle* anchoring
# indirect statement (after a verb of perception): "εἶδε is an
# independent verbal expression, and φεύγουσαν is the verb of the
# indirect statement. The verbal unit wil be anchored to the participle
# φεύγουσαν of syntactic type indirect statement." δέ (postpositive,
# second word) is a sentence connector -> εἶδε, by the same convention as
# dependent_verb_epeide_de_en above.
# ---------------------------------------------------------------------------

_INDIRECT_STATEMENT_PARTICIPLE_EIDE_ANSWER = {
    "reasoning": (
        "εἶδε is the independent main verb (root, transitive active, "
        "'he saw'). δέ is a sentence connector -> εἶδε. φεύγουσαν anchors "
        "the indirect-statement verbal expression (participle after a "
        "verb of perception, intransitive, 'fleeing'), relating back to "
        "εἶδε via 'indirect statement'; βασίλειαν is its accusative "
        "subject, with τήν as its article."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t4", "syntactic_type": "indirect statement", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "εἶδε", "tokentype": "lexical", "lemma": "ὁράω",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "δέ", "tokentype": "lexical", "lemma": "δέ",
         "relatedtoken1": "t0", "relationship1": "sentence connector"},
        {"id": "t2", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "βασίλειαν", "tokentype": "lexical", "lemma": "βασίλεια",
         "relatedtoken1": "t4", "relationship1": "subject"},
        {"id": "t4", "token": "φεύγουσαν", "tokentype": "lexical", "lemma": "φεύγω",
         "verbalunitid": "t4", "relatedtoken1": "t0", "relationship1": "indirect statement"},
        {"id": "t5", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ὁ γὰρ ἀνὴρ ὁ ὑβρίζων εἰς σὲ ἐχθρὸς ὢν ἡμῖν τυγχάνει."
#   t0 ὁ  t1 γάρ  t2 ἀνήρ  t3 ὁ  t4 ὑβρίζων  t5 εἰς  t6 σέ  t7 ἐχθρός
#   t8 ὤν  t9 ἡμῖν  t10 τυγχάνει  t11 .
#
# syntax_model.md's own worked example for the *attributive* participle
# syntactic type and *attributive participle* relation ("the repeated
# article puts the participle ὑβρίζων in attributive relation to the noun
# ἀνήρ. ὑβρίζων will have the id of ἀνήρ as relation1 with attributive
# participle for relationship1") AND, in the same sentence, for the
# *supplementary* participle that is explicitly NOT a verbal expression
# ("the participle ὤν is a supplementary participle with τυγχάνει and does
# not constitute a verbal unit"). Per greek_syntax_dspy.py's own TODO,
# ὤν's relatedtoken1/relationship1 are left unset -- no RelationLabel
# fits a supplementary participle's own relation to its governing verb --
# while it still carries its own predicate complement (ἐχθρός) exactly as
# a linking verb would. ἡμῖν ('hostile to us') is treated as a dative
# depending on the adjective ἐχθρός rather than on τυγχάνει itself.
# ---------------------------------------------------------------------------

_ATTRIBUTIVE_PARTICIPLE_HO_ANER_ANSWER = {
    "reasoning": (
        "τυγχάνει is the independent main verb (root, intransitive, "
        "'happens to be'), with ἀνήρ as its subject and γάρ as its "
        "sentence connector. ὁ (t0) is ἀνήρ's article. The repeated "
        "article ὁ (t3) puts ὑβρίζων in attributive relation to ἀνήρ: "
        "ὑβρίζων anchors its own 'attributive' verbal expression "
        "(intransitive), relating to ἀνήρ via 'attributive participle', "
        "with εἰς σέ an adverbial prepositional phrase modifying it. ὤν "
        "is a supplementary participle completing τυγχάνει's own "
        "predicate idea -- NOT its own verbal expression, and left with "
        "no relatedtoken1/relationship1 of its own (no documented label "
        "fits) -- but still takes ἐχθρός as its own predicate complement, "
        "with ἡμῖν dative depending on ἐχθρός."
    ),
    "verbalunits": [
        {"id": "t4", "syntactic_type": "attributive", "semantic_type": "intransitive"},
        {"id": "t10", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ὁ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t1", "token": "γάρ", "tokentype": "lexical", "lemma": "γάρ",
         "relatedtoken1": "t10", "relationship1": "sentence connector"},
        {"id": "t2", "token": "ἀνήρ", "tokentype": "lexical", "lemma": "ἀνήρ",
         "relatedtoken1": "t10", "relationship1": "subject"},
        {"id": "t3", "token": "ὁ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t4", "relationship1": "article"},
        {"id": "t4", "token": "ὑβρίζων", "tokentype": "lexical", "lemma": "ὑβρίζω",
         "verbalunitid": "t4", "relatedtoken1": "t2", "relationship1": "attributive participle"},
        {"id": "t5", "token": "εἰς", "tokentype": "lexical", "lemma": "εἰς",
         "relatedtoken1": "t4", "relationship1": "adverbial"},
        {"id": "t6", "token": "σέ", "tokentype": "lexical", "lemma": "σύ",
         "relatedtoken1": "t5", "relationship1": "object of preposition"},
        {"id": "t7", "token": "ἐχθρός", "tokentype": "lexical", "lemma": "ἐχθρός",
         "relatedtoken1": "t8", "relationship1": "predicate"},
        {"id": "t8", "token": "ὤν", "tokentype": "lexical", "lemma": "εἰμί"},
        {"id": "t9", "token": "ἡμῖν", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t7", "relationship1": "dative"},
        {"id": "t10", "token": "τυγχάνει", "tokentype": "lexical", "lemma": "τυγχάνω",
         "verbalunitid": "t10", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t11", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "προϊόντος δὲ τοῦ χρόνου ἧκον μὲν ἀπροσδοκήτως ἐξ ἀγροῦ."
#   t0 προϊόντος  t1 δέ  t2 τοῦ  t3 χρόνου  t4 ἧκον  t5 μέν  t6 ἀπροσδοκήτως
#   t7 ἐκ  t8 ἀγροῦ  t9 .
#
# syntax_model.md's own worked example for the *circumstantial* syntactic
# type, the *circumstantial participle* relation, AND the *genitive
# absolute* relation, all in one sentence: "the participle προϊόντος has
# the id of χρόνου as relation1 with circumstantial participle for
# relationship1. χρόνου in turn has the id of the verb ἧκον as relation1
# and has the relationship1 value genitive absolute."
# ---------------------------------------------------------------------------

_CIRCUMSTANTIAL_GENITIVE_ABSOLUTE_PROIONTOS_ANSWER = {
    "reasoning": (
        "ἧκον is the independent main verb (root, intransitive, 'they "
        "came'), with μέν also a sentence connector (postpositive, "
        "marking a transition, with no coordinate partner of its own "
        "within this one-clause sentence -- unlike a 'connecting word', "
        "which would assert an actual series), ἀπροσδοκήτως adverbial, "
        "and ἐκ ἀγροῦ an adverbial prepositional phrase. προϊόντος "
        "anchors a 'circumstantial' verbal expression (intransitive, 'as "
        "time was passing'), relating to χρόνου via 'circumstantial "
        "participle'; χρόνου (a genuine genitive absolute, otherwise "
        "unconnected to the sentence) relates instead to the main verb "
        "ἧκον via 'genitive absolute'. δέ is a sentence connector -> "
        "ἧκον; τοῦ is χρόνου's article."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "circumstantial", "semantic_type": "intransitive"},
        {"id": "t4", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "προϊόντος", "tokentype": "lexical", "lemma": "πρόειμι",
         "verbalunitid": "t0", "relatedtoken1": "t3", "relationship1": "circumstantial participle"},
        {"id": "t1", "token": "δέ", "tokentype": "lexical", "lemma": "δέ",
         "relatedtoken1": "t4", "relationship1": "sentence connector"},
        {"id": "t2", "token": "τοῦ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "χρόνου", "tokentype": "lexical", "lemma": "χρόνος",
         "relatedtoken1": "t4", "relationship1": "genitive absolute"},
        {"id": "t4", "token": "ἧκον", "tokentype": "lexical", "lemma": "ἥκω",
         "verbalunitid": "t4", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t5", "token": "μέν", "tokentype": "lexical", "lemma": "μέν",
         "relatedtoken1": "t4", "relationship1": "sentence connector"},
        {"id": "t6", "token": "ἀπροσδοκήτως", "tokentype": "lexical", "lemma": "ἀπροσδόκητος",
         "relatedtoken1": "t4", "relationship1": "adverbial"},
        {"id": "t7", "token": "ἐκ", "tokentype": "lexical", "lemma": "ἐκ",
         "relatedtoken1": "t4", "relationship1": "adverbial"},
        {"id": "t8", "token": "ἀγροῦ", "tokentype": "lexical", "lemma": "ἀγρός",
         "relatedtoken1": "t7", "relationship1": "object of preposition"},
        {"id": "t9", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐγὼ ἅπαντα ἐπιδείξω τὰ ἐμαυτοῦ πράγματα, οὐδὲν παραλείπων, ἀλλὰ λέγων
# τἀληθῆ."
#   t0 ἐγώ  t1 ἅπαντα  t2 ἐπιδείξω  t3 τά  t4 ἐμαυτοῦ  t5 πράγματα  t6 ,
#   t7 οὐδέν  t8 παραλείπων  t9 ,  t10 ἀλλά  t11 λέγων  t12 τἀληθῆ  t13 .
#
# syntax_model.md's own worked example for the OTHER circumstantial-
# participle case, where the noun the participle agrees with fits
# normally into the surrounding clause instead of forming a genitive
# absolute: "the participles παραλείπων and λέγων are both circumstantial
# participles with the id of ἐγὼ for relation1 and circumstantial
# participle as the value of relationship1. The pronoun ἐγὼ in turn is the
# subject of ἐπιδείξω and will have the id of ἐπιδείξω for relation1 with
# subject as relationship1."
# ---------------------------------------------------------------------------

_CIRCUMSTANTIAL_FITS_CLAUSE_EGO_HAPANTA_ANSWER = {
    "reasoning": (
        "ἐπιδείξω is the independent main verb (root, transitive active, "
        "'I will show'), with ἐγώ as its subject and πράγματα (τὰ ἐμαυτοῦ "
        "πράγματα, 'all my own affairs') as its direct object, ἅπαντα "
        "attributive to πράγματα. παραλείπων and λέγων are both "
        "circumstantial participles (transitive active) relating to ἐγώ "
        "via 'circumstantial participle', with οὐδέν and τἀληθῆ as their "
        "respective direct objects; ἀλλά is a connecting word joining the "
        "two participial clauses, pointing at the first (παραλείπων)."
    ),
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t8", "syntactic_type": "circumstantial", "semantic_type": "transitive active"},
        {"id": "t11", "syntactic_type": "circumstantial", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐγώ", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t2", "relationship1": "subject"},
        {"id": "t1", "token": "ἅπαντα", "tokentype": "lexical", "lemma": "ἅπας",
         "relatedtoken1": "t5", "relationship1": "attributive"},
        {"id": "t2", "token": "ἐπιδείξω", "tokentype": "lexical", "lemma": "ἐπιδείκνυμι",
         "verbalunitid": "t2", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": "τά", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t5", "relationship1": "article"},
        {"id": "t4", "token": "ἐμαυτοῦ", "tokentype": "lexical", "lemma": "ἐμαυτοῦ",
         "relatedtoken1": "t5", "relationship1": "genitive"},
        {"id": "t5", "token": "πράγματα", "tokentype": "lexical", "lemma": "πρᾶγμα",
         "relatedtoken1": "t2", "relationship1": "direct object"},
        {"id": "t6", "token": ",", "tokentype": "punctuation"},
        {"id": "t7", "token": "οὐδέν", "tokentype": "lexical", "lemma": "οὐδείς",
         "relatedtoken1": "t8", "relationship1": "direct object"},
        {"id": "t8", "token": "παραλείπων", "tokentype": "lexical", "lemma": "παραλείπω",
         "verbalunitid": "t8", "relatedtoken1": "t0", "relationship1": "circumstantial participle"},
        {"id": "t9", "token": ",", "tokentype": "punctuation"},
        {"id": "t10", "token": "ἀλλά", "tokentype": "lexical", "lemma": "ἀλλά",
         "relatedtoken1": "t8", "relationship1": "connecting word"},
        {"id": "t11", "token": "λέγων", "tokentype": "lexical", "lemma": "λέγω",
         "verbalunitid": "t11", "relatedtoken1": "t0", "relationship1": "circumstantial participle"},
        {"id": "t12", "token": "τἀληθῆ", "tokentype": "lexical", "lemma": "ἀληθής",
         "relatedtoken1": "t11", "relationship1": "direct object"},
        {"id": "t13", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ὁ νόμος γεγραμμένος ἐστίν."
#   t0 ὁ  t1 νόμος  t2 γεγραμμένος  t3 ἐστίν  t4 .
#
# syntax_model.md's own worked example for the *auxiliary* relation:
# "the conjugated form ἐστίν will be taken as the verb of the verbal
# unit. The associated participle γεγραμμένος will relate to ἐστίν as its
# auxiliary." The compound perfect-passive form is classified transitive
# passive, per the general perfect-system rule in the verbal-expressions
# section.
# ---------------------------------------------------------------------------

_AUXILIARY_HO_NOMOS_GEGRAMMENOS_ANSWER = {
    "reasoning": (
        "ἐστίν anchors the compound perfect-passive verbal expression "
        "('has been written', root, transitive passive) -- per the "
        "compound-form rule, every relation into the verb targets ἐστίν, "
        "not γεγραμμένος. νόμος is its subject, with ὁ as νόμος's "
        "article; γεγραμμένος relates to ἐστίν as its auxiliary."
    ),
    "verbalunits": [
        {"id": "t3", "syntactic_type": "independent", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ὁ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t1", "relationship1": "article"},
        {"id": "t1", "token": "νόμος", "tokentype": "lexical", "lemma": "νόμος",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t2", "token": "γεγραμμένος", "tokentype": "lexical", "lemma": "γράφω",
         "relatedtoken1": "t3", "relationship1": "auxiliary"},
        {"id": "t3", "token": "ἐστίν", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t3", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t4", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "κατηγόρει ὡς μετὰ τὴν ἐκφορὰν αὐτῇ προσίοι."
#   t0 κατηγόρει  t1 ὡς  t2 μετά  t3 τήν  t4 ἐκφοράν  t5 αὐτῇ  t6 προσίοι  t7 .
#
# syntax_model.md's own worked example for the *unit verb* (dependent)
# relation via an ordinary subordinating conjunction (as opposed to a
# relative pronoun): "the verb προσίοι is releated to the subordinating
# conjunction ὡς with the value of unit verb for relationship1."
# ---------------------------------------------------------------------------

_SUBORDINATING_CONJUNCTION_KATEGOREI_HOS_ANSWER = {
    "reasoning": (
        "κατηγόρει is the independent main verb (root, intransitive, "
        "'she accused/charged'). προσίοι is the dependent verb of the "
        "ὡς-clause (intransitive, 'was approaching'), linked to ὡς as "
        "its unit verb; ὡς in turn relates to κατηγόρει as subordinating "
        "conjunction. μετὰ τὴν ἐκφοράν ('after the funeral') is an "
        "adverbial prepositional phrase modifying προσίοι; αὐτῇ is "
        "dative, depending on προσίοι ('was approaching her')."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "intransitive"},
        {"id": "t6", "syntactic_type": "dependent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "κατηγόρει", "tokentype": "lexical", "lemma": "κατηγορέω",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "ὡς", "tokentype": "lexical", "lemma": "ὡς",
         "relatedtoken1": "t0", "relationship1": "subordinating conjunction"},
        {"id": "t2", "token": "μετά", "tokentype": "lexical", "lemma": "μετά",
         "relatedtoken1": "t6", "relationship1": "adverbial"},
        {"id": "t3", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t4", "relationship1": "article"},
        {"id": "t4", "token": "ἐκφοράν", "tokentype": "lexical", "lemma": "ἐκφορά",
         "relatedtoken1": "t2", "relationship1": "object of preposition"},
        {"id": "t5", "token": "αὐτῇ", "tokentype": "lexical", "lemma": "αὐτός",
         "relatedtoken1": "t6", "relationship1": "dative"},
        {"id": "t6", "token": "προσίοι", "tokentype": "lexical", "lemma": "πρόσειμι",
         "verbalunitid": "t6", "relatedtoken1": "t1", "relationship1": "unit verb"},
        {"id": "t7", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἡ ἐμὴ γυνὴ ὑπὸ τούτου τοῦ ἀνθρώπου διαφθείρεται."
#   t0 ἡ  t1 ἐμή  t2 γυνή  t3 ὑπό  t4 τούτου  t5 τοῦ  t6 ἀνθρώπου
#   t7 διαφθείρεται  t8 .
#
# syntax_model.md's own worked example for the *agent* relation ("ὑπὸ will
# have the ID of διαφθείρεται as relation1 and agent for relationship1.
# The noun ἀνθρώπου will be related to ὑπὸ as a normal object of
# preposition") and for *transitive passive* semantic type ("the verb
# διαφθείρεται is transitive passive").
# ---------------------------------------------------------------------------

_AGENT_HE_EME_GYNE_ANSWER = {
    "reasoning": (
        "διαφθείρεται is the independent main verb (root, transitive "
        "passive, 'is being corrupted/seduced'), with γυνή as its "
        "subject (ἡ as its article, ἐμή attributive to γυνή). ὑπό "
        "introduces the passive verb's agent, relating to διαφθείρεται "
        "via 'agent'; ἀνθρώπου is ὑπό's object of preposition (τοῦ its "
        "article, τούτου a demonstrative modifying ἀνθρώπου)."
    ),
    "verbalunits": [
        {"id": "t7", "syntactic_type": "independent", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἡ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t1", "token": "ἐμή", "tokentype": "lexical", "lemma": "ἐμός",
         "relatedtoken1": "t2", "relationship1": "attributive"},
        {"id": "t2", "token": "γυνή", "tokentype": "lexical", "lemma": "γυνή",
         "relatedtoken1": "t7", "relationship1": "subject"},
        {"id": "t3", "token": "ὑπό", "tokentype": "lexical", "lemma": "ὑπό",
         "relatedtoken1": "t7", "relationship1": "agent"},
        {"id": "t4", "token": "τούτου", "tokentype": "lexical", "lemma": "οὗτος",
         "relatedtoken1": "t6", "relationship1": "demonstrative"},
        {"id": "t5", "token": "τοῦ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t6", "relationship1": "article"},
        {"id": "t6", "token": "ἀνθρώπου", "tokentype": "lexical", "lemma": "ἄνθρωπος",
         "relatedtoken1": "t3", "relationship1": "object of preposition"},
        {"id": "t7", "token": "διαφθείρεται", "tokentype": "lexical", "lemma": "διαφθείρω",
         "verbalunitid": "t7", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t8", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἔξεστι ἑλέσθαι."
#   t0 ἔξεστι  t1 ἑλέσθαι  t2 .
#
# syntax_model.md's own worked example for the *complementary infinitive*
# relation: "the verb ἔξεστι ('it is possible') has a complementary
# infinitive ἑλέσθαι, so ἑλέσθαι will have the id of ἔξεστι for relation1
# with complementary infinitive for relationship1." A terminal period is
# added to the doc's own bare two-word example to make it a complete
# sentence.
# ---------------------------------------------------------------------------

_COMPLEMENTARY_INFINITIVE_EXESTI_ANSWER = {
    "reasoning": (
        "ἔξεστι is the independent main verb (root, intransitive, "
        "impersonal 'it is possible'). ἑλέσθαι is its complementary "
        "infinitive, relating to ἔξεστι via 'complementary infinitive' -- "
        "NOT its own verbal expression."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἔξεστι", "tokentype": "lexical", "lemma": "ἔξεστι",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "ἑλέσθαι", "tokentype": "lexical", "lemma": "αἱρέω",
         "relatedtoken1": "t0", "relationship1": "complementary infinitive"},
        {"id": "t2", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐμοίχευεν Ἐρατοσθένης τὴν γυναῖκα τὴν ἐμήν."
#   t0 ἐμοίχευεν  t1 Ἐρατοσθένης  t2 τήν  t3 γυναῖκα  t4 τήν  t5 ἐμήν  t6 .
#
# syntax_model.md's own worked example for both the *subject* and *direct
# object* relations in one sentence: "the subject Ἐρατοσθένης will have as
# relation1 the verb ἐμοίχευεν with subject as its value of relationship1"
# and "the noun γυναῖκα is the direct object. It will have the ID of
# ἐμοίχευεν for relation1, and direct object for relationship1" -- plus
# (from the "articles and adjectives" section's own worked repeated-
# article example, reused here for its literal 'τὴν ἐμήν' phrase) the
# repeated-article *article*/*attributive* pattern for ἐμήν.
# ---------------------------------------------------------------------------

_SUBJECT_DIRECT_OBJECT_EMOICHEUEN_ANSWER = {
    "reasoning": (
        "ἐμοίχευεν is the independent main verb (root, transitive "
        "active, 'was seducing'). Ἐρατοσθένης is its subject; γυναῖκα is "
        "its direct object, with τήν (t2) as γυναῖκα's article and the "
        "repeated article τήν (t4) introducing the attributive adjective "
        "ἐμήν, which itself relates to γυναῖκα via 'attributive'."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐμοίχευεν", "tokentype": "lexical", "lemma": "μοιχεύω",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "Ἐρατοσθένης", "tokentype": "lexical", "lemma": "Ἐρατοσθένης",
         "relatedtoken1": "t0", "relationship1": "subject"},
        {"id": "t2", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "γυναῖκα", "tokentype": "lexical", "lemma": "γυνή",
         "relatedtoken1": "t0", "relationship1": "direct object"},
        {"id": "t4", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t5", "relationship1": "article"},
        {"id": "t5", "token": "ἐμήν", "tokentype": "lexical", "lemma": "ἐμός",
         "relatedtoken1": "t3", "relationship1": "attributive"},
        {"id": "t6", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ᾤμην τὴν ἐμαυτοῦ γυναῖκα πασῶν σωφρονεστάτην εἶναι τῶν ἐν τῇ πόλει."
#   t0 ᾤμην  t1 τήν  t2 ἐμαυτοῦ  t3 γυναῖκα  t4 πασῶν  t5 σωφρονεστάτην
#   t6 εἶναι  t7 τῶν  t8 ἐν  t9 τῇ  t10 πόλει  t11 .
#
# syntax_model.md's own worked example for the *predicate* relation and
# (from the verbal-expressions section's own note "a linking verb") for
# *linking verb* semantic type in one sentence: "we have two verbal
# expressions, an independent expression anchored on ᾤμην, and a dependent
# infinitive in indirect speech, εἶναι, a linking verb... σωφρονεστάτην
# will have the id of εἶναι as its relation1, and predicate as
# relationship1." τῶν ἐν τῇ πόλει ("those in the city") is a substantized
# article further specifying the genitive of comparison πασῶν; this
# sub-structure is not itself spelled out token-by-token in
# syntax_model.md, so its relations are filled in here by direct analogy
# with the doc's own substantized-article and attributive-prepositional-
# phrase rules.
# ---------------------------------------------------------------------------

_PREDICATE_LINKING_OMEN_ANSWER = {
    "reasoning": (
        "ᾤμην is the independent main verb (root, transitive active, "
        "'I thought'). εἶναι anchors the indirect-statement verbal "
        "expression (linking verb), relating back to ᾤμην via 'indirect "
        "statement'; γυναῖκα is its subject (τήν its article, ἐμαυτοῦ "
        "genitive, 'my own wife'), σωφρονεστάτην its predicate adjective. "
        "πασῶν is genitive of comparison, depending on σωφρονεστάτην "
        "('most prudent of all'); τῶν is a substantized article further "
        "specifying πασῶν ('of all [the women]'), with ἐν τῇ πόλει an "
        "attributive prepositional phrase modifying it ('in the city')."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t6", "syntactic_type": "indirect statement", "semantic_type": "linking verb"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ᾤμην", "tokentype": "lexical", "lemma": "οἴομαι",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t2", "token": "ἐμαυτοῦ", "tokentype": "lexical", "lemma": "ἐμαυτοῦ",
         "relatedtoken1": "t3", "relationship1": "genitive"},
        {"id": "t3", "token": "γυναῖκα", "tokentype": "lexical", "lemma": "γυνή",
         "relatedtoken1": "t6", "relationship1": "subject"},
        {"id": "t4", "token": "πασῶν", "tokentype": "lexical", "lemma": "πᾶς",
         "relatedtoken1": "t5", "relationship1": "genitive"},
        {"id": "t5", "token": "σωφρονεστάτην", "tokentype": "lexical", "lemma": "σώφρων",
         "relatedtoken1": "t6", "relationship1": "predicate"},
        {"id": "t6", "token": "εἶναι", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t6", "relatedtoken1": "t0", "relationship1": "indirect statement"},
        {"id": "t7", "token": "τῶν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t4", "relationship1": "genitive"},
        {"id": "t8", "token": "ἐν", "tokentype": "lexical", "lemma": "ἐν",
         "relatedtoken1": "t7", "relationship1": "attributive"},
        {"id": "t9", "token": "τῇ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t10", "relationship1": "article"},
        {"id": "t10", "token": "πόλει", "tokentype": "lexical", "lemma": "πόλις",
         "relatedtoken1": "t8", "relationship1": "object of preposition"},
        {"id": "t11", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἡ μάχη ἡ ἐν Μαραθῶνι ἐγένετο."
#   t0 ἡ  t1 μάχη  t2 ἡ  t3 ἐν  t4 Μαραθῶνι  t5 ἐγένετο  t6 .
#
# CONSTRUCTED example, substituting for syntax_model.md's own worked
# example here, which is still the untranslated Latin "pugna ad Cannas"
# (apparently left over when the document was adapted for Greek -- see
# greek_syntax_dspy.py's own TODO at this exact point in SyntaxAnalysis's
# docstring, which supplies this same Greek sentence: "in 'ἡ μάχη ἡ ἐν
# Μαραθῶνι', ἐν has relatedtoken1 -> μάχη, relationship1 'attributive',
# and Μαραθῶνι has relatedtoken1 -> ἐν, relationship1 'object of
# preposition'"). A finite verb (ἐγένετο, 'happened') is added to make
# this a complete, analyzable sentence.
# ---------------------------------------------------------------------------

_ATTRIBUTIVE_PREPOSITIONAL_PHRASE_HE_MACHE_ANSWER = {
    "reasoning": (
        "ἐγένετο is the independent main verb (root, intransitive, "
        "'happened'), with μάχη as its subject (ἡ its article). The "
        "prepositional phrase ἐν Μαραθῶνι is attributive to μάχη: ἐν "
        "relates to μάχη via 'attributive', with Μαραθῶνι as ἐν's object "
        "of preposition; the repeated article ἡ (t2) introduces this "
        "attributive phrase, relating to ἐν."
    ),
    "verbalunits": [
        {"id": "t5", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἡ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t1", "relationship1": "article"},
        {"id": "t1", "token": "μάχη", "tokentype": "lexical", "lemma": "μάχη",
         "relatedtoken1": "t5", "relationship1": "subject"},
        {"id": "t2", "token": "ἡ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "ἐν", "tokentype": "lexical", "lemma": "ἐν",
         "relatedtoken1": "t1", "relationship1": "attributive"},
        {"id": "t4", "token": "Μαραθῶνι", "tokentype": "lexical", "lemma": "Μαραθών",
         "relatedtoken1": "t3", "relationship1": "object of preposition"},
        {"id": "t5", "token": "ἐγένετο", "tokentype": "lexical", "lemma": "γίγνομαι",
         "verbalunitid": "t5", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t6", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ταύτην ἔλαβον τὴν δίκην."
#   t0 ταύτην  t1 ἔλαβον  t2 τήν  t3 δίκην  t4 .
#
# syntax_model.md's own worked example for the *demonstrative* relation:
# "the demonstrative pronoun ταύτην modifies the noun δίκην, so has the ID
# of δίκην for relation1 and demonstrative for relationship1."
# ---------------------------------------------------------------------------

_DEMONSTRATIVE_TAUTEN_ELABON_ANSWER = {
    "reasoning": (
        "ἔλαβον is the independent main verb (root, transitive active, "
        "'I took/exacted'), with δίκην as its direct object (τήν its "
        "article) and ταύτην a demonstrative modifying δίκην."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ταύτην", "tokentype": "lexical", "lemma": "οὗτος",
         "relatedtoken1": "t3", "relationship1": "demonstrative"},
        {"id": "t1", "token": "ἔλαβον", "tokentype": "lexical", "lemma": "λαμβάνω",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "δίκην", "tokentype": "lexical", "lemma": "δίκη",
         "relatedtoken1": "t1", "relationship1": "direct object"},
        {"id": "t4", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐκείνη μὲν ἀπηλλάγη."
#   t0 ἐκείνη  t1 μέν  t2 ἀπηλλάγη  t3 .
#
# syntax_model.md's own worked example for the substantive use of a
# pronoun: "the pronoun ἐκείνη is the subject of ἀπηλλάγη; it will have
# the id of ἀπηλλάγη as relation1 and subject for relationship1."
# ---------------------------------------------------------------------------

_SUBSTANTIVE_PRONOUN_EKEINE_MEN_ANSWER = {
    "reasoning": (
        "ἀπηλλάγη is the independent main verb (root, transitive "
        "passive, 'was released/dismissed'), with ἐκείνη (substantive "
        "use of the pronoun, standing in for a noun) as its subject; μέν "
        "is a connecting word, sentence-initial, pointing at ἀπηλλάγη."
    ),
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐκείνη", "tokentype": "lexical", "lemma": "ἐκεῖνος",
         "relatedtoken1": "t2", "relationship1": "subject"},
        {"id": "t1", "token": "μέν", "tokentype": "lexical", "lemma": "μέν",
         "relatedtoken1": "t2", "relationship1": "connecting word"},
        {"id": "t2", "token": "ἀπηλλάγη", "tokentype": "lexical", "lemma": "ἀπαλλάττω",
         "verbalunitid": "t2", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "διαρρήδην εἴρηται."
#   t0 διαρρήδην  t1 εἴρηται  t2 .
#
# syntax_model.md's own worked example for a bare *adverbial* relation
# modifying a verb: "the adverb διαρρήδην will have the id of the verb
# εἴρηται as relation1 and adverbial for relationship1."
# ---------------------------------------------------------------------------

_ADVERBIAL_BARE_DIARREDHEN_ANSWER = {
    "reasoning": (
        "εἴρηται is the independent main verb (root, transitive passive, "
        "perfect passive 'it has been said'), with διαρρήδην ('expressly, "
        "explicitly') as an adverbial modifying it."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "διαρρήδην", "tokentype": "lexical", "lemma": "διαρρήδην",
         "relatedtoken1": "t1", "relationship1": "adverbial"},
        {"id": "t1", "token": "εἴρηται", "tokentype": "lexical", "lemma": "λέγω",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "εἰσερχόμεθα ἐκ τοῦ ἐγγύτατα καπηλείου."
#   t0 εἰσερχόμεθα  t1 ἐκ  t2 τοῦ  t3 ἐγγύτατα  t4 καπηλείου  t5 .
#
# syntax_model.md's own worked example for an adverb in ATTRIBUTIVE
# position modifying a noun: "the adverb ἐγγύτατα is in attributive
# position with καπηλείου, so will take the id of καπηλείου for relation1
# and adverbial for relationship1." The doc's own full sentence ("δᾷδας
# λαβόντες ἐκ τοῦ ἐγγύτατα καπηλείου εἰσερχόμεθα") also opens with a
# circumstantial participle (λαβόντες) agreeing with the unexpressed
# subject built into εἰσερχόμεθα's own verb ending -- since there is no
# separate pronoun token for it to agree with, δᾷδας λαβόντες is trimmed
# here to keep this fixture focused on the one relation it's meant to
# exercise, per this file's own license to simplify a worked example
# when a documented worked example would otherwise require inventing an
# additional, undocumented convention.
# ---------------------------------------------------------------------------

_ADVERBIAL_ATTRIBUTIVE_EISERCHOMETHA_ANSWER = {
    "reasoning": (
        "εἰσερχόμεθα is the independent main verb (root, intransitive, "
        "'we enter'), with ἐκ τοῦ ἐγγύτατα καπηλείου an adverbial "
        "prepositional phrase modifying it: ἐκ relates to εἰσερχόμεθα via "
        "'adverbial'; καπηλείου is ἐκ's object of preposition, with τοῦ "
        "its article and ἐγγύτατα ('nearest') an adverb in attributive "
        "position modifying it."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "εἰσερχόμεθα", "tokentype": "lexical", "lemma": "εἰσέρχομαι",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "ἐκ", "tokentype": "lexical", "lemma": "ἐκ",
         "relatedtoken1": "t0", "relationship1": "adverbial"},
        {"id": "t2", "token": "τοῦ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t4", "relationship1": "article"},
        {"id": "t3", "token": "ἐγγύτατα", "tokentype": "lexical", "lemma": "ἐγγύς",
         "relatedtoken1": "t4", "relationship1": "adverbial"},
        {"id": "t4", "token": "καπηλείου", "tokentype": "lexical", "lemma": "καπηλεῖον",
         "relatedtoken1": "t1", "relationship1": "object of preposition"},
        {"id": "t5", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ᾤχετο εἰς τὸ ἱερὸν μετὰ τῆς μητρὸς τῆς ἐκείνου."
#   t0 ᾤχετο  t1 εἰς  t2 τό  t3 ἱερόν  t4 μετά  t5 τῆς  t6 μητρός  t7 τῆς
#   t8 ἐκείνου  t9 .
#
# syntax_model.md's own worked example for the *genitive* relation: "the
# pronoun ἐκείνου is in attributive position modifying μητρὸς, so ἐκείνου
# will have the id of μητρὸς for relation1 with genitive as the value of
# relationship1."
# ---------------------------------------------------------------------------

_GENITIVE_OICHETO_ANSWER = {
    "reasoning": (
        "ᾤχετο is the independent main verb (root, intransitive, 'he "
        "went'), with εἰς τὸ ἱερόν ('to the temple') and μετὰ τῆς μητρός "
        "('with his mother') both adverbial prepositional phrases "
        "modifying it. ἐκείνου is genitive, depending on μητρός ('his "
        "mother'), with τῆς (t7) as ἐκείνου's article."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ᾤχετο", "tokentype": "lexical", "lemma": "οἴχομαι",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "εἰς", "tokentype": "lexical", "lemma": "εἰς",
         "relatedtoken1": "t0", "relationship1": "adverbial"},
        {"id": "t2", "token": "τό", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "ἱερόν", "tokentype": "lexical", "lemma": "ἱερόν",
         "relatedtoken1": "t1", "relationship1": "object of preposition"},
        {"id": "t4", "token": "μετά", "tokentype": "lexical", "lemma": "μετά",
         "relatedtoken1": "t0", "relationship1": "adverbial"},
        {"id": "t5", "token": "τῆς", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t6", "relationship1": "article"},
        {"id": "t6", "token": "μητρός", "tokentype": "lexical", "lemma": "μήτηρ",
         "relatedtoken1": "t4", "relationship1": "object of preposition"},
        {"id": "t7", "token": "τῆς", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t8", "relationship1": "article"},
        {"id": "t8", "token": "ἐκείνου", "tokentype": "lexical", "lemma": "ἐκεῖνος",
         "relatedtoken1": "t6", "relationship1": "genitive"},
        {"id": "t9", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἔχθρα ἐμοὶ καὶ ἐκείνῳ οὐδεμία ἦν."
#   t0 ἔχθρα  t1 ἐμοί  t2 καί  t3 ἐκείνῳ  t4 οὐδεμία  t5 ἦν  t6 .
#
# syntax_model.md's own worked example for the *dative* relation
# (verb-linked): "the two dative pronouns ἐμοὶ and ἐκείνῳ will both take
# the id of the verb ἦν for relation1 with dative for relationship1."
# Trimmed from the doc's own longer, two-clause sentence (which continues
# "...οὐδεμία ἦν πλὴν ταύτης, οὔτε χρημάτων ἕνεκα ἔπραξα ταῦτα") to keep
# this fixture focused on the one relation pair it's meant to exercise;
# the οὔτε...οὔτε correlation and the second clause are not transcribed.
# ---------------------------------------------------------------------------

_DATIVE_VERB_LINKED_ECHTHRA_ANSWER = {
    "reasoning": (
        "ἦν is the independent main verb (root, intransitive, "
        "existential 'there was'), with ἔχθρα as its subject (οὐδεμία "
        "attributive, 'no enmity'). ἐμοί and ἐκείνῳ are both dative, "
        "depending on ἦν; καί is a connecting word joining them, pointing "
        "at the first (ἐμοί)."
    ),
    "verbalunits": [
        {"id": "t5", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἔχθρα", "tokentype": "lexical", "lemma": "ἔχθρα",
         "relatedtoken1": "t5", "relationship1": "subject"},
        {"id": "t1", "token": "ἐμοί", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t5", "relationship1": "dative"},
        {"id": "t2", "token": "καί", "tokentype": "lexical", "lemma": "καί",
         "relatedtoken1": "t1", "relationship1": "connecting word"},
        {"id": "t3", "token": "ἐκείνῳ", "tokentype": "lexical", "lemma": "ἐκεῖνος",
         "relatedtoken1": "t5", "relationship1": "dative"},
        {"id": "t4", "token": "οὐδεμία", "tokentype": "lexical", "lemma": "οὐδείς",
         "relatedtoken1": "t0", "relationship1": "attributive"},
        {"id": "t5", "token": "ἦν", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t5", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t6", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ταῦτα πολὺν χρόνον οὕτως ἐγίγνετο."
#   t0 ταῦτα  t1 πολύν  t2 χρόνον  t3 οὕτως  t4 ἐγίγνετο  t5 .
#
# syntax_model.md's own worked example for the *accusative* relation
# (an accusative of time, linked to a verb): "the accusative χρόνον
# expressions an adverbial idea of time. It will take the id of ἐγίγνετο
# for relation1 with accusative for relationship1."
# ---------------------------------------------------------------------------

_ACCUSATIVE_OF_TIME_TAUTA_ANSWER = {
    "reasoning": (
        "ἐγίγνετο is the independent main verb (root, intransitive, "
        "'kept happening'), with ταῦτα as its subject and οὕτως "
        "adverbial. χρόνον (an accusative of extent of time, 'for a long "
        "time') relates to ἐγίγνετο via 'accusative', with πολύν "
        "attributive to χρόνον."
    ),
    "verbalunits": [
        {"id": "t4", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ταῦτα", "tokentype": "lexical", "lemma": "οὗτος",
         "relatedtoken1": "t4", "relationship1": "subject"},
        {"id": "t1", "token": "πολύν", "tokentype": "lexical", "lemma": "πολύς",
         "relatedtoken1": "t2", "relationship1": "attributive"},
        {"id": "t2", "token": "χρόνον", "tokentype": "lexical", "lemma": "χρόνος",
         "relatedtoken1": "t4", "relationship1": "accusative"},
        {"id": "t3", "token": "οὕτως", "tokentype": "lexical", "lemma": "οὕτως",
         "relatedtoken1": "t4", "relationship1": "adverbial"},
        {"id": "t4", "token": "ἐγίγνετο", "tokentype": "lexical", "lemma": "γίγνομαι",
         "verbalunitid": "t4", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t5", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "Δημοσθένης ὁ ῥήτωρ ἦλθεν."
#   t0 Δημοσθένης  t1 ὁ  t2 ῥήτωρ  t3 ἦλθεν  t4 .
#
# CONSTRUCTED example: syntax_model.md gives only the general definition
# of *apposition* with no worked Greek example of its own. Transcribed
# here verbatim from greek_syntax_dspy.py's own constructed illustration
# at this exact point in SyntaxAnalysis's docstring: "in 'Δημοσθένης ὁ
# ῥήτωρ ἦλθεν', ῥήτωρ has relatedtoken1 -> Δημοσθένης, relationship1
# 'apposition' (and ὁ has relatedtoken1 -> ῥήτωρ, relationship1
# 'article')."
# ---------------------------------------------------------------------------

_APPOSITION_DEMOSTHENES_HO_RHETOR_ANSWER = {
    "reasoning": (
        "ἦλθεν is the independent main verb (root, intransitive, 'he "
        "came'), with Δημοσθένης as its subject. ῥήτωρ ('the orator') "
        "stands in apposition to Δημοσθένης, with ὁ as ῥήτωρ's own "
        "article."
    ),
    "verbalunits": [
        {"id": "t3", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "Δημοσθένης", "tokentype": "lexical", "lemma": "Δημοσθένης",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t1", "token": "ὁ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t2", "token": "ῥήτωρ", "tokentype": "lexical", "lemma": "ῥήτωρ",
         "relatedtoken1": "t0", "relationship1": "apposition"},
        {"id": "t3", "token": "ἦλθεν", "tokentype": "lexical", "lemma": "ἔρχομαι",
         "verbalunitid": "t3", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t4", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ὁ ἀνὴρ ὃν εἶδον ἀπῆλθεν."  ("The man whom I saw departed.")
#   t0 ὁ  t1 ἀνήρ  t2 ὅν  t3 εἶδον  t4 ἀπῆλθεν  t5 .
#
# CONSTRUCTED example, replacing syntax_model.md's own worked *relative
# pronoun* sentence ("οὐκ ἐγώ σε ἀποκτενῶ, ἀλλ' ὁ τῆς πόλεως νόμος, ὃν σὺ
# περὶ ἐλάττονος τῶν ἡδονῶν ἐποιήσω") -- that sentence's antecedent, νόμος,
# has no relation of its own reaching any other verbal expression (the
# clause "ἀλλ' ὁ ... νόμος" elides its own verb -- "[will kill you]" --
# understood from ἀποκτενῶ, and syntax_model.md does not document
# recording that as an 'implied repetition' the way it does for a fully
# parallel construction like ἐγὼ μὲν ἄνω διῃτώμην). Transcribing that
# sentence's tokengraph literally would leave νόμος's antecedent link a
# dead end, which in turn makes ὃν's own outward chase fall through to its
# relatedtoken2 (ἐποιήσω, ITSELF a verbal-unit anchor) and resolve as its
# own parent -- verbal_units.compute_subordination_depths() then reports
# an unresolved self-referential cycle for ἐποιήσω, rather than genuinely
# testing the antecedent-chase the relation is meant to model. This
# replacement keeps exactly the same relation shape syntax_model.md
# documents (relatedtoken1 -> antecedent/'relative pronoun',
# relatedtoken2 -> the pronoun's own role in its clause) in a sentence
# whose antecedent has an ordinary, resolvable relation of its own.
# ---------------------------------------------------------------------------

_RELATIVE_PRONOUN_HO_ANER_HON_EIDON_ANSWER = {
    "reasoning": (
        "ἀπῆλθεν is the independent main verb (root, intransitive, 'he "
        "departed'), with ἀνήρ as its subject (ὁ its article). ὅν is the "
        "relative pronoun linking back to its antecedent ἀνήρ "
        "(relatedtoken1/relationship1) and, simultaneously, the direct "
        "object of the dependent verb εἶδον inside its own relative "
        "clause (relatedtoken2/relationship2). εἶδον is the dependent "
        "verb of the relative clause, linked to ὅν as its unit verb."
    ),
    "verbalunits": [
        {"id": "t3", "syntactic_type": "dependent", "semantic_type": "transitive active"},
        {"id": "t4", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ὁ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t1", "relationship1": "article"},
        {"id": "t1", "token": "ἀνήρ", "tokentype": "lexical", "lemma": "ἀνήρ",
         "relatedtoken1": "t4", "relationship1": "subject"},
        {"id": "t2", "token": "ὅν", "tokentype": "lexical", "lemma": "ὅς",
         "relatedtoken1": "t1", "relationship1": "relative pronoun",
         "relatedtoken2": "t3", "relationship2": "direct object"},
        {"id": "t3", "token": "εἶδον", "tokentype": "lexical", "lemma": "ὁράω",
         "verbalunitid": "t3", "relatedtoken1": "t2", "relationship1": "unit verb"},
        {"id": "t4", "token": "ἀπῆλθεν", "tokentype": "lexical", "lemma": "ἀπέρχομαι",
         "verbalunitid": "t4", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t5", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ταύτην τὴν ὕβριν ἅπαντες ἄνθρωποι δεινοτάτην ἡγοῦνται."
#   t0 ταύτην  t1 τήν  t2 ὕβριν  t3 ἅπαντες  t4 ἄνθρωποι  t5 δεινοτάτην
#   t5_implied [implied eimi]  t6 ἡγοῦνται  t7 .
#
# syntax_model.md's own worked example for the *implied eimi* tokentype
# (elided infinitive of εἰμί in indirect statement): "there is an
# independent verbal expression ἡγοῦνται, and governing a predicate
# statement in indirect discourse with an implied infinitive of 'to be'
# ταύτην τὴν ὕβριν...δεινοτάτην... the syntactic type will be `indirect
# statement` and the semantic type will be *linking verb*." Named
# t5_implied per SyntaxAnalysis's naming rule (appended to δεινοτάτην, the
# last real token before where the elided infinitive would stand).
# ---------------------------------------------------------------------------

_IMPLIED_EIMI_TAUTEN_TEN_HYBRIN_ANSWER = {
    "reasoning": (
        "ἡγοῦνται is the independent main verb (root, transitive active, "
        "'they consider'), governing an implied infinitive of εἰμί in "
        "indirect statement (t5_implied, syntactic type 'indirect "
        "statement', semantic type 'linking verb'), relating to ἡγοῦνται "
        "via 'indirect statement'. ὕβριν is the implied infinitive's "
        "subject (ταύτην a demonstrative modifying it, τήν its article); "
        "δεινοτάτην is its predicate. ἄνθρωποι is the subject of "
        "ἡγοῦνται, ἅπαντες attributive to ἄνθρωποι."
    ),
    "verbalunits": [
        {"id": "t5_implied", "syntactic_type": "indirect statement", "semantic_type": "linking verb"},
        {"id": "t6", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ταύτην", "tokentype": "lexical", "lemma": "οὗτος",
         "relatedtoken1": "t2", "relationship1": "demonstrative"},
        {"id": "t1", "token": "τήν", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t2", "token": "ὕβριν", "tokentype": "lexical", "lemma": "ὕβρις",
         "relatedtoken1": "t5_implied", "relationship1": "subject"},
        {"id": "t3", "token": "ἅπαντες", "tokentype": "lexical", "lemma": "ἅπας",
         "relatedtoken1": "t4", "relationship1": "attributive"},
        {"id": "t4", "token": "ἄνθρωποι", "tokentype": "lexical", "lemma": "ἄνθρωπος",
         "relatedtoken1": "t6", "relationship1": "subject"},
        {"id": "t5", "token": "δεινοτάτην", "tokentype": "lexical", "lemma": "δεινός",
         "relatedtoken1": "t5_implied", "relationship1": "predicate"},
        {"id": "t5_implied", "token": None, "tokentype": "implied eimi",
         "verbalunitid": "t5_implied", "relatedtoken1": "t6", "relationship1": "indirect statement"},
        {"id": "t6", "token": "ἡγοῦνται", "tokentype": "lexical", "lemma": "ἡγέομαι",
         "verbalunitid": "t6", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t7", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐγὼ μὲν ἄνω διῃτώμην, αἱ δὲ γυναῖκες κάτω."
#   t0 ἐγώ  t1 μέν  t2 ἄνω  t3 διῃτώμην  t4 ,  t5 αἱ  t6 δέ  t7 γυναῖκες
#   t8 κάτω  t8_implied [implied repetition]  t9 .
#
# syntax_model.md's own worked example for the *implied repetition*
# tokentype: "there are two verbal expressions coordinating with the
# connecting words μὲν and δὲ. The first has an explicit verb διῃτώμην
# with subject ἐγὼ, the second has a nominative subject γυναῖκες but no
# explicit verb... It will repeat the same values for semantic and
# syntactic type as the implicitly repeated verb διῃτώμην, namely
# intransitive for semantic type, and independent for syntactic type." Per
# SyntaxAnalysis's docstring ("both μέν and δέ have relatedtoken1 ->
# διῃτώμην"), both connecting words point at the FIRST clause's own verb.
# Named t8_implied per the naming rule (appended to κάτω, the last real
# token before where the elided verb would stand).
# ---------------------------------------------------------------------------

_IMPLIED_REPETITION_EGO_MEN_ANO_ANSWER = {
    "reasoning": (
        "διῃτώμην is the first, explicit independent verb (root, "
        "intransitive, 'I was living'), with ἐγώ as its subject, ἄνω "
        "adverbial, and μέν a connecting word pointing at it (starting "
        "the series). The second clause's own verb is elided, repeating "
        "διῃτώμην's own classification: t8_implied (tokentype 'implied "
        "repetition', independent, intransitive), with γυναῖκες as its "
        "subject (αἱ its article), κάτω adverbial, and δέ a connecting "
        "word pointing at διῃτώμην, the first item of the series."
    ),
    "verbalunits": [
        {"id": "t3", "syntactic_type": "independent", "semantic_type": "intransitive"},
        {"id": "t8_implied", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐγώ", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t1", "token": "μέν", "tokentype": "lexical", "lemma": "μέν",
         "relatedtoken1": "t3", "relationship1": "connecting word"},
        {"id": "t2", "token": "ἄνω", "tokentype": "lexical", "lemma": "ἄνω",
         "relatedtoken1": "t3", "relationship1": "adverbial"},
        {"id": "t3", "token": "διῃτώμην", "tokentype": "lexical", "lemma": "διαιτάω",
         "verbalunitid": "t3", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t4", "token": ",", "tokentype": "punctuation"},
        {"id": "t5", "token": "αἱ", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t7", "relationship1": "article"},
        {"id": "t6", "token": "δέ", "tokentype": "lexical", "lemma": "δέ",
         "relatedtoken1": "t3", "relationship1": "connecting word"},
        {"id": "t7", "token": "γυναῖκες", "tokentype": "lexical", "lemma": "γυνή",
         "relatedtoken1": "t8_implied", "relationship1": "subject"},
        {"id": "t8", "token": "κάτω", "tokentype": "lexical", "lemma": "κάτω",
         "relatedtoken1": "t8_implied", "relationship1": "adverbial"},
        {"id": "t8_implied", "token": None, "tokentype": "implied repetition",
         "verbalunitid": "t8_implied", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t9", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "εἶδον δύο ἄνδρας καὶ γʹ γυναῖκας."  ("I saw two men and 3 women.")
#   t0 εἶδον  t1 δύο  t2 ἄνδρας  t3 καί  t4 γʹ  t5 γυναῖκας  t6 .
#
# CONSTRUCTED example illustrating syntax_model.md's tokenization
# clarification that a number written NUMERICALLY (e.g. Milesian γʹ, "3")
# is tokentype *numeral*, while one spelled out as an ordinary word (δύο,
# "two", cf. the doc's own "Ἀτρεΐδα δὲ μάλιστα δύω" example for the
# lexical side of this same distinction) is *lexical* -- syntax_model.md
# gives no full worked sentence combining both, only the isolated
# spelled-out-vs-numeric contrast, so this sentence is constructed to
# exercise both tokentypes side by side in otherwise-identical syntactic
# positions (each modifying its own noun, same relation either way).
# ---------------------------------------------------------------------------

_NUMERAL_VS_LEXICAL_EIDON_DUO_ANSWER = {
    "reasoning": (
        "εἶδον is the independent main verb (root, transitive active, "
        "'I saw'). δύο ('two', spelled out -- tokentype lexical, per "
        "syntax_model.md's own clarification) is attributive to ἄνδρας, "
        "its direct object; γʹ ('3', written numerically in Milesian "
        "notation -- tokentype numeral) is attributive to γυναῖκας, the "
        "second direct object, the same relation δύο uses despite the "
        "different tokentype; καί is a connecting word joining the two "
        "objects, pointing at the first (ἄνδρας)."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "εἶδον", "tokentype": "lexical", "lemma": "ὁράω",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "δύο", "tokentype": "lexical", "lemma": "δύο",
         "relatedtoken1": "t2", "relationship1": "attributive"},
        {"id": "t2", "token": "ἄνδρας", "tokentype": "lexical", "lemma": "ἀνήρ",
         "relatedtoken1": "t0", "relationship1": "direct object"},
        {"id": "t3", "token": "καί", "tokentype": "lexical", "lemma": "καί",
         "relatedtoken1": "t2", "relationship1": "connecting word"},
        {"id": "t4", "token": "γʹ", "tokentype": "numeral",
         "relatedtoken1": "t5", "relationship1": "attributive"},
        {"id": "t5", "token": "γυναῖκας", "tokentype": "lexical", "lemma": "γυνή",
         "relatedtoken1": "t0", "relationship1": "direct object"},
        {"id": "t6", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐγὼ γὰρ οὐδὲν δέομαι λόγων, ἀλλὰ τὸ ἔργον φανερὸν γενέσθαι, εἴπερ οὕτως
# ἔχει."
#   t0 ἐγώ  t1 γάρ  t2 οὐδέν  t3 δέομαι  t4 λόγων  t5 ,  t6 ἀλλά  t7 τό
#   t8 ἔργον  t9 φανερόν  t10 γενέσθαι  t11 ,  t12 εἴ  t13 περ  t14 οὕτως
#   t15 ἔχει  t16 .
#
# syntax_model.md's own worked example for the *enclitic* tokentype: "the
# enclitic περ in the sentence ἐγὼ γὰρ οὐδὲν δέομαι λόγων, ἀλλὰ τὸ ἔργον
# φανερὸν γενέσθαι, εἴπερ οὕτως ἔχει" -- περ splits off of εἴπερ (εἴ +
# περ), the fused-in-the-source case rendering.py's tokengraph_to_text()
# is built to round-trip with NO space before περ (unlike a
# naturally-space-separated enclitic like γε -- see
# direct_quote_hina_su_ge above). γενέσθαι (an infinitive functioning as
# an ordinary noun -- δέομαι's second direct object, "I need... the deed
# to become manifest" -- per greek_syntax_dspy.py's own extrapolated
# "infinitive used as a noun" rule) is NOT itself a verbal expression;
# ἔργον and φανερόν relate to it exactly as a subject and predicate would
# to any linking-sense form of γίγνομαι. οὐδὲν λόγων ("no need of talk")
# is treated as an accusative-plus-genitive construction, οὐδέν itself
# accusative depending on δέομαι and λόγων a partitive genitive on οὐδέν.
# ---------------------------------------------------------------------------

_ENCLITIC_EIPER_HOUTOS_ECHEI_ANSWER = {
    "reasoning": (
        "δέομαι is the independent main verb (root, intransitive, 'I "
        "have need'), with ἐγώ as its subject, γάρ its sentence "
        "connector, and οὐδέν an accusative depending on it ('not at "
        "all'/'no [need]'), with λόγων a partitive genitive on οὐδέν "
        "('of talk'). ἀλλά is a connecting word pointing back at δέομαι, "
        "coordinating a second thing needed: γενέσθαι, an infinitive used "
        "as an ordinary noun (δέομαι's second direct object, 'the deed "
        "to become manifest'), NOT its own verbal expression -- ἔργον is "
        "its subject (τό its article), φανερόν its predicate. εἴ is the "
        "subordinating conjunction of the conditional clause, relating "
        "to δέομαι (the clause it qualifies); περ, split off of εἴπερ, "
        "is a bare enclitic left unrelated (intensifying, 'indeed'); "
        "οὕτως is adverbial; ἔχει is the dependent verb of the "
        "conditional clause (intransitive, idiomatic 'is so'), linked to "
        "εἴ as its unit verb."
    ),
    "verbalunits": [
        {"id": "t3", "syntactic_type": "independent", "semantic_type": "intransitive"},
        {"id": "t15", "syntactic_type": "dependent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐγώ", "tokentype": "lexical", "lemma": "ἐγώ",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t1", "token": "γάρ", "tokentype": "lexical", "lemma": "γάρ",
         "relatedtoken1": "t3", "relationship1": "sentence connector"},
        {"id": "t2", "token": "οὐδέν", "tokentype": "lexical", "lemma": "οὐδείς",
         "relatedtoken1": "t3", "relationship1": "accusative"},
        {"id": "t3", "token": "δέομαι", "tokentype": "lexical", "lemma": "δέομαι",
         "verbalunitid": "t3", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t4", "token": "λόγων", "tokentype": "lexical", "lemma": "λόγος",
         "relatedtoken1": "t2", "relationship1": "genitive"},
        {"id": "t5", "token": ",", "tokentype": "punctuation"},
        {"id": "t6", "token": "ἀλλά", "tokentype": "lexical", "lemma": "ἀλλά",
         "relatedtoken1": "t3", "relationship1": "connecting word"},
        {"id": "t7", "token": "τό", "tokentype": "lexical", "lemma": "ὁ",
         "relatedtoken1": "t8", "relationship1": "article"},
        {"id": "t8", "token": "ἔργον", "tokentype": "lexical", "lemma": "ἔργον",
         "relatedtoken1": "t10", "relationship1": "subject"},
        {"id": "t9", "token": "φανερόν", "tokentype": "lexical", "lemma": "φανερός",
         "relatedtoken1": "t10", "relationship1": "predicate"},
        {"id": "t10", "token": "γενέσθαι", "tokentype": "lexical", "lemma": "γίγνομαι",
         "relatedtoken1": "t3", "relationship1": "direct object"},
        {"id": "t11", "token": ",", "tokentype": "punctuation"},
        {"id": "t12", "token": "εἴ", "tokentype": "lexical", "lemma": "εἰ",
         "relatedtoken1": "t3", "relationship1": "subordinating conjunction"},
        {"id": "t13", "token": "περ", "tokentype": "enclitic", "lemma": "περ"},
        {"id": "t14", "token": "οὕτως", "tokentype": "lexical", "lemma": "οὕτως",
         "relatedtoken1": "t15", "relationship1": "adverbial"},
        {"id": "t15", "token": "ἔχει", "tokentype": "lexical", "lemma": "ἔχω",
         "verbalunitid": "t15", "relatedtoken1": "t12", "relationship1": "unit verb"},
        {"id": "t16", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "οὐκ οἶδα τίς ἦλθεν."  ("I don't know who came.")
#   t0 οὐκ  t1 οἶδα  t2 τίς  t3 ἦλθεν  t4 .
#
# CONSTRUCTED example: syntax_model.md states the rule for indirect
# questions only in passing, as part of the "unit verb (dependent)" rule's
# own parenthetical ("a subordinating conjunction or a relative or
# interrogative pronoun"), without its own worked example. Transcribed
# here verbatim from greek_syntax_dspy.py's own constructed illustration
# at this exact point in SyntaxAnalysis's docstring: "in 'οὐκ οἶδα τίς
# ἦλθεν', τίς has relatedtoken1 -> οἶδα, relationship1 'subordinating
# conjunction', and ἦλθεν has relatedtoken1 -> τίς, relationship1 'unit
# verb'."
# ---------------------------------------------------------------------------

_INDIRECT_QUESTION_OUK_OIDA_TIS_ANSWER = {
    "reasoning": (
        "οἶδα is the independent main verb (root, transitive active, "
        "'I know'), with οὐκ adverbial (negation). τίς, the "
        "interrogative pronoun introducing the indirect question, is "
        "treated like a subordinating conjunction: relatedtoken1 -> "
        "οἶδα, relationship1 'subordinating conjunction'. ἦλθεν is the "
        "dependent verb of the indirect question (intransitive, 'came'), "
        "linked to τίς as its unit verb."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t3", "syntactic_type": "dependent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "οὐκ", "tokentype": "lexical", "lemma": "οὐ",
         "relatedtoken1": "t1", "relationship1": "adverbial"},
        {"id": "t1", "token": "οἶδα", "tokentype": "lexical", "lemma": "οἶδα",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "τίς", "tokentype": "lexical", "lemma": "τίς",
         "relatedtoken1": "t1", "relationship1": "subordinating conjunction"},
        {"id": "t3", "token": "ἦλθεν", "tokentype": "lexical", "lemma": "ἔρχομαι",
         "verbalunitid": "t3", "relatedtoken1": "t2", "relationship1": "unit verb"},
        {"id": "t4", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "ἐπεὶ ᾔδει ἑαυτὸν ἡμαρτηκέναι, ἠνιάθη."
# ("Since he knew he had erred, he was grieved.")
#   t0 ἐπεί  t1 ᾔδει  t2 ἑαυτόν  t3 ἡμαρτηκέναι  t4 ,  t5 ἠνιάθη  t6 .
#
# CONSTRUCTED example, needed for a two-level subordination-depth check
# (verbal_units.compute_subordination_depths()) analogous to
# arsgrammatica's own depth_two_cum_sciret_peccavisse_doluit fixture:
# ἡμαρτηκέναι anchors an indirect statement governed by ᾔδει, itself a
# dependent verb one level below ἠνιάθη -- so ἡμαρτηκέναι sits two levels
# below the root, not one. No single syntax_model.md sentence combines a
# dependent clause with its own nested indirect statement, so this
# sentence is constructed by directly composing two already-documented
# rules (unit verb (dependent) via a subordinating conjunction, and
# indirect statement anchored to an infinitive) rather than any one
# worked example.
# ---------------------------------------------------------------------------

_DEPTH_TWO_EPEI_EDEI_HEMARTEKENAI_ANSWER = {
    "reasoning": (
        "ἠνιάθη is the independent main verb (root, intransitive, 'he "
        "was grieved'). ᾔδει is the dependent verb of the ἐπεί-clause "
        "(transitive active, 'he knew'), linked to ἐπεί as its unit "
        "verb; ἐπεί relates to ἠνιάθη as subordinating conjunction. "
        "ἡμαρτηκέναι anchors the indirect-statement verbal expression "
        "governed by ᾔδει (intransitive, 'that he had erred'), with "
        "ἑαυτόν as its accusative subject."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "dependent", "semantic_type": "transitive active"},
        {"id": "t3", "syntactic_type": "indirect statement", "semantic_type": "intransitive"},
        {"id": "t5", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "ἐπεί", "tokentype": "lexical", "lemma": "ἐπεί",
         "relatedtoken1": "t5", "relationship1": "subordinating conjunction"},
        {"id": "t1", "token": "ᾔδει", "tokentype": "lexical", "lemma": "οἶδα",
         "verbalunitid": "t1", "relatedtoken1": "t0", "relationship1": "unit verb"},
        {"id": "t2", "token": "ἑαυτόν", "tokentype": "lexical", "lemma": "ἑαυτοῦ",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t3", "token": "ἡμαρτηκέναι", "tokentype": "lexical", "lemma": "ἁμαρτάνω",
         "verbalunitid": "t3", "relatedtoken1": "t1", "relationship1": "indirect statement"},
        {"id": "t4", "token": ",", "tokentype": "punctuation"},
        {"id": "t5", "token": "ἠνιάθη", "tokentype": "lexical", "lemma": "ἀνιάω",
         "verbalunitid": "t5", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t6", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# "μεστὸς ἦ ὑποψίας."
#   t0 μεστός  t1 ἦ  t2 ὑποψίας  t3 .
#
# syntax_model.md's own worked example for *linking verb* semantic type:
# "in the sentence μεστὸς ἦ ὑποψίας, the verb ἦ is a linking verb." ὑποψίας
# is a genitive of respect/filling, depending on the predicate adjective
# μεστός, by direct analogy with the doc's own genitive rule for a noun
# modifying another noun (extended here to an adjective on the same
# footing, since the doc draws no principled line between the two for
# this relation).
# ---------------------------------------------------------------------------

_LINKING_VERB_MESTOS_E_ANSWER = {
    "reasoning": (
        "ἦ is a linking verb (root, 'I was'), joining an implicit "
        "first-person subject to the predicate adjective μεστός ('full', "
        "'suspicious'); ὑποψίας is genitive, depending on μεστός ('full "
        "of suspicion')."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "linking verb"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "μεστός", "tokentype": "lexical", "lemma": "μεστός",
         "relatedtoken1": "t1", "relationship1": "predicate"},
        {"id": "t1", "token": "ἦ", "tokentype": "lexical", "lemma": "εἰμί",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "ὑποψίας", "tokentype": "lexical", "lemma": "ὑποψία",
         "relatedtoken1": "t0", "relationship1": "genitive"},
        {"id": "t3", "token": ".", "tokentype": "punctuation"},
    ],
}


# ---------------------------------------------------------------------------
# RelationLabel coverage is complete: all 27 documented labels have at
# least one tagged example above -- unit verb (unit_verb_root_ten_thuran_
# anoixen and many others), direct quote (direct_quote_hina_su_ge_ephe),
# aside (aside_proton_men_oun_dei), indirect statement (indirect_statement_
# infinitive_ephaske_lychnon and others), auxiliary (auxiliary_ho_nomos_
# gegrammenos_estin), agent (agent_he_eme_gyne_hypo_toutou), object of
# preposition (many), circumstantial participle and genitive absolute
# (both in circumstantial_genitive_absolute_proiontos_de_tou_chronou),
# attributive participle (attributive_participle_ho_aner_ho_hybrizon),
# sentence connector (several), connecting word (several), subordinating
# conjunction (dependent_verb_epeide_de_en_hos_hekeen and
# subordinating_conjunction_dependent_kategorei_hos), relative pronoun
# (relative_pronoun_ho_aner_hon_eidon, including the relatedtoken2/
# relationship2 overflow pattern), subject/direct object/predicate (many),
# complementary infinitive (complementary_infinitive_exesti_helesthai and
# aside_proton_men_oun_dei), article/attributive (many), demonstrative
# (demonstrative_tauten_elabon_ten_diken and agent_he_eme_gyne_hypo_
# toutou), adverbial (several), genitive/dative/accusative/vocative
# (genitive_oicheto_eis_to_hieron, dative_verb_linked_echthra_emoi_kai_
# ekeino, accusative_of_time_tauta_polyn_chronon, aside_proton_men_oun_dei
# for vocative), apposition (apposition_demosthenes_ho_rhetor).
#
# VerbalExpression.syntactic_type coverage is also complete: independent,
# dependent, direct quote, aside, indirect statement, attributive
# (attributive_participle_ho_aner_ho_hybrizon), and circumstantial
# (circumstantial_genitive_absolute_proiontos_de_tou_chronou and
# circumstantial_fits_clause_ego_hapanta_epideixo) all have a tagged
# example.
#
# VerbalExpression.semantic_type coverage is also complete: transitive
# active, transitive passive, intransitive, and linking verb (predicate_
# linking_omen_ten_emautou_gynaika, implied_eimi_tauten_ten_hybrin, and
# linking_verb_mestos_e_hypopsias) all have a tagged example.
#
# TokenAnalysis.tokentype coverage is also complete: lexical, punctuation
# (both everywhere), enclitic (enclitic_eiper_houtos_echei -- the
# genuinely fused-then-split case; direct_quote_hina_su_ge_ephe's own γε
# is deliberately tokentype "lexical" instead, since it is NOT fused onto
# a preceding word in the source text -- see that fixture's own comment),
# numeral (numeral_vs_lexical_eidon_duo_andras), implied eimi
# (implied_eimi_tauten_ten_hybrin), and implied repetition
# (implied_repetition_ego_men_ano_dietomen).
# ---------------------------------------------------------------------------

GOLD_EXAMPLES = [
    GoldExample(
        slug="unit_verb_root_ten_thuran_anoixen",
        passage="τὴν θύραν ἀνέῳξεν.",
        tags=["unit verb", "article", "direct object", "independent",
              "transitive active", "lexical", "punctuation"],
        canned_answer=_UNIT_VERB_TEN_THURAN_ANSWER,
    ),
    GoldExample(
        slug="dependent_verb_epeide_de_en_hos_hekeen",
        passage="ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη.",
        tags=["subordinating conjunction", "unit verb", "sentence connector",
              "adverbial", "object of preposition", "subject", "dependent",
              "independent", "intransitive"],
        canned_answer=_DEPENDENT_VERB_EPEIDE_DE_EN_ANSWER,
    ),
    GoldExample(
        slug="direct_quote_hina_su_ge_ephe",
        passage="\"ἵνα σύ γε\" ἔφη \"πειρᾷς ἐνταῦθα τήν παιδίσκην\".",
        tags=["direct quote", "subject", "adverbial", "article",
              "direct object", "independent", "intransitive",
              "transitive active", "lexical", "punctuation"],
        canned_answer=_DIRECT_QUOTE_HINA_SU_GE_ANSWER,
    ),
    GoldExample(
        slug="aside_proton_men_oun_dei",
        passage="πρῶτον μέν οὖν, ὦ ἄνδρες, (δεῖ γάρ καί ταῦθ' ὑμῖν διηγήσασθαι) οἰκίδιον ἔστι μοι διπλοῦν.",
        tags=["aside", "vocative", "complementary infinitive",
              "sentence connector", "adverbial", "direct object", "dative",
              "subject", "attributive", "independent", "intransitive",
              "punctuation"],
        canned_answer=_ASIDE_PROTON_MEN_OUN_DEI_ANSWER,
    ),
    GoldExample(
        slug="indirect_statement_infinitive_ephaske_lychnon",
        passage="ἔφασκε τόν λύχνον ἀποσβεσθῆναι.",
        tags=["indirect statement", "subject", "article", "independent",
              "transitive active", "transitive passive"],
        canned_answer=_INDIRECT_STATEMENT_INFINITIVE_EPHASKE_ANSWER,
    ),
    GoldExample(
        slug="indirect_statement_participle_eide_de_basileian",
        passage="εἶδε δέ τήν βασίλειαν φεύγουσαν.",
        tags=["indirect statement", "sentence connector", "subject",
              "article", "independent", "transitive active", "intransitive"],
        canned_answer=_INDIRECT_STATEMENT_PARTICIPLE_EIDE_ANSWER,
    ),
    GoldExample(
        slug="attributive_participle_ho_aner_ho_hybrizon",
        passage="ὁ γάρ ἀνήρ ὁ ὑβρίζων εἰς σέ ἐχθρός ὤν ἡμῖν τυγχάνει.",
        tags=["attributive participle", "attributive", "article",
              "sentence connector", "subject", "adverbial",
              "object of preposition", "predicate", "dative", "independent",
              "intransitive"],
        canned_answer=_ATTRIBUTIVE_PARTICIPLE_HO_ANER_ANSWER,
    ),
    GoldExample(
        slug="circumstantial_genitive_absolute_proiontos_de_tou_chronou",
        passage="προϊόντος δέ τοῦ χρόνου ἧκον μέν ἀπροσδοκήτως ἐκ ἀγροῦ.",
        tags=["circumstantial participle", "genitive absolute",
              "circumstantial", "sentence connector",
              "article", "adverbial", "object of preposition",
              "independent", "intransitive"],
        canned_answer=_CIRCUMSTANTIAL_GENITIVE_ABSOLUTE_PROIONTOS_ANSWER,
    ),
    GoldExample(
        slug="circumstantial_fits_clause_ego_hapanta_epideixo",
        passage="ἐγώ ἅπαντα ἐπιδείξω τά ἐμαυτοῦ πράγματα, οὐδέν παραλείπων, ἀλλά λέγων τἀληθῆ.",
        tags=["circumstantial participle", "circumstantial", "attributive",
              "genitive", "article", "direct object", "connecting word",
              "subject", "independent", "transitive active"],
        canned_answer=_CIRCUMSTANTIAL_FITS_CLAUSE_EGO_HAPANTA_ANSWER,
    ),
    GoldExample(
        slug="auxiliary_ho_nomos_gegrammenos_estin",
        passage="ὁ νόμος γεγραμμένος ἐστίν.",
        tags=["auxiliary", "article", "subject", "independent",
              "transitive passive"],
        canned_answer=_AUXILIARY_HO_NOMOS_GEGRAMMENOS_ANSWER,
    ),
    GoldExample(
        slug="subordinating_conjunction_dependent_kategorei_hos",
        passage="κατηγόρει ὡς μετά τήν ἐκφοράν αὐτῇ προσίοι.",
        tags=["subordinating conjunction", "unit verb", "dependent",
              "adverbial", "object of preposition", "dative", "article",
              "independent", "intransitive"],
        canned_answer=_SUBORDINATING_CONJUNCTION_KATEGOREI_HOS_ANSWER,
    ),
    GoldExample(
        slug="agent_he_eme_gyne_hypo_toutou",
        passage="ἡ ἐμή γυνή ὑπό τούτου τοῦ ἀνθρώπου διαφθείρεται.",
        tags=["agent", "object of preposition", "demonstrative", "article",
              "attributive", "subject", "independent", "transitive passive"],
        canned_answer=_AGENT_HE_EME_GYNE_ANSWER,
    ),
    GoldExample(
        slug="complementary_infinitive_exesti_helesthai",
        passage="ἔξεστι ἑλέσθαι.",
        tags=["complementary infinitive", "independent", "intransitive"],
        canned_answer=_COMPLEMENTARY_INFINITIVE_EXESTI_ANSWER,
    ),
    GoldExample(
        slug="subject_direct_object_emoicheuen_eratosthenes",
        passage="ἐμοίχευεν Ἐρατοσθένης τήν γυναῖκα τήν ἐμήν.",
        tags=["subject", "direct object", "article", "attributive",
              "independent", "transitive active"],
        canned_answer=_SUBJECT_DIRECT_OBJECT_EMOICHEUEN_ANSWER,
    ),
    GoldExample(
        slug="predicate_linking_omen_ten_emautou_gynaika",
        passage="ᾤμην τήν ἐμαυτοῦ γυναῖκα πασῶν σωφρονεστάτην εἶναι τῶν ἐν τῇ πόλει.",
        tags=["predicate", "linking verb", "indirect statement", "genitive",
              "subject", "article", "attributive", "object of preposition",
              "independent", "transitive active"],
        canned_answer=_PREDICATE_LINKING_OMEN_ANSWER,
    ),
    GoldExample(
        slug="attributive_prepositional_phrase_he_mache_he_en_marathoni",
        passage="ἡ μάχη ἡ ἐν Μαραθῶνι ἐγένετο.",
        tags=["attributive", "article", "object of preposition", "subject",
              "independent", "intransitive"],
        canned_answer=_ATTRIBUTIVE_PREPOSITIONAL_PHRASE_HE_MACHE_ANSWER,
    ),
    GoldExample(
        slug="demonstrative_tauten_elabon_ten_diken",
        passage="ταύτην ἔλαβον τήν δίκην.",
        tags=["demonstrative", "direct object", "article",
              "transitive active", "independent"],
        canned_answer=_DEMONSTRATIVE_TAUTEN_ELABON_ANSWER,
    ),
    GoldExample(
        slug="substantive_pronoun_ekeine_men_apellage",
        passage="ἐκείνη μέν ἀπηλλάγη.",
        tags=["subject", "connecting word", "transitive passive",
              "independent"],
        canned_answer=_SUBSTANTIVE_PRONOUN_EKEINE_MEN_ANSWER,
    ),
    GoldExample(
        slug="adverbial_bare_diarredhen_eiretai",
        passage="διαρρήδην εἴρηται.",
        tags=["adverbial", "transitive passive", "independent"],
        canned_answer=_ADVERBIAL_BARE_DIARREDHEN_ANSWER,
    ),
    GoldExample(
        slug="adverbial_attributive_eiserchometha_engytata_kapeleiou",
        passage="εἰσερχόμεθα ἐκ τοῦ ἐγγύτατα καπηλείου.",
        tags=["adverbial", "object of preposition", "article",
              "independent", "intransitive"],
        canned_answer=_ADVERBIAL_ATTRIBUTIVE_EISERCHOMETHA_ANSWER,
    ),
    GoldExample(
        slug="genitive_oicheto_eis_to_hieron",
        passage="ᾤχετο εἰς τό ἱερόν μετά τῆς μητρός τῆς ἐκείνου.",
        tags=["genitive", "object of preposition", "adverbial", "article",
              "independent", "intransitive"],
        canned_answer=_GENITIVE_OICHETO_ANSWER,
    ),
    GoldExample(
        slug="dative_verb_linked_echthra_emoi_kai_ekeino",
        passage="ἔχθρα ἐμοί καί ἐκείνῳ οὐδεμία ἦν.",
        tags=["dative", "connecting word", "subject", "attributive",
              "independent", "intransitive"],
        canned_answer=_DATIVE_VERB_LINKED_ECHTHRA_ANSWER,
    ),
    GoldExample(
        slug="accusative_of_time_tauta_polyn_chronon",
        passage="ταῦτα πολύν χρόνον οὕτως ἐγίγνετο.",
        tags=["accusative", "attributive", "subject", "adverbial",
              "independent", "intransitive"],
        canned_answer=_ACCUSATIVE_OF_TIME_TAUTA_ANSWER,
    ),
    GoldExample(
        slug="apposition_demosthenes_ho_rhetor",
        passage="Δημοσθένης ὁ ῥήτωρ ἦλθεν.",
        tags=["apposition", "article", "subject", "independent",
              "intransitive"],
        canned_answer=_APPOSITION_DEMOSTHENES_HO_RHETOR_ANSWER,
    ),
    GoldExample(
        slug="relative_pronoun_ho_aner_hon_eidon",
        passage="ὁ ἀνήρ ὅν εἶδον ἀπῆλθεν.",
        tags=["relative pronoun", "unit verb", "direct object", "subject",
              "article", "dependent", "independent", "transitive active",
              "intransitive"],
        canned_answer=_RELATIVE_PRONOUN_HO_ANER_HON_EIDON_ANSWER,
    ),
    GoldExample(
        slug="implied_eimi_tauten_ten_hybrin",
        passage="ταύτην τήν ὕβριν ἅπαντες ἄνθρωποι δεινοτάτην ἡγοῦνται.",
        tags=["implied eimi", "indirect statement", "predicate", "subject",
              "demonstrative", "attributive", "article", "independent",
              "transitive active", "linking verb"],
        canned_answer=_IMPLIED_EIMI_TAUTEN_TEN_HYBRIN_ANSWER,
    ),
    GoldExample(
        slug="implied_repetition_ego_men_ano_dietomen",
        passage="ἐγώ μέν ἄνω διῃτώμην, αἱ δέ γυναῖκες κάτω.",
        tags=["implied repetition", "connecting word", "subject",
              "adverbial", "article", "independent", "intransitive"],
        canned_answer=_IMPLIED_REPETITION_EGO_MEN_ANO_ANSWER,
    ),
    GoldExample(
        slug="numeral_vs_lexical_eidon_duo_andras",
        passage="εἶδον δύο ἄνδρας καί γʹ γυναῖκας.",
        tags=["numeral", "lexical", "attributive", "direct object",
              "connecting word", "independent", "transitive active"],
        canned_answer=_NUMERAL_VS_LEXICAL_EIDON_DUO_ANSWER,
    ),
    GoldExample(
        slug="enclitic_eiper_houtos_echei",
        passage="ἐγώ γάρ οὐδέν δέομαι λόγων, ἀλλά τό ἔργον φανερόν γενέσθαι, εἴπερ οὕτως ἔχει.",
        tags=["enclitic", "subordinating conjunction", "unit verb",
              "sentence connector", "connecting word", "accusative",
              "genitive", "article", "subject", "predicate", "independent",
              "dependent", "intransitive"],
        canned_answer=_ENCLITIC_EIPER_HOUTOS_ECHEI_ANSWER,
    ),
    GoldExample(
        slug="indirect_question_ouk_oida_tis_elthen",
        passage="οὐκ οἶδα τίς ἦλθεν.",
        tags=["subordinating conjunction", "unit verb", "adverbial",
              "dependent", "independent", "transitive active",
              "intransitive"],
        canned_answer=_INDIRECT_QUESTION_OUK_OIDA_TIS_ANSWER,
    ),
    GoldExample(
        slug="depth_two_epei_edei_hemartekenai",
        passage="ἐπεί ᾔδει ἑαυτόν ἡμαρτηκέναι, ἠνιάθη.",
        tags=["subordinating conjunction", "unit verb", "indirect statement",
              "subject", "dependent", "independent", "transitive active",
              "intransitive"],
        canned_answer=_DEPTH_TWO_EPEI_EDEI_HEMARTEKENAI_ANSWER,
    ),
    GoldExample(
        slug="linking_verb_mestos_e_hypopsias",
        passage="μεστός ἦ ὑποψίας.",
        tags=["predicate", "linking verb", "genitive", "independent"],
        canned_answer=_LINKING_VERB_MESTOS_E_ANSWER,
    ),
]
