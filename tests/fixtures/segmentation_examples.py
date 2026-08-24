"""
Gold-annotated examples for the segmentation stage (grammatike/segmentation.py) --
the segmentation-stage counterpart to fixtures/gold_examples.py. Greek
analogue of arsgrammatica's fixtures/segmentation_examples.py.

Each SegmentationExample pairs a list of CitedText sources with the
sentences/tokens segment_sources() (now a deterministic, non-LM function --
see segmentation.py's own module docstring) is expected to produce. Unlike
fixtures/gold_examples.py's canned_answer (fed to DummyLM as a stand-in for
a real LM response), expected_sentences here is simply the correct answer:
segment_sources() is called directly on `sources`, with no DummyLM and no
`dspy` involved at all, and its actual output is compared against
expected_sentences.

These four examples are Greek-appropriate replacements for arsgrammatica's
three (which exercised Latin-only enclitic false-positives, the
context-dependent -ne split, and praenomen/abbreviation recognition --
none of which apply to Greek; see segmentation.py's own docstring: "There
is no praenomen or other abbreviation category in this scheme"). These
cover: (1) the fused-enclitic split (περ off of εἴπερ) versus the
false-positive guard against splitting a word that merely happens to
contain the same letters (περί), (2) numeral vs. spelled-out-word
tokenization, (3) the Greek question mark (U+037E, here spelled with the
ASCII/Latin semicolon it decomposes to) always ending a sentence, and (4)
the classification-not-splitting distinction between the enclitic
indefinite τις and the accented, non-enclitic interrogative τίς -- all
directly from segmentation.py's own docstring.

Three further examples, all real text from Lysias 1 (the same passages
syntax_model.md itself draws its own worked examples from), cover the
current, simplified "Segmentation into sentences" rule: a period or
interrogative ALWAYS ends a sentence, and nothing else does -- in
particular, a raised-dot/colon-type mark (here, an ASCII colon) is now
just ordinary internal punctuation, exactly like a comma, never a sentence
boundary:

- `lysias_1_2_colons_do_not_end_a_sentence` and
  `lysias_1_1_colon_does_not_strand_the_relative_clause` -- both passages
  contain two internal colons and end with a final period; under the
  current rule, neither colon splits anything, so each passage segments as
  ONE sentence running all the way to its own final period. (An earlier
  version of syntax_model.md's "Segmentation into sentences" section
  treated a raised-dot/colon-type mark as a *candidate* boundary, resolved
  by an LM judging syntactic coherence case by case -- these two fixtures
  used to each split into two or three sentences under that rule. The
  section was simplified to remove that judgment call entirely, and these
  fixtures were rewritten to match; see git history for the earlier
  three-sentence/two-sentence versions if that comparison is ever useful.)
  The second of the two also exercises two more genuine fused-enclitic
  splits beyond εἴπερ: οἷοίπερ -> οἷοί + περ, and ἥνπερ -> ἥν + περ.
- `sentence_spans_a_citation_boundary_through_an_internal_colon` -- Lysias
  1.3/1.4 (the same passages the old, now-deleted test_segmentation_live.py
  used to exercise live): one sentence starts in 1.3, runs straight through
  an internal colon (still not a boundary), crosses into 1.4, and only ends
  at 1.4's own final period -- confirming sentence-splitting still ignores
  CitedText boundaries entirely, exactly as it ignores an internal colon.
"""

from dataclasses import dataclass
from typing import Any

from grammatike.models import CitedText


@dataclass
class SegmentationExample:
    slug: str
    sources: list[CitedText]
    tags: list[str]
    expected_sentences: dict[str, Any]


SEGMENTATION_EXAMPLES = [
    SegmentationExample(
        slug="fused_enclitic_split_eiper_vs_false_positive_peri",
        sources=[
            CitedText(citation="ex.1", text="περὶ τούτου λέγει."),
            CitedText(citation="ex.2", text="εἴπερ ἀληθῆ λέγει, πείσομαι."),
        ],
        tags=["enclitic split guard", "enclitic"],
        expected_sentences={
            "reasoning": (
                "περὶ in ex.1 is the ordinary preposition 'concerning' -- it "
                "merely happens to contain the letters περ, but there is no "
                "genuine independent remainder if it were split (*περὶ minus "
                "περ leaves nothing), so it stays one lexical token. εἴπερ in "
                "ex.2 genuinely fuses εἰ ('if') with the enclitic περ, and εἴ "
                "is a real independent word on its own, so it splits into "
                "two tokens: εἴ (lexical) + περ (enclitic)."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "περὶ", "citation": "ex.1"},
                    {"id": "t1", "text": "τούτου", "citation": "ex.1"},
                    {"id": "t2", "text": "λέγει", "citation": "ex.1"},
                    {"id": "t3", "text": ".", "citation": "ex.1"},
                ]},
                {"tokens": [
                    {"id": "t4", "text": "εἴ", "citation": "ex.2"},
                    {"id": "t5", "text": "περ", "citation": "ex.2"},
                    {"id": "t6", "text": "ἀληθῆ", "citation": "ex.2"},
                    {"id": "t7", "text": "λέγει", "citation": "ex.2"},
                    {"id": "t8", "text": ",", "citation": "ex.2"},
                    {"id": "t9", "text": "πείσομαι", "citation": "ex.2"},
                    {"id": "t10", "text": ".", "citation": "ex.2"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="numeral_milesian_vs_spelled_out_lexical",
        sources=[
            CitedText(citation="ex.3", text="τῇ γʹ ἡμέρᾳ ἦλθεν."),
            CitedText(citation="ex.4", text="δύο ἄνδρας εἶδον."),
        ],
        tags=["numeral vs lexical"],
        expected_sentences={
            "reasoning": (
                "γʹ in ex.3 is a number written numerically, in Milesian "
                "notation ('3rd'); its keraia mark is not Unicode "
                "punctuation (category Lm, not P*), so it stays fused to "
                "the digit letter as a single token with no special-casing "
                "needed. δύο in ex.4 is the same kind of quantity spelled "
                "out as an ordinary word ('two'), so it likewise stays a "
                "single token, per syntax_model.md's own δύω example -- "
                "segment_sources() doesn't classify tokentype at all "
                "(that's SyntaxAnalysis's job), only where token boundaries "
                "fall, and neither of these needs a boundary drawn inside it."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "τῇ", "citation": "ex.3"},
                    {"id": "t1", "text": "γʹ", "citation": "ex.3"},
                    {"id": "t2", "text": "ἡμέρᾳ", "citation": "ex.3"},
                    {"id": "t3", "text": "ἦλθεν", "citation": "ex.3"},
                    {"id": "t4", "text": ".", "citation": "ex.3"},
                ]},
                {"tokens": [
                    {"id": "t5", "text": "δύο", "citation": "ex.4"},
                    {"id": "t6", "text": "ἄνδρας", "citation": "ex.4"},
                    {"id": "t7", "text": "εἶδον", "citation": "ex.4"},
                    {"id": "t8", "text": ".", "citation": "ex.4"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="greek_question_mark_always_ends_the_sentence",
        sources=[
            CitedText(citation="ex.5", text="ἐγὼ μὲν ἔμεινα, σὺ δὲ ἀπῆλθες."),
            CitedText(citation="ex.6", text="τίς ἦλθεν;"),
        ],
        tags=["sentence boundary", "punctuation"],
        expected_sentences={
            "reasoning": (
                "The comma in ex.5 separates two coordinate clauses within "
                "a single sentence, which ends only at its final period. "
                "The interrogative mark in ex.6 (written here as the ASCII "
                "semicolon U+003B -- the Greek question mark U+037E's own "
                "valid Unicode decomposition, and what most digital "
                "editions actually type) ALWAYS ends a sentence, per "
                "syntax_model.md's current 'Segmentation into sentences' "
                "rule -- unlike a raised dot or colon-type mark (see the "
                "lysias_1_1/lysias_1_2 examples below), which never ends "
                "one, no matter what surrounds it."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "ἐγὼ", "citation": "ex.5"},
                    {"id": "t1", "text": "μὲν", "citation": "ex.5"},
                    {"id": "t2", "text": "ἔμεινα", "citation": "ex.5"},
                    {"id": "t3", "text": ",", "citation": "ex.5"},
                    {"id": "t4", "text": "σὺ", "citation": "ex.5"},
                    {"id": "t5", "text": "δὲ", "citation": "ex.5"},
                    {"id": "t6", "text": "ἀπῆλθες", "citation": "ex.5"},
                    {"id": "t7", "text": ".", "citation": "ex.5"},
                ]},
                {"tokens": [
                    {"id": "t8", "text": "τίς", "citation": "ex.6"},
                    {"id": "t9", "text": "ἦλθεν", "citation": "ex.6"},
                    {"id": "t10", "text": ";", "citation": "ex.6"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="lysias_1_2_colons_do_not_end_a_sentence",
        sources=[
            CitedText(
                citation="Lysias 1.2",
                text=(
                    "καὶ ταῦτα οὐκ ἂν εἴη μόνον παρ' ὑμῖν οὕτως ἐγνωσμένα, "
                    "ἀλλ' ἐν ἁπάσῃ τῇ Ἑλλάδι: περὶ τούτου γὰρ μόνου τοῦ "
                    "ἀδικήματος καὶ ἐν δημοκρατίᾳ καὶ ὀλιγαρχίᾳ ἡ αὐτὴ "
                    "τιμωρία τοῖς ἀσθενεστάτοις πρὸς τοὺς τὰ μέγιστα "
                    "δυναμένους ἀποδέδοται, ὥστε τὸν χείριστον τῶν αὐτῶν "
                    "τυγχάνειν τῷ βελτίστῳ: οὕτως, ὦ ἄνδρες, ταύτην τὴν "
                    "ὕβριν ἅπαντες ἄνθρωποι δεινοτάτην ἡγοῦνται."
                ),
            ),
        ],
        tags=["sentence boundary", "punctuation"],
        expected_sentences={
            "reasoning": (
                "syntax_model.md's own text for Lysias 1.2, real prose with "
                "two internal colons and a final period. Under the current "
                "rule (period/interrogative only), neither colon ends a "
                "sentence -- they're ordinary internal punctuation tokens, "
                "same as the commas -- so the whole passage is ONE sentence, "
                "ending only at the final period."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "καὶ", "citation": "Lysias 1.2"},
                    {"id": "t1", "text": "ταῦτα", "citation": "Lysias 1.2"},
                    {"id": "t2", "text": "οὐκ", "citation": "Lysias 1.2"},
                    {"id": "t3", "text": "ἂν", "citation": "Lysias 1.2"},
                    {"id": "t4", "text": "εἴη", "citation": "Lysias 1.2"},
                    {"id": "t5", "text": "μόνον", "citation": "Lysias 1.2"},
                    {"id": "t6", "text": "παρ'", "citation": "Lysias 1.2"},
                    {"id": "t7", "text": "ὑμῖν", "citation": "Lysias 1.2"},
                    {"id": "t8", "text": "οὕτως", "citation": "Lysias 1.2"},
                    {"id": "t9", "text": "ἐγνωσμένα", "citation": "Lysias 1.2"},
                    {"id": "t10", "text": ",", "citation": "Lysias 1.2"},
                    {"id": "t11", "text": "ἀλλ'", "citation": "Lysias 1.2"},
                    {"id": "t12", "text": "ἐν", "citation": "Lysias 1.2"},
                    {"id": "t13", "text": "ἁπάσῃ", "citation": "Lysias 1.2"},
                    {"id": "t14", "text": "τῇ", "citation": "Lysias 1.2"},
                    {"id": "t15", "text": "Ἑλλάδι", "citation": "Lysias 1.2"},
                    {"id": "t16", "text": ":", "citation": "Lysias 1.2"},
                    {"id": "t17", "text": "περὶ", "citation": "Lysias 1.2"},
                    {"id": "t18", "text": "τούτου", "citation": "Lysias 1.2"},
                    {"id": "t19", "text": "γὰρ", "citation": "Lysias 1.2"},
                    {"id": "t20", "text": "μόνου", "citation": "Lysias 1.2"},
                    {"id": "t21", "text": "τοῦ", "citation": "Lysias 1.2"},
                    {"id": "t22", "text": "ἀδικήματος", "citation": "Lysias 1.2"},
                    {"id": "t23", "text": "καὶ", "citation": "Lysias 1.2"},
                    {"id": "t24", "text": "ἐν", "citation": "Lysias 1.2"},
                    {"id": "t25", "text": "δημοκρατίᾳ", "citation": "Lysias 1.2"},
                    {"id": "t26", "text": "καὶ", "citation": "Lysias 1.2"},
                    {"id": "t27", "text": "ὀλιγαρχίᾳ", "citation": "Lysias 1.2"},
                    {"id": "t28", "text": "ἡ", "citation": "Lysias 1.2"},
                    {"id": "t29", "text": "αὐτὴ", "citation": "Lysias 1.2"},
                    {"id": "t30", "text": "τιμωρία", "citation": "Lysias 1.2"},
                    {"id": "t31", "text": "τοῖς", "citation": "Lysias 1.2"},
                    {"id": "t32", "text": "ἀσθενεστάτοις", "citation": "Lysias 1.2"},
                    {"id": "t33", "text": "πρὸς", "citation": "Lysias 1.2"},
                    {"id": "t34", "text": "τοὺς", "citation": "Lysias 1.2"},
                    {"id": "t35", "text": "τὰ", "citation": "Lysias 1.2"},
                    {"id": "t36", "text": "μέγιστα", "citation": "Lysias 1.2"},
                    {"id": "t37", "text": "δυναμένους", "citation": "Lysias 1.2"},
                    {"id": "t38", "text": "ἀποδέδοται", "citation": "Lysias 1.2"},
                    {"id": "t39", "text": ",", "citation": "Lysias 1.2"},
                    {"id": "t40", "text": "ὥστε", "citation": "Lysias 1.2"},
                    {"id": "t41", "text": "τὸν", "citation": "Lysias 1.2"},
                    {"id": "t42", "text": "χείριστον", "citation": "Lysias 1.2"},
                    {"id": "t43", "text": "τῶν", "citation": "Lysias 1.2"},
                    {"id": "t44", "text": "αὐτῶν", "citation": "Lysias 1.2"},
                    {"id": "t45", "text": "τυγχάνειν", "citation": "Lysias 1.2"},
                    {"id": "t46", "text": "τῷ", "citation": "Lysias 1.2"},
                    {"id": "t47", "text": "βελτίστῳ", "citation": "Lysias 1.2"},
                    {"id": "t48", "text": ":", "citation": "Lysias 1.2"},
                    {"id": "t49", "text": "οὕτως", "citation": "Lysias 1.2"},
                    {"id": "t50", "text": ",", "citation": "Lysias 1.2"},
                    {"id": "t51", "text": "ὦ", "citation": "Lysias 1.2"},
                    {"id": "t52", "text": "ἄνδρες", "citation": "Lysias 1.2"},
                    {"id": "t53", "text": ",", "citation": "Lysias 1.2"},
                    {"id": "t54", "text": "ταύτην", "citation": "Lysias 1.2"},
                    {"id": "t55", "text": "τὴν", "citation": "Lysias 1.2"},
                    {"id": "t56", "text": "ὕβριν", "citation": "Lysias 1.2"},
                    {"id": "t57", "text": "ἅπαντες", "citation": "Lysias 1.2"},
                    {"id": "t58", "text": "ἄνθρωποι", "citation": "Lysias 1.2"},
                    {"id": "t59", "text": "δεινοτάτην", "citation": "Lysias 1.2"},
                    {"id": "t60", "text": "ἡγοῦνται", "citation": "Lysias 1.2"},
                    {"id": "t61", "text": ".", "citation": "Lysias 1.2"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="lysias_1_1_colon_does_not_strand_the_relative_clause",
        sources=[
            CitedText(
                citation="Lysias 1.1",
                text=(
                    "περὶ πολλοῦ ἂν ποιησαίμην, ὦ ἄνδρες, τὸ τοιούτους "
                    "ὑμᾶς ἐμοὶ δικαστὰς περὶ τούτου τοῦ πράγματος "
                    "γενέσθαι, οἷοίπερ ἂν ὑμῖν αὐτοῖς εἴητε "
                    "τοιαῦτα πεπονθότες: εὖ γὰρ οἶδ' ὅτι, εἰ τὴν "
                    "αὐτὴν γνώμην περὶ τῶν ἄλλων ἔχοιτε, "
                    "ἥνπερ περὶ ὑμῶν αὐτῶν, οὐκ ἂν εἴη: ὅστις "
                    "οὐκ ἐπὶ τοῖς γεγενημένοις ἀγανακτοίη, "
                    "ἀλλὰ πάντες ἂν περὶ τῶν τὰ τοιαῦτα ἐπιτηδευόντων τὰς "
                    "ζημίας μικρὰς ἡγοῖσθε."
                ),
            ),
        ],
        tags=[
            "sentence boundary", "punctuation", "enclitic split guard",
        ],
        expected_sentences={
            "reasoning": (
                "syntax_model.md's own text for Lysias 1.1: two internal "
                "colons and a final period, real prose that used to be the "
                "worked example for the OLD 'candidate boundary, decided by "
                "coherence' rule -- specifically because splitting at the "
                "second colon would strand the dependent ὅστις-relative "
                "clause that belongs with what precedes it, not what "
                "follows. Under the CURRENT rule neither colon is even a "
                "candidate: they're ordinary punctuation, so the passage is "
                "one sentence from περὶ straight through to the final "
                "period, deciding the old dilemma by removing it rather "
                "than by resolving it case by case. This passage also has "
                "two more genuine fused-enclitic splits beyond εἴπερ: "
                "οἷοίπερ splits into οἷοί (lexical, 'such as') + περ "
                "(enclitic), and ἥνπερ splits into ἥν (lexical, the "
                "relative pronoun) + περ (enclitic) -- ἐμοὶ, by contrast, "
                "stays a single ordinary token: it merely happens to share "
                "a root with the enclitic μοι, not an enclitic spelling "
                "itself, and nothing here needs to tell them apart anyway "
                "(that's SyntaxAnalysis's tokentype job, not this stage's)."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "περὶ", "citation": "Lysias 1.1"},
                    {"id": "t1", "text": "πολλοῦ", "citation": "Lysias 1.1"},
                    {"id": "t2", "text": "ἂν", "citation": "Lysias 1.1"},
                    {"id": "t3", "text": "ποιησαίμην", "citation": "Lysias 1.1"},
                    {"id": "t4", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t5", "text": "ὦ", "citation": "Lysias 1.1"},
                    {"id": "t6", "text": "ἄνδρες", "citation": "Lysias 1.1"},
                    {"id": "t7", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t8", "text": "τὸ", "citation": "Lysias 1.1"},
                    {"id": "t9", "text": "τοιούτους", "citation": "Lysias 1.1"},
                    {"id": "t10", "text": "ὑμᾶς", "citation": "Lysias 1.1"},
                    {"id": "t11", "text": "ἐμοὶ", "citation": "Lysias 1.1"},
                    {"id": "t12", "text": "δικαστὰς", "citation": "Lysias 1.1"},
                    {"id": "t13", "text": "περὶ", "citation": "Lysias 1.1"},
                    {"id": "t14", "text": "τούτου", "citation": "Lysias 1.1"},
                    {"id": "t15", "text": "τοῦ", "citation": "Lysias 1.1"},
                    {"id": "t16", "text": "πράγματος", "citation": "Lysias 1.1"},
                    {"id": "t17", "text": "γενέσθαι", "citation": "Lysias 1.1"},
                    {"id": "t18", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t19", "text": "οἷοί", "citation": "Lysias 1.1"},
                    {"id": "t20", "text": "περ", "citation": "Lysias 1.1"},
                    {"id": "t21", "text": "ἂν", "citation": "Lysias 1.1"},
                    {"id": "t22", "text": "ὑμῖν", "citation": "Lysias 1.1"},
                    {"id": "t23", "text": "αὐτοῖς", "citation": "Lysias 1.1"},
                    {"id": "t24", "text": "εἴητε", "citation": "Lysias 1.1"},
                    {"id": "t25", "text": "τοιαῦτα", "citation": "Lysias 1.1"},
                    {"id": "t26", "text": "πεπονθότες", "citation": "Lysias 1.1"},
                    {"id": "t27", "text": ":", "citation": "Lysias 1.1"},
                    {"id": "t28", "text": "εὖ", "citation": "Lysias 1.1"},
                    {"id": "t29", "text": "γὰρ", "citation": "Lysias 1.1"},
                    {"id": "t30", "text": "οἶδ'", "citation": "Lysias 1.1"},
                    {"id": "t31", "text": "ὅτι", "citation": "Lysias 1.1"},
                    {"id": "t32", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t33", "text": "εἰ", "citation": "Lysias 1.1"},
                    {"id": "t34", "text": "τὴν", "citation": "Lysias 1.1"},
                    {"id": "t35", "text": "αὐτὴν", "citation": "Lysias 1.1"},
                    {"id": "t36", "text": "γνώμην", "citation": "Lysias 1.1"},
                    {"id": "t37", "text": "περὶ", "citation": "Lysias 1.1"},
                    {"id": "t38", "text": "τῶν", "citation": "Lysias 1.1"},
                    {"id": "t39", "text": "ἄλλων", "citation": "Lysias 1.1"},
                    {"id": "t40", "text": "ἔχοιτε", "citation": "Lysias 1.1"},
                    {"id": "t41", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t42", "text": "ἥν", "citation": "Lysias 1.1"},
                    {"id": "t43", "text": "περ", "citation": "Lysias 1.1"},
                    {"id": "t44", "text": "περὶ", "citation": "Lysias 1.1"},
                    {"id": "t45", "text": "ὑμῶν", "citation": "Lysias 1.1"},
                    {"id": "t46", "text": "αὐτῶν", "citation": "Lysias 1.1"},
                    {"id": "t47", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t48", "text": "οὐκ", "citation": "Lysias 1.1"},
                    {"id": "t49", "text": "ἂν", "citation": "Lysias 1.1"},
                    {"id": "t50", "text": "εἴη", "citation": "Lysias 1.1"},
                    {"id": "t51", "text": ":", "citation": "Lysias 1.1"},
                    {"id": "t52", "text": "ὅστις", "citation": "Lysias 1.1"},
                    {"id": "t53", "text": "οὐκ", "citation": "Lysias 1.1"},
                    {"id": "t54", "text": "ἐπὶ", "citation": "Lysias 1.1"},
                    {"id": "t55", "text": "τοῖς", "citation": "Lysias 1.1"},
                    {"id": "t56", "text": "γεγενημένοις", "citation": "Lysias 1.1"},
                    {"id": "t57", "text": "ἀγανακτοίη", "citation": "Lysias 1.1"},
                    {"id": "t58", "text": ",", "citation": "Lysias 1.1"},
                    {"id": "t59", "text": "ἀλλὰ", "citation": "Lysias 1.1"},
                    {"id": "t60", "text": "πάντες", "citation": "Lysias 1.1"},
                    {"id": "t61", "text": "ἂν", "citation": "Lysias 1.1"},
                    {"id": "t62", "text": "περὶ", "citation": "Lysias 1.1"},
                    {"id": "t63", "text": "τῶν", "citation": "Lysias 1.1"},
                    {"id": "t64", "text": "τὰ", "citation": "Lysias 1.1"},
                    {"id": "t65", "text": "τοιαῦτα", "citation": "Lysias 1.1"},
                    {"id": "t66", "text": "ἐπιτηδευόντων", "citation": "Lysias 1.1"},
                    {"id": "t67", "text": "τὰς", "citation": "Lysias 1.1"},
                    {"id": "t68", "text": "ζημίας", "citation": "Lysias 1.1"},
                    {"id": "t69", "text": "μικρὰς", "citation": "Lysias 1.1"},
                    {"id": "t70", "text": "ἡγοῖσθε", "citation": "Lysias 1.1"},
                    {"id": "t71", "text": ".", "citation": "Lysias 1.1"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="sentence_spans_a_citation_boundary_through_an_internal_colon",
        sources=[
            CitedText(
                citation="Lysias 1.3",
                text=(
                    "περὶ μὲν οὖν τοῦ μεγέθους τῆς ζημίας ἅπαντας ὑμᾶς "
                    "νομίζω τὴν αὐτὴν διάνοιαν ἔχειν, καὶ οὐδένα οὕτως "
                    "ὀλιγώρως διακεῖσθαι, ὅστις οἴεται δεῖν συγγνώμης "
                    "τυγχάνειν ἢ μικρᾶς ζημίας ἀξίους ἡγεῖται τοὺς τῶν "
                    "τοιούτων ἔργων αἰτίους: ἡγοῦμαι δέ,"
                ),
            ),
            CitedText(
                citation="Lysias 1.4",
                text=(
                    "ὦ ἄνδρες, τοῦτό με δεῖν ἐπιδεῖξαι, ὡς ἐμοίχευεν "
                    "Ἐρατοσθένης τὴν γυναῖκα τὴν ἐμὴν καὶ ἐκείνην τε "
                    "διέφθειρε καὶ τοὺς παῖδας τοὺς ἐμοὺς ᾔσχυνε καὶ ἐμὲ "
                    "αὐτὸν ὕβρισεν εἰς τὴν οἰκίαν τὴν ἐμὴν εἰσιών, καὶ "
                    "οὔτε ἔχθρα ἐμοὶ καὶ ἐκείνῳ οὐδεμία ἦν πλὴν ταύτης, "
                    "οὔτε χρημάτων ἕνεκα ἔπραξα ταῦτα, ἵνα πλούσιος ἐκ "
                    "πένητος γένωμαι, οὔτε ἄλλου κέρδους οὐδενὸς πλὴν τῆς "
                    "κατὰ τοὺς νόμους τιμωρίας."
                ),
            ),
        ],
        tags=["sentence boundary", "punctuation", "citation spanning"],
        expected_sentences={
            "reasoning": (
                "Lysias 1.3 ends mid-sentence, at an internal colon "
                "followed only by 'ἡγοῦμαι δέ,' with no period -- the "
                "colon does not end the sentence (see the two fixtures "
                "above), so it continues straight into 1.4's text and only "
                "ends at 1.4's own final period. One sentence, spanning "
                "both citation units, with each token still attributed to "
                "whichever unit it actually came from."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "περὶ", "citation": "Lysias 1.3"},
                    {"id": "t1", "text": "μὲν", "citation": "Lysias 1.3"},
                    {"id": "t2", "text": "οὖν", "citation": "Lysias 1.3"},
                    {"id": "t3", "text": "τοῦ", "citation": "Lysias 1.3"},
                    {"id": "t4", "text": "μεγέθους", "citation": "Lysias 1.3"},
                    {"id": "t5", "text": "τῆς", "citation": "Lysias 1.3"},
                    {"id": "t6", "text": "ζημίας", "citation": "Lysias 1.3"},
                    {"id": "t7", "text": "ἅπαντας", "citation": "Lysias 1.3"},
                    {"id": "t8", "text": "ὑμᾶς", "citation": "Lysias 1.3"},
                    {"id": "t9", "text": "νομίζω", "citation": "Lysias 1.3"},
                    {"id": "t10", "text": "τὴν", "citation": "Lysias 1.3"},
                    {"id": "t11", "text": "αὐτὴν", "citation": "Lysias 1.3"},
                    {"id": "t12", "text": "διάνοιαν", "citation": "Lysias 1.3"},
                    {"id": "t13", "text": "ἔχειν", "citation": "Lysias 1.3"},
                    {"id": "t14", "text": ",", "citation": "Lysias 1.3"},
                    {"id": "t15", "text": "καὶ", "citation": "Lysias 1.3"},
                    {"id": "t16", "text": "οὐδένα", "citation": "Lysias 1.3"},
                    {"id": "t17", "text": "οὕτως", "citation": "Lysias 1.3"},
                    {"id": "t18", "text": "ὀλιγώρως", "citation": "Lysias 1.3"},
                    {"id": "t19", "text": "διακεῖσθαι", "citation": "Lysias 1.3"},
                    {"id": "t20", "text": ",", "citation": "Lysias 1.3"},
                    {"id": "t21", "text": "ὅστις", "citation": "Lysias 1.3"},
                    {"id": "t22", "text": "οἴεται", "citation": "Lysias 1.3"},
                    {"id": "t23", "text": "δεῖν", "citation": "Lysias 1.3"},
                    {"id": "t24", "text": "συγγνώμης", "citation": "Lysias 1.3"},
                    {"id": "t25", "text": "τυγχάνειν", "citation": "Lysias 1.3"},
                    {"id": "t26", "text": "ἢ", "citation": "Lysias 1.3"},
                    {"id": "t27", "text": "μικρᾶς", "citation": "Lysias 1.3"},
                    {"id": "t28", "text": "ζημίας", "citation": "Lysias 1.3"},
                    {"id": "t29", "text": "ἀξίους", "citation": "Lysias 1.3"},
                    {"id": "t30", "text": "ἡγεῖται", "citation": "Lysias 1.3"},
                    {"id": "t31", "text": "τοὺς", "citation": "Lysias 1.3"},
                    {"id": "t32", "text": "τῶν", "citation": "Lysias 1.3"},
                    {"id": "t33", "text": "τοιούτων", "citation": "Lysias 1.3"},
                    {"id": "t34", "text": "ἔργων", "citation": "Lysias 1.3"},
                    {"id": "t35", "text": "αἰτίους", "citation": "Lysias 1.3"},
                    {"id": "t36", "text": ":", "citation": "Lysias 1.3"},
                    {"id": "t37", "text": "ἡγοῦμαι", "citation": "Lysias 1.3"},
                    {"id": "t38", "text": "δέ", "citation": "Lysias 1.3"},
                    {"id": "t39", "text": ",", "citation": "Lysias 1.3"},
                    {"id": "t40", "text": "ὦ", "citation": "Lysias 1.4"},
                    {"id": "t41", "text": "ἄνδρες", "citation": "Lysias 1.4"},
                    {"id": "t42", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t43", "text": "τοῦτό", "citation": "Lysias 1.4"},
                    {"id": "t44", "text": "με", "citation": "Lysias 1.4"},
                    {"id": "t45", "text": "δεῖν", "citation": "Lysias 1.4"},
                    {"id": "t46", "text": "ἐπιδεῖξαι", "citation": "Lysias 1.4"},
                    {"id": "t47", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t48", "text": "ὡς", "citation": "Lysias 1.4"},
                    {"id": "t49", "text": "ἐμοίχευεν", "citation": "Lysias 1.4"},
                    {"id": "t50", "text": "Ἐρατοσθένης", "citation": "Lysias 1.4"},
                    {"id": "t51", "text": "τὴν", "citation": "Lysias 1.4"},
                    {"id": "t52", "text": "γυναῖκα", "citation": "Lysias 1.4"},
                    {"id": "t53", "text": "τὴν", "citation": "Lysias 1.4"},
                    {"id": "t54", "text": "ἐμὴν", "citation": "Lysias 1.4"},
                    {"id": "t55", "text": "καὶ", "citation": "Lysias 1.4"},
                    {"id": "t56", "text": "ἐκείνην", "citation": "Lysias 1.4"},
                    {"id": "t57", "text": "τε", "citation": "Lysias 1.4"},
                    {"id": "t58", "text": "διέφθειρε", "citation": "Lysias 1.4"},
                    {"id": "t59", "text": "καὶ", "citation": "Lysias 1.4"},
                    {"id": "t60", "text": "τοὺς", "citation": "Lysias 1.4"},
                    {"id": "t61", "text": "παῖδας", "citation": "Lysias 1.4"},
                    {"id": "t62", "text": "τοὺς", "citation": "Lysias 1.4"},
                    {"id": "t63", "text": "ἐμοὺς", "citation": "Lysias 1.4"},
                    {"id": "t64", "text": "ᾔσχυνε", "citation": "Lysias 1.4"},
                    {"id": "t65", "text": "καὶ", "citation": "Lysias 1.4"},
                    {"id": "t66", "text": "ἐμὲ", "citation": "Lysias 1.4"},
                    {"id": "t67", "text": "αὐτὸν", "citation": "Lysias 1.4"},
                    {"id": "t68", "text": "ὕβρισεν", "citation": "Lysias 1.4"},
                    {"id": "t69", "text": "εἰς", "citation": "Lysias 1.4"},
                    {"id": "t70", "text": "τὴν", "citation": "Lysias 1.4"},
                    {"id": "t71", "text": "οἰκίαν", "citation": "Lysias 1.4"},
                    {"id": "t72", "text": "τὴν", "citation": "Lysias 1.4"},
                    {"id": "t73", "text": "ἐμὴν", "citation": "Lysias 1.4"},
                    {"id": "t74", "text": "εἰσιών", "citation": "Lysias 1.4"},
                    {"id": "t75", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t76", "text": "καὶ", "citation": "Lysias 1.4"},
                    {"id": "t77", "text": "οὔτε", "citation": "Lysias 1.4"},
                    {"id": "t78", "text": "ἔχθρα", "citation": "Lysias 1.4"},
                    {"id": "t79", "text": "ἐμοὶ", "citation": "Lysias 1.4"},
                    {"id": "t80", "text": "καὶ", "citation": "Lysias 1.4"},
                    {"id": "t81", "text": "ἐκείνῳ", "citation": "Lysias 1.4"},
                    {"id": "t82", "text": "οὐδεμία", "citation": "Lysias 1.4"},
                    {"id": "t83", "text": "ἦν", "citation": "Lysias 1.4"},
                    {"id": "t84", "text": "πλὴν", "citation": "Lysias 1.4"},
                    {"id": "t85", "text": "ταύτης", "citation": "Lysias 1.4"},
                    {"id": "t86", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t87", "text": "οὔτε", "citation": "Lysias 1.4"},
                    {"id": "t88", "text": "χρημάτων", "citation": "Lysias 1.4"},
                    {"id": "t89", "text": "ἕνεκα", "citation": "Lysias 1.4"},
                    {"id": "t90", "text": "ἔπραξα", "citation": "Lysias 1.4"},
                    {"id": "t91", "text": "ταῦτα", "citation": "Lysias 1.4"},
                    {"id": "t92", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t93", "text": "ἵνα", "citation": "Lysias 1.4"},
                    {"id": "t94", "text": "πλούσιος", "citation": "Lysias 1.4"},
                    {"id": "t95", "text": "ἐκ", "citation": "Lysias 1.4"},
                    {"id": "t96", "text": "πένητος", "citation": "Lysias 1.4"},
                    {"id": "t97", "text": "γένωμαι", "citation": "Lysias 1.4"},
                    {"id": "t98", "text": ",", "citation": "Lysias 1.4"},
                    {"id": "t99", "text": "οὔτε", "citation": "Lysias 1.4"},
                    {"id": "t100", "text": "ἄλλου", "citation": "Lysias 1.4"},
                    {"id": "t101", "text": "κέρδους", "citation": "Lysias 1.4"},
                    {"id": "t102", "text": "οὐδενὸς", "citation": "Lysias 1.4"},
                    {"id": "t103", "text": "πλὴν", "citation": "Lysias 1.4"},
                    {"id": "t104", "text": "τῆς", "citation": "Lysias 1.4"},
                    {"id": "t105", "text": "κατὰ", "citation": "Lysias 1.4"},
                    {"id": "t106", "text": "τοὺς", "citation": "Lysias 1.4"},
                    {"id": "t107", "text": "νόμους", "citation": "Lysias 1.4"},
                    {"id": "t108", "text": "τιμωρίας", "citation": "Lysias 1.4"},
                    {"id": "t109", "text": ".", "citation": "Lysias 1.4"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="enclitic_classification_tis_indefinite_vs_interrogative",
        sources=[
            CitedText(citation="ex.7", text="ἀνήρ τις ἦλθεν."),
            CitedText(citation="ex.8", text="τί λέγεις;"),
        ],
        tags=["enclitic classification guard"],
        expected_sentences={
            "reasoning": (
                "τις in ex.7 is the unaccented indefinite pronoun ('a "
                "certain man'); τί in ex.8 is the accented interrogative "
                "('what?'). Both stay single, already space-delimited "
                "tokens either way -- which one a given spelling actually "
                "is (enclitic vs. not) is a tokentype classification "
                "SyntaxAnalysis makes later, not something this stage's "
                "token-boundary tokenization needs to resolve at all."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "ἀνήρ", "citation": "ex.7"},
                    {"id": "t1", "text": "τις", "citation": "ex.7"},
                    {"id": "t2", "text": "ἦλθεν", "citation": "ex.7"},
                    {"id": "t3", "text": ".", "citation": "ex.7"},
                ]},
                {"tokens": [
                    {"id": "t4", "text": "τί", "citation": "ex.8"},
                    {"id": "t5", "text": "λέγεις", "citation": "ex.8"},
                    {"id": "t6", "text": ";", "citation": "ex.8"},
                ]},
            ],
        },
    ),
]
