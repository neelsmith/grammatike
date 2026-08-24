"""
Gold-annotated examples for the segmentation stage (segmentation_dspy.py) --
the segmentation-stage counterpart to fixtures/gold_examples.py. Greek
analogue of arsgrammatica's fixtures/segmentation_examples.py.

Each SegmentationExample pairs a list of CitedText sources with the
sentences/tokens a *correctly working* context-aware segmenter should
produce. Like fixtures/gold_examples.py, these are structural fixtures:
running them through DummyLM proves the code (models, segment_sources()
plumbing) handles a well-formed, correct answer properly. It does NOT prove
a real LM produces that answer -- see test_segmentation_live.py for that
half of the picture, which hits the actual configured LM instead of
DummyLM and is skipped by default (opt in with `pytest -m live`).

These four examples are Greek-appropriate replacements for arsgrammatica's
three (which exercised Latin-only enclitic false-positives, the
context-dependent -ne split, and praenomen/abbreviation recognition --
none of which apply to Greek; see segmentation_dspy.py's own docstring:
"There is no praenomen or other abbreviation category in this scheme").
Instead these cover: (1) the fused-enclitic split (περ off of εἴπερ) versus
the false-positive guard against splitting a word that merely happens to
contain the same letters (περί), (2) numeral vs. spelled-out-word
tokenization, (3) the raised dot NOT being a sentence boundary versus the
Greek question mark (U+037E) being one, and (4) the classification-not-
splitting distinction between the enclitic indefinite τις and the accented,
non-enclitic interrogative τίς -- all directly from segmentation_dspy.py's
own docstring.
"""

from dataclasses import dataclass
from typing import Any

from grammatike.models import CitedText


@dataclass
class SegmentationExample:
    slug: str
    sources: list[CitedText]
    tags: list[str]
    canned_sentences: dict[str, Any]


SEGMENTATION_EXAMPLES = [
    SegmentationExample(
        slug="fused_enclitic_split_eiper_vs_false_positive_peri",
        sources=[
            CitedText(citation="ex.1", text="περὶ τούτου λέγει."),
            CitedText(citation="ex.2", text="εἴπερ ἀληθῆ λέγει, πείσομαι."),
        ],
        tags=["enclitic split guard", "enclitic"],
        canned_sentences={
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
        canned_sentences={
            "reasoning": (
                "γʹ in ex.3 is a number written numerically, in Milesian "
                "notation ('3rd') -- tokentype numeral. δύο in ex.4 is the "
                "same kind of quantity spelled out as an ordinary word "
                "('two'), so it stays lexical even though it is semantically "
                "a number, per syntax_model.md's own δύω example."
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
        slug="raised_dot_not_boundary_vs_greek_question_mark",
        sources=[
            CitedText(citation="ex.5", text="ἐγὼ μὲν ἔμεινα· σὺ δὲ ἀπῆλθες."),
            CitedText(citation="ex.6", text="τίς ἦλθεν;"),
        ],
        tags=["sentence boundary", "punctuation"],
        canned_sentences={
            "reasoning": (
                "The raised dot (·) in ex.5 separates two coordinate "
                "clauses within a single sentence -- it functions like a "
                "semicolon, not a sentence boundary, so ex.5's whole text "
                "is one sentence ending only at its final period. The Greek "
                "question mark (U+037E ';') in ex.6 DOES end a sentence, "
                "unlike the raised dot, despite looking similar to an "
                "ASCII semicolon."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "ἐγὼ", "citation": "ex.5"},
                    {"id": "t1", "text": "μὲν", "citation": "ex.5"},
                    {"id": "t2", "text": "ἔμεινα", "citation": "ex.5"},
                    {"id": "t3", "text": "·", "citation": "ex.5"},
                    {"id": "t4", "text": "σὺ", "citation": "ex.5"},
                    {"id": "t5", "text": "δὲ", "citation": "ex.5"},
                    {"id": "t6", "text": "ἀπῆλθες", "citation": "ex.5"},
                    {"id": "t7", "text": ".", "citation": "ex.5"},
                ]},
                {"tokens": [
                    {"id": "t8", "text": "τίς", "citation": "ex.6"},
                    {"id": "t9", "text": "ἦλθεν", "citation": "ex.6"},
                    {"id": "t10", "text": ";", "citation": "ex.6"},
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
        canned_sentences={
            "reasoning": (
                "τις in ex.7 is the unaccented indefinite pronoun ('a "
                "certain man') -- genuinely enclitic, a classification "
                "decision rather than a splitting one, per "
                "segmentation_dspy.py's own rule. τί in ex.8 is the "
                "accented interrogative ('what?'), never enclitic despite "
                "sharing a root with τις/τι."
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
                    {"id": "t6", "text": ";", "citation": "ex.8"},
                ]},
            ],
        },
    ),
]
