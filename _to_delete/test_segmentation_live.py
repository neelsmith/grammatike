"""
Live counterpart to test_segmentation_examples.py: runs the same scenarios
against the actual configured LM (via the `real_lm` fixture in
conftest.py) instead of DummyLM. Greek analogue of arsgrammatica's
test_segmentation_live.py.

This is the half of the picture DummyLM tests structurally cannot cover --
whether the LM itself actually performs the context-dependent segmentation
correctly, not just whether the code can represent a correct answer. Costs
real API calls, so it's marked `live` and skipped by default; run it with:

    pytest -m live tests/test_segmentation_live.py

I have not been able to run this against the configured model myself --
verify it once against the real thing before trusting these gaps are
actually closed, not just well-specified.

The raised-dot/colon tests below were rewritten after syntax_model.md
gained its own "Segementation into sentences" section: a raised dot or
colon-type mark is no longer NEVER a sentence boundary -- it's a candidate
boundary, resolved by syntactic coherence (see segmentation_dspy.py's own
docstring for the full rule and its two Lysias worked examples, reused
here as live scenarios).
"""

import pytest

from grammatike.models import CitedText
from grammatike.segmentation_dspy import segment_sources

pytestmark = pytest.mark.live


def _texts(sentence):
    return [t.text for t in sentence.tokens]


def test_live_peri_is_not_split(real_lm):
    sentences = segment_sources([CitedText(citation="ex.1", text="περὶ τούτου λέγει.")])
    assert _texts(sentences[0]) == ["περὶ", "τούτου", "λέγει", "."]


def test_live_eiper_splits_into_ei_plus_per(real_lm):
    sentences = segment_sources(
        [CitedText(citation="ex.2", text="εἴπερ ἀληθῆ λέγει, πείσομαι.")]
    )
    assert _texts(sentences[0]) == ["εἴ", "περ", "ἀληθῆ", "λέγει", ",", "πείσομαι", "."]


def test_live_milesian_numeral_stays_its_own_token(real_lm):
    sentences = segment_sources([CitedText(citation="ex.3", text="τῇ γʹ ἡμέρᾳ ἦλθεν.")])
    assert _texts(sentences[0]) == ["τῇ", "γʹ", "ἡμέρᾳ", "ἦλθεν", "."]


def test_live_greek_question_mark_ends_the_sentence(real_lm):
    sentences = segment_sources([CitedText(citation="ex.6", text="τίς ἦλθεν;")])
    assert len(sentences) == 1
    assert _texts(sentences[0]) == ["τίς", "ἦλθεν", ";"]


def test_live_lysias_1_2_all_three_colons_and_period_are_coherent_boundaries(real_lm):
    """syntax_model.md's own worked example for the common case: every
    raised-dot/colon-type mark in this passage divides syntactically
    coherent material, so it should segment into three sentences, one
    ending at each colon and one at the final period -- not one giant
    run-on sentence, which is what the OLD "raised dot is never a
    boundary" rule would have produced."""
    passage = (
        "καὶ ταῦτα οὐκ ἂν εἴη μόνον παρ' ὑμῖν οὕτως ἐγνωσμένα, "
        "ἀλλ' ἐν ἁπάσῃ τῇ Ἑλλάδι: περὶ τούτου γὰρ μόνου τοῦ "
        "ἀδικήματος καὶ ἐν δημοκρατίᾳ καὶ ὀλιγαρχίᾳ ἡ αὐτὴ "
        "τιμωρία τοῖς ἀσθενεστάτοις πρὸς τοὺς τὰ μέγιστα "
        "δυναμένους ἀποδέδοται, ὥστε τὸν χείριστον τῶν αὐτῶν "
        "τυγχάνειν τῷ βελτίστῳ: οὕτως, ὦ ἄνδρες, ταύτην τὴν "
        "ὕβριν ἅπαντες ἄνθρωποι δεινοτάτην ἡγοῦνται."
    )
    sentences = segment_sources([CitedText(citation="Lysias 1.2", text=passage)])
    assert len(sentences) == 3
    assert _texts(sentences[0])[-1] == ":"
    assert _texts(sentences[1])[-1] == ":"
    assert _texts(sentences[2])[-1] == "."
    assert _texts(sentences[2])[0] == "οὕτως"


def test_live_lysias_1_1_second_colon_does_not_strand_the_relative_clause(real_lm):
    """syntax_model.md's own worked example for the exception: the first
    colon (after πεπονθότες) IS a boundary, but the second (after εἴη) is
    NOT, because splitting there would strand the dependent ὅστις-relative
    clause -- which belongs with what precedes it, not what follows -- as
    if it were its own sentence. So this passage should segment into only
    TWO sentences despite having two colons in it."""
    passage = (
        "περὶ πολλοῦ ἂν ποιησαίμην, ὦ ἄνδρες, τὸ τοιούτους "
        "ὑμᾶς ἐμοὶ δικαστὰς περὶ τούτου τοῦ πράγματος "
        "γενέσθαι, οἷοίπερ ἂν ὑμῖν αὐτοῖς εἴητε "
        "τοιαῦτα πεπονθότες: εὖ γὰρ οἶδ' ὅτι, εἰ τὴν "
        "αὐτὴν γνώμην περὶ τῶν ἄλλων ἔχοιτε, "
        "ἥνπερ περὶ ὑμῶν αὐτῶν, οὐκ ἂν εἴη: ὅστις "
        "οὐκ ἐπὶ τοῖς γεγενημένοις ἀγανακτοίη, "
        "ἀλλὰ πάντες ἂν περὶ τῶν τὰ τοιαῦτα ἐπιτηδευόντων τὰς "
        "ζημίας μικρὰς ἡγοῖσθε."
    )
    sentences = segment_sources([CitedText(citation="Lysias 1.1", text=passage)])
    assert len(sentences) == 2
    assert _texts(sentences[0])[-1] == ":"
    # The second sentence must run all the way through to the final
    # period, past the second colon, WITHOUT starting a new (third)
    # sentence at "ὅστις" -- the whole point of this example.
    assert "ὅστις" in _texts(sentences[1])
    assert _texts(sentences[1])[-1] == "."
    assert len(sentences[1].tokens) > len(sentences[0].tokens)


def test_live_sentence_spans_a_citation_boundary(real_lm):
    """syntax_model.md's own note: segmentation can include sentences
    spanning citation boundaries. Lysias 1.3 ends mid-sentence at
    "ἡγοῦμαι δέ," which continues into 1.4 -- this should come back as one
    sentence whose tokens carry BOTH citations, not two separate
    sentences split at the citation-unit boundary."""
    sources = [
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
    ]
    sentences = segment_sources(sources)
    assert len(sentences) == 2
    assert _texts(sentences[0])[-1] == ":"

    spanning = sentences[1]
    assert _texts(spanning)[0] == "ἡγοῦμαι"
    assert _texts(spanning)[-1] == "."
    citations = {t.citation for t in spanning.tokens}
    assert citations == {"Lysias 1.3", "Lysias 1.4"}
