"""
grammatike: LLM-driven sentence segmentation and tokenization for Ancient
Greek passages. Greek analogue of arsgrammatica's segmentation_dspy.py.

DSPy program that segments a sequence of citation-labeled Greek source
units into sentences, and each sentence into tokens -- replacing a
deterministic, regex-based segmentation with an LLM-driven one, per
syntax_model.md's "Tokenization" section (context-dependent enclitic
recognition, including splitting an enclitic that is orthographically
fused onto a preceding word), while also tracking which citation unit
each token came from.

Input is `sources: List[CitedText]` rather than a single passage string
specifically so that a sentence spanning more than one citation unit (a
verse sentence running across two lines, say) is representable: sentence
boundaries do not need to respect CitedText boundaries, but every token
still records the citation it came from via Token.citation.

This is a separate stage from SyntaxAnalysis (greek_syntax_dspy.py) on
purpose: SegmentPassage's output (List[Sentence]) still feeds
SyntaxAnalysis as `tokens: List[Token]` per sentence, unchanged -- adding
citation tracking here required no changes to SyntaxAnalysis at all, since
Token.citation just rides along and SyntaxAnalysis never needs to look at
it.

Run this file directly for a quick smoke test against the configured LM:
    python segmentation_dspy.py
"""

from typing import List

import dspy

from .models import CitedText, Sentence


class SegmentPassage(dspy.Signature):
    """Segment a sequence of citation-labeled Ancient Greek source units into
    sentences, and each sentence into tokens, following syntax_model.md's
    tokenization scheme.

    `sources` is given in reading order; treat its units' text as one
    continuous passage for sentence-splitting purposes -- a sentence may
    start in one unit's text and finish in the next one's, and often will
    in continuous verse or prose. Every token you produce must carry the
    `citation` of whichever `sources` unit its surface text came from, even
    for a sentence that spans more than one unit.

    - Split into sentences at sentence-ending punctuation: the period (.),
      the Greek question mark (a semicolon-shaped mark, U+037E ";" -- some
      editions instead print an ASCII "?" for the same function; treat
      either as ending the question), and "!" in editions that use it. The
      raised dot / άνω τελεία ("·") functions as a semicolon or colon
      WITHIN a sentence -- e.g. separating coordinate clauses, the way the
      commas do in "ἐγὼ γὰρ οὐδὲν δέομαι λόγων, ἀλλὰ τὸ ἔργον φανερὸν
      γενέσθαι, εἴπερ οὕτως ἔχει." -- and is NEVER by itself a sentence
      boundary, unlike a period or the Greek question mark. (Direct
      quotations may be marked with guillemets («»), straight double
      quotes, or not marked with any quotation punctuation at all and
      signalled only by a verb of speaking such as ἔφη -- none of that
      affects sentence-splitting.)

    - Within each sentence, segment tokens as: lexical, enclitic,
      punctuation, or numeral (a number written numerically, e.g. in
      Milesian/acrophonic notation, as opposed to a number spelled out as
      an ordinary word -- e.g. in "Ἀτρεΐδα δὲ μάλιστα δύω", δύω is lexical,
      not numeral). There is no praenomen or other abbreviation category in
      this scheme -- that is specific to Latin, and Ancient Greek has no
      equivalent construction to special-case.

    - Enclitic recognition depends on context, not just on which word it
      is -- most Greek enclitics are already written as their own
      space-delimited word, so tagging one is a classification decision,
      not a splitting one. Common enclitics include the connectives τε and
      γε, the indefinite pronoun/adjective τις/τι, the unemphatic
      personal-pronoun forms με, μου, μοι, σε, σου, σοι, and most
      present-indicative forms of φημί (φημί, φησί, φαμέν, φατέ, φασί, but
      not the accented φής). These lean prosodically on the preceding word
      (and can shift or add an accent to it) without changing their own
      spelling. Only classify a word as enclitic when context supports that
      reading -- e.g. τις/τι is enclitic only as the indefinite
      ("someone"/"something"), never as the accented interrogative τίς/τί
      ("who?"/"what?"); φησί and its sibling forms are enclitic only as
      unemphatic "says", not as an emphatic or sentence-initial claim.
      Never reclassify a word as enclitic merely because it shares a
      spelling with one.

      Separately, some enclitics are written fused with no space onto the
      preceding word, and those DO need splitting into two tokens, the way
      Latin's -que/-ve/-ne are split. The enclitic περ productively attaches
      this way: syntax_model.md's own example, περ in "ἐγὼ γὰρ οὐδὲν δέομαι
      λόγων, ἀλλὰ τὸ ἔργον φανερὸν γενέσθαι, εἴπερ οὕτως ἔχει", splits
      εἴπερ into εἴ (lexical, "if") + περ (enclitic). Other -περ compounds
      split the same way when the remainder is a real word on its own AND
      context supports that reading -- e.g. ὥσπερ -> ὥς + περ, ὅσπερ ->
      ὅς + περ, καίπερ -> καί + περ, ἐπείπερ -> ἐπεί + περ. Never split off
      περ (or any other enclitic) when there is no genuine independent
      remainder -- the whole spelling is then a single lexical word.

    - Assign token ids sequentially across the WHOLE input, in reading
      order: t0, t1, t2, .... Do not restart numbering at each sentence or
      at each source unit. Every token, across every sentence and every
      source unit, has a unique id, and running this on the same `sources`
      again must produce the same ids for the same tokens.
    """

    sources: List[CitedText] = dspy.InputField(
        desc="Citation-labeled source units, in reading order, to segment as one continuous passage."
    )
    sentences: List[Sentence] = dspy.OutputField(
        desc="The sentences found across all of `sources`, in order. Token ids are global (see instructions); each token's `citation` names the source unit it came from."
    )


segment = dspy.ChainOfThought(SegmentPassage)


def segment_sources(sources: List[CitedText]) -> List[Sentence]:
    """Run the segmentation stage and return its sentences."""
    result = segment(sources=sources)
    return result.sentences
