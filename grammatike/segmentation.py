"""
grammatike: deterministic sentence segmentation and tokenization for Ancient
Greek passages. Greek analogue of arsgrammatica's segmentation module.

Replaces an earlier LLM-driven implementation (formerly `segmentation_dspy.py`,
`SegmentPassage`) with a purely mechanical one, following syntax_model.md's
own simplification of its "Segmentation into sentences" section: sentence
boundaries are now settled by punctuation alone (a period, or an
"interrogative" -- either the Greek question mark U+037E or its valid
Unicode decomposition, the ASCII/Latin semicolon) -- syntax_model.md now
explicitly puts the burden of consistent sentence-ending punctuation on the
text's editor, rather than asking a model to judge syntactic coherence for
an ambiguous raised-dot/colon-type mark the way an earlier version of this
section did. That judgment call is gone from the scheme entirely: a raised
dot ("middle dot"-shaped), an ASCII colon (":"), or a comma (",") never end
a sentence now, no matter what surrounds them -- they're ordinary internal
punctuation, like any other punctuation token.

Because sentence-splitting no longer needs any contextual judgment, this
whole stage can run with no LM call at all, deterministically, from
`sources: List[CitedText]` straight to `List[Sentence]`.

TWO PASSES, IN THIS ORDER -- string-splitting first, tokenizing second:

1. Split each source's own text into sentence-STRINGS with a regex cutting
   right after a sentence terminator (see _SENTENCE_BOUNDARY_RE): purely
   mechanical, no tokenization involved yet. A source whose text doesn't
   end in a terminator leaves its last piece "open" -- carried forward and
   glued (as a still citation-tagged fragment, not literally concatenated
   text; see _group_into_sentences()'s own docstring for why) onto the next
   source's first piece, so a sentence can start in one `sources` unit and
   finish in the next.
2. Tokenize each sentence's string(s) -- splitting off punctuation and any
   fused enclitic (see below) -- assigning ids sequentially (t0, t1, ...)
   across the WHOLE input as they're produced. Ids are NOT restarted per
   sentence: this pass keeps one running counter across every sentence in
   this call, since a Sentence is documented (models.py) as a contiguous
   slice of the passage's global id sequence, not an independently-numbered
   unit.

This is a separate stage from SyntaxAnalysis (greek_syntax_dspy.py) on
purpose: this stage's output (List[Sentence]) still feeds SyntaxAnalysis as
`tokens: List[Token]` per sentence, unchanged.

Tokenization here only decides how many tokens result and where their
surface-text boundaries fall -- NOT their tokentype (lexical/enclitic/
punctuation/numeral): syntax_model.md's "Tokenization" section describes
that classification, but Token (this stage's own output model) has no
tokentype field at all -- SyntaxAnalysis assigns tokentype independently,
per token, as part of its own (still LM-driven) output. So this stage does
not need to decide, for example, whether a word is being used as an
enclitic or an accented, non-enclitic homograph -- only where token
boundaries fall. Two things this stage's tokenization DOES have to decide,
because they affect how many tokens exist rather than merely how one is
classified:

- Splitting off punctuation. Any Unicode punctuation character (general
  category starting with "P") attached to a word, leading or trailing, gets
  its own token -- e.g. "ἀνέῳξεν." splits into "ἀνέῳξεν" + ".". The one
  deliberate exception is an elision apostrophe (ASCII "'" U+0027, or the
  Unicode right single quotation mark U+2019, sometimes used instead) --
  e.g. "παρ'", "ἀλλ'", "οἶδ'" -- which stays fused to its word rather than
  becoming its own token, since it marks an elided vowel as part of that
  word's own spelling, not a separate mark. (A Milesian numeral's keraia,
  e.g. the raised prime mark in "γʹ", needs no special-casing at all here:
  Unicode gives it its own non-punctuation category, GREEK NUMERAL SIGN /
  MODIFIER LETTER PRIME, so the generic punctuation scan already leaves it
  fused to its digit letter.)

- Splitting a fused enclitic. Most Greek enclitics are already their own
  space-delimited word (τε, γε, τις/τι, με/μου/μοι/σε/σου/σοι, unaccented
  φημί forms) -- nothing to split there. But the enclitic περ productively
  fuses onto a preceding word with no space at all (εἴπερ, ὥσπερ, ὅσπερ,
  καίπερ, ἐπείπερ, and, inflected, οἷοίπερ/ἥνπερ), and that DOES need
  splitting into two tokens the way Latin's -que/-ve/-ne is split --
  syntax_model.md's own example: εἴπερ -> εἴ (lexical) + περ (enclitic).
  This module resolves that with a small, explicit lookup table
  (_PER_COMPOUNDS) of the compounds actually documented or exercised in
  this scheme's fixtures, rather than a blind "strip a trailing περ"
  rule -- guarding against the false positive of a word that merely
  happens to end in the letters περ with no genuine independent remainder
  (περὶ, "concerning", is NOT περ + ί: stripping περ would leave a bare
  accented ί that is not a real word on its own). Extend the table as new
  compounds turn up in real text; don't switch to a suffix-strip rule
  without also re-deriving a guard against exactly this false positive.

Run this file directly for a quick smoke test (no LM, no .env needed):
    python segmentation.py
"""

import re
import unicodedata
from typing import List, Tuple

from .models import CitedText, Sentence, Token

# Marks that ALWAYS end a sentence, per syntax_model.md's "Segmentation into
# sentences" section: the period, and the "interrogative" -- either the
# Greek question mark itself (U+037E) or its valid Unicode decomposition,
# the ASCII/Latin semicolon (U+003B). No other mark (comma, raised dot
# U+0387, ASCII colon U+003A, or its Greek near-doubles) ends a sentence --
# editors of texts are responsible for consistent punctuation of sentence
# boundaries, per that section's own closing sentence.
#
# Built from chr(0x...) rather than pasted literal characters -- exactly
# because U+037E and U+003B are visually indistinguishable in most fonts
# (the confusability this whole module's docstring warns about), so nobody
# reading or editing this file has to tell them apart by eye, or trust that
# a copy-paste preserved the right one.
_PERIOD = "."
_GREEK_QUESTION_MARK = chr(0x037E)
_SEMICOLON = chr(0x003B)
_SENTENCE_TERMINATORS = frozenset({_PERIOD, _GREEK_QUESTION_MARK, _SEMICOLON})

# Splits a single source's raw text into sentence-strings: a boundary falls
# right after a terminator character, but only where it's actually followed
# by whitespace (or nothing splits there at all, e.g. a terminator at the
# very end of a source's text with nothing trailing it). Comma, raised dot,
# and colon are absent from this class entirely -- they never introduce a
# boundary, no matter what surrounds them.
_SENTENCE_BOUNDARY_RE = re.compile(
    "(?<=[" + re.escape("".join(sorted(_SENTENCE_TERMINATORS))) + r"])\s+"
)

# Punctuation marks that mark an elision (a dropped final vowel) rather than
# a real punctuation boundary -- these stay fused to their word instead of
# becoming their own token. Both are attested spellings for the same mark:
# a plain ASCII apostrophe is what editions typically use in practice; the
# Unicode right single quotation mark sometimes stands in for it instead.
_ELISION_MARKS = frozenset({
    "'",  # ' APOSTROPHE
    "’",  # RIGHT SINGLE QUOTATION MARK
})

# Known -περ compounds where the enclitic περ is fused onto a preceding word
# with no space, and needs splitting into two tokens (see this module's own
# docstring for the false-positive this guards against). Each entry's
# remainder is independently a genuine word attested in syntax_model.md's
# own worked examples or this package's segmentation fixtures.
_PER_COMPOUNDS = {
    "εἴπερ": ("εἴ", "περ"),
    "ὥσπερ": ("ὥς", "περ"),
    "ὅσπερ": ("ὅς", "περ"),
    "καίπερ": ("καί", "περ"),
    "ἐπείπερ": ("ἐπεί", "περ"),
    "οἷοίπερ": ("οἷοί", "περ"),
    "ἥνπερ": ("ἥν", "περ"),
}


def _split_into_sentence_strings(text: str) -> List[str]:
    """Split one source's own raw text into sentence-strings, purely by
    regex (_SENTENCE_BOUNDARY_RE) -- no tokenization at all yet. Every piece
    except possibly the last ends in a sentence terminator, by construction
    (that's exactly where the regex cuts); the last piece ends in one only
    if `text` itself does -- otherwise it's an "open" fragment that
    _group_into_sentences() carries forward into the next source."""
    text = text.strip()
    if not text:
        return []
    return _SENTENCE_BOUNDARY_RE.split(text)


def _group_into_sentences(sources: List[CitedText]) -> List[List[Tuple[str, str]]]:
    """Pass 1: turn `sources` into a list of sentences, each represented as
    an ordered list of (citation, sentence_string) fragments -- almost
    always just one fragment, since almost every sentence lives entirely
    inside one source. The exception is a sentence that starts in one
    source and finishes in the next (that source's own text didn't end in
    a terminator): its "string representation" is genuinely made of two (or
    more) differently-cited pieces, so it stays a list of fragments rather
    than a single flattened string -- collapsing it into one string here
    would throw away exactly the citation attribution Token.citation (and
    every test that checks it) depends on. Pass 2 (segment_sources() below)
    tokenizes each fragment in turn, in order, which is what actually
    produces the flat per-token citations callers see.
    """
    sentences: List[List[Tuple[str, str]]] = []
    pending: List[Tuple[str, str]] = []

    for source in sources:
        for piece in _split_into_sentence_strings(source.text):
            pending.append((source.citation, piece))
            if piece[-1] in _SENTENCE_TERMINATORS:
                sentences.append(pending)
                pending = []

    if pending:
        sentences.append(pending)

    return sentences


def _is_splittable_punctuation(ch: str) -> bool:
    """True for a Unicode punctuation character that should become its own
    token -- i.e. any general category starting with "P", EXCEPT an elision
    mark (see _ELISION_MARKS), which stays fused to its word instead."""
    return ch not in _ELISION_MARKS and unicodedata.category(ch).startswith("P")


def _split_word(word: str) -> List[str]:
    """Split one whitespace-delimited word into the surface-text pieces it
    becomes as separate tokens: leading punctuation (each character its own
    piece), the word's own core -- itself split into two pieces if it's a
    known fused -περ compound (_PER_COMPOUNDS) -- and trailing punctuation
    (each character its own piece). An elision mark (_ELISION_MARKS) is
    never treated as punctuation to split off; it stays part of the core."""
    i = 0
    while i < len(word) and _is_splittable_punctuation(word[i]):
        i += 1

    j = len(word)
    while j > i and _is_splittable_punctuation(word[j - 1]):
        j -= 1

    leading = list(word[:i])
    core = word[i:j]
    trailing = list(word[j:])

    pieces = leading
    if core:
        pieces = pieces + list(_PER_COMPOUNDS.get(core, (core,)))
    pieces = pieces + trailing
    return pieces


def segment_sources(sources: List[CitedText]) -> List[Sentence]:
    """Deterministically segment `sources` into sentences, in two passes:

    1. _group_into_sentences() splits the input into sentence-strings by
       regex alone (a period or interrogative always ends one; nothing else
       does) -- no tokenization involved yet.
    2. Tokenize each sentence's string(s) in turn (see this module's own
       docstring for the two tokenization decisions this involves), handing
       out ids sequentially (t0, t1, ...) from ONE counter that runs across
       the whole input -- never restarted per sentence, so a Sentence stays
       a contiguous slice of the passage's global id sequence.

    Running this on the same `sources` again always produces the same ids
    for the same tokens -- there is nothing non-deterministic left in this
    stage at all.
    """
    next_id = 0
    sentences: List[Sentence] = []

    for fragments in _group_into_sentences(sources):
        tokens: List[Token] = []
        for citation, text in fragments:
            for word in text.split():
                for piece in _split_word(word):
                    tokens.append(Token(id=f"t{next_id}", text=piece, citation=citation))
                    next_id += 1
        sentences.append(Sentence(tokens=tokens))

    return sentences


if __name__ == "__main__":
    demo = segment_sources([
        CitedText(
            citation="Lysias 1.1",
            text="ἡ ναῦς ἀπόλλυται. τήν θύραν ἀνέῳξεν, εἴπερ ἀληθῆ λέγει.",
        ),
    ])
    for i, sentence in enumerate(demo):
        print(f"{i + 1}. " + " ".join(t.text for t in sentence.tokens))
