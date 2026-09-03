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
boundaries fall. Four things this stage's tokenization DOES have to
decide, because they affect how many tokens exist rather than merely how
one is classified:

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

- Splitting a word written in crasis. syntax_model.md's "Two special
  notes" section: κἀγώ (crasis of καὶ + ἐγώ) must be entered as two
  lexical tokens, κ and ἀγώ -- lemma assignment (κ -> καί, ἀγώ -> ἐγώ) is
  SyntaxAnalysis's job later, not this stage's; a Token here has no lemma
  field at all. Architecturally this is the same shape as the -περ
  enclitic split above -- an explicit lookup table (_CRASIS_COMPOUNDS),
  not a general rule, keyed by the whole fused word -- just for a
  different phenomenon and kept in its own table rather than folded into
  _PER_COMPOUNDS, since crasis and enclitic-fusion are different things
  that happen to want the same "split one already-delimited word into a
  documented tuple of pieces" mechanism. Unlike the -περ table, this one
  isn't limited to single one-off idioms: it also seeds the common
  masculine/neuter paradigm of ὁ αὐτός ("the same"), crased, since that
  recurs constantly in Ionic prose rather than turning up once. Because a
  lookup here is an exact string match, an OXYTONE crasis form needs both
  its accent-states registered separately (e.g. "ὡυτός" isolated/before
  punctuation vs. "ὡυτὸς" mid-clause, where the regular acute-to-grave
  shift applies) -- a form accented with circumflex on its final syllable
  never shifts to grave, so it needs only the one spelling. See
  _CRASIS_COMPOUNDS's own comment for the full paradigm and this same
  caveat applied to κἀγώ itself.

- Merging ὅ τι into one token. The other of syntax_model.md's "Two special
  notes": editors regularly write the neuter nom./acc. singular of ὅστις
  as two words, "ὅ τι", specifically to keep it visually distinct from the
  one-word conjunction ὅτι ("that") -- but "ὅ τι" must be tokenized as a
  SINGLE lexical token, not two. This is the mirror image of the other two
  decisions above: those split one already-delimited word into more than
  one token; this instead MERGES two already-space-delimited words into
  one token. It needs its own pre-pass (_merge_multiword_tokens, driven by
  the explicit _MULTIWORD_TOKENS table) that runs on each fragment's
  word list before the per-word splitting logic ever sees it -- once "ὅ"
  and "τι" are merged into the single string "ὅ τι", the existing
  _split_word() logic handles it with no further changes: a space is
  Unicode category Zs, never "P", so it's never treated as splittable
  punctuation and stays part of the merged word's core.

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

# Words written in crasis that must split into two lexical tokens (see this
# module's own docstring). Seeded with only syntax_model.md's own documented
# example -- extend as new crasis forms turn up in real text, matching
# _PER_COMPOUNDS's own "documented example only" convention above.
_CRASIS_COMPOUNDS = {
    "κἀγώ": ("κ", "ἀγώ"),

    # ὁ αὐτός ("the same"), crased -- pervasive in Ionic/Herodotean prose,
    # unlike κἀγώ's one-off idiom, so seeded here as a small paradigm rather
    # than one entry at a time. Editors spell the crasis "ὡυτός" rather than
    # "αὑτός" specifically to keep it visually distinct from the reflexive
    # pronoun αὑτός ("himself") -- the same visual-confusion concern that
    # motivates "ὅ τι" vs ὅτι in _MULTIWORD_TOKENS below. Masculine/neuter
    # forms only: feminine crasis of ἡ αὐτή is markedly rarer and less
    # consistently attested, so it's left out here rather than guessed at --
    # add it if real text turns it up.
    #
    # Because a dict key here is matched literally, both accent-states of
    # every OXYTONE form need their own entry: an acute on a word's final
    # syllable regularly shifts to grave when the word isn't followed by
    # punctuation, so "ὡυτός" (isolated/before punctuation) and "ὡυτὸς"
    # (mid-clause) are two different strings that must both resolve to the
    # same split. A form accented with circumflex on its final syllable
    # (ὡυτοῦ, ὡυτῷ, ὡυτῶν, ὡυτοῖσι) never shifts to grave, so those need
    # only the one spelling. This acute/grave duplication applies in
    # principle to every entry in this table (κἀγώ included, if it ever
    # turns up non-phrase-finally as κἀγὼ) -- it's just spelled out in full
    # here because this paradigm is large enough to make the pattern clear.
    "ὡυτός": ("ὡ", "υτός"),      # ὁ αὐτός, nom. sg. masc.
    "ὡυτὸς": ("ὡ", "υτὸς"),      # -- grave, mid-clause
    "ὡυτοῦ": ("ὡ", "υτοῦ"),      # τοῦ αὐτοῦ, gen. sg. masc./neut. (circumflex, no grave form)
    "ὡυτῷ": ("ὡ", "υτῷ"),        # τῷ αὐτῷ, dat. sg. masc./neut. (circumflex, no grave form)
    "ὡυτόν": ("ὡ", "υτόν"),      # τὸν αὐτόν, acc. sg. masc.
    "ὡυτὸν": ("ὡ", "υτὸν"),      # -- grave, mid-clause
    "τὠυτό": ("τὠ", "υτό"),      # τὸ αὐτό, nom./acc. sg. neut.
    "τὠυτὸ": ("τὠ", "υτὸ"),      # -- grave, mid-clause
    "ὡυτοί": ("ὡ", "υτοί"),      # οἱ αὐτοί, nom. pl. masc.
    "ὡυτοὶ": ("ὡ", "υτοὶ"),      # -- grave, mid-clause
    "ὡυτῶν": ("ὡ", "υτῶν"),      # τῶν αὐτῶν, gen. pl. (circumflex, no grave form)
    "ὡυτοῖσι": ("ὡ", "υτοῖσι"),  # τοῖσι αὐτοῖσι, dat. pl. (Ionic -οισι; circumflex, no grave form)
    "ὡυτούς": ("ὡ", "υτούς"),    # τοὺς αὐτούς, acc. pl. masc.
    "ὡυτοὺς": ("ὡ", "υτοὺς"),    # -- grave, mid-clause
}

# Pairs of adjacent, already-space-delimited words that must MERGE into one
# lexical token instead (see this module's own docstring): the neuter
# nom./acc. singular of ὅστις, conventionally written "ὅ τι" to keep it
# distinct from the one-word conjunction ὅτι. Compared against each word's
# own punctuation-stripped core (see _strip_punctuation), so "ὅ" and "τι,"
# (with trailing punctuation already attached) still match this entry.
_MULTIWORD_TOKENS = frozenset({
    ("ὅ", "τι"),
})


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


def _strip_punctuation(word: str) -> Tuple[List[str], str, List[str]]:
    """Split one whitespace-delimited word into (leading, core, trailing):
    leading/trailing lists of individual splittable-punctuation characters
    (each its own future token), and the core substring left between them.
    An elision mark (_ELISION_MARKS) is never treated as splittable
    punctuation; it stays part of the core. Shared by _split_word() (which
    goes on to resolve the core into one or more pieces) and
    _merge_multiword_tokens() (which only needs each word's own core, to
    compare its punctuation-stripped spelling against _MULTIWORD_TOKENS)."""
    i = 0
    while i < len(word) and _is_splittable_punctuation(word[i]):
        i += 1

    j = len(word)
    while j > i and _is_splittable_punctuation(word[j - 1]):
        j -= 1

    return list(word[:i]), word[i:j], list(word[j:])


def _split_word(word: str) -> List[str]:
    """Split one whitespace-delimited word into the surface-text pieces it
    becomes as separate tokens: leading punctuation (each character its own
    piece), the word's own core -- itself split into two pieces if it's a
    known fused -περ compound (_PER_COMPOUNDS) or crasis compound
    (_CRASIS_COMPOUNDS) -- and trailing punctuation (each character its own
    piece). An elision mark (_ELISION_MARKS) is never treated as
    punctuation to split off; it stays part of the core.

    Also handles a merged multiword core (see _merge_multiword_tokens):
    "ὅ τι" arrives here already joined into one string by the caller's
    pre-pass, and since a space is never splittable punctuation, it simply
    stays fused as a single core -- neither lookup table matches it, so it
    falls through to the default (core,) and becomes one token, exactly as
    intended."""
    leading, core, trailing = _strip_punctuation(word)

    pieces = leading
    if core:
        pieces = pieces + list(
            _PER_COMPOUNDS.get(core) or _CRASIS_COMPOUNDS.get(core) or (core,)
        )
    pieces = pieces + trailing
    return pieces


def _merge_multiword_tokens(words: List[str]) -> List[str]:
    """Pre-pass over one fragment's whitespace-split word list: scans
    adjacent pairs, and whenever their punctuation-stripped cores match a
    _MULTIWORD_TOKENS entry (e.g. ("ὅ", "τι")), merges the two RAW words
    (punctuation and all) into one space-joined string and advances past
    both; otherwise keeps the word as its own unit and advances by one.

    Run this before _split_word() ever sees the word list -- once "ὅ" and
    "τι," are merged into "ὅ τι,", the existing _split_word() logic handles
    the rest with no changes at all (see its own docstring)."""
    merged: List[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            _, first_core, _ = _strip_punctuation(words[i])
            _, second_core, _ = _strip_punctuation(words[i + 1])
            if (first_core, second_core) in _MULTIWORD_TOKENS:
                merged.append(f"{words[i]} {words[i + 1]}")
                i += 2
                continue
        merged.append(words[i])
        i += 1
    return merged


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
            for word in _merge_multiword_tokens(text.split()):
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
