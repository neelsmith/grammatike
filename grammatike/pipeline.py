"""
grammatike: orchestrates the two-stage pipeline over Greek passages. Greek
analogue of arsgrammatica's pipeline.py.

Orchestrates the two-stage pipeline: segmentation.py's deterministic,
citation-aware sentence/token segmentation, feeding greek_syntax_dspy.py's
unmodified SyntaxAnalysis one sentence at a time.

Kept as its own module, separate from both stages, so neither stage needs
to know the other exists -- segmentation.py doesn't import
greek_syntax_dspy.py or vice versa. This is the only place that does.
"""

from typing import List, Tuple

from .models import CitedText, Sentence
from .segmentation import segment_sources
from .greek_syntax_dspy import validate
from .token_budget import analyze_with_retry


def _render_sentence_text(sentence: Sentence) -> str:
    """Reconstruct a surface string for a sentence from its tokens, to pass
    as SyntaxAnalysis's `passage` field.

    This is an approximation, not a faithful re-rendering: it puts a space
    before every token, including punctuation and enclitics (so "λόγος."
    round-trips as "λόγος ." and a proclitic/enclitic pair as two separate
    space-joined tokens). SyntaxAnalysis uses `passage` for readability
    alongside the authoritative `tokens` list, not for anything validate()
    checks, so exact fidelity isn't required -- but don't reuse this helper
    anywhere that *does* need faithful surface text without tightening it
    first.
    """
    return " ".join(tok.text for tok in sentence.tokens)


def analyze_sources(sources: List[CitedText]) -> Tuple[List[Sentence], list]:
    """Segment `sources` into citation-aware sentences, run each sentence's
    tokens through SyntaxAnalysis, and validate each result.

    Returns (sentences, results): results[i] is the SyntaxAnalysis result
    for sentences[i], same order, one entry per sentence.

    Each sentence's SyntaxAnalysis call goes through
    `token_budget.analyze_with_retry()` rather than calling `analyze()`
    directly, so a sentence whose analysis needs more output than a fixed
    `max_tokens` would allow (a long or deeply subordinated sentence) gets
    an estimated, appropriately-sized budget up front, and a retry with a
    larger one if it still comes back truncated -- see token_budget.py's
    module docstring for the full design.
    """
    sentences = segment_sources(sources)

    results = []
    for sentence in sentences:
        result = analyze_with_retry(passage=_render_sentence_text(sentence), tokens=sentence.tokens)

        problems = validate(sentence.tokens, result)
        if problems:
            first_id = sentence.tokens[0].id if sentence.tokens else "?"
            print(f"Validation warnings (sentence starting at {first_id}):")
            for p in problems:
                print(f"  - {p}")

        results.append(result)

    return sentences, results


def combined_tokengraph(results) -> list:
    """Concatenate every sentence result's tokengraph, in order, into one
    flat list spanning the whole input -- since token ids are global,
    tokengraph_to_mermaid() (mermaid.py) needs no changes at all to render
    this as one diagram for a multi-sentence, multi-citation passage."""
    combined = []
    for result in results:
        combined.extend(result.tokengraph)
    return combined


def analyze_passage(passage: str, citation: str = "") -> Tuple[List[Sentence], list]:
    """Convenience wrapper for the common case of a single string rather
    than a list of citation-labeled CitedText sources -- kept here so
    existing callers (syntaxer_main.py, the marimo notebook) have a
    one-string entry point rather than needing to build a CitedText list
    themselves for the ordinary case of one passage from one source.

    Wraps `passage` as one CitedText (using `citation` if given, else an
    empty string -- fine for callers that don't track citations) and runs
    it through analyze_sources(). Returns (sentences, results) -- the exact
    same shape analyze_sources() returns, one entry per sentence
    segmentation finds in `passage`, in order.

    `passage` may contain any number of sentences: each is segmented and
    analyzed successively, same as if you'd called analyze_sources() with
    one CitedText yourself. (An earlier version of this function raised
    ValueError on multi-sentence input and returned a single (tokens,
    result) pair for exactly one sentence; callers written against that
    contract need to change to unpack (sentences, results) and iterate.)

    Example: `analyze_passage("...", citation="urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1")`
    for a passage of Plato, or `analyze_passage("...", citation="Lysias 1.1")`
    for a citation scheme keyed by author/work/section instead of a URN.
    """
    return analyze_sources([CitedText(citation=citation, text=passage)])
