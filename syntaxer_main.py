"""
A runnable script to run the syntaxer module.

Greek analogue of arsgrammatica's syntaxer_main.py.
"""

# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------
import argparse
from pathlib import Path
import os

import dspy
from dotenv import load_dotenv


# override=True: without this, load_dotenv() leaves any of these vars
# (MODEL, API_BASE, API_KEY, ...) that are ALREADY set in the process's
# environment untouched, instead of applying .env's own value -- and a
# fresh `python syntaxer_main.py` process still inherits whatever the
# launching shell itself has exported (a leftover `export MODEL=...` from
# an earlier test, a value set in .zshrc/.bashrc, etc.), even though it's
# a brand-new process every run. A genuinely stale MODEL this way is a
# very plausible cause of a "prompt truncated for no reason, even on a
# short passage" symptom, since token_budget.py's DEFAULT_CEILING is only
# ever a reasonable ceiling for the model you actually think you're
# calling.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


def _env(name: str, fallback_name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    value = os.getenv(fallback_name)
    if value:
        return value
    return default

def _configure_lm():
    api_base = _env("API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm")
    model = _env("MODEL", "MODEL", "litellm_proxy/anthropic/Claude Opus 5")

    # Distinguish "API_KEY isn't in .env at all" (a likely oversight -- keep
    # raising) from "API_KEY= is there but deliberately empty" (fine for a
    # local, unauthenticated model like Ollama -- see model_bakeoff.py's own
    # "ollama: no API key needed" comment for the same convention). _env()'s
    # own truthiness check can't tell these apart (both look like "falsy"),
    # so this checks os.environ directly instead.
    if "API_KEY" not in os.environ:
        raise RuntimeError(
            "Missing API key. Set API_KEY in your .env file -- an empty "
            "value (API_KEY=) is fine for a local model that doesn't need "
            "one, e.g. Ollama; this only checks that the line exists at all."
        )
    api_key = os.environ["API_KEY"]

    # Only pass api_key through when it's actually non-empty. dspy.LM/litellm
    # don't need one at all for a local Ollama daemon -- passing api_key=""
    # explicitly is unnecessary and, depending on the provider, can behave
    # differently than omitting it outright.
    # An explicit numeric baseline, not None (dspy.LM's own default): every
    # SyntaxAnalysis call here goes through token_budget.analyze_with_retry(),
    # which overrides max_tokens per call anyway, so this baseline is never
    # actually what caps a real analysis -- but leaving it at None means
    # dspy's own truncation warning (dspy.LM._check_truncation) reports
    # "max_tokens=None" for any OTHER call made against this same configured
    # LM that doesn't go through analyze_with_retry() (there's none in this
    # script today, but a notebook or a future script sharing this LM
    # object could add one), which reads as much more alarming/confusing
    # than the truth.
    lm_kwargs = dict(model=model, api_base=api_base, max_tokens=DEFAULT_CEILING)
    if api_key:
        lm_kwargs["api_key"] = api_key

    # Anthropic prompt caching: SyntaxAnalysis's system message (its own
    # instructions plus the TokenAnalysis/VerbalExpression field
    # descriptions) is long and byte-identical on every single call -- only
    # the per-sentence user message actually changes. Marking it with an
    # ephemeral cache_control breakpoint lets a repeat call within
    # Anthropic's cache TTL reuse that whole block at a fraction of its
    # normal input-token price instead of paying full price every time.
    # litellm (which dspy.LM forwards arbitrary kwargs to) applies
    # cache_control_injection_points provider-agnostically based solely on
    # the param's presence, so this is gated on the model actually being
    # Anthropic-routed -- a MODEL override pointing at Ollama/OpenAI/etc.
    # would otherwise just carry an inert, unrecognized field. There are no
    # few-shot demos attached to `analyze` today, so one breakpoint on the
    # system message covers the whole static prefix; if a compiled/
    # optimized program with demos is ever loaded here, add a second point,
    # {"location": "message", "index": -2}, to fold the demo turns into the
    # same cached prefix too (the real, always-different input is always
    # the last message, so -2 is "whatever precedes it," demos or not).
    if "anthropic" in model.lower():
        lm_kwargs["cache_control_injection_points"] = [
            {"location": "message", "role": "system"}
        ]

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return lm



from grammatike import print_analysis, analyze_passage, DEFAULT_CEILING


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Ancient Greek syntax analysis.")
    parser.add_argument(
        "--passage",
        default="τὴν θύραν ἀνέῳξεν.",
        help="Greek passage to analyze (defaults to the built-in sample, from syntax_model.md).",
    )
    parser.add_argument(
        "--citation",
        default="",
        help="Optional citation label for the passage (e.g. 'urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1'), "
             "recorded on every token via Token.citation. Defaults to no citation.",
    )
    args = parser.parse_args()

    _configure_lm()
    sentences, results = analyze_passage(args.passage, citation=args.citation)

    for i, (sentence, result) in enumerate(zip(sentences, results), start=1):
        if len(sentences) > 1:
            print(f"\n=== Sentence {i} ===")
        print_analysis(sentence.tokens, result)
