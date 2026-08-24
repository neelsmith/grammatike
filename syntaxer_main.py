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


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


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
    lm_kwargs = dict(model=model, api_base=api_base)
    if api_key:
        lm_kwargs["api_key"] = api_key

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return lm



from grammatike import print_analysis, analyze_passage


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
