"""
Calibrates grammatike/token_budget.py's max_tokens estimate against real LM
output, instead of the untuned fallback constants token_budget.py ships
with. Greek analogue of arsgrammatica's calibrate_max_tokens.py.

Why: SyntaxAnalysis's output (a `reasoning` field plus JSON-serialized
`verbalunits`/`tokengraph` lists) grows with how long and how syntactically
complex a passage is, not by a fixed amount -- so the right `max_tokens`
budget for a call is a function of the input token count, not a constant.
This script measures that function directly: it runs every GOLD_EXAMPLES
passage (tests/fixtures/gold_examples.py) through the real configured LM
with a generous max_tokens ceiling so nothing truncates, records how many
completion tokens each one actually used, and fits
`completion_tokens ~ intercept + slope * num_input_tokens` by ordinary
least squares. The fitted (intercept, slope) is written to
grammatike/grammatike_token_budget_calibration.json, where
token_budget.estimate_max_tokens() picks it up automatically.

Usage:

    python3 utilities/calibrate_max_tokens.py

Needs the same .env this project's other scripts use (see USAGE.md's
"Running an analysis from the command line"):

    API_BASE=https://localmodel/api
    MODEL=litellm/modelname
    API_KEY=your-key-here

This is a live-LM script with real API cost -- one call per GOLD_EXAMPLES
entry. Re-run it whenever the configured MODEL, the SyntaxAnalysis prompt,
or the TokenAnalysis/VerbalExpression schema changes substantially, since
any of those shifts how many output tokens a given passage actually needs.
GOLD_EXAMPLES itself is a corpus of short, illustrative single-construction
sentences, not long real-world passages -- the fit is a genuine measurement
over that range, but treat max_tokens estimates for much longer passages as
an extrapolation, and lean on token_budget.py's safety_margin/ceiling/retry
machinery rather than trusting the raw line far past the calibrated range.

--calibration-ceiling controls the max_tokens used *during calibration
itself* (not the fitted result) -- generous by default so calibration runs
aren't the ones getting truncated; --limit runs a quick smoke test over
just the first N examples instead of the whole corpus.

Every calibration call always bypasses DSPy's own LM response cache
(`config={"cache": False, ...}`), the same "force a fresh call" bypass
`token_budget.analyze_with_retry()`'s own `disable_cache` parameter offers a
human resubmitting a passage. This isn't optional here the way it is there:
DSPy's cache layer deliberately blanks out `usage` on every cache hit --
"no LM call is made", so there's nothing to report (see dspy's own
`clients/cache.py`, `_prepare_cached_response()`) -- and GOLD_EXAMPLES
passages are exactly the ones most likely to already be sitting in that
cache (this script's own prior runs, the test suite, or other real-LM
sessions over the same fixtures). Without the bypass, a warm cache silently
turns every single example into a "no completion_tokens in usage" skip --
not a provider problem at all, just a stale local cache -- and calibration
can look completely broken (0/N calibrated) even though the LM itself is
working fine.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import dspy
from dotenv import load_dotenv

from grammatike import analyze, IMPLIED_TOKENTYPES, Token

# tests/ isn't an installed package, and this script now lives in utilities/
# rather than the repo root -- a plain `from tests.fixtures.gold_examples
# import ...` would otherwise only resolve when Python's automatic sys.path[0]
# (a directly-run script's own directory) happened to be the repo root.
# Add the repo root itself, so `tests` still resolves as a (namespace)
# package the same way it did before this script moved.
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

CALIBRATION_FILE = Path(__file__).parent.parent / "grammatike" / "grammatike_token_budget_calibration.json"


def _env(name: str, fallback_name: str, default: "str | None" = None) -> "str | None":
    return os.getenv(name) or os.getenv(fallback_name) or default


def _configure_lm():
    """Same .env-driven LM setup as syntaxer_main.py's own _configure_lm()
    -- duplicated rather than imported, following this repo's existing
    convention of each runnable script owning its own copy (see
    model_bakeoff.py's several near-identical _configure_*_lm() helpers)."""
    api_base = _env("API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm")
    model = _env("MODEL", "MODEL", "litellm_proxy/anthropic/Claude Opus 5")

    # Same "API_KEY= is fine for a local model, API_KEY missing entirely is
    # not" distinction as syntaxer_main.py's own _configure_lm() -- see that
    # function's comment for the full rationale.
    if "API_KEY" not in os.environ:
        raise RuntimeError(
            "Missing API key. Set API_KEY in your .env file -- an empty "
            "value (API_KEY=) is fine for a local model that doesn't need "
            "one, e.g. Ollama; this only checks that the line exists at all."
        )
    api_key = os.environ["API_KEY"]

    lm_kwargs = dict(model=model, api_base=api_base)
    if api_key:
        lm_kwargs["api_key"] = api_key

    # Anthropic prompt caching -- see syntaxer_main.py's own _configure_lm()
    # for the full rationale. Especially worth it here: this script fires
    # one real LM call per GOLD_EXAMPLES entry back to back, all sharing the
    # exact same system message, so every call after the first should hit
    # the cache within Anthropic's TTL and cost a fraction of what it
    # otherwise would on that portion.
    if "anthropic" in model.lower():
        lm_kwargs["cache_control_injection_points"] = [
            {"location": "message", "role": "system"}
        ]

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return lm


def _tokens_from_canned_answer(canned_answer):
    """Mirrors tests/conftest.py's tokens_from_canned_answer(): builds the
    Token list a GoldExample's own passage should be paired with directly
    from its canned_answer's tokengraph (id + surface text), rather than
    re-tokenizing the passage string by some other means. Excludes implied/
    elided entries (tokentype in IMPLIED_TOKENTYPES) -- those are never
    part of the *input* token list; analyze() adds them to its output."""
    return [
        Token(id=entry["id"], text=entry["token"])
        for entry in canned_answer["tokengraph"]
        if entry.get("tokentype") not in IMPLIED_TOKENTYPES
    ]


def _fit_line(xs, ys):
    """Ordinary least squares for y = a + b*x, plain Python (no numpy
    dependency needed for a fit this simple). Returns (a, b). Raises
    ValueError if there are fewer than 2 distinct x values -- a line isn't
    identifiable from a single point."""
    n = len(xs)
    if len({x for x in xs}) < 2:
        raise ValueError(
            "Need at least 2 examples with different token counts to fit a "
            "line; every calibrated example had the same num_tokens."
        )

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    b = cov / var
    a = mean_y - b * mean_x
    return a, b


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate token_budget.py's max_tokens estimate against the real configured LM."
    )
    parser.add_argument(
        "--calibration-ceiling",
        type=int,
        default=8000,
        help="max_tokens used for calibration calls themselves (default: 8000) -- "
             "should comfortably exceed anything GOLD_EXAMPLES needs; raise it if "
             "examples are still getting skipped as truncated even at the default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only calibrate against the first N GOLD_EXAMPLES (for a quick smoke run).",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
    lm = _configure_lm()

    examples = GOLD_EXAMPLES[: args.limit] if args.limit else GOLD_EXAMPLES

    rows = []  # (slug, num_tokens, completion_tokens)
    skipped = []
    for example in examples:
        tokens = _tokens_from_canned_answer(example.canned_answer)
        try:
            analyze(
                passage=example.passage,
                tokens=tokens,
                # cache=False always -- see this script's own module
                # docstring for why calibration can never afford a cache
                # hit here (DSPy blanks out usage entirely on one).
                config={"max_tokens": args.calibration_ceiling, "cache": False},
            )
        except Exception as exc:  # noqa: BLE001 -- report and keep calibrating
            skipped.append((example.slug, f"raised {exc.__class__.__name__}: {exc}"))
            continue

        history_entry = lm.history[-1]

        # Recent DSPy versions (>= ~3.x) store each history entry as an
        # `LMHistoryEntry` object -- dict-STYLE access (`entry.get(...)`,
        # `entry["..."]`) is supported for backward compatibility, but it
        # goes through that object's own `Mapping` implementation, which
        # re-serializes `entry.response` via `model_dump()` -- and that
        # serialization deliberately DROPS the raw provider response
        # (`provider_response` is marked `exclude=True`, since it isn't
        # JSON-safe in general). Reading `history_entry.response` as a
        # plain ATTRIBUTE instead bypasses that re-serialization and gives
        # back the real, unserialized `LMResponse`, `provider_response`
        # (the actual litellm/provider object, with its real `.choices`)
        # included. On an OLDER DSPy version, `history_entry` is a plain
        # dict rather than an object with a `.response` attribute at all --
        # there, `entry["response"]` WAS always the raw provider response
        # directly, so the `.get()` fallback below still gets the right
        # thing.
        response_obj = (
            history_entry.response
            if hasattr(history_entry, "response")
            else history_entry.get("response")
        )

        usage = history_entry.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            # Diagnose rather than just report "missing" -- this exact
            # message used to fire for every example whenever the cache
            # bypass above was missing (see this script's own docstring),
            # so a genuine provider-reporting gap is now worth telling
            # apart from that at a glance if it ever recurs.
            cache_hit = bool(getattr(response_obj, "cache_hit", False))
            detail = "usage dict is empty" if not usage else f"usage={usage!r}"
            if cache_hit:
                detail += " (served from DSPy's own cache despite cache=False -- unexpected; check DSPy's cache config)"
            skipped.append((example.slug, f"no completion_tokens in usage -- provider didn't report it ({detail})"))
            continue

        # `.provider_response` is only present on the newer, normalized
        # `LMResponse` (see above) -- on the older shape, response_obj IS
        # already the raw provider response, so getattr's default just
        # returns it unchanged.
        provider_response = getattr(response_obj, "provider_response", None) or response_obj
        choices = getattr(provider_response, "choices", [])
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            skipped.append((example.slug, f"still truncated even at max_tokens={args.calibration_ceiling}"))
            continue

        rows.append((example.slug, len(tokens), completion_tokens))

    print(f"Calibrated against {len(rows)}/{len(examples)} examples.")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")

    if len(rows) < 2:
        raise RuntimeError(
            f"Only {len(rows)} usable example(s) -- need at least 2 to fit a line. "
            "Check the skipped list above."
        )

    print("\nslug                                       num_tokens  completion_tokens")
    for slug, num_tokens, completion_tokens in rows:
        print(f"{slug:<42}  {num_tokens:>10}  {completion_tokens:>17}")

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    intercept, slope = _fit_line(xs, ys)

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    max_abs_residual = max(abs(r) for r in residuals)

    print(f"\nFitted: completion_tokens ~= {intercept:.1f} + {slope:.2f} * num_tokens")
    print(f"Largest residual over the calibration set: {max_abs_residual:.1f} tokens")
    print(
        "(token_budget.estimate_max_tokens() applies its own safety_margin on top "
        "of this fit -- the margin is what actually covers residual variance like "
        "this, not the fit itself.)"
    )

    payload = {
        "intercept": intercept,
        "slope": slope,
        "sample_size": len(rows),
        "model": lm.model,
        "calibrated_at": datetime.datetime.now().isoformat(),
    }

    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
