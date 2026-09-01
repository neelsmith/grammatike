"""
grammatike: estimating and enforcing a `max_tokens` output budget for
SyntaxAnalysis calls over Greek passages. Greek analogue of arsgrammatica's
token_budget.py.

Estimating and enforcing a `max_tokens` output budget for SyntaxAnalysis
calls, and retrying with a larger one when a call actually gets truncated.

Background: `greek_syntax_dspy.analyze` (a `dspy.ChainOfThought`) produces a
free-text `reasoning` field plus JSON-serialized `verbalunits` and
`tokengraph` lists -- one `TokenAnalysis` entry per input token, plus extra
entries for implied/elided tokens. That output's size scales with how long
and how syntactically nested the passage is, not with a fixed constant, so
any single hard-coded `max_tokens` value is eventually wrong for either a
short passage (wastes budget) or a long/complex one (truncates mid-output --
DSPy logs a warning for this itself, see `dspy.LM._check_truncation`, but
that's a warning after the fact, not a fix).

This module takes a hybrid approach instead:

1. `estimate_max_tokens()` picks a per-call budget from a simple linear
   model (`completion_tokens ≈ intercept + slope * num_input_tokens`),
   calibrated empirically by `calibrate_max_tokens.py` against real LM
   output over the gold-example corpus (see that script's own docstring),
   with a safety margin on top. Until you've run that script, a
   conservative, deliberately-generous fallback fit is used instead (see
   `_FALLBACK_INTERCEPT`/`_FALLBACK_SLOPE` below) -- one that's meant to
   overestimate rather than truncate, not to be a good fit.
2. `analyze_with_retry()` wraps `analyze()` and, if a call still comes back
   truncated despite that estimate (either the LM's own `finish_reason`
   says so, or -- the more robust, LM-independent check -- the result is
   missing entries for input token ids it should have covered), retries
   with a larger budget rather than silently returning an incomplete
   result or leaving the caller to guess a bigger number by hand. It also
   retries once (at the same budget, with the LM's own response cache
   explicitly bypassed) on a parse failure that ISN'T a truncation -- e.g.
   the LM emitting one malformed `tokengraph` entry in an otherwise
   well-terminated response -- since that kind of malformation is usually
   a one-off sampling glitch a fresh call clears up, not something a
   bigger budget would fix.

Re-run `calibrate_max_tokens.py` whenever the configured model, the
SyntaxAnalysis prompt, or the shape of `TokenAnalysis`/`VerbalExpression`
changes substantially -- all three shift how many output tokens a given
passage actually needs.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

import dspy
from dspy.utils.exceptions import AdapterParseError

from .greek_syntax_dspy import analyze
from .models import Token

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

# calibrate_max_tokens.py writes its fitted (intercept, slope) here. Kept
# next to this module (not under tests/) since it's runtime configuration,
# not test fixture data -- any script or notebook using grammatike benefits
# from it, not just the test suite. Named distinctly from arsgrammatica's
# own `token_budget_calibration.json` so the two packages' calibration
# files never collide if both are ever installed/checked out side by side.
CALIBRATION_FILE = Path(__file__).with_name("grammatike_token_budget_calibration.json")

# Untuned stand-ins, used only until calibrate_max_tokens.py has actually
# been run once against the real configured model. Deliberately generous
# (60 output tokens per input token, plus a 500-token allowance for the
# reasoning field and the verbalunits list) -- an overestimate here just
# spends a bit more of the model's output budget than necessary; an
# underestimate is what causes the truncation this module exists to avoid.
#
# Inherited as-is from arsgrammatica's own Latin fallback fit -- neither
# language has actually run calibrate_max_tokens.py against a real model
# yet, so there's no Greek-specific measurement to prefer over it. Re-tune
# for Greek once real calibration data exists (calibrate_max_tokens.py
# writes CALIBRATION_FILE, which takes over from these two constants
# automatically -- see _load_calibration() below).
_FALLBACK_INTERCEPT = 500.0
_FALLBACK_SLOPE = 60.0

DEFAULT_SAFETY_MARGIN = 1.4
DEFAULT_FLOOR = 256
# Stand-in for "this model's real max output tokens". There's no single
# correct value across providers/models -- override this with whatever your
# configured MODEL actually allows (check its provider's documentation)
# rather than relying on this default for anything but a rough starting
# point. Raised from an earlier 8192: that value was itself never confirmed
# against any real configured model's actual limit, just a guess, and
# analyze_with_retry()'s own growth_factor doubling means a single retry
# starting from a mid-size estimate can legitimately need to clear 8192
# for a long or deeply subordinated Greek sentence well before hitting any
# real provider limit. If your model's own true ceiling is lower than this,
# a request for more than it allows should surface as an explicit error
# from the provider (naming the real limit) rather than a silent
# truncation -- lower this to match if that happens.
DEFAULT_CEILING = 32000


def _load_calibration() -> dict:
    """Read calibrate_max_tokens.py's saved fit, if any.

    Returns a dict with at least "intercept", "slope", and "source" keys.
    "source" is "calibrated" when CALIBRATION_FILE was read successfully,
    or "fallback" when it's missing, unreadable, or malformed -- callers
    that want to know which one is active (or tests that want to force the
    fallback) can check that field rather than re-deriving it.
    """
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "intercept": float(data["intercept"]),
            "slope": float(data["slope"]),
            "source": "calibrated",
            "sample_size": data.get("sample_size"),
            "model": data.get("model"),
            "calibrated_at": data.get("calibrated_at"),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "intercept": _FALLBACK_INTERCEPT,
            "slope": _FALLBACK_SLOPE,
            "source": "fallback",
            "sample_size": None,
            "model": None,
            "calibrated_at": None,
        }


def get_calibration() -> dict:
    """Public introspection: what (intercept, slope) is estimate_max_tokens()
    currently using, and did it come from calibrate_max_tokens.py's fit or
    from this module's untuned fallback? See _load_calibration()'s
    docstring for the shape returned."""
    return _load_calibration()


def estimate_max_tokens(
    num_tokens: int,
    *,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
) -> int:
    """Estimate a `max_tokens` budget for a SyntaxAnalysis call over a
    sentence with `num_tokens` input tokens.

    `raw = intercept + slope * num_tokens` comes from the calibrated (or
    fallback) linear fit (see _load_calibration()); `safety_margin`
    multiplies that to leave room for the reasoning field's length being
    only roughly, not exactly, a function of passage length. The result is
    clamped to `[floor, ceiling]` -- `floor` guards against a degenerate
    tiny estimate for a 1-2 token sentence, `ceiling` is a hard cap you
    should set to your actual model's real max-output-tokens limit (see
    DEFAULT_CEILING's docstring note).

    Raises ValueError if `num_tokens` is negative.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

    calibration = _load_calibration()
    raw = calibration["intercept"] + calibration["slope"] * num_tokens
    budget = math.ceil(raw * safety_margin)
    return max(floor, min(ceiling, budget))


# ---------------------------------------------------------------------------
# Retry-on-truncation wrapper
# ---------------------------------------------------------------------------


def _finish_reason_was_length() -> bool:
    """Best-effort check of whether the most recent call made against the
    currently configured LM was cut off for hitting max_tokens, via the
    same `finish_reason == "length"` signal dspy.LM._check_truncation()
    itself warns on.

    This is a *secondary* corroborating signal, not the primary detector --
    see analyze_with_retry()'s docstring for why -- so it fails safe: any
    missing attribute, empty history, or non-dspy.LM configured LM (e.g.
    DummyLM in tests, which doesn't populate `.history` the same way)
    just returns False rather than raising.
    """
    try:
        lm = dspy.settings.lm
        entry = lm.history[-1]
        response = entry["response"]
        # Dict-style access on `response` itself, matching dspy.LM's own
        # _check_truncation() exactly (`results["choices"]`); attribute-style
        # access on each choice, same as that method's `c.finish_reason`.
        return any(getattr(c, "finish_reason", None) == "length" for c in response["choices"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def _missing_token_ids(tokens: List[Token], result) -> set:
    """Which of `tokens`' own ids never showed up in `result.tokengraph`.

    This is the primary truncation signal: unlike finish_reason (which
    depends on the LM class actually surfacing it, and isn't exercised at
    all by DummyLM-backed tests), a real truncation -- the model's
    per-token loop getting cut off partway through -- always shows up here,
    regardless of provider, and it's checked even when the JSON still
    happened to parse successfully. A non-empty result means the analysis
    is missing coverage for at least one real input token, which validate()
    (models.py's own referential check) doesn't itself catch -- validate()
    flags ids that shouldn't exist, not ids that should have.
    """
    seen_ids = {tok.id for tok in result.tokengraph}
    return {t.id for t in tokens} - seen_ids


def analyze_with_retry(
    passage: str,
    tokens: List[Token],
    *,
    max_retries: int = 1,
    growth_factor: float = 2.0,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
    initial_max_tokens: Optional[int] = None,
    disable_cache: bool = False,
):
    """Call `analyze()`, detecting truncation and retrying with a larger
    `max_tokens` budget instead of either crashing or silently returning an
    incomplete result.

    The starting budget is `initial_max_tokens` if given, else
    `estimate_max_tokens(len(tokens), safety_margin=safety_margin,
    floor=floor, ceiling=ceiling)`.

    `disable_cache` is the caller-facing counterpart to the automatic
    cache-bypass described below: pass `True` to force EVERY attempt in
    this call (not just an internal malformed-output retry) to bypass
    DSPy's own LM response cache. This is for a human deliberately
    resubmitting the exact same `passage`/`tokens` -- e.g. from a "disable
    cache" checkbox in a notebook, while iterating on the configured
    model, the `SyntaxAnalysis` prompt, or the schema -- where the whole
    point is a genuinely fresh LM call, not a replay of whatever this
    passage returned last time. Leave it `False` (the default) for
    ordinary pipeline use, where the cache is a real cost/latency win.

    After each attempt, truncation is checked two ways: `_missing_token_ids`
    against the result (the primary, LM-independent signal -- works
    whenever a result exists at all, parsed or not, including under
    DummyLM in tests) and, if the call raised `AdapterParseError` instead
    of returning a result (the JSON was cut off badly enough to not parse
    at all), `_finish_reason_was_length()` as a corroborating check.

    An `AdapterParseError` whose `finish_reason` ISN'T "length" means the
    response was well-terminated but still malformed somewhere -- e.g. one
    `tokengraph` entry coming back as a bare `["id"]` list instead of a
    full `TokenAnalysis` object. A bigger budget wouldn't have fixed that,
    but the malformation itself is very often a one-off sampling glitch
    rather than a systematic prompt/schema problem, so it's retried once
    too (still counted against `max_retries`, at the SAME budget) with
    dspy's own response cache explicitly bypassed for that one attempt
    (`config={"cache": False, ...}`) -- without that, an identical request
    would just replay the identical broken response, retrying nothing. If
    that retry also fails to parse, or `max_retries` is already exhausted,
    the exception propagates. (When `disable_cache` is already `True`, this
    one-shot bypass is redundant but harmless -- the cache is off either
    way.)

    If truncation is detected and there's still a retry available (fewer
    than `max_retries` attempts so far, and the budget hasn't already hit
    `ceiling`), the budget is multiplied by `growth_factor` (capped at
    `ceiling`) and the call is retried. `max_tokens` is part of DSPy's own
    LM cache key, so a retry with a different budget always reaches the LM
    again rather than replaying a cached truncated response.

    Once retries are exhausted: if the last attempt raised, that exception
    propagates (there's no result to fall back to). If the last attempt
    returned a still-incomplete result, it's returned anyway -- with a
    `UserWarning` naming the missing token ids -- rather than raising,
    matching this codebase's existing convention of surfacing analysis
    problems as warnings (see pipeline.py's own validate() warning-printing
    and this module's docstring) instead of treating an imperfect LM
    result as fatal.
    """
    budget = initial_max_tokens if initial_max_tokens is not None else estimate_max_tokens(
        len(tokens), safety_margin=safety_margin, floor=floor, ceiling=ceiling
    )

    attempt = 0
    retry_bypass_cache = False  # one-shot: set only for the malformed-output retry below
    while True:
        old_budget = budget
        call_config = {"max_tokens": budget}
        if disable_cache or retry_bypass_cache:
            call_config["cache"] = False
        retry_bypass_cache = False  # only meant for the one attempt it was set for
        try:
            result = analyze(passage=passage, tokens=tokens, config=call_config)
        except AdapterParseError as exc:
            if attempt >= max_retries:
                raise
            if budget < ceiling and _finish_reason_was_length():
                attempt += 1
                budget = min(ceiling, math.ceil(budget * growth_factor))
                warnings.warn(
                    f"SyntaxAnalysis call truncated at max_tokens={old_budget} before it "
                    f"could be parsed at all; retrying with max_tokens={budget} "
                    f"(attempt {attempt}/{max_retries}).",
                    stacklevel=2,
                )
                continue
            # Not a (detectable) truncation -- the response finished
            # normally but was malformed somewhere (see this function's own
            # docstring). Retry once more at the SAME budget, but with
            # dspy's cache explicitly bypassed for that one attempt, so a
            # retry is a genuinely fresh LM call rather than a replay of
            # the same broken response -- otherwise, if this exact request
            # was already served from cache (e.g. a repeat of an earlier,
            # already-broken run), simply calling analyze() again would
            # just return the identical malformed result again and again,
            # even with dspy's cache enabled as normal for every other call.
            attempt += 1
            retry_bypass_cache = True
            warnings.warn(
                f"SyntaxAnalysis call at max_tokens={old_budget} returned output that "
                f"failed to parse, but doesn't look like a truncation (finish_reason "
                f"wasn't 'length'): {exc} Retrying once at the same budget with the LM "
                f"cache bypassed, in case this was a one-off malformed-output glitch "
                f"(attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        missing = _missing_token_ids(tokens, result)
        truncated = bool(missing) or _finish_reason_was_length()
        if truncated and attempt < max_retries and budget < ceiling:
            attempt += 1
            budget = min(ceiling, math.ceil(budget * growth_factor))
            warnings.warn(
                f"SyntaxAnalysis call at max_tokens={old_budget} returned a tokengraph "
                f"missing {len(missing)} input token id(s) ({sorted(missing)}); retrying "
                f"with a larger max_tokens={budget} (attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        if truncated:
            missing_desc = sorted(missing) if missing else "(finish_reason indicated truncation, but no ids are directly missing)"
            warnings.warn(
                f"SyntaxAnalysis call still looks truncated after {attempt} retry(ies) "
                f"(max_tokens={old_budget}) -- returning it anyway. Missing input token "
                f"id(s): {missing_desc}.",
                stacklevel=2,
            )

        return result
