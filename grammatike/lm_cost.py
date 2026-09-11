"""
Summarizes a `dspy.LM` instance's own `.history` -- the list `dspy.LM`
appends one entry to per call it makes (see `dspy.clients.base_lm.BaseLM`).
Each entry is a plain dict carrying, among other keys, `cost` -- that
call's price in US dollars, taken from litellm's own
`response._hidden_params["response_cost"]` -- except `cost` is `None`
specifically when THAT call was served from cache (dspy's own per-call
response cache, keyed on model + messages + config, or a provider-side
cache) rather than actually billed; see `BaseLM._process_lm_response()`'s
own comment, "cost is None on cache hit". `lm.history` itself is always a
list -- possibly empty, e.g. before any call has been made -- never `None`.

This module exists because every marimo notebook in this codebase that
displays "cost" had converged on the same small, subtly broken pattern:

    last_call = lm.history[-1] if lm.history else None
    cost = last_call.get('cost')

Two problems with that: (1) it raises `AttributeError` on `None.get(...)`
whenever `lm.history` is empty -- true of every one of these notebooks the
FIRST time it's opened, before its Analyze button has ever been clicked,
so this was the default, not an edge case; and (2) even once there's a
`last_call`, it only reports the cost of the single LAST LM call, which
understates a multi-sentence analysis -- `analyze_sources()`/
`analyze_passage()` make one LM call for segmentation plus one more per
sentence, so "cost of the last call" isn't the cost of the click that
triggered them, despite one notebook (`greek_syntaxer_textinput.py`)
having labeled it "Total cost" as if it were.

Greek analogue of arsgrammatica's `arsgrammatica/lm_cost.py`
(https://github.com/neelsmith/arsgrammatica), the same author's parallel
package for analyzing Latin syntax.

Usage (see any of the marimo notebooks' own "Cost" cells):

    from grammatike import summarize_lm_cost, format_lm_cost

    cost_summary = summarize_lm_cost(lm.history)
    mo.md(f"**LM cost so far**: {format_lm_cost(cost_summary)}")

`summarize_lm_cost()` never raises on an empty or all-cache-hit history --
see its own docstring for exactly what it returns in each case.
"""

from typing import Any, List, NamedTuple, Optional


class LMCostSummary(NamedTuple):
    """The result of `summarize_lm_cost()`.

    `total_cost` is the dollar sum of every history entry that actually
    recorded a cost, or `None` if there were no such entries at all --
    either because `history` was empty, or because every call in it was a
    cache hit (see this module's own docstring). Callers should treat
    `None` as "unknown", not as "$0" -- summing an empty/all-`None` set of
    costs is not the same claim as "this cost nothing".

    `priced_calls` and `uncosted_calls` count entries that did and didn't
    record a `cost`, respectively (a `None` `cost` value on any individual
    entry -- see the module docstring for why that happens); `total_calls`
    is their sum, i.e. `len(history)`."""

    total_cost: Optional[float]
    priced_calls: int
    uncosted_calls: int

    @property
    def total_calls(self) -> int:
        return self.priced_calls + self.uncosted_calls


def _entry_cost(entry: Any) -> Optional[float]:
    """Read one history entry's own `cost`, however it's shaped: a plain
    dict (every entry `dspy.LM` itself produces, as of this writing) via
    `.get('cost')`, or -- defensively, in case a future dspy version ever
    represents an entry as an object instead -- via a `cost` attribute.
    Missing either way reads as `None`, same as an explicit cache-hit
    `None`; this function doesn't distinguish "no such key/attribute at
    all" from "the key/attribute is there and set to None", since neither
    case has a known cost to report."""
    if isinstance(entry, dict):
        return entry.get("cost")
    return getattr(entry, "cost", None)


def summarize_lm_cost(history: List[Any]) -> LMCostSummary:
    """Sum the `cost` recorded on every entry of `history` (a `dspy.LM`
    instance's own `.history` list, or any list shaped like it) that
    actually has one, and count how many did versus didn't.

    Never raises: an empty `history` returns
    `LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=0)`,
    and a `history` where every entry's `cost` is `None` (every call
    served from cache) returns `LMCostSummary(total_cost=None,
    priced_calls=0, uncosted_calls=len(history))` -- `total_cost` is only
    ever a number when at least one entry actually recorded one.
    """
    priced_calls = 0
    uncosted_calls = 0
    total_cost = 0.0

    for entry in history:
        cost = _entry_cost(entry)
        if cost is None:
            uncosted_calls += 1
        else:
            priced_calls += 1
            total_cost += cost

    if priced_calls == 0:
        return LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=uncosted_calls)
    return LMCostSummary(total_cost=total_cost, priced_calls=priced_calls, uncosted_calls=uncosted_calls)


def format_lm_cost(summary: LMCostSummary) -> str:
    """Render an `LMCostSummary` as one short, human-readable line for
    display (e.g. a notebook's own "Cost" markdown cell) -- covering every
    case `summarize_lm_cost()` can return without the caller needing to
    branch on `None` itself:

    - no calls at all -> "no LM calls yet"
    - calls, but every one served from cache -> says so explicitly,
      rather than printing a bare, unexplained "None"
    - a mix of priced and cached calls -> the priced total, plus a note
      that some calls aren't included in it
    - every call priced -> the total alone
    """
    if summary.total_calls == 0:
        return "no LM calls yet"

    if summary.total_cost is None:
        call_word = "call" if summary.uncosted_calls == 1 else "calls"
        return f"$0.00 billed -- {summary.uncosted_calls} {call_word}, all served from cache (no cost recorded)"

    priced_word = "call" if summary.priced_calls == 1 else "calls"
    base = f"${summary.total_cost:.4f} across {summary.priced_calls} {priced_word}"
    if summary.uncosted_calls:
        cached_word = "call" if summary.uncosted_calls == 1 else "calls"
        return f"{base} (+ {summary.uncosted_calls} more {cached_word} served from cache, not included)"
    return base
