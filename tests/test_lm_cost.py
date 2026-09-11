"""
Tests for grammatike/lm_cost.py: summarize_lm_cost()'s handling of every
shape a dspy.LM's own `.history` list can take (empty, all-cache-hit,
mixed, all-priced, dict-shaped and object-shaped entries), and
format_lm_cost()'s human-readable rendering of each resulting summary.
Greek analogue of arsgrammatica's test_lm_cost.py.
"""

import pytest

from grammatike import LMCostSummary, summarize_lm_cost, format_lm_cost


# ---------------------------------------------------------------------------
# summarize_lm_cost()
# ---------------------------------------------------------------------------


def test_summarize_empty_history():
    summary = summarize_lm_cost([])
    assert summary == LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=0)
    assert summary.total_calls == 0


def test_summarize_all_cache_hits():
    history = [{"cost": None}, {"cost": None}, {"cost": None}]
    summary = summarize_lm_cost(history)
    assert summary.total_cost is None
    assert summary.priced_calls == 0
    assert summary.uncosted_calls == 3
    assert summary.total_calls == 3


def test_summarize_all_priced():
    history = [{"cost": 0.01}, {"cost": 0.02}, {"cost": 0.03}]
    summary = summarize_lm_cost(history)
    assert summary.priced_calls == 3
    assert summary.uncosted_calls == 0
    assert summary.total_calls == 3
    assert summary.total_cost == pytest.approx(0.06)


def test_summarize_mixed_priced_and_cached():
    history = [{"cost": 0.01}, {"cost": None}, {"cost": 0.02}]
    summary = summarize_lm_cost(history)
    assert summary.priced_calls == 2
    assert summary.uncosted_calls == 1
    assert summary.total_calls == 3
    assert summary.total_cost == pytest.approx(0.03)


def test_summarize_entry_missing_cost_key_entirely():
    # A dict entry with no "cost" key at all reads the same as an
    # explicit cost=None -- both mean "no known cost for this call".
    history = [{"model": "some-model"}, {"cost": 0.05}]
    summary = summarize_lm_cost(history)
    assert summary.priced_calls == 1
    assert summary.uncosted_calls == 1
    assert summary.total_cost == pytest.approx(0.05)


class _FakeHistoryEntry:
    """Object-shaped history entry (rather than a plain dict), exercising
    _entry_cost()'s defensive getattr() fallback for a hypothetical future
    dspy version that represents entries as objects."""

    def __init__(self, cost):
        self.cost = cost


def test_summarize_object_shaped_entries():
    history = [_FakeHistoryEntry(0.10), _FakeHistoryEntry(None), _FakeHistoryEntry(0.20)]
    summary = summarize_lm_cost(history)
    assert summary.priced_calls == 2
    assert summary.uncosted_calls == 1
    assert summary.total_cost == pytest.approx(0.30)


class _NoAttrEntry:
    """Object with no `cost` attribute at all -- getattr() fallback should
    read this the same as an explicit cost=None."""


def test_summarize_object_entry_missing_cost_attribute():
    history = [_NoAttrEntry(), _FakeHistoryEntry(0.07)]
    summary = summarize_lm_cost(history)
    assert summary.priced_calls == 1
    assert summary.uncosted_calls == 1
    assert summary.total_cost == pytest.approx(0.07)


def test_total_calls_is_sum_of_priced_and_uncosted():
    summary = LMCostSummary(total_cost=1.23, priced_calls=4, uncosted_calls=2)
    assert summary.total_calls == 6


# ---------------------------------------------------------------------------
# format_lm_cost()
# ---------------------------------------------------------------------------


def test_format_no_calls_yet():
    summary = summarize_lm_cost([])
    assert format_lm_cost(summary) == "no LM calls yet"


def test_format_all_cache_hits_singular():
    summary = summarize_lm_cost([{"cost": None}])
    text = format_lm_cost(summary)
    assert "1 call," in text  # singular, not "1 calls,"
    assert "all served from cache" in text
    assert "$0.00 billed" in text


def test_format_all_cache_hits_plural():
    summary = summarize_lm_cost([{"cost": None}, {"cost": None}])
    text = format_lm_cost(summary)
    assert "2 calls" in text
    assert "all served from cache" in text


def test_format_all_priced_no_cache_mention():
    summary = summarize_lm_cost([{"cost": 0.01}, {"cost": 0.02}])
    text = format_lm_cost(summary)
    assert "$0.0300" in text
    assert "2 calls" in text
    assert "cache" not in text


def test_format_all_priced_singular_call():
    summary = summarize_lm_cost([{"cost": 0.05}])
    text = format_lm_cost(summary)
    assert "$0.0500" in text
    assert "1 call" in text
    assert "1 calls" not in text


def test_format_mixed_notes_uncosted_calls():
    summary = summarize_lm_cost([{"cost": 0.01}, {"cost": None}, {"cost": None}])
    text = format_lm_cost(summary)
    assert "$0.0100" in text
    assert "1 call" in text
    assert "2 more" in text
    assert "not included" in text


def test_format_mixed_singular_uncosted_call():
    summary = summarize_lm_cost([{"cost": 0.01}, {"cost": None}])
    text = format_lm_cost(summary)
    assert "1 more call " in text or text.endswith("1 more call served from cache, not included)")
    assert "1 more calls" not in text
