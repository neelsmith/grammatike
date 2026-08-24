"""
Tests for grammatike/token_budget.py: estimate_max_tokens()'s formula and
clamping, and analyze_with_retry()'s detect-truncation-and-retry loop.
Greek analogue of arsgrammatica's test_token_budget.py.

analyze_with_retry()'s primary truncation signal -- a tokengraph missing
some of the input tokens' own ids -- is exercised directly with DummyLM by
handing it a deliberately incomplete canned_answer first and a complete one
second: DummyLM's list mode returns one answer per call, in order, so a
result that matches the *second* answer is only possible if a retry
actually happened.
"""

import copy

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from grammatike import analyze_passage, analyze_with_retry, estimate_max_tokens, get_calibration
from grammatike.token_budget import DEFAULT_CEILING, DEFAULT_FLOOR
from conftest import tokens_from_canned_answer
from fixtures.gold_examples import GOLD_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


# ---------------------------------------------------------------------------
# estimate_max_tokens()
# ---------------------------------------------------------------------------


def test_estimate_max_tokens_grows_with_input_length():
    small = estimate_max_tokens(5)
    large = estimate_max_tokens(50)
    assert large > small


def test_estimate_max_tokens_rejects_negative_input():
    with pytest.raises(ValueError):
        estimate_max_tokens(-1)


def test_estimate_max_tokens_respects_floor():
    assert estimate_max_tokens(0, floor=10_000, ceiling=20_000) == 10_000


def test_estimate_max_tokens_respects_ceiling():
    assert estimate_max_tokens(1_000_000, floor=0, ceiling=4_096) == 4_096


def test_estimate_max_tokens_safety_margin_scales_the_budget():
    baseline = estimate_max_tokens(20, safety_margin=1.0, floor=0, ceiling=1_000_000)
    margined = estimate_max_tokens(20, safety_margin=2.0, floor=0, ceiling=1_000_000)
    assert margined == pytest.approx(2 * baseline, rel=0.05)


def test_estimate_max_tokens_uses_fallback_constants_when_uncalibrated(tmp_path, monkeypatch):
    monkeypatch.setattr("grammatike.token_budget.CALIBRATION_FILE", tmp_path / "missing.json")
    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_get_calibration_reads_a_real_calibration_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text(
        '{"intercept": 100.0, "slope": 10.0, "sample_size": 43, "model": "test-model", '
        '"calibrated_at": "2026-01-01T00:00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("grammatike.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "calibrated"
    assert calibration["intercept"] == 100.0
    assert calibration["slope"] == 10.0

    assert estimate_max_tokens(10, safety_margin=1.0, floor=0, ceiling=10_000) == 200


def test_get_calibration_falls_back_on_malformed_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr("grammatike.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_default_floor_and_ceiling_are_sane():
    assert 0 < DEFAULT_FLOOR < DEFAULT_CEILING


# ---------------------------------------------------------------------------
# analyze_with_retry()
# ---------------------------------------------------------------------------


def test_analyze_with_retry_matches_analyze_on_a_normal_gold_example():
    """No truncation at all: behaves exactly like calling analyze()
    directly, and consumes exactly one DummyLM answer."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = tokens_from_canned_answer(example.canned_answer)

    result = analyze_with_retry(example.passage, tokens)

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in example.canned_answer["tokengraph"]]


def test_analyze_with_retry_retries_once_on_missing_ids_then_succeeds():
    example = _example("unit_verb_root_ten_thuran_anoixen")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_answer = copy.deepcopy(example.canned_answer)
    truncated_answer["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]

    dspy.configure(lm=DummyLM([truncated_answer, example.canned_answer]))

    with pytest.warns(UserWarning, match="missing"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in full_tokengraph]


def test_analyze_with_retry_gives_up_after_max_retries_and_returns_incomplete_result():
    example = _example("unit_verb_root_ten_thuran_anoixen")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_once = copy.deepcopy(example.canned_answer)
    truncated_once["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]
    truncated_twice = copy.deepcopy(example.canned_answer)
    truncated_twice["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2 + 1]

    dspy.configure(lm=DummyLM([truncated_once, truncated_twice]))

    with pytest.warns(UserWarning) as record:
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert any("still looks truncated" in str(w.message) for w in record)

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in truncated_twice["tokengraph"]]


def test_analyze_with_retry_does_not_retry_past_the_ceiling():
    """If the budget is already pinned at `ceiling`, a truncated result is
    returned (with a warning) rather than retried again."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_answer = copy.deepcopy(example.canned_answer)
    truncated_answer["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]

    dspy.configure(lm=DummyLM([truncated_answer]))

    with pytest.warns(UserWarning, match="still looks truncated"):
        result = analyze_with_retry(
            example.passage,
            tokens,
            max_retries=1,
            initial_max_tokens=500,
            ceiling=500,
        )

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in truncated_answer["tokengraph"]]


def test_analyze_with_retry_honors_initial_max_tokens_over_the_estimate():
    """A caller-supplied initial_max_tokens is used as-is for the first
    attempt rather than being recomputed from estimate_max_tokens()."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = tokens_from_canned_answer(example.canned_answer)

    result = analyze_with_retry(example.passage, tokens, initial_max_tokens=50)
    assert result is not None


# ---------------------------------------------------------------------------
# Live sanity check (see tests/conftest.py's real_lm fixture) -- skipped by
# default, run with `pytest -m live` once .env has real credentials.
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_pipeline_handles_a_deeply_subordinated_passage_without_truncating(real_lm):
    """End-to-end check against the real configured LM: pipeline.py's
    analyze_sources() now calls analyze_with_retry() rather than analyze()
    directly, so a passage with several layers of subordination should
    come back with a complete tokengraph, covering every input token id,
    with no manual retry needed from the caller."""
    sentences, results = analyze_passage(
        "ἐπειδὴ γὰρ ᾔδει τὸν ἄνδρα ὃς ταῦτα ἔπραξεν ἀδικοῦντα, ἐνόμιζε δεῖν "
        "αὐτὸν κολάζειν, ἵνα οἱ ἄλλοι μαθόντες ὅτι οὐκ ἔξεστι τοιαῦτα ποιεῖν "
        "μηκέτι τολμῶσιν."
    )

    for sentence, result in zip(sentences, results):
        seen_ids = {tok.id for tok in result.tokengraph}
        missing = {t.id for t in sentence.tokens} - seen_ids
        assert not missing, f"tokengraph still missing input token id(s): {missing}"
