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
from dspy.utils.exceptions import AdapterParseError

from grammatike import analyze_passage, analyze_with_retry, estimate_max_tokens, get_calibration
from grammatike.greek_syntax_dspy import SyntaxAnalysis
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


def test_analyze_with_retry_default_max_retries_is_three():
    """max_retries defaults to 3 (raised from an earlier 1) -- see
    analyze_with_retry()'s own docstring for why one retry wasn't enough
    headroom in practice for a real, longer passage. Confirms the DEFAULT
    itself, not just behavior when max_retries is passed explicitly (every
    other test in this file does): four consecutive truncated DummyLM
    answers (the initial attempt plus all 3 retries) should all get
    consumed before analyze_with_retry() gives up and returns the last
    (still incomplete) one with a warning."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated = copy.deepcopy(example.canned_answer)
    truncated["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]

    dspy.configure(lm=DummyLM([truncated, truncated, truncated, truncated]))

    with pytest.warns(UserWarning) as record:
        result = analyze_with_retry(example.passage, tokens)  # max_retries omitted -- exercises the default

    assert any("still looks truncated" in str(w.message) for w in record)
    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in truncated["tokengraph"]]


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


def test_analyze_with_retry_retries_once_on_malformed_non_truncated_output_then_succeeds(monkeypatch):
    """A parse failure whose finish_reason ISN'T "length" (e.g. one
    tokengraph entry coming back as a bare `["id"]` list instead of a
    TokenAnalysis object, matching a real bug report against arsgrammatica)
    is a different failure mode from truncation -- retried once anyway, at
    the SAME budget, with the LM cache explicitly bypassed for that one
    retry. DummyLM's own ChatAdapter/JSONAdapter fallback dance makes it
    awkward to simulate this precisely (a malformed answer can get silently
    "fixed" by the JSONAdapter fallback consuming a second queued answer),
    so `analyze()` itself is monkeypatched here instead -- letting this
    test assert exactly what analyze_with_retry() does with the config
    dict on each attempt, not just the end-to-end outcome."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    tokens = tokens_from_canned_answer(example.canned_answer)

    good_result = dspy.Prediction(
        reasoning="fine",
        verbalunits=[],
        tokengraph=[dspy.Prediction(**e) for e in example.canned_answer["tokengraph"]],
    )

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        if len(calls) == 1:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=SyntaxAnalysis,
                lm_response="[[ ## tokengraph ## ]]\n[...]",
                message="Failed to parse field tokengraph with value [...]. Error message: "
                "1 validation error for list[TokenAnalysis]\n21\n  Input should be a valid "
                "dictionary or instance of TokenAnalysis [type=model_type, input_value=['id'], "
                "input_type=list]",
            )
        return good_result

    monkeypatch.setattr("grammatike.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("grammatike.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="doesn't look like a truncation"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert result is good_result
    assert len(calls) == 2
    # Same budget both times (no growth -- a bigger budget wouldn't have
    # fixed a malformed entry), but the cache is bypassed only for the
    # retry, not the first (normal) attempt.
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"]
    assert "cache" not in calls[0]
    assert calls[1]["cache"] is False


def test_analyze_with_retry_gives_up_after_max_retries_on_persistent_malformed_output(monkeypatch):
    """If the retry ALSO comes back malformed, the exception propagates
    once max_retries is exhausted -- same as any other unrecoverable
    failure, not silently swallowed."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    tokens = tokens_from_canned_answer(example.canned_answer)

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=SyntaxAnalysis,
            lm_response="[[ ## tokengraph ## ]]\n[...]",
            message="persistently malformed",
        )

    monkeypatch.setattr("grammatike.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("grammatike.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="doesn't look like a truncation"):
        with pytest.raises(AdapterParseError, match="persistently malformed"):
            analyze_with_retry(example.passage, tokens, max_retries=1)

    # One normal attempt, one cache-bypassed retry, then give up -- no
    # third attempt beyond max_retries=1.
    assert len(calls) == 2
    assert calls[1]["cache"] is False


def test_analyze_with_retry_disable_cache_bypasses_cache_on_every_attempt(monkeypatch):
    """disable_cache=True is the caller-facing knob (e.g. a notebook's
    "disable cache" checkbox) for a human deliberately resubmitting the
    same passage/tokens and wanting a genuinely fresh LM call each time --
    unlike the internal one-shot bypass for a malformed-output retry, this
    stays on for every attempt in the loop, including a truncation-and-grow
    retry, not just one."""
    example = _example("unit_verb_root_ten_thuran_anoixen")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_result = dspy.Prediction(
        reasoning="fine",
        verbalunits=[],
        tokengraph=[dspy.Prediction(**e) for e in full_tokengraph[: len(full_tokengraph) // 2]],
    )
    complete_result = dspy.Prediction(
        reasoning="fine",
        verbalunits=[],
        tokengraph=[dspy.Prediction(**e) for e in full_tokengraph],
    )

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        return truncated_result if len(calls) == 1 else complete_result

    monkeypatch.setattr("grammatike.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("grammatike.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="missing"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1, disable_cache=True)

    assert result is complete_result
    assert len(calls) == 2
    assert calls[0]["cache"] is False
    assert calls[1]["cache"] is False


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
