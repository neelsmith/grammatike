# Testing without network access

First time in a fresh checkout or virtual environment: `pytest` itself is a
test-only dependency, not something a plain `pip install -e .` pulls in (see
`pyproject.toml`'s `dependencies` vs. `[project.optional-dependencies]`), so
install it explicitly before your first run:

```bash
pip install -e ".[test]"
```

(`.[dev]` instead of `.[test]` also pulls in `pdoc` and `marimo`, for
`docs/build_api_docs.py` and the notebooks in `marimo/`.) Then:

```bash
pytest
```

Runs the whole suite (536 tests, `tests/`) — the `SyntaxAnalysis` stage's tests use DSPy's `DummyLM` in place of a real LM call, while the segmentation stage's tests call its deterministic `segment_sources()` directly with no LM involved at all (see `segmentation.py`) — useful for confirming the models/signatures/pipeline still fit together after you change something, without spending API calls. Tests that call the actual configured LM are marked `live` and skipped by default (`pytest.ini`'s `addopts = -m "not live"`) — they're the only way to check the LM itself gets a scenario right, not just that the code can represent a correct answer; run them explicitly with:

```bash
pytest -m live
```

(`live` tests need a working `.env`, same as `syntaxer_main.py`; the `real_lm` fixture in `tests/conftest.py` skips gracefully if `API_KEY` isn't set.)

Some standard `pytest` shorthands:

- `pytest` to run all.
- `pytest -v` for per-test names instead of dots.
- `pytest tests/test_gold_examples.py` to run just one file.
- `pytest -k agent` to run only tests matching a substring.
- `pytest --collect-only` if you just want to see what it discovered without running anything.


## Layout

- `tests/conftest.py` — shared helpers: `tokens_from_canned_answer()` builds a `Token` list straight from a `GoldExample`'s own `canned_answer` tokengraph; `run_gold_example()` runs a `GoldExample`'s passage through `analyze()` with `DummyLM` standing in for the real LM; `run_segmentation_example()` does the same for `segment_sources()`, though that one needs no `DummyLM` at all -- segmentation is deterministic (see `segmentation.py`); the `real_lm` fixture configures `dspy` against the real `.env`-configured model for `live`-marked tests (only `SyntaxAnalysis` itself uses this now).
- `tests/fixtures/gold_examples.py` — `GOLD_EXAMPLES`, a list of `GoldExample(slug, passage, tags, canned_answer)` entries: a real Greek sentence, the relation(s)/construction it's meant to exercise, and a hand-written, `syntax_model.md`-correct `canned_answer` in the exact dict shape `DummyLM` expects. Wherever possible each fixture transcribes one of `syntax_model.md`'s own worked examples verbatim; a few are necessarily constructed, and say so in a comment.
- `tests/fixtures/segmentation_examples.py` — `SegmentationExample` entries exercising `segment_sources()`'s citation-aware sentence/token segmentation: enclitic vs. false-positive splits, numeral tokenization, a period or interrogative always ending a sentence, indefinite vs. interrogative enclitics, and (per `syntax_model.md`'s current "Segmentation into sentences" section) a raised-dot/colon-type mark NEVER ending a sentence, no matter what surrounds it -- three fixtures use real text from Lysias 1.1-1.4 (the same passages `syntax_model.md` itself draws its worked examples from) to confirm this, including one sentence that spans a citation-unit boundary through an internal colon that isn't a boundary.
- `tests/test_analyze_passage.py`, `tests/test_gold_examples.py` — run every `GoldExample` through `analyze()`/`analyze_passage()` under `DummyLM` and check the result round-trips correctly.
- `tests/test_segmentation.py`, `tests/test_segmentation_examples.py` — the segmentation stage; both run `segment_sources()` directly with no LM in the loop at all, so there's no `live` counterpart left for this stage (there used to be one, `test_segmentation_live.py`, back when segmentation was itself LM-driven).
- `tests/test_validate.py` — `validate()`'s referential-integrity checks (unknown ids, malformed implied tokens, the reserved `'root'` sentinel).
- `tests/test_coverage.py` — enforces that every documented `RelationLabel`, `tokentype`, and `syntactic_type`/`semantic_type` value in `models.py` has at least one `GOLD_EXAMPLES` entry tagged for it, so the fixture corpus can't silently drift out of sync with the scheme.
- `tests/test_verbal_units.py` — `assign_verbal_units()`, `assign_verbal_unit_colors()`, `compute_subordination_depths()`, `max_subordination_depth()`, and the `find_unanchored_coordinated_verbs()` heuristic.
- `tests/test_rendering.py` — `tokengraph_to_text()`, `tokengraph_to_html()`, `tokengraph_to_depth_html()`.
- `tests/test_mermaid_coloring.py` — `tokengraph_to_mermaid()` and its verbal-unit coloring.
- `tests/test_serialization.py` — `serialize_analyses()`/`write_analyses()`/`read_analyses()`/`read_llm_notes()`/`split_analysis_by_sentence()`, including the optional `results` parameter's `#!llm` blocks (model/reasoning round trip, internal-blank-line preservation vs. the writer's own trailing separator, and the malformed-block errors both readers share).
- `tests/test_ctsdata.py` — `read_ctsdata()`.
- `tests/test_token_budget.py` — `estimate_max_tokens()`, `get_calibration()`, and `analyze_with_retry()`'s truncation-detection/retry logic, including its separate retry-once-with-cache-bypassed path for a non-truncation parse failure (`analyze()` itself is monkeypatched for that path — DummyLM can't cleanly simulate a malformed-but-well-terminated response) and its caller-facing `disable_cache` parameter (bypasses the cache on every attempt, not just one retry).
- `tests/test_gepa_metric.py` — `grammatike.gepa_metric.syntax_metric()` in isolation, including that swapping the `relatedtoken1`/`relatedtoken2` overflow slots scores as a perfect match, not an error.
- `tests/test_harvest.py` — a thin, `live`-marked stub. `arsgrammatica`'s `tests/fixtures/harvest.py` (`gold_example_from_analysis()`/`format_gold_example_source()`, for turning a real analysis into a paste-ready `GoldExample`) has not been ported to `grammatike` yet; see that test file's own docstring.
