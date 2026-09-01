

## Files

- `syntaxer_main.py` — command-line entry point: loads `.env`, configures the LM, and runs an analysis for a passage given on the command line.
- `calibrate_max_tokens.py` — one-off (well, re-run-when-things-change) live-LM script that fits `token_budget.py`'s calibration file against your own configured model (see "Estimating and enforcing a `max_tokens` budget" above).
- `grammatike/` — the package with the actual analysis logic:
  - `models.py` — pydantic models for `CitedText`, `Token`, `Sentence`, `VerbalExpression`, and `TokenAnalysis`, matching the fields and relation labels from `syntax_model.md`.
  - `segmentation.py` — `segment_sources()`, a deterministic (no LM call) function that segments citation-labeled source text into sentences and tokens, assigning stable ids and tracking which citation each token came from. Sentence boundaries are settled purely by punctuation, per `syntax_model.md`'s "Segmentation into sentences" section: a period or an interrogative (the Greek question mark U+037E or its valid decomposition, the ASCII semicolon) always ends a sentence; nothing else (comma, raised dot, colon) does.
  - `greek_syntax_dspy.py` — the DSPy signature (`SyntaxAnalysis`) that takes a sentence's tokens and produces `verbalunits` + `tokengraph`, plus `validate()` and `print_analysis()`.
  - `pipeline.py` — ties the two stages together: `analyze_sources()` runs the full pipeline over citation-labeled input and analyzes every sentence it finds (via `token_budget.analyze_with_retry()`, not `analyze()` directly); `analyze_passage()` is the convenience wrapper for a single bare passage string; `combined_tokengraph()` concatenates results for diagramming.
  - `token_budget.py` — `estimate_max_tokens()`/`analyze_with_retry()` (see "Estimating and enforcing a `max_tokens` budget" above).
  - `serialization.py` — `serialize_analyses()`/`write_analyses()`/`read_analyses()`/`read_llm_notes()`/`split_analysis_by_sentence()` (see "Saving and loading analyses" above).
  - `ctsdata.py` — `read_ctsdata()` (see "Reading passages from a delimited-text source file" above).
  - `mermaid.py` — turns a `tokengraph` into a Mermaid flowchart: one node per non-punctuation token, one labelled edge per `relatedtoken1`/`relationship1` and `relatedtoken2`/`relationship2` pair, colored by verbal unit.
  - `verbal_units.py` — `assign_verbal_units()` partitions a `tokengraph` into the verbal units its own relations imply, purely from the existing graph structure (no extra LM call); `assign_verbal_unit_colors()` builds on that to assign each verbal unit a stable palette color, the single shared source both `mermaid.py` and `rendering.py` draw on so their colorings always agree. `compute_subordination_depths()`/`max_subordination_depth()` compute each verbal expression's depth of subordination. `find_unanchored_coordinated_verbs()` is a heuristic sanity check, separate from `validate()` (see "Using `grammatike` in a script" above).
  - `rendering.py` — `tokengraph_to_text()` reconstructs a continuous, readable plain-text string from a `tokengraph`, with correct spacing around punctuation, enclitics, and proclitics. `tokengraph_to_html()` does the same join as an HTML string, with lexical/numeral tokens (and coordinating conjunctions) wrapped in verbal-unit-colored `<span>`s. `tokengraph_to_depth_html()` renders the same colored tokens grouped into per-verbal-unit blocks, each CSS-indented by its depth of subordination.
  - `gepa_metric.py` — `syntax_metric()`, the scoring/feedback function used by `optimize_gepa.py` and `model_bakeoff.py` (see OPTIMIZING.md/BAKEOFF.md).
  - `__init__.py` — re-exports the public names above, so callers do `from grammatike import ...` rather than reaching into submodules.
- `tests/` — a pytest suite covering models, segmentation, analysis, validation, and coverage of the scheme's relation/type vocabulary (see TESTING.md).
- `docs/build_api_docs.py` — regenerates `docs/grammatike-api-docs.html`, a single self-contained HTML page documenting every name in `grammatike.__all__`, built with `pdoc` straight from the package's own docstrings and type hints. Run `python docs/build_api_docs.py` after changing a public docstring or signature to refresh it; requires `pdoc` (`pip install pdoc --break-system-packages`).

Not yet ported from `arsgrammatica`: `tests/fixtures/harvest.py` (turning a real analysis into a paste-ready gold example) and `diagnose_max_tokens.py` (a one-off diagnostic for a specific historical proxy error, not a general facility — see "Estimating and enforcing a `max_tokens` budget" above). Neither affects the core pipeline described above; both are development/tooling conveniences worth adding as follow-up work.

