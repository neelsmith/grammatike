# Ancient Greek Syntax Analyzer — Usage Guide

A DSPy program that analyzes an Ancient Greek passage into two structures: a table of verbal expressions, and a token-by-token dependency graph. The analytic scheme itself is documented in `syntax_model.md`.


## Running an analysis from the command line

You can run an analysis from the command line with the wrapper script `syntaxer_main.py`. It needs an `.env` file in this folder with your LM credentials, like this:

```
API_BASE=https://localmodel/api
MODEL=litellm/modelname
API_KEY=your-key-here
```

Then:

```bash
python3 syntaxer_main.py --passage "τὴν θύραν ἀνέῳξεν."
```

`--citation` is an optional second argument giving a citation label for the passage (e.g. a CTS URN), recorded on every resulting token via `Token.citation`; it defaults to no citation if omitted:

```bash
python3 syntaxer_main.py --passage "τὴν θύραν ἀνέῳξεν." --citation "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1"
```

`syntaxer_main.py` reads `API_BASE` / `MODEL` / `API_KEY` from `.env`, configures the LM, and prints the analysis.

For a local, unauthenticated model (e.g. Ollama), leave `API_KEY` present but empty:

```
API_BASE=http://localhost:11434
MODEL=ollama_chat/llama3
API_KEY=
```

`syntaxer_main.py` only raises "Missing API key" when `API_KEY` isn't in `.env` at all; an empty value is treated as "this model doesn't need one" and is left out of the LM call entirely, rather than sent through as an empty credential.


## Using `grammatike` in a script

To call the pipeline from your own script or a REPL instead of the CLI, configure a `dspy.LM` yourself and use `grammatike` directly:

```python
import dspy
from grammatike import analyze_passage, print_analysis

dspy.configure(lm=dspy.LM(model="litellm_proxy/anthropic/Claude Opus 5",
                           api_base="https://api_url/litellm",
                           api_key="your-key-here"))

sentences, results = analyze_passage("τὴν θύραν ἀνέῳξεν.")
for sentence, result in zip(sentences, results):
    print_analysis(sentence.tokens, result)
```

Explanation:

- `analyze_passage()` returns `(sentences, results)`: that is, one `Sentence` and one `SyntaxAnalysis` result per sentence it finds in `passage`.
- `result.verbalunits` is a list of `VerbalExpression` objects.
- `result.tokengraph` is a list of `TokenAnalysis` objects, one per token in that sentence, in order.

`analyze_sources()`/`analyze_passage()` already call `validate()` for you and print a warning if the LM refers to a token id that doesn't exist in its sentence's input tokens. (That's a sign that the output needs a re-run or a prompt tweak, and does not necessarily mean that your code is broken.)

`validate()` only catches referential problems like that one — ids that don't exist. It can't tell you an otherwise well-formed analysis is probably still wrong. For one specific, observed failure mode — a coordinating conjunction correctly pairing two verbal expressions, but the second one silently missing its own `verbalunitid` — call `find_unanchored_coordinated_verbs()` on the result:

```python
from grammatike import find_unanchored_coordinated_verbs

for sentence, result in zip(sentences, results):
    for warning in find_unanchored_coordinated_verbs(result.tokengraph):
        print(f"Possible mistake: {warning}")
```

It's a heuristic, not a guarantee — see its own docstring — but a clean result costs nothing to check, and a flagged one is worth a manual read before you trust the analysis.


## Analyzing citable sources

`grammatike` supports analyzing texts identified by some canonical citation. Under the hood, `analyze_passage()` wraps `passage` as a `CitedText` and hands this to `analyze_sources()`, which is what actually does the work. You can call `analyze_sources()` directly like this:

```python
from grammatike import analyze_sources, combined_tokengraph
from grammatike.models import CitedText

apology = "urn:cts:greekLit:tlg0059.tlg002.perseus-grc2:"
sources = [
    CitedText(citation=f"{apology}17a", text="ὅτι μὲν ὑμεῖς, ὦ ἄνδρες Ἀθηναῖοι, πεπόνθατε ὑπὸ τῶν ἐμῶν κατηγόρων, οὐκ οἶδα·"),
    CitedText(citation=f"{apology}17b", text="ἐγὼ δ᾽ οὖν καὶ αὐτὸς ὑπ᾽ αὐτῶν ὀλίγου ἐμαυτοῦ ἐπελαθόμην."),
]
sentences, results = analyze_sources(sources)
tokengraph = combined_tokengraph(results)  # one flat list, spanning every sentence
```

`analyze_sources()` handles any number of sentences and citation units; sentence boundaries don't need to respect citation-unit boundaries (one sentence may span two source lines), and every token still records which source unit it came from via `Token.citation`.


## Saving and loading analyses

`write_analyses()`/`read_analyses()` (in `grammatike/serialization.py`) save and reload a full analysis — `sentences`, `verbalunits` (concatenated across every sentence's result), and `tokengraph` (via `combined_tokengraph()`) — as one deterministic, pipe-delimited plain-text file, so you can persist an analysis, diff it, hand-edit it, or reload it later without re-running the LM:

```python
from grammatike import write_analyses, read_analyses, combined_tokengraph

verbalunits = [vu for result in results for vu in result.verbalunits]
tokengraph = combined_tokengraph(results)

warnings = write_analyses(sentences, verbalunits, tokengraph, "analysis.txt")
for w in warnings:
    print(f"Warning: {w}")

tokengraph, verbalunits, sentences = read_analyses("analysis.txt")
```

`serialize_analyses(sentences, verbalunits, tokengraph)` builds the exact same text and returns it as a string (plus the same warnings) instead of writing it to a file — `write_analyses()` is just a thin wrapper around it.

The file has three labelled, pipe-delimited blocks (`#!sentences`, `#!verbal_units`, `#!tokens`), each with its own fixed header row — see `serialization.py`'s module docstring for the exact format, why `sentences` is needed at all (it's the only place a citation is actually attached to a token id), and what `write_analyses()`'s warnings vs. `read_analyses()`'s errors each catch. Each of the three labels may appear more than once in the file; `read_analyses()` merges every instance of a label into that label's combined row list, in file order, so simply concatenating several `write_analyses()`/`serialize_analyses()` outputs together and reading the result back gives you one combined analysis. `read_analyses()` is otherwise deliberately strict: a malformed or internally inconsistent file raises `ValueError` naming the exact line and problem, rather than silently reconstructing something partial.

Pass `results` (the same list `analyze_sources()`/`analyze_passage()` return alongside `sentences`) to keep each sentence's own LM reasoning trace with the saved analysis, for later review or for curating a GEPA trainset (see `OPTIMIZING.md`):

```python
warnings = write_analyses(sentences, verbalunits, tokengraph, "analysis.txt", results=results)
```

This adds one `#!llm` block per sentence — `MODEL=<value of the `MODEL` environment variable>` followed by that sentence's `result.reasoning` text, verbatim. It's purely additive: omit `results` (the default) and no `#!llm` blocks are written at all; `read_analyses()` skips over any it finds (still checking they're well-formed) without changing its own return shape. Read the reasoning traces back out with the dedicated `read_llm_notes()`, which returns `[(model, reasoning), ...]` in file order:

```python
from grammatike import read_llm_notes

notes = read_llm_notes("analysis.txt")
for model, reasoning in notes:
    print(f"[{model}] {reasoning}")
```

`read_analyses()` hands back flat, whole-file lists — every sentence's `tokengraph`/`verbalunits` concatenated together, the same shape `combined_tokengraph()` produces. `split_analysis_by_sentence(tokengraph, verbalunits, sentences)` splits that back into one `(sentence_tokengraph, sentence_verbalunits)` slice per sentence, aligned with `sentences` itself:

```python
from grammatike import read_analyses, split_analysis_by_sentence

tokengraph, verbalunits, sentences = read_analyses("analysis.txt")
slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)

for sentence, (sentence_tokengraph, sentence_verbalunits) in zip(sentences, slices):
    ...  # render or inspect this one sentence's own analysis
```


## Reading passages from a delimited-text source file

`read_ctsdata()` (in `grammatike/ctsdata.py`) reads a list of citable passages — each one a CTS URN paired with its own text — out of a pipe-delimited file, the input-side counterpart to `write_analyses()`/`read_analyses()` above (which handle an analysis's *results*, not the passages you're about to analyze):

```python
from grammatike import read_ctsdata

rows = read_ctsdata("passages.txt")
for row in rows:
    citation = row.urnbase + row.citation  # reconstructs the full URN
    print(citation, "--", row.text)
```

The file has one or more `#!ctsdata` blocks, each with its own `urn|text` header row:

```
#!ctsdata
urn|text
urn:cts:greekLit:tlg0059.tlg002.perseus-grc2:17a|ὅτι μὲν ὑμεῖς, ὦ ἄνδρες Ἀθηναῖοι, πεπόνθατε ὑπὸ τῶν ἐμῶν κατηγόρων, οὐκ οἶδα·
```

Each row's `urn` column must be a 5-part, colon-separated CTS URN; `read_ctsdata()` splits it into `urnbase` (the first 4 parts, rejoined with `:`, plus a trailing `:`) and `citation` (the 5th part). Pass `delimiter=...` if the file itself uses something other than `|`. Like `read_analyses()`, this is deliberately strict (a malformed row or a urn that doesn't split into exactly 5 parts raises `ValueError`, naming the line) and merges multiple `#!ctsdata` blocks in file order.


## Estimating and enforcing a `max_tokens` budget

`SyntaxAnalysis`'s output (a `reasoning` field plus JSON-serialized `verbalunits`/`tokengraph`) grows with how long and how syntactically complex a sentence is, not by a fixed amount, so a single hard-coded `max_tokens` value is eventually wrong: too small for a long or deeply subordinated sentence (truncation), too large for a short one (wasted budget). `grammatike/token_budget.py` addresses this with a calibrate-then-retry approach, and `pipeline.py`'s `analyze_sources()` already uses it — both `analyze_sources()` and `analyze_passage()` get this for free, with nothing to change in your own calling code. This mirrors `arsgrammatica`'s own "managing prompt size" facility (see its `guides/managing_prompt_size` docs page), ported here for `SyntaxAnalysis`/Greek in place of `SentenceAnalysis`/Latin.

Until `calibrate_max_tokens.py` (below) has been run against your own configured model, `estimate_max_tokens()` falls back to an untuned, deliberately generous placeholder fit inherited from `arsgrammatica`'s own Latin fallback — safe, but not a real measurement of your model.

```python
from grammatike import estimate_max_tokens

budget = estimate_max_tokens(num_tokens=25)  # -> an int max_tokens value
```

`estimate_max_tokens()` takes the calibrated (or fallback) fit, multiplies it by a `safety_margin` (default `1.4`, covering reasoning-length variance the fit alone doesn't), and clamps the result to `[floor, ceiling]`. `DEFAULT_CEILING` is `32000` (raised from an earlier, never-confirmed `8192` guess — see `token_budget.py`'s own comment) — still only a placeholder stand-in, since a model's real max-output-tokens limit varies by provider and there's no single correct default; set `ceiling` explicitly to whatever your configured model actually allows if you know it.

For the retry half, `analyze_with_retry()` wraps `analyze()`:

```python
from grammatike import analyze_with_retry

result = analyze_with_retry(passage, tokens)
```

It starts from `estimate_max_tokens(len(tokens))` (or `initial_max_tokens`, if you pass one), and checks the result two ways: whether the returned `tokengraph` is missing any of `tokens`' own ids (the primary, provider-independent signal — a real truncation, LM-JSON getting cut off mid-list, always shows up here), and, as a corroborating check, whether the LM's own `finish_reason` was `"length"`. If either signals truncation and a retry is still available (`max_retries`, default `1`) with budget left before `ceiling`, it multiplies the budget by `growth_factor` (default `2.0`) and calls again — `max_tokens` is part of DSPy's own LM cache key, so the retry always reaches the LM again rather than replaying the same truncated cached response.

Separately, a parse failure that ISN'T a truncation (the response finished normally but came back malformed somewhere, e.g. one `tokengraph` entry as a bare list instead of a full object) is also retried once, at the *same* budget, with DSPy's own LM response cache explicitly bypassed for that one attempt (`config={"cache": False}`) — a bigger budget wouldn't fix a malformed entry, but the malformation itself is usually a one-off sampling glitch a fresh call clears up.

If retries run out: a call that raised re-raises (nothing to fall back to); a call that returned an incomplete result is returned anyway, with a `UserWarning` naming the missing ids, rather than treated as fatal — consistent with `validate()`'s own warn-don't-raise convention for imperfect LM output.

Pass `disable_cache=True` to bypass DSPy's own LM response cache on *every* attempt of the call, not just the one-shot bypass above — this is for a human deliberately resubmitting the exact same passage/tokens (typically while iterating on the configured model, the `SyntaxAnalysis` prompt, or the schema) who wants a genuinely fresh LM call each time rather than a replay of whatever this passage returned last time. `greek_syntaxer_ctsdata.py`'s *Disable LM cache* checkbox (next to the *Analyze selected sentences* button) is wired straight to this parameter; leave it unchecked for ordinary browsing, where the cache is a real cost/latency win.

`get_calibration()` reports which fit is currently active (a real calibrated one, or the untuned fallback) if you want to check before relying on an estimate. The calibration file, once one exists, is `grammatike/grammatike_token_budget_calibration.json` — named distinctly from `arsgrammatica`'s own `token_budget_calibration.json` so the two packages' calibration files never collide if both are ever installed/checked out side by side. Run

```bash
python3 calibrate_max_tokens.py
```

against your own `.env`-configured model to produce one: it runs every `GOLD_EXAMPLES` passage through the real LM with a generous ceiling, records actual completion-token usage, fits `completion_tokens ~ intercept + slope * num_tokens` by ordinary least squares, and writes the result to that file, where `estimate_max_tokens()` picks it up automatically on the next call. It's a live-LM script with real API cost (one call per `GOLD_EXAMPLES` entry) — use `--limit N` for a quick smoke run, or `--calibration-ceiling` if examples are still getting skipped as truncated even at its default. Re-run it whenever the configured `MODEL`, the `SyntaxAnalysis` prompt, or the `TokenAnalysis`/`VerbalExpression` schema changes substantially.

`arsgrammatica`'s `diagnose_max_tokens.py` — a one-off diagnostic written to track down a specific proxy error it once hit when passing an explicit `max_tokens` — hasn't been ported, since it isn't a general facility, just a historical debugging aid for that one incident; write an analogous throwaway script if grammatike ever needs to debug something similar.

`syntaxer_main.py`'s `_configure_lm()` also carries two related pieces of this facility: it now passes `max_tokens=DEFAULT_CEILING` explicitly to `dspy.LM(...)` (a numeric baseline rather than DSPy's own `None` default, so `dspy.LM`'s own truncation warning reports the real ceiling rather than `max_tokens=None` for any call that doesn't go through `analyze_with_retry()`), and it sets litellm's `cache_control_injection_points` on `SyntaxAnalysis`'s system message whenever the configured model is Anthropic-routed — that message is long and byte-identical on every call, so Anthropic's prompt caching lets a repeat call within its cache TTL reuse it at a fraction of the normal input-token price. `load_dotenv()` there also now passes `override=True`, so a stale `MODEL`/`API_BASE`/`API_KEY` left over in the shell environment from an earlier session can't silently shadow `.env`'s own values.


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


## `marimo` notebooks

`greek_syntaxer_workflow.py` and `greek_syntaxer_ctsdata.py` take different approaches to segmentation, matching how each notebook's input arrives:

- `greek_syntaxer_workflow.py`: enter a base URN, passage reference, and text to analyze as one form (nothing happens until you click *Analyze*, rather than on every keystroke). Submitting it calls `analyze_passage()` directly, which segments the text into sentences internally and runs full `SyntaxAnalysis` (via `analyze_with_retry()`) on each one in turn, then shows the discussion, Mermaid diagram, and highlighted/indented HTML for the whole passage. There's no separate sentence-selection step here -- the text you type is manually entered, so it's up to you to submit it one sentence at a time if you want to control segmentation boundaries yourself. There's a `cex`/`txt` extension choice (default `cex`) plus a *Download analysis* button that hands the current analysis (built with `serialize_analyses()`, see "Saving and loading analyses" above) to the browser's own download mechanism -- no folder path to type, at the cost of the browser (not the notebook) deciding where the file actually lands. The filename defaults to the submitted citation (base URN + passage) with the chosen extension.
- `greek_syntaxer_ctsdata.py`: the input comes from a `#!ctsdata` source file (see "Reading passages from a delimited-text source file" above) instead of being typed in by hand, so segmentation and analysis stay two distinct steps -- segmentation is free (no LM call at all, see `segmentation.py`), so you can see every sentence it finds, and choose which ones are actually worth the cost of a full analysis, before spending anything on `SyntaxAnalysis`. Browse for the file and pick one or more passages from the *Passage(s)* multiselect menu (labelled `<citation>: <first few words>…`), then click *OK* to segment the selection -- every selected passage becomes its own `CitedText` source and is segmented together via `segment_sources()`, always in the file's own order regardless of the order they were selected in, since consecutive sources can share a sentence across their boundary. A *Sentence(s) to analyze* menu then lists every sentence segmentation found across all the selected passages (labelled `<n>. <citation>: <first eight words>…`, since sentences here can come from different citations); pick some and click *Analyze selected sentences* to run `SyntaxAnalysis` on just those. A *Disable LM cache* checkbox next to that button forces the click to bypass DSPy's own LM response cache (`analyze_with_retry()`'s `disable_cache` parameter — see "Estimating and enforcing a `max_tokens` budget" above), so re-clicking *Analyze* on the exact same sentence(s) after changing the configured model, the prompt, or the schema always reaches the LM again instead of silently replaying the previous response; leave it unchecked for ordinary browsing, where the cache saves real cost and latency. Everything downstream (Mermaid diagram, highlighted/indented HTML, save-to-file) covers only the analyzed subset. Both this notebook and `greek_syntaxer_workflow.py` also have *See list of tokens*/*See cost*/*See prompts* checkboxes, each toggling a hidden display of the raw token list, the last LM call's own reported cost, or `dspy.inspect_history()`'s prompt/response transcript.
- `greek_syntaxer_review.py`: no LM access at all -- browse for a file previously written by `write_analyses()` (see "Saving and loading analyses" above), pick a sentence from the menu that appears (labelled `<n>. <citation>: <first six words>…`, via `split_analysis_by_sentence()`), and it displays that one sentence's own Mermaid diagram, plain (uncolored) text, verbal-unit-colored HTML, and colored-and-indented-by-subordination-depth HTML -- reconstructed entirely from the saved file, which is already complete, so there's no separate segment-then-analyze step here the way there is in the other two notebooks. A slider above the indented view caps it to that sentence's own `max_subordination_depth()` or shallower, the same depth-cap control the other two notebooks offer, except here it only appears once a sentence with at least one token has actually been picked. A *Download Mermaid diagram (.mmd)* button next to the diagram hands that sentence's raw Mermaid source (the same text `mo.mermaid()` renders) to the browser's own download mechanism. Useful for reviewing or presenting an already-completed analysis (e.g. one harvested into `GOLD_EXAMPLES`) without spending an LM call, or working at all when the LM is unreachable.

All three notebooks default their sample text/URN to `syntax_model.md`'s own worked examples -- most of which are drawn from Lysias 1, *On the Murder of Eratosthenes* (`urn:cts:greekLit:tlg0540.tlg001.perseus-grc2:`) -- as a placeholder; point `urnbase` at your own corpus. Run any of them with `marimo edit marimo/greek_syntaxer_workflow.py` (requires `pip install marimo --break-system-packages`, or `pip install -e ".[dev]"`, see TESTING.md) from the repo root, and a `.env` with `API_BASE`/`MODEL`/`API_KEY` set (see DEVELOPMENT.md), except `greek_syntaxer_review.py`, which needs no LM access at all.


## Extending the scheme

`syntax_model.md` says the current relation set is partial. To add a new relation:

1. Add the new label to `RelationLabel` in `grammatike/models.py`.
2. Describe when to use it in `SyntaxAnalysis`'s docstring in
   `grammatike/greek_syntax_dspy.py`, following the pattern of the existing relations
   (which token gets `relatedtoken1`/`relationship1`, which gets the
   corresponding value on the other end).
3. Add a gold example exercising it to `tests/fixtures/gold_examples.py` and
   re-run `pytest` to confirm the models still validate before trying it
   against the real LM.
