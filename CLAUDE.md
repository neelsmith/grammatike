# CLAUDE.md

Guidance for Claude when working in this repo. Read this before making changes.

## What this is

`grammatike` is a Python package (using [dspy](https://dspy.ai)) that analyzes the
syntax of Ancient Greek passages with LLMs, using an original analytic scheme
(documented in `syntax_model.md`) rather than Universal Dependencies. It's an
early-stage research project by Neel Smith (Holy Cross), developed in parallel
with his `arsgrammatica` (Latin) and `diqduq` (Biblical Hebrew) packages — ports
between them are common, so a comment referencing "arsgrammatica's own X" is
normal and not a mistake.

This repo is worked on from more than one machine via Cowork; there is no
single canonical checkout.

## Ownership boundaries — read/write rules

- **Read-only for Claude:** `README.md`, `releases.md`, and everything under
  `quarto/`. These are Neel's own writing. Read them freely for context, but
  never edit them, even to fix a typo — flag it instead and let him decide.
- **Generated, never hand-edited:** `docs/`. This is the build output of
  `quarto render` (from `quarto/`) and of `utilities/build_api_docs.py` (which
  writes `docs/grammatike-api-docs.html`). If something in `docs/` looks wrong,
  fix the source it's generated from, then regenerate — don't patch the HTML.
- **Claude's own notes and documentation go in `notes/`.** This is where
  design notes, investigation write-ups, and anything else Claude produces
  about the code should live (see the existing files there for the tone/style:
  `dot_diagrams.md`, `extend-scheme.md`, `files.md`, `quarto.md`).
- **Never `git commit` or `git push`.** Make changes and leave them in the
  working tree (staged or not) for Neel to review and commit himself.

## Before reporting any change as done

1. **Run the full `pytest` suite and confirm it passes** — not just tests
   near the change. Don't tell Neel something is finished until it does.
   - Default `pytest` runs offline against `DummyLM` only (`addopts = -m "not
     live"` in `pytest.ini`). Tests marked `@pytest.mark.live` hit a real
     configured LM via the `real_lm` fixture in `tests/conftest.py`, which
     needs a `.env` (untracked) with `API_KEY` (and optionally `API_BASE`,
     `MODEL`) — it skips gracefully if that's missing. Don't assume live
     tests ran just because the default suite passed.
2. **If the change touches a public docstring, signature, or anything in
   `grammatike.__all__`**, re-run `python utilities/build_api_docs.py`
   (requires `pdoc`, part of the `dev` extra) and confirm it regenerates
   `docs/grammatike-api-docs.html` cleanly, with no errors.
3. **If the change produces output meant for another downstream tool**
   (Graphviz DOT text from `dot.py`/`analysis_to_dot.py`/`analyses_to_png.py`,
   a Mermaid diagram, a Quarto page, etc.), actually run that real tool on the
   real output at least once — e.g. `dot -Tsvg file.dot > file.svg` — rather
   than relying only on unit tests or reading the generated text/DOT source by
   eye. This project's own convention (see `notes/dot_diagrams.md`, "Tests")
   is that generation is unit-tested but rendering is spot-checked against the
   real `dot` binary; keep following that pattern for anything new.

## Setup

```sh
pip install -e ".[dev]"   # pytest, python-dotenv, pdoc, marimo, graphviz
# or, for just running tests:
pip install -e ".[test]"  # pytest, python-dotenv
```

`graphviz` here is only the Python subprocess wrapper — actually rendering a
`.dot` file needs Graphviz's own `dot` executable installed separately and on
`PATH` (`brew install graphviz` / `apt install graphviz`), which is why step 3
above can't be skipped even when the `graphviz` package imports fine.

## Repo layout (see `notes/files.md` for the full file-by-file map)

- `grammatike/` — the package itself (models, segmentation, the DSPy
  signature, pipeline, rendering, serialization, dot/mermaid diagram export,
  token-budget handling, LM cost tracking).
- `syntaxer_main.py` — the one CLI entry point kept at repo root; everything
  else runnable lives in `utilities/`.
- `utilities/` — CLI scripts and live-LM tools (GEPA optimization, model
  bakeoff, DOT/PNG rendering, API-doc generation, token-budget calibration).
- `marimo/` — interactive notebooks for reviewing/analyzing passages.
- `tests/` — the pytest suite; offline/`DummyLM`-backed by default.
- `notes/` — Claude's own documentation (see above).
- `quarto/` — Neel's hand-written docs source (read-only for Claude);
  renders to `docs/` for GitHub Pages (`quarto render` from repo root,
  `quarto preview` from `quarto/`; see `notes/quarto.md`).
- `docs/` — generated output only (see above).
- `_to_delete/` — retired modules an earlier cleanup couldn't actually
  delete (no delete permission on the connected folder at the time); excluded
  from test collection via `norecursedirs` in `pytest.ini`. Delete permission
  has since been granted for this Desktop folder, so this can be cleaned up
  properly — but only when Neel asks for it, not on Claude's own initiative.

## Where to look for more detail

- `syntax_model.md` — the syntactic scheme itself (relation labels, token
  types, segmentation rules).
- `quarto/guides/*.qmd` — Neel's own guides (development loop, testing,
  optimizing with GEPA, model bakeoff, serialization, managing prompt size),
  rendered at `docs/guides/*.html` and published to
  https://neelsmith.github.io/grammatike/. Read these for workflow context;
  don't edit them.
- `notes/extend-scheme.md` — the steps for adding a new relation label to the
  scheme.
- `notes/dot_diagrams.md` — the Graphviz DOT renderer's design and the
  generation-vs-rendering test convention referenced above.
