# Graphviz DOT diagrams (`dot.py`)

`grammatike.tokengraph_to_dot()` draws the same tokengraph diagram as `tokengraph_to_mermaid()` -- same nodes, same edges, same verbal-unit coloring, same implied-token treatment, same sentence-connector treatment -- as Graphviz DOT source instead of Mermaid source. See `dot.py`'s own module docstring for the full rationale, including two adaptations from arsgrammatica's own `dot.py` (grammatike has no `compute_aat_depths()`, and grammatike's sentence-connector coloring has no Latin-side counterpart); the short version is below.

Ported from arsgrammatica 0.6.0. Everything documentation-related about this feature -- this file included -- lives under `notes/`, not `USAGE.md`: see this file's own place in the repo for that convention.

## Why a second renderer

Graphviz's `dot` layout engine has an actual primitive for forcing same-depth nodes onto one level: a `{rank=same; id1; id2; ...}` subgraph statement, which *forces* those nodes onto the same rank -- not a heuristic a layout engine could override, a hard constraint. `tokengraph_to_dot()`'s `rank_by_depth` builds exactly that, grouped by `verbal_units.compute_subordination_depths()` -- the same clause-level depth function `rendering.tokengraph_to_depth_html()` already uses for its own indented-HTML view (grammatike substitutes this for arsgrammatica's `compute_aat_depths()`, which has no counterpart in grammatike's `verbal_units.py` -- see `dot.py`'s own docstring, "Two adaptations", for the full reasoning and the one real consequence: unlike `compute_aat_depths()`, `compute_subordination_depths()` CAN leave an anchor's depth unresolved, in which case it's simply excluded from ranking and its own resolution warning is folded into `tokengraph_to_dot()`'s returned warnings).

## Usage

```python
from grammatike import tokengraph_to_dot

dot_source, warnings = tokengraph_to_dot(tokengraph)
```

Same signature shape as `tokengraph_to_mermaid()`, plus two parameters `tokengraph_to_mermaid()` doesn't have:

- `orientation` (default `"BT"`) -- maps straight onto DOT's `rankdir` attribute (`BT`/`TB`/`LR`/`RL`).
- `color_by_verbal_unit` (default `True`) -- colors each node by its verbal unit, via `fillcolor`/`color`/`fontcolor` node attributes (DOT has no reusable named class the way Mermaid's `classDef` does, so it's inline per node instead). An implied/elided token gets the same dedicated amber `tokengraph_to_mermaid()` uses; a sentence-connector token (relationship1 == "sentence connector", e.g. γάρ) gets the same dedicated neon-yellow/strong-border treatment `tokengraph_to_mermaid()`'s own `sentenceconnector` class uses, via an added `penwidth="4"` attribute (DOT's counterpart to a Mermaid class's `stroke-width`).
- `rank_by_depth` (default `True`) -- the `{rank=same; ...}` alignment described above.
- `depth` (default `None`) -- caps the diagram to nodes within a given *graph* distance of a root verb, dropping farther ones entirely. See "Depth filtering" below -- this is a THIRD depth notion, distinct from `rank_by_depth`'s, and easy to conflate with either that one or `tokengraph_to_depth_html()`'s own `depth`.
- `aat_depth` (default `None`) -- a SECOND, independent node-omission cutoff, additive alongside `depth`: a node is kept only if it survives BOTH. Uses the SAME clause-level notion `rank_by_depth` groups by (`compute_subordination_depths()`), so a whole clause's subject, object, and other ordinary dependents share ONE `aat_depth` with their governing verb -- unlike `depth`, which gives each of them its own hop. See "A second cutoff: `aat_depth`" below.
- Returns `(dot_source, warnings)` -- same warnings as the Mermaid version (a skipped edge pointing at punctuation, a token excluded by the `depth` or `aat_depth` cutoff, or a missing id; more than 8 verbal units repeating colors) plus, when `rank_by_depth` or `aat_depth` is active, any warning `compute_subordination_depths()` itself produces (a relation cycle, or no governing verbal expression found) -- see "Why a second renderer" above. When both `rank_by_depth` and `aat_depth` are active, `compute_subordination_depths()` is only called once, so its warning (if any) appears only once in the returned list. `depth` filtering itself never adds a warning.

`save_dot(tokengraph, path, ...)` writes the diagram straight to a `.dot` file, and takes the same `depth` and `aat_depth` parameters. Unlike arsgrammatica's own convention (`save_dot()` not re-exported from its top-level package), grammatike exports both `tokengraph_to_dot` and `save_dot` from the top level -- matching `save_mermaid()`'s own existing, already-established treatment in this package.

## Depth filtering

`depth` caps the diagram to nodes at or within a given *graph* distance -- number of edges -- from the nearest root/independent verbal-unit anchor, following the exact same `relatedtoken1`/`relatedtoken2` edges drawn as `->` lines (`dot.compute_graph_depths()`). This is deliberately **not** the same notion as `rendering.tokengraph_to_depth_html()`'s own `depth` (`verbal_units.compute_subordination_depths()`, a CLAUSE-level notion where a whole clause's subject, object, and other ordinary dependents all share ONE depth with their governing verb) -- even though `rank_by_depth` above uses that SAME function for a different purpose. All of these can disagree on the same tokengraph -- don't assume a `rank=same` grouping, an indented-HTML block, and a `depth` cutoff here line up, even when two of them are ultimately computed from the same underlying function.

`depth=0` shows ONLY root verbal-unit anchors:

```python
# Only root verb(s) -- no dependents at all:
dot_source, warnings = tokengraph_to_dot(tokengraph, depth=0)
```

`depth=1` adds every token exactly one edge from a root anchor (its subject, object, adverbials, ...); `depth=2` adds tokens two edges away; and so on. A token farther than `depth` is dropped entirely -- omitted as a node, exactly as if it had never been in `tokengraph`. Omit `depth` (or pass `None`) to show everything, same as before this parameter existed; a `depth` at or beyond `dot.max_graph_depth()`'s own return value for the tokengraph shows everything too; a negative `depth` raises `ValueError`.

A token's depth follows its `relatedtoken1` relation, falling back to `relatedtoken2` only when `relatedtoken1` itself doesn't resolve -- the same preference `compute_subordination_depths()` already uses to chase a verbal expression's own governor. This matters for a token playing two roles at once, most notably a relative pronoun that's both an anaphoric pointer (`relatedtoken1` -> its antecedent) and its own dependent clause's subject or object (`relatedtoken2` -> that clause's verb, which points back at the pronoun via its own "unit verb" relation) -- a genuine two-way link the data model allows (see e.g. `relative_pronoun_ho_aner_hon_eidon` in `tests/fixtures/gold_examples.py`, ὅν in "ὁ ἀνήρ ὅν εἶδον ἀπῆλθεν"). Preferring `relatedtoken1` and never averaging or taking a minimum over both avoids letting that second, forward-pointing edge collapse the pronoun's depth down to whatever the (only computable FROM the pronoun) dependent verb happens to resolve to. A token with no resolvable relation at all defaults to depth 0, the same "can't determine, default to root level" fallback `compute_subordination_depths()` and `tokengraph_to_depth_html()` both use.

Dropping a node can leave a KEPT node's edge pointing at a now-excluded one -- e.g. a relative pronoun kept at its own depth while the dependent verb it's also the direct object of is still excluded one depth further out. `tokengraph_to_dot()` skips that edge (with a warning) rather than emitting a dangling `->` line Graphviz would reject.

## A second cutoff: `aat_depth` (subordination/AAT depth)

Ported from arsgrammatica's own `tokengraph_to_mermaid()`/`tokengraph_to_dot()`, which each take an `aat_depth` parameter alongside (in `tokengraph_to_dot()`'s case) their own `depth` -- "AAT" is arsgrammatica's own term for its Agent-Action-Target graph; grammatike has no literal `compute_aat_depths()` of its own, so, exactly as `rank_by_depth` already does (see "Why a second renderer" above), `aat_depth` here is implemented against `verbal_units.compute_subordination_depths()` instead -- the same CLAUSE-level notion `rendering.tokengraph_to_depth_html()`'s own `depth` parameter uses at the block level.

Three renderers now carry a member of this same "depth in the AAT graph" family, each adapted to what that renderer actually does:

- `rendering.tokengraph_to_html(tokengraph, depth=None)` -- **color-limiting only**. A verbal unit whose subordination depth is `> depth` still renders as plain (escaped) text -- nothing is ever omitted, since this function reconstructs continuous prose. Named plain `depth` (no other depth notion exists on this function to collide with).
- `mermaid.tokengraph_to_mermaid(tokengraph, aat_depth=None)` -- **node omission**. A token whose verbal unit's subordination depth is `> aat_depth` is dropped as a node entirely, and any surviving node's edge that would dangle from a dropped one is skipped with a warning, same convention as `depth` filtering below.
- `dot.tokengraph_to_dot(tokengraph, depth=None, aat_depth=None)` -- **node omission, as a SECOND, independent cutoff** alongside the pre-existing graph-distance `depth` (see "Depth filtering" above): a node survives only if it passes BOTH. `depth=1, aat_depth=0` on the same tokengraph, for instance, can keep a token `depth` alone would have kept (graph distance 1) but `aat_depth` alone would have excluded (its own verbal unit turns out to be a subordinate clause, not the root one) -- the two cutoffs really are independent, not one subsuming the other.

All three treat a token with no verbal-unit assignment, or a verbal unit whose subordination depth couldn't be resolved (a relation cycle, or no governing verbal expression found), as depth 0 -- kept/colored, not dropped -- folding `compute_subordination_depths()`'s own warning about the unresolved case into the function's returned warnings (mermaid.py and dot.py only; `tokengraph_to_html()` has no warnings return at all, matching its pre-existing signature). Omit the parameter (or pass `None`, the default) on any of the three to get the old, unfiltered behavior; a negative value raises `ValueError` on all three, same convention as `depth`.

The three notebooks that show all four display types together (`greek_syntaxer_ctsdata.py`, `greek_syntaxer_review.py`, `greek_syntaxer_textinput.py` -- see "Diagram-tool toggle in the other notebooks" below) each drive all four from ONE `*Maximum depth of subordination to display*:` slider (the same slider that already existed for indented HTML): a `depth_value` cell reads it (guarded against the slider being `None` before anything's been analyzed/selected) and feeds `tokengraph_to_html(depth=depth_value)`, `tokengraph_to_depth_html(depth=depth_value)` (unchanged), `tokengraph_to_mermaid(aat_depth=depth_value)`, and `tokengraph_to_dot(aat_depth=depth_value)` -- `tokengraph_to_dot()`'s own, differently-scoped `depth` parameter is left at its default (unset) in these three notebooks, same as before; only `greek_syntaxer_dot.py` exposes that one, via its own graph-depth slider (see "`marimo/greek_syntaxer_dot.py`" above).

## Rendering

Generating the DOT text needs no dependency at all -- pure string building, same as `tokengraph_to_mermaid()`. Turning it into a picture needs Graphviz installed separately (the `dot` command-line tool is not a pip package):

```sh
dot -Tsvg analysis.dot > analysis.svg
dot -Tpng analysis.dot > analysis.png
```

Other options: paste the `.dot` text into an online Graphviz viewer (e.g. edotor.net), or Quarto's own fenced ```` ```{dot} ```` code-block support (needs Graphviz on the machine building the site) -- Quarto's own hand-written docs live in `quarto/`, separate from this file.

**marimo notebooks**: unlike `mo.mermaid()`, marimo has no built-in Graphviz display helper as of this writing. Showing a DOT diagram in a notebook cell needs one extra step -- render to SVG yourself (`subprocess` to the `dot` binary, or the `graphviz` PyPI package, which does the same subprocess call for you) and wrap the result in `mo.Html(svg_text)`:

```python
import graphviz
import marimo as mo

src = graphviz.Source(dot_source)
mo.Html(src.pipe(format="svg").decode("utf-8"))
```

The `graphviz` PyPI package is covered by this project's own `dev` extra: `pip install -e ".[dev]"` (see `pyproject.toml`). That's only the subprocess wrapper -- Graphviz's own `dot` executable still needs installing separately and on PATH (`brew install graphviz` on macOS, `apt install graphviz` on Linux).

## `marimo/greek_syntaxer_dot.py`

A dedicated notebook doing exactly that, end to end: browse for a previously-saved analysis file (`read_analyses()`'s own format), pick a sentence from a menu (`split_analysis_by_sentence()`), then generate and display its Graphviz diagram inline, with `orientation`/`color_by_verbal_unit`/`rank_by_depth` exposed as live toggles, a graph-depth slider (bounded by `dot.max_graph_depth()` -- see "Depth filtering" above; NOT the same depth notion as `greek_syntaxer_review.py`'s own indented-HTML slider, despite the visual similarity), and a "Download Graphviz DOT source (.dot)" button. No LM access needed -- it only reads an already-saved analysis, the same way `greek_syntaxer_review.py` does.

It degrades visibly through both Graphviz failure modes rather than crashing the cell: the `graphviz` package missing entirely (`pip install -e ".[dev]"` covers it) versus the package present but the `dot` executable not on PATH (`graphviz.ExecutableNotFound`, only raised once you actually try to render) -- either way you still get the "Download .dot source" button to render elsewhere.

This notebook remains the place for the FULL set of DOT-specific controls (`orientation`/`color_by_verbal_unit`/`rank_by_depth`/depth slider) -- the lighter diagram-tool toggle described next, ported into the other notebooks, deliberately doesn't duplicate all of that everywhere; it just lets you switch a notebook that already shows a diagram between Mermaid and Graphviz's own (default-parameter) rendering of the same tokengraph.

## Diagram-tool toggle in the other notebooks

Ported from arsgrammatica's own `latin_syntaxer_ctsdata.py`/`latin_syntaxer_textinput.py`/`latin_syntaxer_review.py`: `marimo/greek_syntaxer_ctsdata.py`, `marimo/greek_syntaxer_textinput.py`, and `marimo/greek_syntaxer_review.py` -- every notebook in this project that shows a Mermaid diagram of a tokengraph -- now also offer a `*Diagram tool*:` radio (`mermaid` / `graphviz`) right above that diagram. Both diagrams are always computed (`tokengraph_to_mermaid()` and `tokengraph_to_dot()` are both pure string building -- see "Rendering" above), but only the selected one is actually rendered and only it is downloadable at a time -- switching the radio swaps both the display and what the download button underneath it offers, the same "follows the widget" pattern each of these notebooks' own `save_extension` radio already used for its serialized-analysis download.

Unlike `greek_syntaxer_dot.py`, these three don't expose `tokengraph_to_dot()`'s own `orientation`/`color_by_verbal_unit`/`rank_by_depth`/`depth` parameters as separate widgets -- the Graphviz branch here just calls `tokengraph_to_dot(tokengraph)` with its own defaults (matching `tokengraph_to_mermaid()`'s defaults: `orientation="BT"`, colored). Reach for `greek_syntaxer_dot.py` instead when you want to tune those.

`graphviz` availability is checked the same way `greek_syntaxer_dot.py` checks it (see "Rendering" above): the radio only offers `"graphviz"` as an option when the package actually imported, and rendering still degrades visibly (a `mo.callout` pointing back at *Mermaid*, or at "download the source and render elsewhere") if the `dot` executable itself isn't on PATH.

Each notebook's download button keeps its own established filename/format convention for the Mermaid branch (`greek_syntaxer_ctsdata.py`/`greek_syntaxer_textinput.py`: a ```` ```mermaid ```` fenced code block saved as `.md`; `greek_syntaxer_review.py`: raw Mermaid source saved as `.mmd`) and adds a raw `.dot` file for the Graphviz branch, same as `greek_syntaxer_dot.py`'s own download.

## `utilities/analysis_to_dot.py`

A command-line counterpart to the notebook above, for scripting/piping instead of interactive use: reads a saved analysis file (`read_analyses()`'s own format) and writes its tokengraph's Graphviz DOT source to standard output, with `--orientation`/`--no-color`/`--no-rank` flags covering `tokengraph_to_dot()`'s own parameters. Only the DOT source goes to stdout -- warnings go to stderr instead -- so redirection and piping both work cleanly:

```sh
python utilities/analysis_to_dot.py analysis.cex > analysis.dot
python utilities/analysis_to_dot.py analysis.cex --orientation LR > analysis.dot
python utilities/analysis_to_dot.py analysis.cex --no-color --no-rank > analysis.dot

# Piped straight into Graphviz, if it's installed:
python utilities/analysis_to_dot.py analysis.cex | dot -Tsvg > analysis.svg
```

Unlike the notebook, it operates on the file's whole tokengraph as `read_analyses()` returns it -- one flat list spanning every sentence in the file, not split by sentence -- so use `marimo/greek_syntaxer_dot.py` instead if you want to pick a single sentence out of a multi-sentence file. No LM access needed, same as the notebook. No dedicated test file, matching `syntaxer_main.py`'s own precedent of no pytest coverage for CLI entry points -- it's a thin wrapper around `read_analyses()` and `tokengraph_to_dot()`, which are both already covered. It doesn't yet expose `tokengraph_to_dot()`'s `depth` parameter as a flag -- only the notebook's slider (and `analyses_to_png.py`'s own `--depth` flag, below) does, for now.

## `utilities/analyses_to_png.py`

A grammatike-specific addition, with no arsgrammatica counterpart to port: batch-renders one or more saved analysis files straight to PNG -- one image per FILE (the file's whole tokengraph, same granularity `analysis_to_dot.py` uses, not one image per sentence), written into a given output directory:

```sh
python utilities/analyses_to_png.py --outdir diagrams analysis1.cex analysis2.cex
python utilities/analyses_to_png.py --outdir diagrams *.cex --orientation LR
python utilities/analyses_to_png.py --outdir diagrams *.cex --no-color --no-rank --depth 2
```

Each output filename is the input file's own stem plus `.png` (e.g. `iliad_1.cex` -> `iliad_1.png`); two inputs that would collide on that stem (e.g. from different directories) get `_2`, `_3`, ... appended rather than one silently overwriting another. Requires both the `graphviz` package AND the Graphviz `dot` executable up front (it always renders all the way to PNG, unlike the notebook's graceful two-tier degradation) -- a missing package or executable is reported once, clearly, rather than as a wall of identical per-file errors. If you want a single sentence's own diagram out of a multi-sentence file, use `marimo/greek_syntaxer_dot.py`'s interactive sentence picker instead -- this script is for turning a batch of already-separate analysis files into a batch of pictures, not for splitting one file up.

## Tests

`tests/test_dot.py` covers generation only (node/edge selection, coloring, ranking, depth filtering) via plain string assertions on the DOT text -- no Graphviz binary needed to run `pytest`, matching this codebase's offline/DummyLM test philosophy. It reuses grammatike's own gold fixtures (`tests/fixtures/gold_examples.py`) rather than porting arsgrammatica's Latin ones, plus two hand-built `TokenAnalysis` fixtures for cases no gold example naturally exercises: a direct two-anchor relation cycle (confirming `rank_by_depth` excludes unresolved anchors and folds in `compute_subordination_depths()`'s own warning -- see "Why a second renderer" above), and a sentence-connector node (grammatike-specific coloring, no arsgrammatica precedent). The depth-filtering tests include a property check across every gold example, at every depth level from 0 up to that passage's own `max_graph_depth()`, confirming no edge is ever left dangling -- a KEPT node's edge pointing at a token `depth` excluded. Every gold example was also spot-checked against a real `dot` binary during development (rendered cleanly to SVG/PNG, `rank=same` statements included) -- not part of the automated suite, since that would add a system dependency to `pytest` itself.

`aat_depth` (see "A second cutoff" above) gets its own analogous coverage in each affected test file: `tests/test_rendering.py` for `tokengraph_to_html(depth=...)` (color-limiting, never omission -- checked against the same `ταῦρον` relative-clause fixture `tokengraph_to_depth_html()`'s own tests already use), `tests/test_mermaid_coloring.py` and `tests/test_dot.py` for the two renderers' `aat_depth` (node omission, composing with `color_by_verbal_unit`/`rank_by_depth`, and -- in `test_dot.py` only -- composing independently alongside the pre-existing `depth`), plus a property check in both files, mirroring `depth`'s own, confirming no edge is ever left dangling across every gold example at every `aat_depth` from 0 up to `max_subordination_depth()`. A dedicated `test_dot.py` case also confirms `rank_by_depth` and `aat_depth` share one `compute_subordination_depths()` call rather than duplicating its warning when both are active.
