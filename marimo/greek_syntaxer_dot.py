import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Explore a saved Greek analysis as a Graphviz diagram
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > No LM access needed -- browse a previously-saved analysis file (the same format `write_analyses()` produces), pick one sentence, and view its dependency graph as a Graphviz diagram. Unlike the Mermaid diagram (`greek_syntaxer_review.py`), same-subordination-depth verbal expressions are forced onto the same rank, not just left to Graphviz's own layout heuristics -- see `notes/dot_diagrams.md` for why. A depth slider lets you cap the diagram to nodes within a given number of edges of a root verb -- `0` shows only the root verb(s) themselves.
    """)
    return


@app.cell(hide_code=True)
def _(analysis_file_browser):
    analysis_file_browser
    return


@app.cell(hide_code=True)
def _(mo, read_error, sentence_dropdown, sentences, split_error):
    if read_error is not None:
        analysis_status = mo.callout(
            mo.md(f"Could not read this file as a saved analysis: {read_error}"),
            kind="danger",
        )
    elif split_error is not None:
        analysis_status = mo.callout(
            mo.md(f"Could not split this analysis by sentence: {split_error}"),
            kind="danger",
        )
    elif not sentences:
        analysis_status = mo.md("*Choose an analysis file above to list its sentences.*")
    else:
        analysis_status = mo.md(f"## Sentence selection\n\n*{len(sentences)} sentence(s) loaded from this file.*")

    mo.vstack([analysis_status, sentence_dropdown])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Diagram options
    """)
    return


@app.cell(hide_code=True)
def _(color_checkbox, mo, orientation_dropdown, rank_checkbox):
    mo.hstack([orientation_dropdown, rank_checkbox, color_checkbox], justify="start")
    return


@app.cell(hide_code=True)
def _(depth_slider):
    depth_slider
    return


@app.cell(hide_code=True)
def _(dot_display):
    dot_display
    return


@app.cell(hide_code=True)
def _(dot_download):
    dot_download
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("<hr/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI selections
    """)
    return


@app.cell
def _(Path, mo):
    # Browse for a previously-written analysis file (write_analyses()'s own
    # format -- see USAGE.md's "Saving and loading analyses" and
    # notes/dot_diagrams.md). A file_browser is used for the same reason
    # greek_syntaxer_review.py's own analysis_file_browser is: selecting a
    # single FILE by clicking it just works, unlike mo.ui.file_browser's
    # "directory" selection mode.
    analysis_file_browser = mo.ui.file_browser(
        initial_path=Path(__file__).parent.parent,
        selection_mode="file",
        multiple=False,
        label="*Analysis file*:",
    )
    return (analysis_file_browser,)


@app.cell
def _(analysis_file_browser, read_analyses):
    # Re-read the file every time the file_browser's own selection changes.
    # No LM call anywhere in this notebook -- read_analyses() reconstructs
    # everything from the file's own text.
    analysis_path = analysis_file_browser.path(index=0)
    tokengraph, verbalunits, sentences = [], [], []
    read_error = None
    if analysis_path is not None:
        try:
            tokengraph, verbalunits, sentences = read_analyses(str(analysis_path))
        except (ValueError, OSError) as e:
            read_error = str(e)
    return read_error, sentences, tokengraph, verbalunits


@app.cell
def _(sentences, split_analysis_by_sentence, tokengraph, verbalunits):
    # split_analysis_by_sentence() gives us each sentence's own tokengraph/
    # verbalunits slice out of the file's flat, whole-passage lists -- see
    # grammatike/serialization.py -- so the rest of this notebook only ever
    # has to think about "the currently selected sentence's tokengraph".
    sentence_slices = []
    split_error = None
    if sentences:
        try:
            sentence_slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)
        except ValueError as e:
            split_error = str(e)
    return sentence_slices, split_error


@app.function
# Label one menu entry as "<n>. <citation>: <first six words>…" -- numbered
# so entries are always unique even when several sentences share (or lack)
# a citation, or happen to start with the same words. Same convention
# greek_syntaxer_review.py's own sentence_label() uses.
def sentence_label(index, citation, sentence_tokengraph, tokengraph_to_text):
    preview_text = tokengraph_to_text(sentence_tokengraph)
    words = preview_text.split()
    preview = " ".join(words[:6])
    ellipsis = "…" if len(words) > 6 else ""
    prefix = f"{citation}: " if citation else ""
    return f"{index + 1}. {prefix}{preview}{ellipsis}"


@app.cell
def _(mo, sentence_slices, sentences, tokengraph_to_text):
    # Menu for selecting a sentence. Maps each label directly to that
    # sentence's own index, so sentence_dropdown.value is an int usable to
    # index into sentence_slices below -- zip() with sentences stops at
    # whichever list is shorter, so a split_analysis_by_sentence() failure
    # (sentence_slices left empty, sentences possibly not) can't produce a
    # mismatched, out-of-range index here.
    sentence_options = {}
    if sentence_slices:
        for i, (sentence, (sentence_tokengraph, _sentence_verbalunits)) in enumerate(
            zip(sentences, sentence_slices)
        ):
            citation = sentence.tokens[0].citation if sentence.tokens else None
            sentence_options[sentence_label(i, citation, sentence_tokengraph, tokengraph_to_text)] = i

    sentence_dropdown = mo.ui.dropdown(
        options=sentence_options,
        label="*Sentence*:",
    )
    return (sentence_dropdown,)


@app.cell
def _(sentence_dropdown, sentence_slices, sentences):
    # The currently selected sentence's own tokengraph -- empty until a
    # sentence is actually picked, which tokengraph_to_dot() already
    # handles gracefully (an empty digraph). selected_citation rides along
    # for the downloaded .dot file's name below.
    selected_tokengraph = []
    selected_citation = None
    if sentence_dropdown.value is not None and 0 <= sentence_dropdown.value < len(sentence_slices):
        selected_tokengraph, _selected_verbalunits = sentence_slices[sentence_dropdown.value]
        selected_sentence = sentences[sentence_dropdown.value]
        selected_citation = selected_sentence.tokens[0].citation if selected_sentence.tokens else None
    return selected_citation, selected_tokengraph


@app.cell
def _(mo):
    # tokengraph_to_dot()'s own three display-affecting parameters, exposed
    # directly so you can see rank_by_depth's effect (Graphviz's
    # `rank=same` forcing same-subordination-depth verbal expressions onto
    # one level -- see notes/dot_diagrams.md) by toggling it.
    orientation_dropdown = mo.ui.dropdown(
        options=["BT", "TB", "LR", "RL"],
        value="BT",
        label="*Orientation*:",
    )
    rank_checkbox = mo.ui.checkbox(value=True, label="Rank by subordination depth (`rank=same`)")
    color_checkbox = mo.ui.checkbox(value=True, label="Color by verbal unit")
    return color_checkbox, orientation_dropdown, rank_checkbox


@app.cell
def _(max_graph_depth, mo, selected_tokengraph):
    # Left None until a sentence with at least one token is selected, same
    # guard greek_syntaxer_review.py's own maxdepth slider uses. This caps
    # by GRAPH depth (dot.compute_graph_depths()) -- the number of edges
    # back to the nearest root verb, following the same relatedtoken1/
    # relatedtoken2 edges drawn as `->` lines -- NOT
    # verbal_units.compute_subordination_depths() (the CLAUSE-level notion
    # behind tokengraph_to_depth_html()'s own indented-HTML slider, where a
    # whole clause's subject/object/etc. share ONE depth with their verb,
    # and also what rank_checkbox's `rank=same` alignment above uses).
    # depth=0 shows ONLY root verbal-unit anchors, not the whole root
    # clause. See tokengraph_to_dot()'s own docstring and
    # notes/dot_diagrams.md.
    depth_slider = None
    if selected_tokengraph:
        depth_slider = mo.ui.slider(
            start=0,
            stop=max_graph_depth(selected_tokengraph),
            label="*Maximum graph depth from a root verb to include*:",
            show_value=True,
            value=max_graph_depth(selected_tokengraph),
        )
    return (depth_slider,)


@app.cell
def _(
    color_checkbox,
    depth_slider,
    orientation_dropdown,
    rank_checkbox,
    selected_tokengraph,
    tokengraph_to_dot,
):
    # Guard against depth_slider being None (nothing selected yet) rather
    # than calling .value unconditionally -- same guard
    # greek_syntaxer_review.py's own otherwise-identical cell uses for
    # maxdepth.
    depth = depth_slider.value if depth_slider is not None else None
    dot_source, dot_warnings = tokengraph_to_dot(
        selected_tokengraph,
        orientation=orientation_dropdown.value,
        color_by_verbal_unit=color_checkbox.value,
        rank_by_depth=rank_checkbox.value,
        depth=depth,
    )
    return dot_source, dot_warnings


@app.cell
def _(dot_source, dot_warnings, graphviz, graphviz_available, mo, selected_tokengraph):
    # Pipe the DOT source straight through Graphviz to SVG and drop it into
    # mo.Html() -- marimo has no built-in Graphviz display helper the way
    # mo.mermaid() exists for Mermaid, so this is the "one extra step"
    # notes/dot_diagrams.md describes. Two distinct failure modes to
    # degrade visibly from, same "don't just crash the cell" convention
    # every other notebook in this project uses:
    #   - the `graphviz` package itself isn't installed (graphviz_available,
    #     checked at import time below);
    #   - it IS installed, but the Graphviz `dot` executable isn't on PATH
    #     (graphviz.ExecutableNotFound, only raised once you actually try
    #     to render something).
    if not selected_tokengraph:
        dot_display = mo.md("*Choose a sentence above to see its Graphviz diagram.*")
    elif not graphviz_available:
        dot_display = mo.callout(
            mo.md(
                "The `graphviz` package isn't installed, so this diagram "
                "can't be rendered here. Install it with `pip install "
                "graphviz` (already covered by `pip install -e \".[dev]\"` "
                "-- see notes/dot_diagrams.md), or download the `.dot` source "
                "below and render it elsewhere (`dot -Tsvg analysis.dot > "
                "analysis.svg`). See notes/dot_diagrams.md."
            ),
            kind="warn",
        )
    else:
        try:
            svg_bytes = graphviz.Source(dot_source).pipe(format="svg")
            dot_display = mo.vstack(
                [mo.Html(svg_bytes.decode("utf-8"))]
                + (
                    [mo.callout(mo.md("\n".join(f"- {w}" for w in dot_warnings)), kind="warn")]
                    if dot_warnings
                    else []
                )
            )
        except graphviz.ExecutableNotFound:
            dot_display = mo.callout(
                mo.md(
                    "The `graphviz` package is installed, but the Graphviz "
                    "`dot` command itself isn't on your system's PATH -- "
                    "install Graphviz separately (e.g. `brew install "
                    "graphviz` on macOS, `apt install graphviz` on Linux), "
                    "or download the `.dot` source below and render it "
                    "elsewhere. See notes/dot_diagrams.md."
                ),
                kind="warn",
            )
    return (dot_display,)


@app.cell
def _(selected_citation, sentence_dropdown):
    # Same alphanumeric-sanitizing convention greek_syntaxer_workflow.py's
    # own filename_base and greek_syntaxer_review.py's own
    # mermaid_filename_stem use -- the sentence's own 1-based menu number
    # goes first (matching sentence_label()'s "<n>. ..." prefix) so every
    # download gets a distinct, stable name even across sentences that
    # share (or lack) a citation.
    dot_filename_stem = "sentence"
    if sentence_dropdown.value is not None:
        raw = f"{sentence_dropdown.value + 1}_{selected_citation or ''}"
        dot_filename_stem = "".join(c if c.isalnum() else "_" for c in raw).strip("_") or "sentence"
    return (dot_filename_stem,)


@app.cell
def _(dot_filename_stem, dot_source, mo, selected_tokengraph):
    # Hands the raw DOT source (the same text piped into Graphviz above) to
    # the browser's own download mechanism -- same mo.download() pattern
    # greek_syntaxer_review.py's own mermaid_download uses. Renderable
    # anywhere else Graphviz-aware (`dot -Tsvg`, an online DOT viewer,
    # Quarto's own fenced ```{dot}``` blocks), not just here.
    dot_download = mo.download(
        data=dot_source.encode("utf-8"),
        filename=f"{dot_filename_stem}.dot",
        label="Download Graphviz DOT source (.dot)",
        mimetype="text/plain",
        disabled=not selected_tokengraph,
    )
    return (dot_download,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from grammatike import (
        max_graph_depth,
        read_analyses,
        split_analysis_by_sentence,
        tokengraph_to_dot,
        tokengraph_to_text,
    )

    # graphviz (the PyPI package -- a thin subprocess wrapper around the
    # separately-installed Graphviz `dot` executable) is optional: importable
    # or not, checked once here, rather than every display cell catching
    # ImportError itself. Whether the `dot` executable is actually on PATH
    # is a SEPARATE check (graphviz.ExecutableNotFound), made only when a
    # diagram is actually rendered -- see the dot_display cell above.
    try:
        import graphviz

        graphviz_available = True
    except ImportError:
        graphviz = None
        graphviz_available = False
    return (
        Path,
        graphviz,
        graphviz_available,
        max_graph_depth,
        read_analyses,
        split_analysis_by_sentence,
        tokengraph_to_dot,
        tokengraph_to_text,
    )


if __name__ == "__main__":
    app.run()
