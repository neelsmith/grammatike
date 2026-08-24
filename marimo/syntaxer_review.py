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
    # Review a saved Greek syntax analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > No LM access needed -- browse a previously-saved analysis file (the same format `write_analyses()` produces), pick one sentence, and inspect it.
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
def _(diagram, mo):
    mo.mermaid(diagram)
    return


@app.cell(hide_code=True)
def _(mermaid_download):
    mermaid_download
    return


@app.cell(hide_code=True)
def _(plaintext_html):
    plaintext_html
    return


@app.cell(hide_code=True)
def _(vuhtml):
    vuhtml
    return


@app.cell(hide_code=True)
def _(maxdepth):
    maxdepth
    return


@app.cell(hide_code=True)
def _(indentpsg):
    indentpsg
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
    # format -- see USAGE.md's "Saving and loading analyses"). A
    # file_browser is used for the same reason syntaxer_ctsdata.py's own
    # ctsdata_file_browser is: selecting a single FILE by clicking it just
    # works, unlike mo.ui.file_browser's "directory" selection mode.
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
    # grammatike/serialization.py -- so the rest of this notebook only
    # ever has to think about "the currently selected sentence's data".
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
# a citation, or happen to start with the same words.
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
    # The currently selected sentence's own tokengraph/verbalunits slice --
    # empty until a sentence is actually picked, which every rendering
    # utility below already handles gracefully (an empty diagram/string).
    # selected_citation rides along for the Mermaid download's filename
    # below -- pulled from the Sentence's own first Token, same source
    # sentence_label()'s own menu-entry citation comes from.
    selected_tokengraph, selected_verbalunits = [], []
    selected_citation = None
    if sentence_dropdown.value is not None and 0 <= sentence_dropdown.value < len(sentence_slices):
        selected_tokengraph, selected_verbalunits = sentence_slices[sentence_dropdown.value]
        selected_sentence = sentences[sentence_dropdown.value]
        selected_citation = selected_sentence.tokens[0].citation if selected_sentence.tokens else None
    return selected_citation, selected_tokengraph, selected_verbalunits


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Format output display
    """)
    return


@app.cell
def _(selected_tokengraph, tokengraph_to_mermaid):
    diagram, mermaid_warnings = tokengraph_to_mermaid(selected_tokengraph)
    return (diagram,)


@app.cell
def _(selected_citation, sentence_dropdown):
    # Same alphanumeric-sanitizing convention syntaxer_workflow.py's own
    # filename_base uses -- the sentence's own 1-based menu number goes
    # first (matching sentence_label()'s "<n>. ..." prefix) so every
    # download gets a distinct, stable name even across sentences that
    # share (or lack) a citation.
    mermaid_filename_stem = "sentence"
    if sentence_dropdown.value is not None:
        raw = f"{sentence_dropdown.value + 1}_{selected_citation or ''}"
        mermaid_filename_stem = "".join(c if c.isalnum() else "_" for c in raw).strip("_") or "sentence"
    return (mermaid_filename_stem,)


@app.cell
def _(diagram, mermaid_filename_stem, mo, selected_tokengraph):
    # mo.download() hands the raw Mermaid source (the same text
    # mo.mermaid() renders above) to the browser's own download mechanism
    # -- see syntaxer_workflow.py's "Download analysis" button for the
    # same pattern. Reusable directly in any other Mermaid-aware tool
    # (mermaid.live, a README code block, etc.), not just here.
    mermaid_download = mo.download(
        data=diagram.encode("utf-8"),
        filename=f"{mermaid_filename_stem}_mermaid.mmd",
        label="Download Mermaid diagram (.mmd)",
        mimetype="text/plain",
        disabled=not selected_tokengraph,
    )
    return (mermaid_download,)


@app.cell
def _(mo, selected_tokengraph, tokengraph_to_text):
    # Plain, uncolored text -- tokengraph_to_text() never emits HTML, but
    # the underlying surface text is still escaped before going into
    # mo.Html(), same as syntaxer_ctsdata.py's own raw-passage-preview cell.
    import html as _html

    plaintext_html = mo.Html(
        "<b><i>Plain text</i></b>: " + _html.escape(tokengraph_to_text(selected_tokengraph))
    )
    return (plaintext_html,)


@app.cell
def _(mo, selected_tokengraph, tokengraph_to_html):
    vuhtml = mo.Html("<b><i>Highlighted by verbal unit</i></b>: " + tokengraph_to_html(selected_tokengraph))
    return (vuhtml,)


@app.cell
def _(max_subordination_depth, mo, selected_tokengraph):
    # Same depth-cap slider as syntaxer_workflow.py/syntaxer_ctsdata.py --
    # left None until a sentence with at least one token is selected.
    maxdepth = None
    if selected_tokengraph:
        maxdepth = mo.ui.slider(
            start=0,
            stop=max_subordination_depth(selected_tokengraph),
            label="*Maximum depth of subordination to display*:",
            show_value=True,
            value=max_subordination_depth(selected_tokengraph),
        )
    return (maxdepth,)


@app.cell
def _(maxdepth, mo, selected_tokengraph, tokengraph_to_depth_html):
    # Guard against maxdepth being None (nothing selected yet) rather than
    # calling .value unconditionally -- unlike the other two notebooks'
    # otherwise-identical cell, which would raise AttributeError here.
    depth = maxdepth.value if maxdepth is not None else None
    indenthtml, indentwarnings = tokengraph_to_depth_html(selected_tokengraph, depth=depth)
    indentpsg = mo.Html("<b><i>Indented by verbal unit</i></b>: " + indenthtml)
    return (indentpsg,)


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
        max_subordination_depth,
        read_analyses,
        split_analysis_by_sentence,
        tokengraph_to_depth_html,
        tokengraph_to_html,
        tokengraph_to_mermaid,
        tokengraph_to_text,
    )

    return (
        Path,
        max_subordination_depth,
        read_analyses,
        split_analysis_by_sentence,
        tokengraph_to_depth_html,
        tokengraph_to_html,
        tokengraph_to_mermaid,
        tokengraph_to_text,
    )


if __name__ == "__main__":
    app.run()
