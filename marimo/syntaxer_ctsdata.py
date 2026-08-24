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
    # Analyze Ancient Greek syntax with a configured LM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > Read citable text from a delimited-text (CEX) file, then choose one or more passages -- selecting them automatically segments them into sentences -- then choose which sentence(s) to analyze.
    """)
    return


@app.cell(hide_code=True)
def _(ctsdata_file_browser):
    ctsdata_file_browser
    return


@app.cell(hide_code=True)
def _(ctsdata_error, ctsdata_rows, mo, passage_multiselect):
    if ctsdata_error is not None:
        ctsdata_status = mo.callout(
            mo.md(f"Could not read this file as a `#!ctsdata` source: {ctsdata_error}"),
            kind="danger",
        )
    elif not ctsdata_rows:
        ctsdata_status = mo.md("*Choose a source data file above to list its passages.*")
    else:
        ctsdata_status = mo.md(f"## Passage selection\n\n*{len(ctsdata_rows)} passage(s) loaded from this file. Selecting one or more below automatically segments them into sentences.*")

    mo.vstack(
        [ctsdata_status, mo.hstack([passage_multiselect], justify="start")]
    )
    return


@app.cell(hide_code=True)
def _(rawpreview):
    rawpreview
    return


@app.cell(hide_code=True)
def _(all_sentences, analyze_button, mo, sentence_multiselect):
    if not all_sentences:
        sentence_status = mo.md("*Select passage(s) above to automatically list their sentences.*")
    else:
        sentence_status = mo.md(f"## Sentence selection\n\n*{len(all_sentences)} sentence(s) found.*")

    mo.vstack(
        [sentence_status, mo.hstack([sentence_multiselect, analyze_button], justify="start")]
    )
    return


@app.cell(hide_code=True)
def _(sentence_rawpreview):
    sentence_rawpreview
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md("**Discussion**:\n\n" + "\n\n".join(f"> {result.reasoning}" for result in results))
    return


@app.cell(hide_code=True)
def _(psghtml):
    psghtml
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
def _(diagram, mo):
    mo.mermaid(diagram)
    return


@app.cell(hide_code=True)
def _(analysis_warnings, download_widget, mo, save_extension):
    mo.vstack(
        [
            mo.hstack([save_extension, download_widget], justify="start"),
        ]
        + (
            [mo.callout(mo.md("\n".join(f"- {w}" for w in analysis_warnings)), kind="warn")]
            if analysis_warnings
            else []
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    seetokens = mo.ui.checkbox(label="*See list of tokens*")
    seecost = mo.ui.checkbox(label="*See cost*")
    seeprompts = mo.ui.checkbox(label="*See prompts*")
    mo.hstack([seetokens, seeprompts, seecost], justify="start")
    return seecost, seeprompts, seetokens


@app.cell(hide_code=True)
def _(finaltokens, seetokens):
    tokendisplay = None
    if seetokens.value:
        tokendisplay = finaltokens

    tokendisplay
    return


@app.cell(hide_code=True)
def _(cost, mo, seecost):
    costdisplay = None
    if seecost.value:
        costdisplay = mo.md(f"**Cost of last LM call**: {cost}")
    costdisplay
    return


@app.cell(hide_code=True)
def _(dspy, seeprompts):
    prompts = None
    if seeprompts.value:
        prompts = dspy.inspect_history()
    prompts
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
    ## UI selections for source passages
    """)
    return


@app.cell
def _(ctsdata_file_browser, read_ctsdata):
    # Choose file with CEX source data.
    # Re-read the file every time the file_browser's own selection changes.
    ctsdata_path = ctsdata_file_browser.path(index=0)
    ctsdata_rows = []
    ctsdata_error = None
    if ctsdata_path is not None:
        try:
            ctsdata_rows = read_ctsdata(str(ctsdata_path))
        except (ValueError, OSError) as e:
            ctsdata_error = str(e)
    return ctsdata_error, ctsdata_rows


@app.cell
def _(Path, mo):
    # Browse for the delimited-text file listing passages to analyze (see
    # grammatike/ctsdata.py for the '#!ctsdata' block format). Unlike
    # the "choose a folder to save to" field syntaxer_workflow.py used to
    # have (see that notebook's own history: mo.ui.file_browser's
    # "directory" selection mode has no way to select the folder currently
    # being browsed, only a subfolder shown in its listing), selecting a
    # single FILE by clicking it works correctly -- there's no equivalent
    # gap for selection_mode="file" -- so a file_browser is used here
    # rather than a typed path.
    ctsdata_file_browser = mo.ui.file_browser(
        initial_path=Path(__file__).parent.parent,
        selection_mode="file",
        multiple=False,
        label="*Source data file*:",
    )
    return (ctsdata_file_browser,)


@app.function
# Format label for one menu entry as "<citation>: <first four words>…"
# The trailing "…" is only added when the passage actually has more words than the preview
# shows -- a passage that's already 4 words or shorter is shown in full.
def passage_label(row):
    words = row.text.split()
    preview = " ".join(words[:4])
    ellipsis = "…" if len(words) > 4 else ""
    return f"{row.citation}: {preview}{ellipsis}"


@app.cell
def _(ctsdata_rows, mo):
    # Menu for selecting one or more passages -- a multiselect rather than
    # a dropdown, since segment_sources() (see the Segmentation cell below)
    # accepts a list of sources and segments them together, not just one at
    # a time. Maps each label directly to a CtsDataRow, so
    # passage_multiselect.value is a list of the selected CtsDataRows (in
    # whatever order the widget itself reports them -- see selected_rows
    # below for why that order isn't used directly).
    passage_options = {passage_label(row): row for row in ctsdata_rows}
    passage_multiselect = mo.ui.multiselect(
        options=passage_options,
        label="*Passage(s)*:",
    )
    return (passage_multiselect,)


@app.cell
def _(ctsdata_rows, passage_multiselect):
    # Always segment selected passages in their original file order, not
    # whatever order passage_multiselect.value happens to report them in
    # (multiselect widgets are free to report selections in click order) --
    # segment_sources() treats consecutive sources as potentially sharing a
    # sentence, so an out-of-file-order source list could segment
    # incorrectly or produce citations in a confusing order.
    selected_rows = [row for row in ctsdata_rows if row in passage_multiselect.value]
    return (selected_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Segmentation and sentence selection
    """)
    return


@app.cell
def _(CitedText, segment_sources, selected_rows):
    # Segment every selected passage, together, automatically as soon as
    # the passage selection is non-empty -- no separate button, since
    # segmentation here is cheap enough (and useful enough as an immediate
    # preview of what a passage selection contains) not to gate behind an
    # extra click, unlike full SyntaxAnalysis below. Each selected row
    # becomes its own CitedText source (same urnbase+citation concatenation
    # the single-passage cell used to build) -- segment_sources() segments
    # across all of them at once (a sentence may span two consecutive
    # sources), in the file order selected_rows already established.
    all_sentences = []
    if selected_rows:
        sources = [
            CitedText(citation=row.urnbase + row.citation, text=row.text)
            for row in selected_rows
        ]
        all_sentences = segment_sources(sources)
    return (all_sentences,)


@app.function
# A readable (if approximate) surface rendering of one segmented Sentence's
# tokens, for menu labels and previews -- puts a space before every token,
# including punctuation and enclitics, same approximation pipeline.py's own
# (private) _render_sentence_text() uses internally. Not meant to be exact;
# tokengraph_to_text() is the faithful renderer, but it needs an *analyzed*
# tokengraph (tokentype, relations, etc.), which a freshly-segmented
# Sentence doesn't have yet.
def sentence_preview_text(sentence):
    return " ".join(tok.text for tok in sentence.tokens)


@app.function
# Format one menu entry as "<n>. <citation>: <first eight words>…" -- the
# citation comes from the sentence's own first token (sentences drawn from
# different selected passages carry different citations, unlike
# syntaxer_workflow.py's single-source case), and the number keeps entries
# unique even when two sentences share a citation or opening words.
def sentence_menu_label(index, sentence):
    preview_text = sentence_preview_text(sentence)
    words = preview_text.split()
    preview = " ".join(words[:8])
    ellipsis = "…" if len(words) > 8 else ""
    citation = sentence.tokens[0].citation if sentence.tokens else None
    prefix = f"{citation}: " if citation else ""
    return f"{index + 1}. {prefix}{preview}{ellipsis}"


@app.cell
def _(all_sentences, mo):
    # Menu for selecting which segmented sentence(s) to analyze -- maps
    # each label directly to that sentence's own index into all_sentences.
    sentence_options = {
        sentence_menu_label(i, sentence): i for i, sentence in enumerate(all_sentences)
    }
    sentence_multiselect = mo.ui.multiselect(
        options=sentence_options,
        label="*Sentence(s) to analyze*:",
    )
    return (sentence_multiselect,)


@app.cell
def _(mo, sentence_multiselect):
    # A new instance is created (and analyze_button.value resets to False)
    # every time sentence_multiselect's own selection changes, since this
    # cell depends on sentence_multiselect.value -- so changing the
    # selection always requires a fresh, deliberate Analyze click rather
    # than silently re-using a previous click.
    analyze_button = mo.ui.run_button(
        label="Analyze selected sentences",
        disabled=not sentence_multiselect.value,
    )
    return (analyze_button,)


@app.cell
def _(finaltokens, max_subordination_depth, mo):
    maxdepth = None
    if finaltokens:
        maxdepth = mo.ui.slider(start=0,stop=max_subordination_depth(finaltokens),label="*Maximum depth of subordination to display*:",show_value=True,value=max_subordination_depth(finaltokens))
    return (maxdepth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI selections for serialization
    """)
    return


@app.cell
def _(mo):
    save_extension = mo.ui.radio(
        options=["cex", "txt"], value="cex", inline=True, label="*File extension*:"
    )
    return (save_extension,)


@app.cell
def _(sentences):
    # A readable default filename base, drawn from every analyzed
    # sentence's own citation (falling back to "analysis" if nothing's
    # been analyzed yet), deduplicated and in analysis order -- unlike
    # syntaxer_workflow.py's single passage, here `sentences` is already
    # the *selected, analyzed* subset (see the Analysis cell below), not
    # every sentence segmentation found -- so the filename reflects what's
    # actually in the downloaded file. This can get long with many
    # sentences selected across several citations, but stays deterministic
    # and collision-resistant; the extension is chosen separately.
    _citation_values = []
    for _sentence in sentences:
        _citation = _sentence.tokens[0].citation if _sentence.tokens else None
        if _citation and _citation not in _citation_values:
            _citation_values.append(_citation)
    filename_base = "_".join(_citation_values)
    filename_base = "".join(c if c.isalnum() else "_" for c in filename_base).strip("_") or "analysis"
    return (filename_base,)


@app.cell
def _(analysis_text, filename_base, mo, results, save_extension):
    # mo.download() puts the browser in charge of where the file lands.
    # filename reactively follows both citation-derived filename_base and whichever extension is chosen.
    download_widget = mo.download(
        data=analysis_text.encode("utf-8"),
        filename=f"{filename_base}.{save_extension.value}",
        label="Download analysis",
        mimetype="text/plain",
        disabled=not results,
    )
    return (download_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Analysis
    """)
    return


@app.cell
def _(all_sentences, analyze_button, analyze_with_retry, sentence_multiselect, validate):
    # Run full syntax analysis only on the selected sentences, in original
    # sentence order -- not everything segmentation found -- so you only
    # spend an LM call on the sentence(s) you actually chose from the menu
    # above. analyze_button.value is True for exactly the one reactive
    # cycle triggered by a click. Mirrors pipeline.py's own
    # analyze_sources(), which does the same segment-then-analyze-each-
    # sentence loop over every selected passage; this is the same loop
    # scoped to a subset of the sentences segmentation actually found.
    sentences, results = [], []
    if analyze_button.value and sentence_multiselect.value:
        selected_indices = sorted(sentence_multiselect.value)
        sentences = [all_sentences[i] for i in selected_indices]
        for sentence in sentences:
            result = analyze_with_retry(passage=sentence_preview_text(sentence), tokens=sentence.tokens)
            problems = validate(sentence.tokens, result)
            if problems:
                first_id = sentence.tokens[0].id if sentence.tokens else "?"
                print(f"Validation warnings (sentence starting at {first_id}):")
                for p in problems:
                    print(f"  - {p}")
            results.append(result)
    return results, sentences


@app.cell
def _(combined_tokengraph, results, tokengraph_to_mermaid):
    # Compose Mermaid diagram:
    finaltokens = combined_tokengraph(results)
    diagram, mermaid_warnings = tokengraph_to_mermaid(finaltokens)
    return diagram, finaltokens


@app.cell
def _(sentences):
    tokens = [tok for sentence in sentences for tok in sentence.tokens]
    return


@app.cell
def _(results):
    vus = [res.verbalunits for res in results]
    return


@app.cell
def _(lm):
    last_call = None
    if lm.history:
        last_call = lm.history[-1]
    return (last_call,)


@app.cell
def _(last_call):
    cost = last_call.get('cost')
    return (cost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Format output display
    """)
    return


@app.cell
def _(mo, selected_rows):
    # Show the raw, as-selected passage text (one block per selection, in
    # file order) as soon as the menu selection changes -- no LM call
    # involved, so this can update immediately and independently of
    # whether Segment has been clicked yet. Lets the reader browse a whole
    # text passage-by-passage before spending an LM call on segmentation
    # at all.
    import html as _html

    if selected_rows:
        _blocks = [
            f"## Selected passage: {_html.escape(_row.citation)}\n\n{_html.escape(_row.text)}"
            for _row in selected_rows
        ]
        rawpreview = mo.md("\n\n---\n\n".join(_blocks))
    else:
        rawpreview = mo.md("")
    return (rawpreview,)


@app.cell
def _(all_sentences, mo, sentence_multiselect):
    # Show the selected sentences' own segmented text (tokenization only,
    # no syntax analysis) as soon as the sentence menu selection changes --
    # no additional LM call beyond the segmentation that already ran, so
    # this can update immediately and independently of whether Analyze has
    # been clicked yet. Always shown in original sentence order, not
    # whatever order the multiselect widget happens to report selections
    # in.
    import html as _html

    _selected_indices = sorted(sentence_multiselect.value) if sentence_multiselect.value else []
    if _selected_indices:
        _blocks = [
            f"## Sentence {i + 1}\n\n{_html.escape(sentence_preview_text(all_sentences[i]))}"
            for i in _selected_indices
        ]
        sentence_rawpreview = mo.md("\n\n---\n\n".join(_blocks))
    else:
        sentence_rawpreview = mo.md("")
    return (sentence_rawpreview,)


@app.cell
def _(finaltokens, mo, sentences, tokengraph_to_text):
    _citation_values = []
    for _sentence in sentences:
        _citation = _sentence.tokens[0].citation if _sentence.tokens else None
        if _citation and _citation not in _citation_values:
            _citation_values.append(_citation)
    citation_label = ", ".join(_citation_values)
    psghtml = mo.Html(
        f"<b><i>Reconstructed passage {citation_label}</i></b>: " + tokengraph_to_text(finaltokens)
    )
    return (psghtml,)


@app.cell
def _(finaltokens, mo, tokengraph_to_html):
    vuhtml = mo.Html("<b><i>Highlighted by verbal unit</i></b>: " + tokengraph_to_html(finaltokens))
    return (vuhtml,)


@app.cell
def _(finaltokens, maxdepth, mo, tokengraph_to_depth_html):
    indenthtml, indentwarnings = tokengraph_to_depth_html(finaltokens,depth=maxdepth.value)
    indentpsg = mo.Html("<b><i>Indented by verbal unit</i></b>: " + indenthtml)
    return (indentpsg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Serialize analysis to file
    """)
    return


@app.cell
def _(finaltokens, results, sentences, serialize_analyses):
    # Flatten every sentence's own verbalunits into the one flat list
    # serialize_analyses()/write_analyses() expect, matching how
    # combined_tokengraph() already flattens tokengraph across sentences.
    # `sentences` here is the selected-and-analyzed subset (see the
    # Analysis cell above), so a saved file only ever covers what was
    # actually analyzed, not every sentence segmentation found.
    all_verbalunits = [vu for result in results for vu in result.verbalunits]
    analysis_text, analysis_warnings = serialize_analyses(sentences, all_verbalunits, finaltokens)
    return analysis_text, analysis_warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    import dspy
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    return Path, dspy, load_dotenv, os


@app.cell
def _(Path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from grammatike import (
        CitedText,
        segment_sources,
        analyze_with_retry,
        validate,
        tokengraph_to_mermaid,
        combined_tokengraph,
        tokengraph_to_html,
        tokengraph_to_text,
        tokengraph_to_depth_html,
        serialize_analyses,
        read_ctsdata,
        max_subordination_depth,
    )

    return (
        CitedText,
        analyze_with_retry,
        combined_tokengraph,
        max_subordination_depth,
        read_ctsdata,
        segment_sources,
        serialize_analyses,
        tokengraph_to_depth_html,
        tokengraph_to_html,
        tokengraph_to_mermaid,
        tokengraph_to_text,
        validate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration of LM
    """)
    return


@app.cell
def _(Path, load_dotenv):
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    return


@app.cell
def _(os):
    api_base = os.getenv("API_BASE")
    model = os.getenv("MODEL")
    api_key = os.getenv("API_KEY")
    return


@app.cell
def _(os):
    def getenv(name: str, fallback_name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value:
            return value
        value = os.getenv(fallback_name)
        if value:
            return value
        return default


    return (getenv,)


@app.cell
def _(dspy, getenv):
    def configure_lm():
        if dspy.settings.lm is not None:
            return dspy.settings.lm

        api_base = getenv("API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm")
        model = getenv("MODEL", "MODEL", "litellm_proxy/anthropic/Claude Opus 5")
        api_key = getenv("API_KEY", "API_KEY")

        if not api_key:
            raise RuntimeError(
                "Missing API key. Set API_KEY (preferred) or API_KEY in your .env file."
            )

        lm = dspy.LM(model=model, api_base=api_base, api_key=api_key)
        dspy.configure(lm=lm)
        return lm


    return (configure_lm,)


@app.cell
def _(configure_lm):
    lm = configure_lm()
    return (lm,)


if __name__ == "__main__":
    app.run()
