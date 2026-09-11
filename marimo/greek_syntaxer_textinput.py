import marimo

__generated_with = "0.24.0"
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
    *Enter values for a base URN, passage reference, and text to analyze, then submit the form with the `Analyze` button. Segmenting the text into sentences (and how many there turn out to be) is handled internally -- if you want to control segmentation yourself, enter one sentence per submission.*
    """)
    return


@app.cell(hide_code=True)
def _(input_form):
    input_form
    return


@app.cell(hide_code=True)
def _(seecost):
    seecost
    return


@app.cell(hide_code=True)
def _(cost_summary, format_lm_cost, mo, seecost):
    costdisplay = mo.md("**Cost**: no LM calls yet.")
    if seecost.value:
        costdisplay = mo.md(f"**Total cost**: {format_lm_cost(cost_summary)}")
    costdisplay    
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
def _(diagram_tool):
    diagram_tool
    return


@app.cell(hide_code=True)
def _(diagram, diagram_tool, dot_source, dot_warnings, graphviz, mo):
    # Two distinct failure modes to degrade visibly from when
    # diagram_tool.value == "graphviz", same convention greek_syntaxer_dot.py's
    # own dot_display cell uses (see notes/dot_diagrams.md):
    #   - the `graphviz` package itself isn't installed -- not actually
    #     reachable here, since diagram_tool's own options only offer
    #     "graphviz" at all when graphviz_available is True (see that
    #     widget's own definition), but the "mermaid"-only fallback is
    #     what a user without the package ever sees instead;
    #   - it IS installed, but the Graphviz `dot` executable isn't on PATH
    #     (graphviz.ExecutableNotFound, only raised once you actually try
    #     to render something) -- this one genuinely can't be known ahead
    #     of time without trying, so it's still handled here.
    if diagram_tool.value == "graphviz":
        try:
            svg_bytes = graphviz.Source(dot_source).pipe(format="svg")
            diagram_display = mo.vstack(
                [mo.Html(svg_bytes.decode("utf-8"))]
                + (
                    [mo.callout(mo.md("\n".join(f"- {w}" for w in dot_warnings)), kind="warn")]
                    if dot_warnings
                    else []
                )
            )
        except graphviz.ExecutableNotFound:
            diagram_display = mo.callout(
                mo.md(
                    "The `graphviz` package is installed, but the Graphviz "
                    "`dot` command itself isn't on your system's PATH -- "
                    "install Graphviz separately (e.g. `brew install "
                    "graphviz` on macOS, `apt install graphviz` on Linux), "
                    "or switch back to *Mermaid* above. See "
                    "notes/dot_diagrams.md."
                ),
                kind="warn",
            )
    else:
        diagram_display = mo.mermaid(diagram)

    diagram_display
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
def _(diagram_download):
    diagram_download
    return


@app.cell
def _(mo):
    seetokens = mo.ui.checkbox(label="*See list of tokens*")
    seecost = mo.ui.checkbox(label="*See cost*")
    seeprompts = mo.ui.checkbox(label="*See prompts*")
    mo.hstack([seetokens, seeprompts], justify="start")
    return seecost, seeprompts, seetokens


@app.cell(hide_code=True)
def _(finaltokens, seetokens):
    tokendisplay = None
    if seetokens.value:
        tokendisplay = finaltokens

    tokendisplay
    return


@app.cell(hide_code=True)
def _():


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
    ## Analysis
    """)
    return


@app.cell
def _(analyze_passage, input_form):
    # Analyze the submitted text directly -- only once the form has been
    # submitted at least once (input_form.value is None until then), and
    # again on each subsequent submission, not on every keystroke in the
    # form's own inputs. analyze_passage() segments the passage into
    # sentences internally and analyzes each one in turn; there's no
    # separate sentence-selection step here, so a passage with more than
    # one sentence in it gets all of them analyzed together. If you want
    # to control segmentation yourself, enter one sentence per submission.
    sentences, results = [], []
    if input_form.value and input_form.value.get("text_area"):
        passage = input_form.value["text_area"]
        citation = input_form.value["urnbase"] + input_form.value["citation_context"]
        sentences, results = analyze_passage(passage, citation=citation)
    return results, sentences


@app.cell
def _(combined_tokengraph, results):
    # finaltokens lives in its own cell (rather than bundled into the
    # Mermaid-composing cell below, as it used to be) specifically so that
    # depth_value -- itself derived, via maxdepth, FROM finaltokens -- can
    # feed back into the Mermaid diagram without creating a circular
    # dependency (finaltokens -> maxdepth -> depth_value -> diagram would
    # otherwise need finaltokens again to produce diagram in the same
    # cell).
    finaltokens = combined_tokengraph(results)
    return (finaltokens,)


@app.cell
def _(depth_value, finaltokens, tokengraph_to_mermaid):
    # Compose Mermaid diagram. aat_depth=depth_value shares the same
    # subordination-depth cutoff as the colored HTML, indented HTML, and
    # Graphviz DOT views below -- see the depth_value cell's own comment --
    # omitting whole nodes (and any edge that would dangle from one) beyond
    # that depth, rather than just re-coloring them.
    diagram, mermaid_warnings = tokengraph_to_mermaid(finaltokens, aat_depth=depth_value)
    return (diagram,)


@app.cell
def _(depth_value, finaltokens, tokengraph_to_dot):
    # Compose Graphviz diagram: cheap to always compute regardless of
    # which tool is currently selected -- tokengraph_to_dot() is pure
    # string building with no dependency of its own (see
    # notes/dot_diagrams.md), unlike actually rendering it, which needs
    # the graphviz package and the `dot` executable (handled in
    # diagram_display above). aat_depth=depth_value is the same
    # subordination-depth cutoff the Mermaid diagram above uses -- a SECOND,
    # independent cutoff from this function's own `depth` parameter (graph-
    # edge distance), which is left at its default (unset) here.
    dot_source, dot_warnings = tokengraph_to_dot(finaltokens, aat_depth=depth_value)
    return dot_source, dot_warnings


@app.cell
def _(graphviz_available, mo):
    # "graphviz" is only ever offered as a choice when the graphviz PyPI
    # package actually imported successfully below -- this can't rule out
    # the OTHER failure mode (the package installed but the `dot`
    # executable missing from PATH), which is why diagram_display still
    # has to handle graphviz.ExecutableNotFound even though this list is
    # filtered. See notes/dot_diagrams.md.
    diagram_tool = mo.ui.radio(
        options=["mermaid", "graphviz"] if graphviz_available else ["mermaid"],
        value="mermaid",
        inline=True,
        label="*Diagram tool*:",
    )
    return (diagram_tool,)


@app.cell
def _(sentences):
    tokens = [tok for sentence in sentences for tok in sentence.tokens]
    return


@app.cell
def _(results):
    vus = [res.verbalunits for res in results]
    return


@app.cell
def _(lm, results, summarize_lm_cost):
    _ = results
    #
    # summarize_lm_cost() (grammatike/lm_cost.py) sums cost across EVERY
    # call in lm.history, not just the last one -- analyze_passage()
    # segments the submitted text into sentences internally and makes one
    # LM call per sentence, so this is what actually makes "Total cost"
    # above a total rather than just the last individual call's own cost.
    # It also never crashes on an empty history (true before the form's
    # first submission) or on a call served from dspy's own cache
    # (cost=None) -- see that module's own docstring.
    cost_summary = summarize_lm_cost(lm.history)
    return (cost_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Format output display
    """)
    return


@app.cell
def _(finaltokens, input_form, mo, tokengraph_to_text):
    citation_label = input_form.value["citation_context"] if input_form.value else ""
    psghtml = mo.Html(f"<b><i>Analyzed passage {citation_label}</i></b>: " + tokengraph_to_text(finaltokens))
    return (psghtml,)


@app.cell
def _(depth_value, finaltokens, mo, tokengraph_to_html):
    vuhtml = mo.Html("<b><i>Highlighted by verbal unit</i></b>: " + tokengraph_to_html(finaltokens, depth=depth_value))
    return (vuhtml,)


@app.cell
def _(finaltokens, max_subordination_depth, mo):
    # Same depth-cap slider as greek_syntaxer_ctsdata.py/greek_syntaxer_review.py --
    # left None until at least one sentence has been analyzed.
    maxdepth = None
    if finaltokens:
        maxdepth = mo.ui.slider(
            start=0,
            stop=max_subordination_depth(finaltokens),
            label="*Maximum depth of subordination to display*:",
            show_value=True,
            value=max_subordination_depth(finaltokens),
        )
    return (maxdepth,)


@app.cell
def _(maxdepth):
    # One shared depth value driving all four display types (colored HTML,
    # indented HTML, Mermaid, and Graphviz DOT) -- guarded against maxdepth
    # being None (nothing analyzed yet) rather than each of those four
    # cells calling maxdepth.value unconditionally.
    depth_value = maxdepth.value if maxdepth is not None else None
    return (depth_value,)


@app.cell
def _(depth_value, finaltokens, mo, tokengraph_to_depth_html):
    indenthtml, indentwarnings = tokengraph_to_depth_html(finaltokens, depth=depth_value)
    indentpsg = mo.Html("<b><i>Indented by verbal unit</i></b>: " + indenthtml)
    return (indentpsg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Save analysis
    """)
    return


@app.cell
def _(finaltokens, results, sentences, serialize_analyses):
    # Flatten every sentence's own verbalunits into the one flat list
    # serialize_analyses()/write_analyses() expect, matching how
    # combined_tokengraph() already flattens tokengraph across sentences.
    # results=results adds one '#!llm' block per sentence (MODEL env var +
    # that sentence's own result.reasoning) -- see serialization.py's
    # module docstring. Purely additive: read_analyses() ignores these
    # blocks, so older saved files (and this one, read back) still work.
    all_verbalunits = [vu for result in results for vu in result.verbalunits]
    analysis_text, analysis_warnings = serialize_analyses(
        sentences, all_verbalunits, finaltokens, results=results
    )
    return analysis_text, analysis_warnings


@app.cell
def _(input_form):
    # A readable default filename base, drawn from whatever citation the
    # form was submitted with (falling back to "analysis" if the passage
    # field was left blank) -- the extension is chosen separately, via
    # save_extension below.
    filename_base = ""
    if input_form.value:
        filename_base = (input_form.value.get("urnbase") or "") + (input_form.value.get("citation_context") or "")
    filename_base = "".join(c if c.isalnum() else "_" for c in filename_base).strip("_") or "analysis"
    return (filename_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(mo):
    # Placeholder base URN pointing at Lysias 1 ("On the Murder of
    # Eratosthenes"), matching syntax_model.md's own worked examples -- a
    # real user should point this at their own corpus, same as the Latin
    # original pointed at the author's own Livy-derived corpus.
    urnbase = mo.ui.text(value="urn:cts:greekLit:tlg0540.tlg001.perseus-grc2:", label="*Base URN*:")
    return (urnbase,)


@app.cell
def _(mo):
    citation_context = mo.ui.text(placeholder="urn:cts:greekLit:....", label="*Passage*:")
    return (citation_context,)


@app.cell
def _(mo):
    text_area = mo.ui.text_area(value = "τὴν θύραν ἀνέῳξεν.", full_width=True, label="*Text to analyze*:")
    return (text_area,)


@app.cell
def _(citation_context, mo, text_area, urnbase):
    # All three inputs as one form -- marimo only updates input_form.value
    # (and so only re-triggers the Analysis cell below) when the whole form
    # is submitted, never on every keystroke in an individual field.
    input_form = (
        mo.md(
            """
            {urnbase}

            {citation_context}

            {text_area}
            """
        )
        .batch(urnbase=urnbase, citation_context=citation_context, text_area=text_area)
        .form(submit_button_label="Analyze")
    )
    return (input_form,)


@app.cell
def _(mo):
    save_extension = mo.ui.radio(
        options=["cex", "txt"], value="cex", inline=True, label="*File extension*:"
    )
    return (save_extension,)


@app.cell
def _(analysis_text, filename_base, mo, results, save_extension):
    # mo.download() puts the browser in charge of where the file lands --
    # no folder-path field to mistype, at the cost of not choosing a
    # location up front (the browser's own download prompt/default
    # download folder decides that). filename reactively follows both the
    # citation-derived filename_base and whichever extension is chosen
    # above.
    download_widget = mo.download(
        data=analysis_text.encode("utf-8"),
        filename=f"{filename_base}.{save_extension.value}",
        label="Download analysis",
        mimetype="text/plain",
        disabled=not results,
    )
    return (download_widget,)


@app.cell
def _(diagram, diagram_tool, dot_source, filename_base, finaltokens, mo):
    # Downloads whichever diagram is currently selected/displayed above,
    # not both -- same reactive "follows the widget" convention
    # download_widget above uses for save_extension. Mermaid source is
    # wrapped in a ```mermaid fenced code block and saved as .md; Graphviz
    # source is saved raw as .dot -- both are renderable elsewhere (a
    # Markdown viewer with Mermaid support, `dot -Tsvg`, an online DOT
    # viewer, Quarto's fenced ```{dot}```/```{mermaid}``` blocks) without
    # needing this notebook.
    if diagram_tool.value == "graphviz":
        diagram_download = mo.download(
            data=dot_source.encode("utf-8"),
            filename=f"{filename_base}.dot",
            label="Download Graphviz DOT source (.dot)",
            mimetype="text/plain",
            disabled=not finaltokens,
        )
    else:
        diagram_download = mo.download(
            data=("```mermaid\n\n" + diagram + "\n```\n").encode("utf-8"),
            filename=f"{filename_base}.md",
            label="Download Mermaid diagram (.md)",
            mimetype="text/plain",
            disabled=not finaltokens,
        )
    return (diagram_download,)


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
        analyze_passage,
        tokengraph_to_mermaid,
        tokengraph_to_dot,
        combined_tokengraph,
        tokengraph_to_html,
        tokengraph_to_text,
        tokengraph_to_depth_html,
        serialize_analyses,
        summarize_lm_cost,
        format_lm_cost,
        max_subordination_depth,
    )

    # graphviz (the PyPI package -- a thin subprocess wrapper around the
    # separately-installed Graphviz `dot` executable) is optional: importable
    # or not, checked once here, rather than every display cell catching
    # ImportError itself. Whether the `dot` executable is actually on PATH
    # is a SEPARATE check (graphviz.ExecutableNotFound), made only when a
    # diagram is actually rendered -- see the diagram_display cell above.
    try:
        import graphviz

        graphviz_available = True
    except ImportError:
        graphviz = None
        graphviz_available = False
    return (
        analyze_passage,
        combined_tokengraph,
        format_lm_cost,
        graphviz,
        graphviz_available,
        max_subordination_depth,
        serialize_analyses,
        summarize_lm_cost,
        tokengraph_to_depth_html,
        tokengraph_to_dot,
        tokengraph_to_html,
        tokengraph_to_mermaid,
        tokengraph_to_text,
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
