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
def _(citation_context, mo, urnbase):
    mo.hstack([urnbase, citation_context], justify="start")
    return


@app.cell(hide_code=True)
def _(text_area):
    text_area
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
def _(indentpsg):
    indentpsg
    return


@app.cell(hide_code=True)
def _(diagram, mo):
    mo.mermaid(diagram)
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
def _(analyze_passage, citation_context, text_area, urnbase):
    # Analyze text passage:
    passage = ''
    sentences, results = [], []
    if text_area.value:
        passage = text_area.value
        sentences, results = analyze_passage(passage, citation = urnbase.value + citation_context.value)
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
    return (vus,)


@app.cell
def _(vus):
    vus[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Format output display
    """)
    return


@app.cell
def _(citation_context, finaltokens, mo, tokengraph_to_text):
    psghtml = mo.Html(f"<b><i>Passage {citation_context.value}</i></b>: " + tokengraph_to_text(finaltokens))
    return (psghtml,)


@app.cell
def _(finaltokens, mo, tokengraph_to_html):
    vuhtml = mo.Html("<b><i>Highlighted by verbal unit</i></b>: " + tokengraph_to_html(finaltokens))
    return (vuhtml,)


@app.cell
def _(finaltokens, mo, tokengraph_to_depth_html):
    indenthtml, indentwarnings = tokengraph_to_depth_html(finaltokens)
    indentpsg = mo.Html("<b><i>Indented by verbal unit</i></b>: " + indenthtml)
    return (indentpsg,)


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
def _():
    #input_form = mo.vstack([mo.hstack([urnbase, citation_context], justify="start"), text_area]).form()
    #
    # Instead:
    # Define individual input elements
    #name = mo.ui.text(label="Name")
    #age = mo.ui.number(start=0, stop=120, label="Age")
    #category = mo.ui.dropdown(options=["A", "B", "C"], label="Category")

    # Create a layout inside mo.md and bind them into a batch form
    #my_form = (
    #    mo.md(
    #        f"""
    #       ### Complex Input Form

    #       {name}

    #        {age}

    #        {category}
    #        """
    #    )
    #    .batch(name=name, age=age, category=category)
    #    .form()
    #

    # Display the form
    #my_form
    return


@app.cell
def _():
    return


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

    return Path, dspy, os


@app.cell
def _():
    from dotenv import load_dotenv

    return (load_dotenv,)


@app.cell
def _(Path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from grammatike import print_analysis, analyze_passage, tokengraph_to_mermaid, combined_tokengraph, tokengraph_to_html, tokengraph_to_text, tokengraph_to_depth_html

    return (
        analyze_passage,
        combined_tokengraph,
        tokengraph_to_depth_html,
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
    configure_lm()
    return


if __name__ == "__main__":
    app.run()
