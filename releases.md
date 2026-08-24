# Release history

Current version: **0.1.0**.


**0.1.0**, *Aug. 23, 2026*: Initial public release, built using Opus 5. Includes a complete framework for developing, testing and optimizing Ancient Greek syntactic analyzers with a wide variety of language models using `dspy`. This release includes:

    - a python package with a complete implementation of the initial syntactic scheme, adapted from arsgrammatica's Latin scheme to Greek's own constructions (e.g. a three-way split of participles into attributive/circumstantial/indirect-statement verbal expressions, and the implied `εἰμί`/implied-repetition token types)
    - 483 tests verifying the structure of the code and its data structures
    - configuration for any LM via litellm API using environmental variables or settings in `.env` file
    - a command-line script (`syntaxer_main.py`) for interactive analysis of citable passages of Ancient Greek
    - utilities for visualizing syntactic analyses as Mermaid graphs, and as HTML display with a variety of syntactic highlighting
    - serialization and loading of syntactic analyses to/from plain-text files
    - utilities supporting automated loading of validated analyses into training set or evaluation data set
    - optimization pipeline scaffolding against a given model using GEPA (`optimize_gepa.py`, `grammatike/gepa_metric.py`)
    - "bakeoff" utility script to automate comparative testing of open models from Hugging Face or running locally on ollama
