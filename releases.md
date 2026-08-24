# Release history

No public releases yet. For known gaps and issues see the project [issue tracker](https://github.com/neelsmith/grammatike/issues).

Current work in progress in `main` branch includes a complete framework for developing, testing and optimizing ancient Greek syntactic analyzers with a wide variety of language models using `dspy`, including:

- a python package with a complete implementation of the initial syntactic scheme
- more than 480 tests verifying the structure of the code and its data structures
- configuration for any LM via litellm API using environmental variables or settings in `.env` file
- a command-line script (`syntaxer_main.py`) for interactive analysis of citable passages of ancient Greek
- utilities for visualizing syntactic analyses as Mermaid graphs, and as HTML display with a variety of syntactic highlighting
- serialization and loading of syntactic analyses to/from plain-text files
- utilities supporting automated loading of validated analyses into training set or evaluation data set
- optimization pipeline scaffolding against a given model using GEPA (`optimize_gepa.py`, `grammatike/gepa_metric.py`)
- "bakeoff" utility script to automate comparative testing of open models from Hugging Face or running locally on ollama