"""
Read a saved analysis file (write_analyses()'s own pipe-delimited format --
see USAGE.md's "Saving and loading analyses", or notes/dot_diagrams.md) and
print its tokengraph as Graphviz DOT source, via tokengraph_to_dot(). The
command-line counterpart to marimo/greek_syntaxer_dot.py's own interactive
DOT display, for scripting/piping instead of notebook use.

No LM access needed -- read_analyses() reconstructs everything from the
file's own text, the same way marimo/greek_syntaxer_dot.py's own
analysis_file_browser cell does.

Operates on the file's whole tokengraph as read_analyses() returns it --
already one flat list spanning every sentence in the file, the same shape
write_analyses() saved it in -- not one sentence at a time. Use
marimo/greek_syntaxer_dot.py instead if you want to pick a single sentence
out of a multi-sentence file.

Usage:
    python analysis_to_dot.py analysis.cex > analysis.dot
    python analysis_to_dot.py analysis.cex --orientation LR > analysis.dot
    python analysis_to_dot.py analysis.cex --no-color --no-rank > analysis.dot

    # Piped straight into Graphviz, if it's installed (see notes/dot_diagrams.md):
    python analysis_to_dot.py analysis.cex | dot -Tsvg > analysis.svg
"""

import argparse
import sys

from grammatike import read_analyses, tokengraph_to_dot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a saved analysis file's tokengraph as Graphviz DOT source."
    )
    parser.add_argument(
        "analysis_file",
        help="Path to a saved analysis file (write_analyses()'s own format).",
    )
    parser.add_argument(
        "--orientation",
        choices=["BT", "TB", "LR", "RL"],
        default="BT",
        help="DOT rankdir -- tokengraph_to_dot()'s own orientation values (default: %(default)s).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable verbal-unit coloring (tokengraph_to_dot()'s color_by_verbal_unit=False).",
    )
    parser.add_argument(
        "--no-rank",
        action="store_true",
        help="Disable forcing same-subordination-depth verbal expressions onto the same "
             "rank (tokengraph_to_dot()'s rank_by_depth=False).",
    )
    args = parser.parse_args()

    try:
        tokengraph, verbalunits, sentences = read_analyses(args.analysis_file)
    except (ValueError, OSError) as e:
        print(f"Could not read {args.analysis_file!r} as a saved analysis: {e}", file=sys.stderr)
        sys.exit(1)

    dot_source, warnings = tokengraph_to_dot(
        tokengraph,
        orientation=args.orientation,
        color_by_verbal_unit=not args.no_color,
        rank_by_depth=not args.no_rank,
    )

    # Only the DOT source goes to stdout, so `... > analysis.dot` or
    # `... | dot -Tsvg > analysis.svg` redirects/pipes cleanly; warnings go
    # to stderr instead -- same split syntaxer_main.py's own stdout output
    # uses.
    print(dot_source)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)
