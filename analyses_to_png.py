"""
Batch-render one or more saved analysis files (write_analyses()'s own
format -- see USAGE.md's "Saving and loading analyses", or
notes/dot_diagrams.md) straight to PNG, one image per FILE, written into a
given output directory. A grammatike-specific addition alongside
analysis_to_dot.py -- arsgrammatica has no equivalent script -- built the
same way: no LM access, read_analyses() + tokengraph_to_dot(), Graphviz's
own `dot` executable doing the actual rendering (via the `graphviz` PyPI
package, a thin subprocess wrapper -- see notes/dot_diagrams.md).

Operates on each file's WHOLE tokengraph as read_analyses() returns it --
one image per file, spanning every sentence that file contains -- not one
image per sentence. If you want a single sentence's own diagram out of a
multi-sentence file, use marimo/greek_syntaxer_dot.py's interactive sentence
picker and its own "Download" controls instead; this script is for turning
a batch of already-separate analysis files into a batch of pictures, not
for splitting one file up.

Each output filename is the input file's own stem plus '.png' (e.g.
'iliad_1.cex' -> 'iliad_1.png'), written into --outdir (created if it
doesn't exist). Two input files that would produce the same stem (e.g. from
different directories) get '_2', '_3', ... appended to keep every output
distinct rather than one silently overwriting another.

Usage:
    python analyses_to_png.py --outdir diagrams analysis1.cex analysis2.cex
    python analyses_to_png.py --outdir diagrams *.cex --orientation LR
    python analyses_to_png.py --outdir diagrams *.cex --no-color --no-rank --depth 2

Requires the `graphviz` package (`pip install graphviz`, already covered by
`pip install -e ".[dev]"` -- see notes/dot_diagrams.md) AND Graphviz's own `dot`
command-line tool installed separately and on PATH (e.g. `brew install
graphviz` on macOS, `apt install graphviz` on Linux) -- generating the DOT
source itself needs neither, but this script always renders all the way to
PNG, so both are required up front rather than degrading gracefully the way
marimo/greek_syntaxer_dot.py's interactive display does.
"""

import argparse
import sys
from pathlib import Path

from grammatike import read_analyses, tokengraph_to_dot

try:
    import graphviz
except ImportError:
    graphviz = None


def _unique_stem(stem, used):
    """Return `stem`, or `stem` with a numeric suffix ('_2', '_3', ...) the
    first time it collides with an already-used name in `used` -- keeps
    every output file in one run distinct rather than letting one
    overwrite another."""
    if stem not in used:
        used.add(stem)
        return stem
    n = 2
    while f"{stem}_{n}" in used:
        n += 1
    candidate = f"{stem}_{n}"
    used.add(candidate)
    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-render saved analysis files' tokengraphs to PNG via Graphviz."
    )
    parser.add_argument(
        "analysis_files",
        nargs="+",
        help="One or more saved analysis files (write_analyses()'s own format).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write PNGs into (created if it doesn't exist).",
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
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Cap each diagram to nodes within this many edges of a root verbal-unit "
             "anchor (tokengraph_to_dot()'s own depth parameter -- see its docstring, "
             "or notes/dot_diagrams.md, for what this depth notion means). Omit for "
             "no cap.",
    )
    args = parser.parse_args()

    if graphviz is None:
        print(
            "The `graphviz` package isn't installed -- install it with "
            "`pip install graphviz` (already covered by `pip install -e \".[dev]\"` "
            "-- see notes/dot_diagrams.md) before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    used_stems = set()
    had_error = False
    executable_missing = False

    for analysis_file in args.analysis_files:
        if executable_missing:
            # Every remaining file would fail the exact same way -- one
            # message is enough, not one per file.
            break

        try:
            tokengraph, verbalunits, sentences = read_analyses(analysis_file)
        except (ValueError, OSError) as e:
            print(f"Could not read {analysis_file!r} as a saved analysis: {e}", file=sys.stderr)
            had_error = True
            continue

        dot_source, warnings = tokengraph_to_dot(
            tokengraph,
            orientation=args.orientation,
            color_by_verbal_unit=not args.no_color,
            rank_by_depth=not args.no_rank,
            depth=args.depth,
        )
        for w in warnings:
            print(f"Warning ({analysis_file}): {w}", file=sys.stderr)

        stem = _unique_stem(Path(analysis_file).stem, used_stems)
        out_path = outdir / f"{stem}.png"
        try:
            png_bytes = graphviz.Source(dot_source).pipe(format="png")
        except graphviz.ExecutableNotFound:
            print(
                "The `graphviz` package is installed, but the Graphviz `dot` command "
                "itself isn't on your system's PATH -- install Graphviz separately "
                "(e.g. `brew install graphviz` on macOS, `apt install graphviz` on "
                "Linux). See notes/dot_diagrams.md.",
                file=sys.stderr,
            )
            had_error = True
            executable_missing = True
            continue

        out_path.write_bytes(png_bytes)
        print(f"{analysis_file} -> {out_path}")

    sys.exit(1 if had_error else 0)
