"""grammatike: a DSPy program analyzing the syntax of Ancient Greek passages
according to the scheme documented in syntax_model.md.

Greek analogue of `arsgrammatica` (https://github.com/neelsmith/arsgrammatica),
the same author's parallel package for analyzing Latin syntax.
"""

from .models import (
    Token,
    CitedText,
    Sentence,
    VerbalExpression,
    TokenAnalysis,
    RelationLabel,
    IMPLIED_TOKENTYPES,
)
from .mermaid import tokengraph_to_mermaid, save_mermaid
from .verbal_units import (
    assign_verbal_units,
    assign_verbal_unit_colors,
    compute_subordination_depths,
    max_subordination_depth,
    find_unanchored_coordinated_verbs,
)
from .rendering import tokengraph_to_text, tokengraph_to_html, tokengraph_to_depth_html
from .greek_syntax_dspy import (
    SyntaxAnalysis,
    analyze,
    validate,
    print_analysis,
)
from .segmentation import segment_sources
from .pipeline import analyze_sources, combined_tokengraph, analyze_passage
from .serialization import (
    serialize_analyses,
    write_analyses,
    read_analyses,
    read_llm_notes,
    split_analysis_by_sentence,
)
from .ctsdata import CtsDataRow, read_ctsdata
from .token_budget import (
    estimate_max_tokens,
    analyze_with_retry,
    get_calibration,
    DEFAULT_CEILING,
)
from .gepa_metric import syntax_metric

__all__ = [
    "Token",
    "CitedText",
    "Sentence",
    "VerbalExpression",
    "TokenAnalysis",
    "RelationLabel",
    "IMPLIED_TOKENTYPES",
    "tokengraph_to_mermaid",
    "save_mermaid",
    "assign_verbal_units",
    "assign_verbal_unit_colors",
    "compute_subordination_depths",
    "max_subordination_depth",
    "find_unanchored_coordinated_verbs",
    "tokengraph_to_text",
    "tokengraph_to_html",
    "tokengraph_to_depth_html",
    "SyntaxAnalysis",
    "analyze",
    "validate",
    "print_analysis",
    "segment_sources",
    "analyze_sources",
    "combined_tokengraph",
    "analyze_passage",
    "serialize_analyses",
    "write_analyses",
    "read_analyses",
    "read_llm_notes",
    "split_analysis_by_sentence",
    "CtsDataRow",
    "read_ctsdata",
    "estimate_max_tokens",
    "analyze_with_retry",
    "get_calibration",
    "DEFAULT_CEILING",
    "syntax_metric",
]
