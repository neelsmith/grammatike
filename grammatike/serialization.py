"""
grammatike: serializing/deserializing Greek syntax analyses to and from
disk. Greek analogue of arsgrammatica's serialization.py.

Deterministic plain-text serialization for a set of analyses: writes and
reads back the three flat lists analyze_sources()/analyze_passage() (plus
pipeline.py's combined_tokengraph()) naturally produce across however many
sentences and citation sources were analyzed --

    sentences:  List[Sentence]        (each Sentence.tokens: List[Token])
    verbalunits: List[VerbalExpression]
    tokengraph:  List[TokenAnalysis]

-- to and from one plain-text file, using '|' as the column separator, so
an analysis can be saved, diffed, hand-edited, or loaded back into exactly
the same three Python types without needing a database or a pickle file.

write_analyses() writes straight to a file; serialize_analyses() builds
the exact same text and warnings but returns the string instead of
writing it anywhere -- useful whenever the caller wants to embed this
format in something else (a log, a prompt, another file's contents, an
in-memory test fixture) rather than write a standalone file. The two
share one implementation: write_analyses() is a thin wrapper that calls
serialize_analyses() and writes its result to `path`.

File shape: three line-oriented, pipe-delimited blocks, each introduced by
a label line (one of '#!sentences', '#!verbal_units', '#!tokens' alone on
its own line) immediately followed by a fixed header line naming that
block's columns, then one data line per record. Blocks may appear in any
order (the label is what identifies a block, not its position), blank
lines between blocks are ignored, and all three blocks are required. A
fourth, optional kind of block, labelled '#!llm', holds the LM's own
`reasoning` output for one sentence's analysis -- see below.

Each of the three core labels may also appear MORE THAN ONCE -- e.g.
several '#!tokens' blocks, each with its own repeated header line,
scattered anywhere in the file. read_analyses() concatenates every block
sharing a label into that label's single combined row list, in file
order, before doing anything else with it -- so a file built by literally
concatenating several write_analyses()/serialize_analyses() outputs (each
a complete, self-contained trio of blocks) reads back exactly as if all
their sentences/verbalunits/tokengraph rows had been passed to a single
write_analyses() call to begin with. write_analyses() itself still only
ever emits one instance of each of the three core blocks; multiple
instances are something read_analyses() accepts, not something this
module produces for them. '#!llm' blocks are the one exception: when
`results` is given, write_analyses()/serialize_analyses() emit ONE
'#!llm' block per sentence (in the same order as `sentences`), so a
single write already produces several.

    #!sentences
    context_begin|first_token|context_end|last_token
    Lysias 1.1|t0|Lysias 1.1|t9

    #!verbal_units
    context|token|syntactic_type|semantic_type
    Lysias 1.1|t5|independent|transitive active

    #!tokens
    context|id|tokentype|text|lemma|verbalunit|related1|relationship1|related2|relationship2
    Lysias 1.1|t0|lexical|ἐγὼ|ἐγώ|||||

    #!llm
    MODEL=openai/gpt-4o-mini
    ἄειδε is the independent main verb (root, transitive active), with
    μῆνιν as its direct object and θεά as its subject.

An '#!llm' block is unlike the three core blocks: it has no fixed header
line or pipe-delimited columns. Its first line always has the form
'MODEL=<value>', recording the `MODEL` environment variable's value at
analysis time (written as an empty field, 'MODEL=', if that variable was
unset -- same None-as-empty-field convention as everywhere else in this
format); every following line, up to (but not including) the next
recognized block-label line or end of file, is that sentence's own
`result.reasoning` text, written verbatim, one file line per line of the
original string -- so, unlike every other block here, blank lines INSIDE
an '#!llm' block are significant (a real paragraph break in the
reasoning) and are not skipped the way blank lines between blocks
normally are. Exactly one blank line is still written after each '#!llm'
block as a separator (matching every other block), and exactly one
trailing blank line is stripped back off when reading -- so a reasoning
string that happens to end in its own blank line loses just that one
trailing blank line, not the paragraph breaks before it. This block is
purely additive: it exists so a saved analysis keeps the LM's own
rationale alongside it (useful for later review, or for building/curating
a GEPA optimization trainset -- see OPTIMIZING.md), not because any of
Sentence/VerbalExpression/TokenAnalysis has a `reasoning` field to
reconstruct one into. Passing `results` is optional everywhere it's
accepted; a file with no '#!llm' blocks at all reads back exactly as
before. read_analyses() itself does not return '#!llm' content (its
return shape has nowhere to put it, and this way every existing caller's
3-tuple unpacking keeps working unchanged) -- it only checks each '#!llm'
block it encounters is well-formed and skips over it. Use the dedicated
read_llm_notes() to get the (model, reasoning) pairs back out.

Why sentences/verbalunits/tokengraph aren't each self-contained: neither
VerbalExpression nor TokenAnalysis carries its own citation (only the
pre-analysis Token does -- see models.py's own note on why), and token ids
are global across a whole multi-sentence, multi-citation passage rather
than restarting per sentence. So `sentences` is what actually supplies
"context" (Token.citation) for a given token id, plus each sentence's own
boundaries; write_analyses() looks up every tokengraph/verbalunits row's
context by matching its id against `sentences`' own tokens, rather than
requiring TokenAnalysis/VerbalExpression to carry a redundant copy.

Round-tripping sentence boundaries back out of the file relies on one
invariant: the #!tokens block's row order is the same overall reading
order `sentences` implies when its tokens are read sentence-by-sentence,
token-by-token (exactly what combined_tokengraph() already assumes when
concatenating multiple sentences' tokengraphs -- see pipeline.py). Given
that, a sentence's tokens are recovered by finding its first_token/
last_token ids' *positions* in that row order and slicing between them,
rather than by parsing or sorting id strings -- ids are treated as opaque,
matching how models.py itself only guarantees they're "stable" and
"globally unique", not that they follow any particular numbering scheme.
write_analyses() checks this invariant itself and returns a warning (not
an error -- the file is still written) for any sentence whose own token
ids don't form a contiguous, matching-order run in the given tokengraph;
a file written with such a warning may not round-trip its sentence
boundaries correctly through read_analyses().

Field encoding: None serializes as an empty field (two adjacent '|'s, or
an empty field at the start/end of a line) and parses back as None --
this is the normal case for many fields (e.g. Token.citation is None for
any citation-free caller, and most tokens have no lemma/verbalunitid/
relatedtoken*/relationship* at all, per syntax_model.md's "Incomplete
status"). The literal sentinel string 'root' (an independent verb's own
relatedtoken1, per syntax_model.md) is written and read back verbatim,
like any other non-None string value -- it is never confused with an
empty/None field. Every field value is validated at write time to
contain neither '|' nor a newline (this format has no escaping mechanism
for either); Greek surface text/lemmas are not expected to ever contain
either character, so this is a defensive check, not an expected case.

Implied/elided tokens (tokentype in IMPLIED_TOKENTYPES -- 'implied eimi' or
'implied repetition'; see models.py's TokenAnalysis)
round-trip like any other #!tokens row -- their `text` column is empty,
same as any other None field, and reads back as None (not ''), same as
every other optional column. But they're excluded from a sentence's own
reconstructed `tokens` list in both directions: write_analyses() ignores
them when checking a sentence's tokens form a contiguous run in
`tokengraph`, and read_analyses() skips them when rebuilding each
Sentence's `tokens` -- since an implied token was never part of the
original per-sentence token list segmentation produced, only something
the analysis stage added afterward.

read_analyses() is deliberately strict, not "degrade visibly" like
tokengraph_to_mermaid()'s or compute_subordination_depths()'s warnings-
returning functions: a missing block, a header line that doesn't match
exactly, a wrong column count, a token id referenced by #!sentences or
#!verbal_units but absent from #!tokens, or a #!sentences/#!verbal_units
row whose own context column disagrees with what #!tokens recorded for
that same id, all raise ValueError immediately rather than silently
reconstructing something partial or wrong. The whole point of this format
is a faithful round trip; a malformed file should fail loudly and
specifically (naming the line and the problem) rather than hand back
subtly incorrect objects. An '#!llm' block is held to the same standard:
read_analyses() (and read_llm_notes()) both raise ValueError, naming the
line, for one whose first line doesn't have the form 'MODEL=...' -- even
though read_analyses() itself goes on to discard that block's content,
since it's not returned unless '#!llm' is well-formed either.
"""

import os
from typing import Dict, List, Optional, Tuple

from .models import IMPLIED_TOKENTYPES, Sentence, Token, TokenAnalysis, VerbalExpression

SENTENCES_LABEL = "#!sentences"
VERBAL_UNITS_LABEL = "#!verbal_units"
TOKENS_LABEL = "#!tokens"
LLM_LABEL = "#!llm"

SENTENCES_HEADER = "context_begin|first_token|context_end|last_token"
VERBAL_UNITS_HEADER = "context|token|syntactic_type|semantic_type"
TOKENS_HEADER = (
    "context|id|tokentype|text|lemma|verbalunit|"
    "related1|relationship1|related2|relationship2"
)

_EXPECTED_HEADERS = {
    SENTENCES_LABEL: SENTENCES_HEADER,
    VERBAL_UNITS_LABEL: VERBAL_UNITS_HEADER,
    TOKENS_LABEL: TOKENS_HEADER,
}

# Every recognized block-label line, core blocks plus '#!llm' -- used to
# tell where an '#!llm' block's own free-form body ends (see
# _scan_llm_block() below): unlike the three core blocks, '#!llm' has no
# fixed-column data rows to count, so its extent is bounded purely by
# "until the next line that is itself one of these labels, or EOF".
_ALL_LABELS = frozenset(_EXPECTED_HEADERS) | {LLM_LABEL}

_MODEL_PREFIX = "MODEL="


def _field(value: Optional[str], *, where: str) -> str:
    """Render one column value: None -> '' (see module docstring), any
    other string verbatim -- after checking it contains neither '|' (this
    format's only column separator, with no escaping) nor a newline,
    either of which would silently corrupt the line-oriented structure."""
    if value is None:
        return ""
    if "|" in value or "\n" in value or "\r" in value:
        raise ValueError(
            f"{where}: value {value!r} contains a '|' or a newline, which "
            "this pipe-delimited format has no way to escape"
        )
    return value


def _parse_optional(value: str) -> Optional[str]:
    """Inverse of `_field` for an optional column: '' -> None, anything
    else verbatim (including the literal string 'root', which is a real
    value, never a stand-in for empty)."""
    return value if value != "" else None


def _validate_llm_body_line(line: str, *, where: str) -> None:
    """A reasoning line must not itself equal a recognized block label
    (`_ALL_LABELS`) -- otherwise reading the file back would mistake it
    for the start of the next block, silently truncating the reasoning
    text at that point. Vanishingly unlikely for real LM-generated prose,
    but checked anyway, matching `_field`'s own "no escaping mechanism, so
    reject whatever would corrupt the structure" philosophy."""
    if line in _ALL_LABELS:
        raise ValueError(
            f"{where}: a reasoning line is exactly {line!r}, which this "
            "format would misread as the start of a new block"
        )


def serialize_analyses(
    sentences: List[Sentence],
    verbalunits: List[VerbalExpression],
    tokengraph: List[TokenAnalysis],
    results: Optional[list] = None,
) -> Tuple[str, List[str]]:
    """Build the exact text write_analyses() would write to a file, and
    return it directly as `(content, warnings)` instead of writing it
    anywhere -- see the module docstring for why this exists alongside
    write_analyses(). All three lists are flat and span however many
    sentences/citation sources were analyzed -- the same shape
    analyze_sources() (for `sentences`) and combined_tokengraph() (for
    `tokengraph`; `verbalunits` needs the analogous concatenation, which
    this function does not do for you) already produce.

    `results` is optional -- the same list analyze_sources()/
    analyze_passage() return alongside `sentences` (one entry per
    sentence, each with a `.reasoning` attribute; a dspy prediction from
    SyntaxAnalysis has exactly this shape). When given, it must have
    exactly one entry per entry of `sentences` (raises ValueError
    otherwise, naming the mismatched lengths) -- one '#!llm' block is
    written per sentence, in order, each recording the `MODEL` environment
    variable's current value and that sentence's own `result.reasoning`
    text (see the module docstring for the exact block shape). Omit
    `results` (the default) to write a file with no '#!llm' blocks at all,
    exactly as before this parameter existed.

    `content` is the complete file body, including its trailing newline,
    exactly as write_analyses() would have written it. `warnings` is a
    list of warning strings (empty if nothing looks wrong), matching this
    codebase's "degrade visibly, don't raise" convention for warnings
    distinct from hard errors:

    - a tokengraph or verbalunits entry whose id isn't found among any
      given sentence's tokens (so no citation is known for it -- an empty
      context is written, same as a token that legitimately has no
      citation at all, but this case specifically means the id wasn't
      found anywhere in `sentences` -- EXCEPT for an implied token
      (tokentype in IMPLIED_TOKENTYPES), which never appears in any sentence's own
      `tokens` by design, so this warning is suppressed for those
      specifically rather than flagged as an anomaly);
    - a sentence whose own tokens don't form a contiguous, matching-order
      run in `tokengraph`'s given order -- see the module docstring for
      why this matters for read_analyses() to recover sentence boundaries
      correctly.

    Raises ValueError for a sentence with no tokens at all (nothing to
    derive first_token/last_token from), or if any field value contains
    '|' or a newline (see `_field`).
    """
    warnings: List[str] = []

    id_to_citation: Dict[str, Optional[str]] = {}
    for sentence in sentences:
        for tok in sentence.tokens:
            id_to_citation[tok.id] = tok.citation

    # Implied tokens (tokentype in IMPLIED_TOKENTYPES) never appear in any sentence's
    # own `tokens` list by design (see the module docstring's note above)
    # -- so having no recorded citation is expected and correct for them,
    # not the kind of anomaly the "not found among the given sentences'
    # tokens" warning below exists to flag.
    implied_ids = {tok.id for tok in tokengraph if tok.tokentype in IMPLIED_TOKENTYPES}

    tg_index = {tok.id: i for i, tok in enumerate(tokengraph)}

    lines: List[str] = []

    lines.append(SENTENCES_LABEL)
    lines.append(SENTENCES_HEADER)
    for s_idx, sentence in enumerate(sentences):
        if not sentence.tokens:
            raise ValueError(
                f"sentence at index {s_idx} has no tokens -- cannot derive "
                "first_token/last_token for an empty sentence"
            )
        first_tok = sentence.tokens[0]
        last_tok = sentence.tokens[-1]

        first_pos = tg_index.get(first_tok.id)
        last_pos = tg_index.get(last_tok.id)
        if first_pos is None or last_pos is None:
            warnings.append(
                f"sentence at index {s_idx} (tokens {first_tok.id!r}.."
                f"{last_tok.id!r}) has a boundary token not present in the "
                "given tokengraph -- reading this file back may not "
                "reconstruct this sentence's tokens correctly"
            )
        else:
            expected_ids = [t.id for t in sentence.tokens]
            # Implied tokens (tokentype in IMPLIED_TOKENTYPES) were never part of the
            # original per-sentence `tokens` list -- they're synthesized by
            # analysis itself -- so exclude them here before comparing, or
            # every sentence containing one would spuriously warn.
            actual_ids = [
                tok.id
                for tok in tokengraph[first_pos : last_pos + 1]
                if tok.tokentype not in IMPLIED_TOKENTYPES
            ]
            if actual_ids != expected_ids:
                warnings.append(
                    f"sentence at index {s_idx} (tokens {first_tok.id!r}.."
                    f"{last_tok.id!r}) is not a contiguous, matching-order "
                    "run in the given tokengraph -- reading this file back "
                    "may not reconstruct this sentence's tokens correctly"
                )

        where = f"#!sentences row for sentence {s_idx}"
        lines.append(
            "|".join(
                [
                    _field(first_tok.citation, where=where),
                    _field(first_tok.id, where=where),
                    _field(last_tok.citation, where=where),
                    _field(last_tok.id, where=where),
                ]
            )
        )

    lines.append("")
    lines.append(VERBAL_UNITS_LABEL)
    lines.append(VERBAL_UNITS_HEADER)
    for vu in verbalunits:
        if vu.id not in id_to_citation and vu.id not in implied_ids:
            warnings.append(
                f"verbal expression {vu.id!r} not found among the given "
                "sentences' tokens -- writing an empty context for it"
            )
        where = f"#!verbal_units row for {vu.id}"
        lines.append(
            "|".join(
                [
                    _field(id_to_citation.get(vu.id), where=where),
                    _field(vu.id, where=where),
                    _field(vu.syntactic_type, where=where),
                    _field(vu.semantic_type, where=where),
                ]
            )
        )

    lines.append("")
    lines.append(TOKENS_LABEL)
    lines.append(TOKENS_HEADER)
    for tok in tokengraph:
        if tok.id not in id_to_citation and tok.id not in implied_ids:
            warnings.append(
                f"token {tok.id!r} not found among the given sentences' "
                "tokens -- writing an empty context for it"
            )
        where = f"#!tokens row for {tok.id}"
        lines.append(
            "|".join(
                [
                    _field(id_to_citation.get(tok.id), where=where),
                    _field(tok.id, where=where),
                    _field(tok.tokentype, where=where),
                    _field(tok.token, where=where),
                    _field(tok.lemma, where=where),
                    _field(tok.verbalunitid, where=where),
                    _field(tok.relatedtoken1, where=where),
                    _field(tok.relationship1, where=where),
                    _field(tok.relatedtoken2, where=where),
                    _field(tok.relationship2, where=where),
                ]
            )
        )

    if results is not None:
        if len(results) != len(sentences):
            raise ValueError(
                f"results has {len(results)} entries but sentences has "
                f"{len(sentences)} -- serialize_analyses() needs exactly "
                "one result per sentence to label each '#!llm' block"
            )
        model = os.environ.get("MODEL")
        for s_idx, result in enumerate(results):
            where = f"#!llm block for sentence {s_idx}"
            lines.append("")
            lines.append(LLM_LABEL)
            lines.append(_MODEL_PREFIX + _field(model, where=where))
            normalized_reasoning = str(result.reasoning).replace("\r\n", "\n").replace("\r", "\n")
            for reasoning_line in normalized_reasoning.split("\n"):
                _validate_llm_body_line(reasoning_line, where=where)
                lines.append(reasoning_line)

    return "\n".join(lines) + "\n", warnings


def write_analyses(
    sentences: List[Sentence],
    verbalunits: List[VerbalExpression],
    tokengraph: List[TokenAnalysis],
    path: str,
    results: Optional[list] = None,
) -> List[str]:
    """Write `sentences`/`verbalunits`/`tokengraph` to `path` in the format
    this module's docstring describes -- see serialize_analyses() (which
    this is a thin wrapper around) for what's actually written and for the
    full list of warnings this can return. `results` is optional and
    passed straight through -- see serialize_analyses()'s own docstring
    for the '#!llm' blocks it produces when given.

    Returns a list of warning strings (empty if nothing looks wrong); see
    serialize_analyses()'s docstring for what each one means. Raises
    ValueError for a sentence with no tokens at all (nothing to derive
    first_token/last_token from), if any field value contains '|' or a
    newline (see `_field`), if `results` is given with a different length
    than `sentences`, or if a reasoning line collides with a block label
    (see `_validate_llm_body_line`) -- all raised by serialize_analyses()
    before this function ever opens `path`.
    """
    content, warnings = serialize_analyses(sentences, verbalunits, tokengraph, results=results)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return warnings


def _scan_llm_block(
    raw_lines: List[str], start: int, label_line_no: int
) -> Tuple[Optional[str], str, int]:
    """Parse one '#!llm' block's body, given `raw_lines` and `start` (the
    0-based index of the line right after the '#!llm' label line itself,
    which was found at 1-based `label_line_no`). Consumes every line up to
    (but not including) the next line that is itself one of `_ALL_LABELS`,
    or end of file -- see the module docstring for why blank lines are
    significant inside this block, unlike everywhere else in the format.

    Returns `(model, reasoning, next_index)`: `model` is the 'MODEL='
    line's value (None if empty, same None-as-empty-field convention as
    `_parse_optional`); `reasoning` is every following line joined by
    '\\n', with exactly one trailing blank line stripped (the writer's own
    separator, not part of the reasoning itself -- see the module
    docstring); `next_index` is the 0-based index of the first line NOT
    consumed (the next label line, or `len(raw_lines)` at EOF), for the
    caller to resume scanning from.

    Raises ValueError, naming `label_line_no` or the offending line, if
    the block has no line at all before the next label/EOF, or if its
    first line doesn't start with 'MODEL='.
    """
    if start >= len(raw_lines) or raw_lines[start] in _ALL_LABELS:
        raise ValueError(
            f"line {label_line_no}: {LLM_LABEL!r} block has a label line "
            "but no 'MODEL=' line before the next block starts (or the "
            "file ends)"
        )
    first_line = raw_lines[start]
    if not first_line.startswith(_MODEL_PREFIX):
        raise ValueError(
            f"line {start + 1}: expected an {LLM_LABEL!r} block's first "
            f"line to start with {_MODEL_PREFIX!r}, got {first_line!r}"
        )
    model = _parse_optional(first_line[len(_MODEL_PREFIX):])

    i = start + 1
    body_lines: List[str] = []
    while i < len(raw_lines) and raw_lines[i] not in _ALL_LABELS:
        body_lines.append(raw_lines[i])
        i += 1
    while body_lines and body_lines[-1] == "":
        body_lines.pop()

    return model, "\n".join(body_lines), i


def read_analyses(
    path: str,
) -> Tuple[List[TokenAnalysis], List[VerbalExpression], List[Sentence]]:
    """Read `path` (as written by write_analyses()/serialize_analyses()) and
    reconstruct `(tokengraph, verbalunits, sentences)` -- in that order,
    matching the order these three types are usually discussed in this
    codebase (the token-level graph, then the verbal-expression table,
    then the sentence/citation structure that supplies context for both).

    Each of the three block labels may appear more than once in `path`
    (see the module docstring) -- every instance contributes its own rows,
    in file order, to that label's combined row list, as if the file were
    the concatenation of however many separate write_analyses()/
    serialize_analyses() outputs it actually is. Any '#!llm' blocks in
    `path` are checked for well-formedness and then skipped -- their
    (model, reasoning) content isn't part of this function's return shape;
    use read_llm_notes() to get it.

    Raises ValueError, naming the offending line and problem, for anything
    that isn't a faithful, internally-consistent file written by
    write_analyses() -- see this module's own docstring for exactly what's
    checked. This function does not accept a file with warnings-worthy
    inconsistencies silently patched over; if write_analyses() returned
    warnings when the file was written, fix the input and re-write it
    rather than expecting read_analyses() to compensate.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    # blocks[label] accumulates (line_no, line) data rows across every
    # instance of that label found in the file, in file order. A label
    # line always starts a new instance and must be immediately followed
    # by that label's header line (`awaiting_header` tracks this) before
    # any more data rows can be appended to it -- this holds per instance,
    # not just for the label's first appearance, so every repeated block
    # must repeat its own header line too.
    blocks: Dict[str, List[Tuple[int, str]]] = {label: [] for label in _EXPECTED_HEADERS}
    seen_labels = set()
    current_label: Optional[str] = None
    awaiting_header = False

    i = 0
    n = len(raw_lines)
    while i < n:
        line_no = i + 1
        line = raw_lines[i]

        if line == LLM_LABEL:
            if awaiting_header:
                raise ValueError(
                    f"line {line_no}: block {current_label!r} has a label "
                    "line but no header line before the next block starts"
                )
            # Validated and discarded here -- current_label/awaiting_header
            # are left exactly as they were, so an '#!llm' block can sit
            # between two other blocks (or inside one's own data run,
            # though this module never writes it that way itself) without
            # disturbing whatever block was already in progress.
            _model, _reasoning, i = _scan_llm_block(raw_lines, i + 1, line_no)
            continue

        if line.strip() == "":
            i += 1
            continue

        if line in _EXPECTED_HEADERS:
            if awaiting_header:
                raise ValueError(
                    f"line {line_no}: block {current_label!r} has a label "
                    "line but no header line before the next block starts"
                )
            current_label = line
            seen_labels.add(line)
            awaiting_header = True
            i += 1
            continue

        if current_label is None:
            raise ValueError(
                f"line {line_no}: data line {line!r} appears before any "
                "'#!' block label"
            )

        if awaiting_header:
            expected = _EXPECTED_HEADERS[current_label]
            if line != expected:
                raise ValueError(
                    f"line {line_no}: expected header {expected!r} for "
                    f"block {current_label!r}, got {line!r}"
                )
            awaiting_header = False
            i += 1
            continue

        blocks[current_label].append((line_no, line))
        i += 1

    missing = sorted(set(_EXPECTED_HEADERS) - seen_labels)
    if missing:
        raise ValueError(f"file is missing required block(s): {missing}")
    if awaiting_header:
        raise ValueError(
            f"block {current_label!r} has a label line but no header line "
            "(and no data) -- the file ends too early"
        )

    # --- #!tokens: build the TokenAnalysis list, the id->citation map,
    # and the row-order index sentence reconstruction relies on. ---
    tokengraph: List[TokenAnalysis] = []
    id_to_citation: Dict[str, Optional[str]] = {}
    row_order: List[str] = []

    for line_no, line in blocks[TOKENS_LABEL]:
        parts = line.split("|")
        if len(parts) != 10:
            raise ValueError(
                f"line {line_no}: #!tokens row has {len(parts)} columns, "
                f"expected 10: {line!r}"
            )
        (
            context,
            tok_id,
            tokentype,
            text,
            lemma,
            verbalunit,
            related1,
            relationship1,
            related2,
            relationship2,
        ) = parts
        if tok_id == "":
            raise ValueError(f"line {line_no}: #!tokens row has an empty id")
        if tok_id in id_to_citation:
            raise ValueError(f"line {line_no}: duplicate token id {tok_id!r} in #!tokens")

        tokengraph.append(
            TokenAnalysis(
                id=tok_id,
                token=_parse_optional(text),
                tokentype=tokentype,
                lemma=_parse_optional(lemma),
                verbalunitid=_parse_optional(verbalunit),
                relatedtoken1=_parse_optional(related1),
                relationship1=_parse_optional(relationship1),
                relatedtoken2=_parse_optional(related2),
                relationship2=_parse_optional(relationship2),
            )
        )
        id_to_citation[tok_id] = _parse_optional(context)
        row_order.append(tok_id)

    id_position = {tid: i for i, tid in enumerate(row_order)}

    # --- #!verbal_units ---
    verbalunits: List[VerbalExpression] = []
    for line_no, line in blocks[VERBAL_UNITS_LABEL]:
        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"line {line_no}: #!verbal_units row has {len(parts)} "
                f"columns, expected 4: {line!r}"
            )
        context, vu_id, syntactic_type, semantic_type = parts
        if vu_id == "":
            raise ValueError(f"line {line_no}: #!verbal_units row has an empty token id")
        if vu_id not in id_to_citation:
            raise ValueError(
                f"line {line_no}: #!verbal_units references token id "
                f"{vu_id!r}, which does not appear in the #!tokens block"
            )
        recorded_context = _parse_optional(context)
        expected_context = id_to_citation[vu_id]
        if recorded_context != expected_context:
            raise ValueError(
                f"line {line_no}: #!verbal_units row's context "
                f"{recorded_context!r} for token {vu_id!r} does not match "
                f"the #!tokens block's recorded context {expected_context!r} "
                "for the same id"
            )

        verbalunits.append(
            VerbalExpression(
                id=vu_id,
                syntactic_type=syntactic_type,
                semantic_type=semantic_type,
            )
        )

    # --- #!sentences ---
    sentences: List[Sentence] = []
    for line_no, line in blocks[SENTENCES_LABEL]:
        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"line {line_no}: #!sentences row has {len(parts)} "
                f"columns, expected 4: {line!r}"
            )
        context_begin, first_id, context_end, last_id = parts
        if first_id == "" or last_id == "":
            raise ValueError(
                f"line {line_no}: #!sentences row is missing first_token "
                f"or last_token: {line!r}"
            )
        if first_id not in id_position or last_id not in id_position:
            raise ValueError(
                f"line {line_no}: #!sentences references a first_token/"
                "last_token id not found in the #!tokens block"
            )

        start = id_position[first_id]
        end = id_position[last_id]
        if start > end:
            raise ValueError(
                f"line {line_no}: #!sentences row's first_token "
                f"{first_id!r} comes after last_token {last_id!r} in the "
                "#!tokens block's row order"
            )

        parsed_begin = _parse_optional(context_begin)
        parsed_end = _parse_optional(context_end)
        if parsed_begin != id_to_citation[first_id]:
            raise ValueError(
                f"line {line_no}: #!sentences row's context_begin "
                f"{parsed_begin!r} does not match the #!tokens block's "
                f"recorded context {id_to_citation[first_id]!r} for token "
                f"{first_id!r}"
            )
        if parsed_end != id_to_citation[last_id]:
            raise ValueError(
                f"line {line_no}: #!sentences row's context_end "
                f"{parsed_end!r} does not match the #!tokens block's "
                f"recorded context {id_to_citation[last_id]!r} for token "
                f"{last_id!r}"
            )

        sentence_ids = [
            tid
            for tid in row_order[start : end + 1]
            if tokengraph[id_position[tid]].tokentype not in IMPLIED_TOKENTYPES
        ]
        sentences.append(
            Sentence(
                tokens=[
                    Token(
                        id=tid,
                        text=tokengraph[id_position[tid]].token,
                        citation=id_to_citation[tid],
                    )
                    for tid in sentence_ids
                ]
            )
        )

    return tokengraph, verbalunits, sentences


def read_llm_notes(path: str) -> List[Tuple[Optional[str], str]]:
    """Read `path` (as written by write_analyses()/serialize_analyses())
    and return every '#!llm' block's own `(model, reasoning)` pair, in
    file order -- the counterpart to read_analyses(), which parses the
    same file but deliberately discards '#!llm' content (see the module
    docstring for why: none of Sentence/VerbalExpression/TokenAnalysis has
    a `reasoning` field to reconstruct one into, and changing
    read_analyses()'s own 3-tuple return would break every existing
    caller). Concatenates every '#!llm' block found in `path`, the same
    "multiple instances, in file order" convention read_analyses() already
    applies to the three core blocks -- so a file built by literally
    concatenating several write_analyses(..., results=...) outputs returns
    every one of their reasoning entries, in order, exactly as if they'd
    all been written by a single call with a longer `results` list.

    `model` is None wherever the 'MODEL' environment variable was unset at
    write time (an empty 'MODEL=' line, same None-as-empty-field
    convention used everywhere else in this format); `reasoning` is the
    exact, verbatim multiline text originally passed as that sentence's
    own `result.reasoning`, with exactly one trailing blank line stripped
    (the writer's own block separator -- see the module docstring).

    Returns an empty list for a file with no '#!llm' blocks at all --
    including any file written before this parameter existed, or any
    write_analyses()/serialize_analyses() call that omitted `results`.

    Raises ValueError, naming the line, for a malformed '#!llm' block (a
    label line with nothing after it before the next block or EOF, or a
    first line that doesn't start with 'MODEL=') -- the same check
    read_analyses() applies to every '#!llm' block it skips over, so a
    file that reads cleanly with one of these two functions reads cleanly
    with the other.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    notes: List[Tuple[Optional[str], str]] = []
    i = 0
    n = len(raw_lines)
    while i < n:
        if raw_lines[i] == LLM_LABEL:
            model, reasoning, i = _scan_llm_block(raw_lines, i + 1, i + 1)
            notes.append((model, reasoning))
        else:
            i += 1

    return notes


def split_analysis_by_sentence(
    tokengraph: List[TokenAnalysis],
    verbalunits: List[VerbalExpression],
    sentences: List[Sentence],
) -> List[Tuple[List[TokenAnalysis], List[VerbalExpression]]]:
    """The inverse of what write_analyses()/serialize_analyses() flatten
    together: given the same `(tokengraph, verbalunits, sentences)` triple
    read_analyses() returns (or that analyze_sources()/combined_tokengraph()
    produce before ever being written to a file), split `tokengraph` and
    `verbalunits` back into one slice per sentence.

    Returns a list the same length and order as `sentences` -- entry i is
    `(sentence_tokengraph, sentence_verbalunits)` for `sentences[i]`. Useful
    for anything that wants to review or render one sentence's analysis at
    a time (e.g. a sentence-picker UI, like marimo/syntaxer_review.py)
    without re-running analysis or re-deriving the same id-position
    bookkeeping read_analyses()/write_analyses() already do internally.

    Relies on the same invariant read_analyses() and write_analyses()
    already depend on: a sentence's own tokens form a contiguous,
    matching-order run in `tokengraph` (see this module's own docstring).
    `sentence_tokengraph` is the slice of `tokengraph` between that
    sentence's first and last token's positions, inclusive -- which also
    picks up any implied/elided tokens (tokentype in IMPLIED_TOKENTYPES)
    interspersed within that range, since those were never part of
    `sentence.tokens` to begin with but do belong to that sentence's own
    analysis. `sentence_verbalunits` is every VerbalExpression whose id
    falls within that same slice.

    One consequence of using [first, last] *real* token positions as the
    slice boundary, shared with read_analyses()'s own sentence
    reconstruction: an implied token placed AFTER a sentence's last real
    token (rather than nested between two real tokens) falls just outside
    that slice, since there's no further real token of the same sentence
    to bound it from above -- e.g. a one-real-token sentence whose only
    verbal expression is an implied eimi that comes after it (see
    tests/test_serialization.py's
    test_split_excludes_a_trailing_implied_token_past_the_sentences_last_real_token).
    An implied token nested between two real tokens of the same sentence
    is included as expected; only this specific trailing case isn't.

    Raises ValueError for a sentence with no tokens at all, or whose first
    or last token id isn't present in `tokengraph` -- both should be
    impossible for a triple that actually came from read_analyses(), which
    already guarantees this by construction, but this function checks
    explicitly anyway rather than trusting the caller, since nothing stops
    it being called with a hand-built triple too.
    """
    id_position: Dict[str, int] = {tok.id: i for i, tok in enumerate(tokengraph)}

    result: List[Tuple[List[TokenAnalysis], List[VerbalExpression]]] = []
    for s_idx, sentence in enumerate(sentences):
        if not sentence.tokens:
            raise ValueError(f"sentence at index {s_idx} has no tokens")

        first_id = sentence.tokens[0].id
        last_id = sentence.tokens[-1].id
        if first_id not in id_position or last_id not in id_position:
            raise ValueError(
                f"sentence at index {s_idx} (tokens {first_id!r}.."
                f"{last_id!r}) has a boundary token not present in the "
                "given tokengraph"
            )

        start = id_position[first_id]
        end = id_position[last_id]
        sentence_tokengraph = tokengraph[start : end + 1]
        sentence_ids = {tok.id for tok in sentence_tokengraph}
        sentence_verbalunits = [vu for vu in verbalunits if vu.id in sentence_ids]
        result.append((sentence_tokengraph, sentence_verbalunits))

    return result
