"""
Tests for grammatike/serialization.py's serialize_analyses()/
write_analyses()/read_analyses(). Greek analogue of arsgrammatica's
test_serialization.py.

Covers: a full round trip (object equality, and a second write producing
byte-identical output) across sentences with and without citations, every
kind of optional field (None, the 'root' sentinel, the relatedtoken2/
relationship2 overflow slot); every warning write_analyses() can return;
every malformed-file error read_analyses() can raise; that
serialize_analyses() and write_analyses() agree exactly; that
read_analyses() accepts a file with more than one instance of a block
label, merging them in file order; and a round trip built directly from
real gold fixtures for realistic coverage of the scheme's relation shapes.
"""

import pytest

from grammatike.models import Sentence, Token, TokenAnalysis, VerbalExpression
from grammatike.serialization import (
    read_analyses,
    read_llm_notes,
    serialize_analyses,
    split_analysis_by_sentence,
    write_analyses,
)
from fixtures.gold_examples import GOLD_EXAMPLES


class _FakeResult:
    """Stands in for a dspy prediction: the only thing serialize_analyses()/
    write_analyses() ever reads off a `results` entry is `.reasoning`."""

    def __init__(self, reasoning):
        self.reasoning = reasoning


def _tokengraph_for(slug):
    example = next(e for e in GOLD_EXAMPLES if e.slug == slug)
    return [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]


def _verbalunits_for(slug):
    example = next(e for e in GOLD_EXAMPLES if e.slug == slug)
    return [VerbalExpression(**vu) for vu in example.canned_answer["verbalunits"]]


def _sentence_from_tokengraph(tokengraph, citation=None):
    """Build a single Sentence spanning every real (non-implied) token in
    `tokengraph`, in order, with a uniform citation (None by default,
    matching how the gold fixtures -- built directly from canned
    tokengraphs, not through segmentation.py -- never populate
    Token.citation)."""
    return Sentence(
        tokens=[
            Token(id=tok.id, text=tok.token, citation=citation)
            for tok in tokengraph
            if tok.token is not None
        ]
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def _two_sentence_fixture():
    """A hand-built two-sentence, two-citation-state passage covering: a
    citation on every token, a sentence with no citation at all, the
    relatedtoken2/relationship2 relative-pronoun overflow slot, the
    'root' sentinel, and a dependent verb's ordinary relation --
    deliberately not reusing a single gold fixture so the test also
    exercises multiple sentences/citations in one file."""
    s1_tokens = [
        Token(id="t0", text="μῆνιν", citation="Iliad 1.1"),
        Token(id="t1", text="ἄειδε", citation="Iliad 1.1"),
        Token(id="t2", text="θεά", citation="Iliad 1.1"),
        Token(id="t3", text=".", citation="Iliad 1.1"),
    ]
    s2_tokens = [
        Token(id="t4", text="ὁ"),
        Token(id="t5", text="ἀνήρ"),
        Token(id="t6", text="ὅν"),
        Token(id="t7", text="εἶδον"),
        Token(id="t8", text="ἀπῆλθεν"),
        Token(id="t9", text="."),
    ]
    sentences = [Sentence(tokens=s1_tokens), Sentence(tokens=s2_tokens)]

    tokengraph = [
        TokenAnalysis(id="t0", token="μῆνιν", tokentype="lexical", lemma="μῆνις",
                      relatedtoken1="t1", relationship1="direct object"),
        TokenAnalysis(id="t1", token="ἄειδε", tokentype="lexical", lemma="ἀείδω",
                      verbalunitid="t1", relatedtoken1="root", relationship1="unit verb"),
        TokenAnalysis(id="t2", token="θεά", tokentype="lexical", lemma="θεά",
                      relatedtoken1="t1", relationship1="subject"),
        TokenAnalysis(id="t3", token=".", tokentype="punctuation"),
        TokenAnalysis(id="t4", token="ὁ", tokentype="lexical", lemma="ὁ",
                      relatedtoken1="t5", relationship1="article"),
        TokenAnalysis(id="t5", token="ἀνήρ", tokentype="lexical", lemma="ἀνήρ",
                      relatedtoken1="t8", relationship1="subject"),
        TokenAnalysis(id="t6", token="ὅν", tokentype="lexical", lemma="ὅς",
                      relatedtoken1="t5", relationship1="relative pronoun",
                      relatedtoken2="t7", relationship2="direct object"),
        TokenAnalysis(id="t7", token="εἶδον", tokentype="lexical", lemma="ὁράω",
                      verbalunitid="t7", relatedtoken1="t6", relationship1="unit verb"),
        TokenAnalysis(id="t8", token="ἀπῆλθεν", tokentype="lexical", lemma="ἀπέρχομαι",
                      verbalunitid="t8", relatedtoken1="root", relationship1="unit verb"),
        TokenAnalysis(id="t9", token=".", tokentype="punctuation"),
    ]

    verbalunits = [
        VerbalExpression(id="t1", syntactic_type="independent", semantic_type="transitive active"),
        VerbalExpression(id="t7", syntactic_type="dependent", semantic_type="transitive active"),
        VerbalExpression(id="t8", syntactic_type="independent", semantic_type="intransitive"),
    ]
    return sentences, verbalunits, tokengraph


def test_round_trip_preserves_every_object_exactly(tmp_path):
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"

    warnings = write_analyses(sentences, verbalunits, tokengraph, str(path))
    assert warnings == []

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits
    assert got_sentences == sentences


def test_round_tripped_data_writes_byte_identical_output(tmp_path):
    """Writing the objects read_analyses() reconstructs should reproduce
    the exact same file -- the whole point of a *deterministic*
    serialization."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path1 = tmp_path / "first.txt"
    path2 = tmp_path / "second.txt"

    write_analyses(sentences, verbalunits, tokengraph, str(path1))
    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path1))
    write_analyses(got_sentences, got_verbalunits, got_tokengraph, str(path2))

    assert path1.read_text() == path2.read_text()


def test_file_contents_match_the_documented_format(tmp_path):
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path))

    text = path.read_text()
    assert "#!sentences" in text
    assert "#!verbal_units" in text
    assert "#!tokens" in text
    assert "context_begin|first_token|context_end|last_token" in text
    assert "context|token|syntactic_type|semantic_type" in text
    assert (
        "context|id|tokentype|text|lemma|verbalunit|"
        "related1|relationship1|related2|relationship2"
    ) in text
    # The 'root' sentinel is written verbatim, not as an empty field.
    assert "|t1|lexical|ἄειδε|ἀείδω|t1|root|unit verb||" in text
    # A citation-free sentence's rows have an empty leading context column.
    assert "|t5|lexical|ἀνήρ|ἀνήρ||t8|subject||" in text
    # The relatedtoken2/relationship2 overflow slot round-trips too.
    assert "|t6|lexical|ὅν|ὅς||t5|relative pronoun|t7|direct object" in text


# ---------------------------------------------------------------------------
# '#!llm' blocks (the optional `results` parameter)
# ---------------------------------------------------------------------------


def test_llm_blocks_written_with_model_env_and_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL", "openai/gpt-4o-mini")
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    results = [
        _FakeResult("ἄειδε is the main verb, with μῆνιν as its direct object."),
        _FakeResult("ἀπῆλθεν is the main verb; ὅν is the relative pronoun."),
    ]
    path = tmp_path / "analysis.txt"

    warnings = write_analyses(sentences, verbalunits, tokengraph, str(path), results=results)
    assert warnings == []

    text = path.read_text()
    assert text.count("#!llm") == 2
    assert "MODEL=openai/gpt-4o-mini" in text
    assert "ἄειδε is the main verb, with μῆνιν as its direct object." in text
    assert "ἀπῆλθεν is the main verb; ὅν is the relative pronoun." in text


def test_llm_blocks_omitted_when_results_not_given(tmp_path):
    """The default (`results=None`) writes a file with no '#!llm' blocks at
    all, exactly as before this parameter existed."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path))
    assert "#!llm" not in path.read_text()
    assert read_llm_notes(str(path)) == []


def test_read_llm_notes_round_trips_model_and_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL", "anthropic/claude-sonnet-5")
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    results = [_FakeResult("First sentence's reasoning."), _FakeResult("Second sentence's reasoning.")]
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path), results=results)

    notes = read_llm_notes(str(path))
    assert notes == [
        ("anthropic/claude-sonnet-5", "First sentence's reasoning."),
        ("anthropic/claude-sonnet-5", "Second sentence's reasoning."),
    ]


def test_read_analyses_ignores_llm_blocks_and_still_reconstructs_everything(tmp_path, monkeypatch):
    """read_analyses() must not choke on '#!llm' blocks, and must still
    reconstruct the three core objects exactly -- it just doesn't return
    the reasoning content (see read_llm_notes() for that)."""
    monkeypatch.setenv("MODEL", "openai/gpt-4o-mini")
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    results = [_FakeResult("reasoning one"), _FakeResult("reasoning two")]
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path), results=results)

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits
    assert got_sentences == sentences


def test_mismatched_results_length_raises(tmp_path):
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    with pytest.raises(ValueError, match="results has 1 entries but sentences has 2"):
        serialize_analyses(sentences, verbalunits, tokengraph, results=[_FakeResult("only one")])


def test_reasoning_with_internal_blank_line_round_trips_but_trailing_blank_is_stripped(tmp_path):
    """A blank line INSIDE the reasoning (a paragraph break) is
    significant and must survive; the one blank line the writer itself
    appends as a block separator must not be mistaken for part of it."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    multi_paragraph = "First paragraph.\n\nSecond paragraph, after a blank line."
    results = [_FakeResult(multi_paragraph), _FakeResult("second sentence, unremarkable")]
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path), results=results)

    notes = read_llm_notes(str(path))
    assert notes[0][1] == multi_paragraph  # internal blank line preserved exactly
    assert notes[1][1] == "second sentence, unremarkable"


def test_model_env_unset_writes_empty_and_reads_back_none(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    results = [_FakeResult("r1"), _FakeResult("r2")]
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path), results=results)

    assert "\nMODEL=\n" in path.read_text()
    notes = read_llm_notes(str(path))
    assert notes[0][0] is None
    assert notes[1][0] is None


def test_malformed_llm_block_missing_model_prefix_raises(tmp_path):
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n#!llm\nnot a model line\nsome reasoning\n")

    with pytest.raises(ValueError, match="MODEL="):
        read_llm_notes(str(path))
    with pytest.raises(ValueError, match="MODEL="):
        read_analyses(str(path))


def test_reasoning_line_colliding_with_a_block_label_raises(tmp_path):
    """A reasoning line that's exactly '#!tokens' (or any other block
    label) would be misread as the start of a new block on read -- caught
    at write time instead, per this format's usual 'no escaping, so
    reject what would corrupt the structure' policy."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    results = [_FakeResult("#!tokens"), _FakeResult("fine")]
    with pytest.raises(ValueError, match="misread as the start of a new block"):
        serialize_analyses(sentences, verbalunits, tokengraph, results=results)


# ---------------------------------------------------------------------------
# serialize_analyses() -- same content as write_analyses(), returned as a
# string instead of written to a file
# ---------------------------------------------------------------------------


def test_serialize_analyses_matches_what_write_analyses_writes_to_disk(tmp_path):
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"

    write_warnings = write_analyses(sentences, verbalunits, tokengraph, str(path))
    content, serialize_warnings = serialize_analyses(sentences, verbalunits, tokengraph)

    assert content == path.read_text()
    assert serialize_warnings == write_warnings


def test_serialize_analyses_content_round_trips_through_read_analyses(tmp_path):
    """serialize_analyses()'s string, written to a file by the caller
    itself (not through write_analyses()), should read back identically to
    a file write_analyses() produced directly."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    content, warnings = serialize_analyses(sentences, verbalunits, tokengraph)
    assert warnings == []

    path = tmp_path / "from_string.txt"
    path.write_text(content, encoding="utf-8")

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits
    assert got_sentences == sentences


def test_serialize_analyses_surfaces_the_same_warnings_and_raises(tmp_path):
    """serialize_analyses() must reproduce write_analyses()'s "degrade
    visibly" warnings (not raise for them) and its hard ValueErrors alike,
    since write_analyses() is now just a thin wrapper around it."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    # Detach t9's citation from any sentence, like
    # test_warns_when_a_token_is_not_covered_by_any_sentence below does.
    sentences[1].tokens = sentences[1].tokens[:-1]
    content, warnings = serialize_analyses(sentences, verbalunits, tokengraph)
    assert any("not found among the given sentences' tokens" in w for w in warnings)
    assert isinstance(content, str) and content  # still produced despite the warning

    with pytest.raises(ValueError, match="has no tokens"):
        serialize_analyses([Sentence(tokens=[])], [], [])


@pytest.mark.parametrize(
    "slug",
    [
        "unit_verb_root_ten_thuran_anoixen",
        "relative_pronoun_ho_aner_hon_eidon",
        "direct_quote_hina_su_ge_ephe",
        "indirect_statement_infinitive_ephaske_lychnon",
        "circumstantial_genitive_absolute_proiontos_de_tou_chronou",
        "attributive_participle_ho_aner_ho_hybrizon",
        "implied_eimi_tauten_ten_hybrin",
        "implied_repetition_ego_men_ano_dietomen",
        "apposition_demosthenes_ho_rhetor",
        "enclitic_eiper_houtos_echei",
    ],
    ids=lambda s: s,
)
def test_round_trip_against_real_gold_fixtures(tmp_path, slug):
    """Realistic coverage: several documented relation shapes currently in
    gold_examples.py, run through an actual write/read round trip rather
    than a hand-built minimal example."""
    tokengraph = _tokengraph_for(slug)
    verbalunits = _verbalunits_for(slug)
    sentences = [_sentence_from_tokengraph(tokengraph)]

    path = tmp_path / "analysis.txt"
    warnings = write_analyses(sentences, verbalunits, tokengraph, str(path))
    assert warnings == []

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits
    assert got_sentences == sentences


def test_empty_lists_round_trip_to_an_empty_but_valid_file(tmp_path):
    path = tmp_path / "empty.txt"
    warnings = write_analyses([], [], [], str(path))
    assert warnings == []

    tokengraph, verbalunits, sentences = read_analyses(str(path))
    assert tokengraph == []
    assert verbalunits == []
    assert sentences == []


# ---------------------------------------------------------------------------
# write_analyses() warnings
# ---------------------------------------------------------------------------


def test_warns_when_a_token_is_not_covered_by_any_sentence(tmp_path):
    tokengraph = [
        TokenAnalysis(id="t0", token="χαῖρε", tokentype="lexical", verbalunitid="t0",
                      relatedtoken1="root", relationship1="unit verb"),
    ]
    verbalunits = [VerbalExpression(id="t0", syntactic_type="independent", semantic_type="intransitive")]
    path = tmp_path / "uncovered.txt"

    warnings = write_analyses([], verbalunits, tokengraph, str(path))
    assert any("t0" in w and "not found among the given sentences" in w for w in warnings)
    got_tokengraph, got_verbalunits, _ = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits


def test_warns_when_a_sentence_is_not_contiguous_in_the_tokengraph(tmp_path):
    tokengraph = [
        TokenAnalysis(id="t0", token="a", tokentype="lexical"),
        TokenAnalysis(id="t2", token="c", tokentype="lexical"),  # t1 missing here
        TokenAnalysis(id="t1", token="b", tokentype="lexical"),
    ]
    sentences = [Sentence(tokens=[Token(id="t0", text="a"), Token(id="t1", text="b")])]
    path = tmp_path / "noncontiguous.txt"

    warnings = write_analyses(sentences, [], tokengraph, str(path))
    assert any(
        "sentence at index 0" in w and "not a contiguous" in w for w in warnings
    )


def test_raises_on_an_empty_sentence(tmp_path):
    with pytest.raises(ValueError, match="no tokens"):
        write_analyses([Sentence(tokens=[])], [], [], str(tmp_path / "bad.txt"))


def test_raises_on_a_pipe_character_in_a_field(tmp_path):
    tokengraph = [TokenAnalysis(id="t0", token="a|b", tokentype="lexical")]
    with pytest.raises(ValueError, match=r"\|"):
        write_analyses([], [], tokengraph, str(tmp_path / "bad.txt"))


def test_raises_on_a_newline_in_a_field(tmp_path):
    tokengraph = [TokenAnalysis(id="t0", token="a", tokentype="lexical", lemma="a\nb")]
    with pytest.raises(ValueError, match="newline"):
        write_analyses([], [], tokengraph, str(tmp_path / "bad.txt"))


# ---------------------------------------------------------------------------
# read_analyses() errors
# ---------------------------------------------------------------------------

_TOKENS_HEADER = (
    "context|id|tokentype|text|lemma|verbalunit|"
    "related1|relationship1|related2|relationship2"
)


def _write_raw(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_missing_block_raises(tmp_path):
    content = f"#!tokens\n{_TOKENS_HEADER}\nctx|t0|lexical|a||||||\n"
    path = _write_raw(tmp_path, "missing.txt", content)
    with pytest.raises(ValueError, match="missing required block"):
        read_analyses(path)


def test_repeated_block_labels_are_merged_in_file_order(tmp_path):
    """Each of the three labels may appear more than once (see the module
    docstring) -- this is what makes a file built by literally
    concatenating two separate write_analyses() outputs (each a complete,
    self-contained trio of blocks) read back as one combined analysis,
    rather than raising or silently keeping only one instance."""
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\n"
        "Iliad 1.1|t0|lexical|foo||t0|root|unit verb||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "Iliad 1.1|t0|independent|intransitive\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "Iliad 1.1|t0|Iliad 1.1|t0\n"
        f"#!tokens\n{_TOKENS_HEADER}\n"
        "Iliad 1.2|t1|lexical|bar||t1|root|unit verb||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "Iliad 1.2|t1|independent|intransitive\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "Iliad 1.2|t1|Iliad 1.2|t1\n"
    )
    path = _write_raw(tmp_path, "repeated.txt", content)
    tokengraph, verbalunits, sentences = read_analyses(path)

    assert [tok.id for tok in tokengraph] == ["t0", "t1"]
    assert [vu.id for vu in verbalunits] == ["t0", "t1"]
    assert len(sentences) == 2
    assert sentences[0].tokens == [Token(id="t0", text="foo", citation="Iliad 1.1")]
    assert sentences[1].tokens == [Token(id="t1", text="bar", citation="Iliad 1.2")]


def test_block_label_without_its_own_header_raises(tmp_path):
    content = (
        "#!tokens\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
    )
    path = _write_raw(tmp_path, "noheader.txt", content)
    with pytest.raises(ValueError, match="label line but no header line"):
        read_analyses(path)


def test_wrong_header_raises(tmp_path):
    content = (
        "#!tokens\nwrong|header\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
    )
    path = _write_raw(tmp_path, "badheader.txt", content)
    with pytest.raises(ValueError, match="expected header"):
        read_analyses(path)


def test_wrong_column_count_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\nctx|t0|lexical|a\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
    )
    path = _write_raw(tmp_path, "badcols.txt", content)
    with pytest.raises(ValueError, match="expected 10"):
        read_analyses(path)


def test_data_before_any_block_label_raises(tmp_path):
    content = "some stray line\n#!tokens\n" + _TOKENS_HEADER + "\n"
    path = _write_raw(tmp_path, "stray.txt", content)
    with pytest.raises(ValueError, match="before any"):
        read_analyses(path)


def test_verbal_units_referencing_unknown_token_id_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\nctx|t0|lexical|a||||||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "ctx|t99|independent|intransitive\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
    )
    path = _write_raw(tmp_path, "unknownvu.txt", content)
    with pytest.raises(ValueError, match="t99"):
        read_analyses(path)


def test_sentences_referencing_unknown_token_id_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\nctx|t0|lexical|a||||||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "ctx|t0|ctx|t99\n"
    )
    path = _write_raw(tmp_path, "unknownsent.txt", content)
    with pytest.raises(ValueError, match="not found in the #!tokens block"):
        read_analyses(path)


def test_sentence_context_mismatch_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\nIliad 1.1|t0|lexical|a||||||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "WRONG|t0|Iliad 1.1|t0\n"
    )
    path = _write_raw(tmp_path, "mismatch.txt", content)
    with pytest.raises(ValueError, match="does not match"):
        read_analyses(path)


def test_sentence_first_after_last_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\n"
        "ctx|t0|lexical|a||||||\n"
        "ctx|t1|lexical|b||||||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "ctx|t1|ctx|t0\n"
    )
    path = _write_raw(tmp_path, "reversed.txt", content)
    with pytest.raises(ValueError, match="comes after"):
        read_analyses(path)


def test_duplicate_token_id_raises(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\n"
        "ctx|t0|lexical|a||||||\n"
        "ctx|t0|lexical|b||||||\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
    )
    path = _write_raw(tmp_path, "dupid.txt", content)
    with pytest.raises(ValueError, match="duplicate token id"):
        read_analyses(path)


def test_blocks_may_appear_in_any_order(tmp_path):
    content = (
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "Iliad 1.1|t0|independent|intransitive\n"
        "#!tokens\n" + _TOKENS_HEADER + "\n"
        "Iliad 1.1|t0|lexical|foo||t0|root|unit verb||\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "Iliad 1.1|t0|Iliad 1.1|t0\n"
    )
    path = _write_raw(tmp_path, "reordered.txt", content)
    tokengraph, verbalunits, sentences = read_analyses(path)
    assert len(tokengraph) == 1
    assert len(verbalunits) == 1
    assert len(sentences) == 1
    assert sentences[0].tokens == [Token(id="t0", text="foo", citation="Iliad 1.1")]


def test_blank_lines_between_blocks_are_tolerated(tmp_path):
    content = (
        f"#!tokens\n{_TOKENS_HEADER}\n"
        "Iliad 1.1|t0|lexical|foo||t0|root|unit verb||\n"
        "\n\n"
        "#!verbal_units\ncontext|token|syntactic_type|semantic_type\n"
        "Iliad 1.1|t0|independent|intransitive\n"
        "\n"
        "#!sentences\ncontext_begin|first_token|context_end|last_token\n"
        "Iliad 1.1|t0|Iliad 1.1|t0\n"
    )
    path = _write_raw(tmp_path, "blank.txt", content)
    tokengraph, verbalunits, sentences = read_analyses(path)
    assert len(tokengraph) == 1
    assert len(verbalunits) == 1
    assert len(sentences) == 1


# ---------------------------------------------------------------------------
# Implied/elided tokens (tokentype='implied eimi'/'implied repetition'; see
# models.py's TokenAnalysis and IMPLIED_TOKENTYPES)
# ---------------------------------------------------------------------------


def _sentence_with_implied_token_fixture():
    """"ταῦτα [ἐστι] καλά." -- one sentence, one real token (t0) plus one
    implied token (t0_implied) anchoring its own linking-verb verbal
    expression."""
    sentences = [Sentence(tokens=[Token(id="t0", text="ταῦτα", citation="Livy 1.1")])]
    tokengraph = [
        TokenAnalysis(id="t0", token="ταῦτα", tokentype="lexical",
                      relatedtoken1="t0_implied", relationship1="subject"),
        TokenAnalysis(id="t0_implied", token=None, tokentype="implied eimi",
                      verbalunitid="t0_implied", relatedtoken1="root", relationship1="unit verb"),
    ]
    verbalunits = [
        VerbalExpression(id="t0_implied", syntactic_type="independent", semantic_type="linking verb"),
    ]
    return sentences, verbalunits, tokengraph


def test_implied_token_round_trips_with_none_text_not_empty_string(tmp_path):
    sentences, verbalunits, tokengraph = _sentence_with_implied_token_fixture()
    path = tmp_path / "implied.txt"

    warnings = write_analyses(sentences, verbalunits, tokengraph, str(path))
    assert warnings == [], (
        "an implied token sitting inside a sentence's own token range should "
        "not trigger the 'not a contiguous run' warning"
    )

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    assert got_tokengraph == tokengraph
    assert got_verbalunits == verbalunits

    implied = next(tok for tok in got_tokengraph if tok.id == "t0_implied")
    assert implied.token is None, (
        f"expected the implied token's text to round-trip as None, got {implied.token!r}"
    )


def test_implied_token_is_excluded_from_its_sentences_reconstructed_tokens(tmp_path):
    sentences, verbalunits, tokengraph = _sentence_with_implied_token_fixture()
    path = tmp_path / "implied.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path))

    _got_tokengraph, _got_verbalunits, got_sentences = read_analyses(str(path))
    assert len(got_sentences) == 1
    assert got_sentences[0].tokens == [Token(id="t0", text="ταῦτα", citation="Livy 1.1")]


# ---------------------------------------------------------------------------
# split_analysis_by_sentence()
# ---------------------------------------------------------------------------


def test_split_returns_one_slice_per_sentence_in_order():
    sentences, verbalunits, tokengraph = _two_sentence_fixture()

    slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)

    assert len(slices) == 2
    s1_tokengraph, s1_verbalunits = slices[0]
    s2_tokengraph, s2_verbalunits = slices[1]

    assert [tok.id for tok in s1_tokengraph] == ["t0", "t1", "t2", "t3"]
    assert [tok.id for tok in s2_tokengraph] == ["t4", "t5", "t6", "t7", "t8", "t9"]

    assert [vu.id for vu in s1_verbalunits] == ["t1"]
    assert [vu.id for vu in s2_verbalunits] == ["t7", "t8"]


def test_split_round_trips_through_a_written_and_reread_file(tmp_path):
    """The realistic path: write a multi-sentence analysis, read it back,
    then split it -- rather than splitting the in-memory objects directly."""
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    path = tmp_path / "analysis.txt"
    write_analyses(sentences, verbalunits, tokengraph, str(path))

    got_tokengraph, got_verbalunits, got_sentences = read_analyses(str(path))
    slices = split_analysis_by_sentence(got_tokengraph, got_verbalunits, got_sentences)

    assert len(slices) == 2
    assert [tok.id for tok in slices[0][0]] == ["t0", "t1", "t2", "t3"]
    assert [tok.id for tok in slices[1][0]] == ["t4", "t5", "t6", "t7", "t8", "t9"]


def _sentence_with_medial_implied_token_fixture():
    """Two real tokens (t0, t1) with an implied token (t0_implied) sitting
    *between* them in tokengraph's own order -- "κόρη [ἐστι] καλή."
    Unlike _sentence_with_implied_token_fixture()'s single-real-token case
    (where the implied token trails the sentence's only real token, with
    nothing to bound it from above), this exercises an implied token
    genuinely nested inside a sentence's own [first, last] real-token
    range."""
    sentences = [Sentence(tokens=[Token(id="t0", text="κόρη"), Token(id="t1", text="καλή")])]
    tokengraph = [
        TokenAnalysis(id="t0", token="κόρη", tokentype="lexical",
                      relatedtoken1="t0_implied", relationship1="subject"),
        TokenAnalysis(id="t0_implied", token=None, tokentype="implied eimi",
                      verbalunitid="t0_implied", relatedtoken1="root", relationship1="unit verb"),
        TokenAnalysis(id="t1", token="καλή", tokentype="lexical",
                      relatedtoken1="t0_implied", relationship1="predicate"),
    ]
    verbalunits = [
        VerbalExpression(id="t0_implied", syntactic_type="independent", semantic_type="linking verb"),
    ]
    return sentences, verbalunits, tokengraph


def test_split_includes_an_implied_token_nested_within_a_sentences_range():
    """An implied token positioned between two of a sentence's own real
    tokens belongs in that sentence's slice, even though it was never part
    of sentence.tokens -- it's part of the analysis, just with no surface
    realization."""
    sentences, verbalunits, tokengraph = _sentence_with_medial_implied_token_fixture()

    slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)

    assert len(slices) == 1
    sentence_tokengraph, sentence_verbalunits = slices[0]
    assert [tok.id for tok in sentence_tokengraph] == ["t0", "t0_implied", "t1"]
    assert [vu.id for vu in sentence_verbalunits] == ["t0_implied"]


def test_split_excludes_a_trailing_implied_token_past_the_sentences_last_real_token():
    """A known, pre-existing limitation shared with read_analyses()'s own
    sentence reconstruction: an implied token placed AFTER a sentence's
    last real token (rather than nested between two real tokens) falls
    outside the [first, last] real-token range this function -- like
    read_analyses() -- uses to slice a sentence's own tokengraph."""
    sentences, verbalunits, tokengraph = _sentence_with_implied_token_fixture()

    slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)

    assert len(slices) == 1
    sentence_tokengraph, sentence_verbalunits = slices[0]
    assert [tok.id for tok in sentence_tokengraph] == ["t0"]
    assert sentence_verbalunits == []


def test_split_rejects_a_sentence_with_no_tokens():
    _sentences, verbalunits, tokengraph = _two_sentence_fixture()

    with pytest.raises(ValueError, match="no tokens"):
        split_analysis_by_sentence(tokengraph, verbalunits, [Sentence(tokens=[])])


def test_split_rejects_a_boundary_token_missing_from_tokengraph():
    sentences, verbalunits, tokengraph = _two_sentence_fixture()
    truncated_tokengraph = [tok for tok in tokengraph if tok.id != "t9"]

    with pytest.raises(ValueError, match="not present in the given tokengraph"):
        split_analysis_by_sentence(truncated_tokengraph, verbalunits, sentences)
