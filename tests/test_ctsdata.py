"""
Tests for grammatike/ctsdata.py's read_ctsdata(). Greek analogue of
arsgrammatica's test_ctsdata.py.

Covers: a basic single-row read (urn split into urnbase/citation, text
verbatim); multiple rows and multiple '#!ctsdata' blocks merged in file
order; a custom delimiter; and every malformed-file error read_ctsdata()
can raise.
"""

import pytest

from grammatike.ctsdata import CtsDataRow, read_ctsdata


def _write_raw(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_single_row_splits_urn_into_urnbase_and_citation(tmp_path):
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
    )
    path = _write_raw(tmp_path, "ctsdata.txt", content)
    rows = read_ctsdata(path)
    assert rows == [
        CtsDataRow(
            urnbase="urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:",
            citation="1",
            text="τίνες ὑμῖν διειλεγμένοι εἰσί;",
        )
    ]


def test_multiple_rows_in_one_block_preserve_file_order(tmp_path):
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:2|δῆλον ὅτι σοφισταί.\n"
    )
    path = _write_raw(tmp_path, "ctsdata.txt", content)
    rows = read_ctsdata(path)
    assert [r.citation for r in rows] == ["1", "2"]
    assert [r.text for r in rows] == [
        "τίνες ὑμῖν διειλεγμένοι εἰσί;",
        "δῆλον ὅτι σοφισταί.",
    ]
    assert all(r.urnbase == "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:" for r in rows)


def test_repeated_blocks_are_merged_in_file_order(tmp_path):
    """Same convention as serialization.py's read_analyses(): more than one
    '#!ctsdata' block, each with its own header line, concatenates into one
    row list rather than raising or keeping only the first."""
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1|μῆνιν ἄειδε θεά.\n"
    )
    path = _write_raw(tmp_path, "ctsdata.txt", content)
    rows = read_ctsdata(path)
    assert [r.citation for r in rows] == ["1", "1.1"]
    assert rows[0].urnbase == "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:"
    assert rows[1].urnbase == "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:"


def test_blank_lines_between_and_within_blocks_are_ignored(tmp_path):
    content = (
        "\n"
        "#!ctsdata\n"
        "\n"
        "urn|text\n"
        "\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
        "\n"
    )
    path = _write_raw(tmp_path, "ctsdata.txt", content)
    rows = read_ctsdata(path)
    assert len(rows) == 1


def test_custom_delimiter(tmp_path):
    content = (
        "#!ctsdata\n"
        "urn\ttext\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1\tτίνες ὑμῖν διειλεγμένοι εἰσί;\n"
    )
    path = _write_raw(tmp_path, "ctsdata.tsv", content)
    rows = read_ctsdata(path, delimiter="\t")
    assert rows == [
        CtsDataRow(
            urnbase="urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:",
            citation="1",
            text="τίνες ὑμῖν διειλεγμένοι εἰσί;",
        )
    ]


def test_text_may_contain_colons(tmp_path):
    """The delimiter is '|', not ':' -- a passage's own text is free to
    contain colons without being mistaken for part of the urn column."""
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|ἔφη: οὐ δύναμαι.\n"
    )
    path = _write_raw(tmp_path, "ctsdata.txt", content)
    rows = read_ctsdata(path)
    assert rows[0].text == "ἔφη: οὐ δύναμαι."


def test_missing_block_raises(tmp_path):
    content = "\n\n"
    path = _write_raw(tmp_path, "missing.txt", content)
    with pytest.raises(ValueError, match="no '#!ctsdata' block"):
        read_ctsdata(path)


def test_data_line_before_any_label_raises(tmp_path):
    content = "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|τίνες.\n#!ctsdata\nurn|text\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="appears before any"):
        read_ctsdata(path)


def test_label_with_no_header_before_next_block_raises(tmp_path):
    content = "#!ctsdata\n#!ctsdata\nurn|text\nurn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|x\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="label line but no header line"):
        read_ctsdata(path)


def test_label_with_no_header_at_end_of_file_raises(tmp_path):
    content = "#!ctsdata\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="ends too early"):
        read_ctsdata(path)


def test_wrong_header_raises(tmp_path):
    content = "#!ctsdata\ntext|urn\nurn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|x\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="expected header"):
        read_ctsdata(path)


def test_wrong_column_count_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|extra|columns\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="expected 2"):
        read_ctsdata(path)


def test_empty_urn_column_raises(tmp_path):
    content = "#!ctsdata\nurn|text\n|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="empty urn"):
        read_ctsdata(path)


def test_empty_text_column_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:greekLit:tlg0059.tlg030.perseus-grc2:1|\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="empty text"):
        read_ctsdata(path)


def test_urn_with_wrong_part_count_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:greekLit:1|τίνες ὑμῖν διειλεγμένοι εἰσί;\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="colon-separated part"):
        read_ctsdata(path)


def test_urn_with_empty_final_part_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:greekLit:tlg0059.tlg030.perseus-grc2:|τίνες.\n"
    path = _write_raw(tmp_path, "bad.txt", content)
    with pytest.raises(ValueError, match="empty final"):
        read_ctsdata(path)
