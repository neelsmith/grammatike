"""
Thin stub for arsgrammatica's test_harvest.py.

arsgrammatica's fixtures/harvest.py (gold_example_from_analysis()/
format_gold_example_source(), turning a real analyze() result into a
paste-ready GoldExample) has not been ported to grammatike as part of this
test suite -- it's a fixture-authoring convenience tool, not something any
other module here depends on, and porting it faithfully would mean
re-deriving its pydantic-to-dict conversion and source-rendering logic
for grammatike's own Greek-specific field shapes (the relatedtoken2/
relationship2 overflow slot, IMPLIED_TOKENTYPES's two distinct values)
without a corresponding requirement in this task's own fixture list beyond
"skip test_harvest.py's live-network parts unless meaningfully offline; a
thin stub is fine."

Marked `live` (skipped by default, same as every other network-adjacent
test in this suite) purely so it's visibly accounted for in the test
listing rather than silently absent, and so `pytest -m live` -- the one
mode that would actually try to build fixtures/harvest.py's tooling out --
is the one that surfaces this as a to-do rather than a silent gap in
`pytest -q`'s default run.
"""

import pytest


@pytest.mark.live
def test_harvest_module_not_yet_ported():
    pytest.skip(
        "fixtures/harvest.py (gold_example_from_analysis()/"
        "format_gold_example_source()) has not been ported to grammatike; "
        "see this file's module docstring."
    )
