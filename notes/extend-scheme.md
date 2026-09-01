

## Extending the scheme

To add a new relation:

1. Add the new label to `RelationLabel` in `grammatike/models.py`.
2. Describe when to use it in `SyntaxAnalysis`'s docstring in
   `grammatike/greek_syntax_dspy.py`, following the pattern of the existing relations
   (which token gets `relatedtoken1`/`relationship1`, which gets the
   corresponding value on the other end).
3. Add a gold example exercising it to `tests/fixtures/gold_examples.py` and
   re-run `pytest` to confirm the models still validate before trying it
   against the real LM.
