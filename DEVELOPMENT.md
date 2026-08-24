# Developing `grammatike`

`USAGE.md` describes how to call the pipeline, `TESTING.md` how to run the offline test suite, `OPTIMIZING.md` how to tune `SyntaxAnalysis`'s prompt with GEPA, and `BAKEOFF.md` how to compare candidate models. Each of those describes one tool. This document describes how they fit together into a repeatable development loop, centered on testing the analyzer against real Ancient Greek text rather than just the hand-picked `GOLD_EXAMPLES` corpus.


## Why real-world testing, not just the gold-example suite

`tests/fixtures/gold_examples.py`'s `GOLD_EXAMPLES` is a deliberately curated corpus: `test_coverage.py` enforces that every documented relation label, verbal-expression classification, and token type in `syntax_model.md` has at least one example exercising it, and the whole `pytest` suite runs against `DummyLM`, not a real model. That combination is exactly right for what it's for -- proving the code (`models.py`'s pydantic models, `validate()`, `verbal_units.py`, `rendering.py`, `mermaid.py`, `serialization.py`) correctly *represents* a correct answer -- but it can't tell you whether a live model actually *produces* one, and it can't surface a construction nobody has thought to write a fixture for yet.

Running the analyzer against real Greek passages -- actual prose, not fixtures written to order -- is where you find the things the gold-example suite structurally can't show you:

- a construction `syntax_model.md` doesn't document at all yet (`arsgrammatica`'s own history has a worked precedent for exactly this: the implied/elided-token feature -- `implied eimi` and `implied repetition` in `TokenAnalysis.tokentype` -- came about this way for Latin's `implied sum`/`continued discourse`, and this port carried the same two-case distinction over for Greek's `εἰμί`; see "A worked precedent" below);
- a construction the scheme already documents, but the current prompt still gets wrong;
- a genuinely ambiguous case that exposes a modeling choice worth deciding and flagging explicitly -- `greek_syntax_dspy.py`'s own module docstring already lists several places this port had to extrapolate beyond what `syntax_model.md` states explicitly (the scope of "implied εἰμί," the "attributive participle" vs. "attributive" relation split, an infinitive functioning as an ordinary noun, καί's adverbial reading, a substituted example for a still-untranslated Latin one, indirect questions, and the supplementary participle's own relation); flag any *new* judgment call the same way, in a comment near the fixture, rather than quietly picking one and moving on;
- an ordinary, everyday construction the model already handles correctly -- not new information about the scheme, but real evidence worth locking in as a regression guardrail so a future prompt or model change can't silently break it without anyone noticing.


## The core loop: analyze, check automatically, review by hand, triage, act

1. **Analyze a real passage.** `analyze_passage(passage)` for a single string, or `analyze_sources(sources)` for a list of citation-labeled `CitedText` (see USAGE.md's "Analyzing citable sources") -- either returns `(sentences, results)`, one `SyntaxAnalysis` result per sentence found.

2. **Run the automated checks first, before reading anything by hand.** `analyze_sources()` already calls `validate()` for you and prints any referential problems it finds (a token id that doesn't exist, a malformed implied token); `find_unanchored_coordinated_verbs(result.tokengraph)` (`verbal_units.py`) catches one more specific, observed live-LM mistake (a coordinating-conjunction pair where only one side anchors its own verbal unit) that's self-consistent enough to slip past `validate()`. Both are cheap and mechanical -- let them rule out the "obviously broken" cases before you spend a human read on anything.

3. **Read the surviving result against `syntax_model.md` by hand.** This is the one step nothing in the codebase can do for you: `validate()` only checks referential integrity, never correctness, and neither `find_unanchored_coordinated_verbs()` nor any metric can substitute for actually knowing the Greek. `tokengraph_to_html()`/`tokengraph_to_depth_html()` (`rendering.py`) or `tokengraph_to_mermaid()` (`mermaid.py`) are worth rendering here -- seeing the verbal-unit coloring and subordination depth laid out is usually faster to check by eye than reading the raw `tokengraph` rows.

4. **Triage what you found into one of three outcomes, and act accordingly** (see the next section). `arsgrammatica`'s own `tests/fixtures/harvest.py` (`gold_example_from_analysis()`, turning a real analysis into a paste-ready `GoldExample`) has not yet been ported to `grammatike` -- until it is, building a fixture from a real analysis means transcribing `result.verbalunits`/`result.tokengraph` into `gold_examples.py`'s existing dict shape by hand, following the pattern of whichever existing fixture is closest to the new construction.


## The three outcomes, and what to do with each

### Outcome A: a failure

Something is referentially broken (`validate()`/`find_unanchored_coordinated_verbs()` caught it) or substantively wrong (you caught it by hand). Don't just note it and move on -- a failure is the most valuable signal this loop produces, because it's the main way the scheme and the prompt actually improve. Triage it further, in this order:

1. **Is `syntax_model.md` actually silent or ambiguous about this construction?** If so, this is a scheme gap, not a model mistake. Extend `syntax_model.md` first, then follow USAGE.md's "Extending the scheme" steps: add the new relation label / `tokentype` / `syntactic_type` value to the relevant `Literal` in `grammatike/models.py`, describe when to use it in `SyntaxAnalysis`'s docstring in `grammatike/greek_syntax_dspy.py`, and hand-write a `GoldExample` with a *correct* `canned_answer` exercising it (since the live model, by definition, didn't produce one) in `tests/fixtures/gold_examples.py`.
2. **Is the scheme already clear, but the prompt/model got it wrong anyway?** Hand-write a corrected `GoldExample` for the passage the same way, so the failure becomes a concrete, checkable trainset entry rather than an anecdote. If you're unsure between two defensible readings, say so explicitly in a comment above the fixture (matching the existing convention in `gold_examples.py`, e.g. `relative_pronoun_ho_aner_hon_eidon`'s own comment about a graph-resolution judgment call) rather than silently committing to one.
3. Re-run `pytest` (fast, `DummyLM`-backed -- see TESTING.md) to confirm the new/corrected fixture actually validates and that `test_coverage.py` is satisfied, *before* spending any real API budget re-testing it against the live model.

Either way, the corrected fixture lands in `GOLD_EXAMPLES` -- see "How this feeds `OPTIMIZING.md` and `BAKEOFF.md`" below for what that means downstream.

### Outcome B: a success against a rare or tricky construction

The model got something genuinely uncommon or structurally hard right -- a deep subordination chain, a construction with few other examples in `GOLD_EXAMPLES`, anything you'd be nervous betting the model gets right consistently. This is worth *reinforcing*, not just recording: hand-build the `GoldExample` from the real `sentences`/`result.verbalunits`/`result.tokengraph` (recording `result.reasoning` too, since `dspy.ChainOfThought` gives you a real one for free here, not a placeholder) and add it straight to `GOLD_EXAMPLES` as ordinary trainset material -- a correct demonstration of a rare case is precisely the kind of thing `optimize_gepa.py`'s trainset benefits from having more of, and precisely what `model_bakeoff.py`'s `bootstrap` stage is designed to lock in as a few-shot demo for a candidate model that doesn't reliably get it right zero-shot.

### Outcome C: a success against a common, ordinary construction

The model got something right that it was already expected to get right -- a plain independent clause, an ordinary direct object, nothing structurally novel. This is real evidence, but low-value as *training* signal: `optimize_gepa.py` has no held-out split at all today (see below), so anything added to `GOLD_EXAMPLES` is immediately part of what GEPA both trains against and scores itself against, and an easy case the model already nails teaches the optimizer nothing new -- it just dilutes the trainset with redundant coverage. The better default is to harvest it the same way, add it to `GOLD_EXAMPLES`, *and* add its slug to `model_bakeoff.py`'s `HELD_OUT_SLUGS` (see BAKEOFF.md's "The held-out evaluation set"). That turns it into a regression check: something that already works today, now protected against silently breaking as the prompt, the scheme, or the underlying model changes later -- exactly the role `model_bakeoff.py`'s held-out set exists to play, and one that only gets more useful as it grows more diverse.

The dividing line, in short: rare-and-tricky successes are worth teaching the optimizer with; common-and-already-reliable successes are worth protecting with a regression check. "Which bucket does this belong in" is a judgment call about how common the construction already is in `GOLD_EXAMPLES`, not about whether the analysis happened to be correct -- both outcomes started from a correct analysis.


## How this feeds `OPTIMIZING.md` and `BAKEOFF.md`

`optimize_gepa.py` (OPTIMIZING.md) trains `SyntaxAnalysis`'s prompt against *all* of `GOLD_EXAMPLES`, with no separate held-out valset at all -- per `dspy.GEPA`'s own behavior when none is given, the trainset doubles as the Pareto-tracking set GEPA scores itself against. That means every fixture this loop adds to `GOLD_EXAMPLES` -- a corrected failure (Outcome A) or a harvested rare-construction success (Outcome B) -- becomes real trainset material the next time `optimize_gepa.py` runs, which is the main way the shipped prompt actually improves over time: not synthetic examples, but real passages that either broke something or demonstrated something worth reinforcing.

**A known gap worth being aware of**: `optimize_gepa.py` has no mechanism today to *exclude* anything from its trainset -- unlike `model_bakeoff.py`, it doesn't consult `HELD_OUT_SLUGS` (or anything else) to keep held-out examples out of what it trains against. So an Outcome-C fixture you add specifically to protect as a held-out regression check for `model_bakeoff.py` purposes still gets folded into `optimize_gepa.py`'s trainset the next time it runs -- the two scripts don't currently share one consistent notion of "held out." Until that's addressed (e.g. by teaching `optimize_gepa.py` to skip `HELD_OUT_SLUGS` too), treat the held-out/trainset distinction described above as meaningful specifically *for `model_bakeoff.py`'s cross-model comparisons*, not as an airtight guarantee that held-out examples never influence the production prompt.

`model_bakeoff.py` (BAKEOFF.md) is the one place in this repo with an actual train/eval firewall: its `gepa`/`bootstrap` stages never train against `HELD_OUT_SLUGS`, so every candidate model is compared on the same untouched slice. That slice was deliberately stratified by hand when it was first built (a plain independent clause, a subordinating-conjunction dependent clause, a relative pronoun, an indirect statement, a circumstantial genitive absolute, an attributive participle, a depth-2 nesting case, and three further relations); every Outcome-C fixture this loop produces is a natural way to keep growing that stratification with real passages rather than more hand-constructed ones, making `model_bakeoff.py`'s cross-model comparisons more representative as the held-out set grows.

Put together: this loop is the thing that keeps both downstream tools honest over time. Without it, `optimize_gepa.py` only ever optimizes against a fixed, hand-written snapshot of the scheme, and `model_bakeoff.py` only ever compares candidates against that same fixed snapshot -- neither one improves just by running the scripts again. Real-world testing is what actually grows and diversifies the corpus both scripts depend on.


## A worked precedent, inherited from `arsgrammatica`: the implied/elided-token feature

`grammatike`'s `IMPLIED_TOKENTYPES` (`implied eimi`, `implied repetition`) is `arsgrammatica`'s Latin `implied sum`/`continued discourse` feature carried over to Greek, following the same loop this document describes:

1. A real Latin passage needed a construction `syntax_model.md` didn't document: an elided form of the copula, and a shared governing verb of indirect discourse left unrepeated across coordinate clauses.
2. `syntax_model.md` was extended with an "understood or implied verbal expressions" section describing both cases and the id-naming convention for a token with no surface realization -- ported into the Greek scheme with Greek's own copula, εἰμί, in place of Latin's *sum*.
3. `models.py`'s `TokenAnalysis.tokentype` gained the two matching `Literal` values (plus the shared `IMPLIED_TOKENTYPES` constant so every consumer -- `validate()`, `rendering.py`, `serialization.py`, `conftest.py` -- checks membership in one place rather than hardcoding either string), and `SyntaxAnalysis`'s docstring in `greek_syntax_dspy.py` documents exactly when to use each one and how to name the new token's id -- extrapolating beyond `syntax_model.md`'s own single worked example, as that module's own docstring says explicitly.
4. Hand-written `GoldExample` entries (`implied_eimi_tauten_ten_hybrin`, `implied_repetition_ego_men_ano_dietomen`) were added to `gold_examples.py`, and `test_coverage.py` confirms both `tokentype` values are exercised.

A rare-construction success (Outcome B) or a common-construction success (Outcome C) follows the same shape starting from step 4 instead, using a real `analyze_passage()` result rather than a hand-invented one.


## Suggested cadence

- After any batch of real-world testing, run `pytest` (TESTING.md) first -- it's fast and `DummyLM`-backed, and will immediately tell you if a new or corrected fixture doesn't actually validate or if `test_coverage.py` regressed.
- Run `optimize_gepa.py` (OPTIMIZING.md) periodically to refresh the shipped, production prompt against whatever `GOLD_EXAMPLES` has grown into since the last run -- this is a live-LM script with real API cost, so batch it rather than running it after every single new fixture.
- Run `model_bakeoff.py` (BAKEOFF.md) periodically -- before adopting a new candidate model, or whenever the held-out set has grown enough to be worth re-checking -- to confirm existing candidates still stand where you last measured them, now against a larger, more representative held-out slice.
