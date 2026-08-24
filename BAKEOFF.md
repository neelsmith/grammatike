# Comparing task models with `model_bakeoff.py`

`model_bakeoff.py` answers a practical question: **which models, if any, could realistically replace the model this program was developed against?** `grammatike` was developed with Claude Opus 5. This script lets you check open-weight models hosted on Hugging Face, or running locally under Ollama, against the same held-out slice of gold examples, and see not just whether each one is "good enough" but *where* it falls short and *whether that performance can be improved with prompting or demos*.

It's a separate, standalone script from `optimize_gepa.py` — it doesn't touch `optimized_syntax_analysis.json` or anything `syntaxer_main.py` uses at runtime. Run it whenever you want to evaluate a candidate; it never changes the production pipeline.


## The three questions, as three stages

"Can model X run this program" is really three separate questions. `model_bakeoff.py` answers each one as its own stage, selectable with `--stages`:

1. **`baseline`** (zero-shot) — does the candidate do anything useful with the exact instructions already written and tuned against Opus 5, no optimization at all? This is the cheapest stage by far (one call per held-out example) and worth running alone first.
2. **`gepa`** — if `dspy.GEPA` is allowed to rewrite the instructions specifically for this candidate — reading feedback on the candidate's *own* attempts and proposing better wording — how much of the gap closes? The candidate still solves every example itself, both during optimization and at final scoring; only the wording changes.
3. **`bootstrap`** — if Opus itself is allowed to solve some of the training passages (acting as a *teacher*), and the verified-correct traces are attached to the candidate's own prompt as worked few-shot examples, how much closes? Here the candidate never has to solve a training example itself — only the held-out scoring examples afterward.

Comparing `baseline` against `gepa`/`bootstrap` separates "does this model already get it" from "can it be brought up to speed"; comparing `gepa` against `bootstrap` separates two different ways of doing that bringing-up. A candidate that's mediocre at `baseline` but closes most of the gap at `gepa` or `bootstrap` is a much better prospect than one that's mediocre at all three — the latter suggests a real capability ceiling, not just a prompting mismatch.

`gepa`/`bootstrap` make many more LM calls than `baseline` (both the candidate's own calls and the teacher's), so start with `--stages baseline` across every candidate you're curious about, then spend the heavier stages only on the ones worth the API usage.


## Two providers, two different rhythms

**`--provider huggingface`** (the default) routes each candidate through Hugging Face's Inference Providers layer. You can run several candidates in one invocation, one after another, since each is just an API call.

**`--provider ollama`** is for models you pull and run locally with `ollama`. Realistically you can only have one model loaded in Ollama at a time, so this script's shape is built around that constraint: `--provider ollama` refuses to run unless exactly one candidate is selected. The intended workflow is:

```bash
ollama pull llama3.1:8b            # or whichever candidate; make sure ollama serve is running
python model_bakeoff.py --provider ollama --candidates llama-3.1-8b --stages baseline
# ...stop that model, pull/load the next one...
python model_bakeoff.py --provider ollama --candidates gpt-oss-20b --stages baseline
```

Every invocation is a normal one-shot run, not a long-lived process — the results CSV (see below) is what accumulates scores across separate invocations, so you can score one currently-loaded model, save, swap models, and repeat, and nothing you scored earlier gets overwritten.


## The held-out evaluation set

`optimize_gepa.py` trains against every example in `tests/fixtures/gold_examples.py`, with no separate held-out set — a reasonable choice for tuning one model, but it would make cross-model comparison unreliable here: a candidate's post-optimization score would partly reflect how well its own optimized prompt/demos fit the exact examples it's judged on.

`model_bakeoff.py` instead carves out a fixed slice of ten gold examples (`HELD_OUT_SLUGS`, near the top of the file) that no candidate's `gepa`/`bootstrap` stage ever trains against. Every candidate optimizes against the same remaining 22 examples and is scored against the same untouched held-out set, so scores are actually comparable across models and across stages. The slice is deliberately stratified across `syntax_model.md`'s own Greek constructions — a plain independent clause, a subordinating-conjunction dependent clause, a relative pronoun (exercising the `relatedtoken2` overflow slot), an indirect statement, a circumstantial genitive absolute, an attributive participle (one of Greek's three predicate-participle categories, with no single-value Latin analogue), a depth-2 nesting case, and three further relations (apposition, indirect question, complementary infinitive) — so a low score can be traced to a specific construction rather than just "worse overall."


## Reading the scores

Every stage reports a blended `mean` score in [0, 1] over the held-out set, plus three unblended sub-scores from `grammatike.gepa_metric.syntax_metric()`:

- **`field_mean`** — basic per-token fields (tokentype, lemma, which verbal unit a token belongs to).
- **`relation_mean`** — the actual dependency relations, weighted highest (0.5 of the blend) since they're the heart of the scheme.
- **`vu_mean`** — verbal-expression classification (`syntactic_type`/`semantic_type`).

A candidate that nails `field_mean` but collapses on `relation_mean` is failing at multi-hop structural reasoning specifically — a different (and probably less prompt-fixable) problem than one that's just generally worse across the board. The console output also prints, per held-out example, any that raised an outright error (a request that failed, or output that didn't parse into the expected shape) rather than silently folding those into the average — worth reading before trusting a low score.

This script doesn't compute a malformed-output rate as its own number (`validate()`/`find_unanchored_coordinated_verbs()` in the main package can do that over raw predictions if you want it), and it doesn't do real weight-level fine-tuning — the `bootstrap` stage attaches Opus-solved examples to the candidate's *prompt*, never touching its weights. `dspy.BootstrapFinetune` is real fine-tuning, but needs different infrastructure than either provider here and isn't wired in.


## Setting up `.env`

`--provider huggingface` needs a Hugging Face access token:

```
HUGGINGFACE_API_KEY=hf_...
```

(a token with Inference Providers access — see huggingface.co/settings/tokens). Some candidates specify their own `api_key_env`/`api_base_env` in `CANDIDATES` if they need a different credential or a dedicated endpoint.

`--provider ollama` needs nothing by default — it talks to `http://localhost:11434`. Set `OLLAMA_API_BASE` in `.env`, or pass `--ollama-api-base`, if Ollama is running somewhere else on your network.

The **teacher model** — used as GEPA's `reflection_lm` in the `gepa` stage, and as the program that actually solves training passages in the `bootstrap` stage — is fixed across every candidate and provider, and defaults to this repo's already-configured main model (`API_BASE`/`MODEL`/`API_KEY`, the same `.env` `syntaxer_main.py` and `optimize_gepa.py` use). Override it with `REFLECTION_MODEL` (and `REFLECTION_API_BASE`/`REFLECTION_API_KEY` if they differ) — exactly like `optimize_gepa.py`. If you only run `--stages baseline`, no teacher credentials are needed at all.


## Running it

```bash
# Cheap first pass across every Hugging Face candidate
python model_bakeoff.py --stages baseline

# Everything, every candidate (expensive -- gepa + bootstrap make many calls)
python model_bakeoff.py --stages baseline gepa bootstrap

# Just a couple of candidates
python model_bakeoff.py --candidates llama-3.1-8b gpt-oss-20b

# Skip gepa/bootstrap for anything that didn't clear a baseline threshold
python model_bakeoff.py --min-baseline-to-optimize 0.3

# Ollama, one candidate at a time (see workflow above)
python model_bakeoff.py --provider ollama --candidates llama-3.1-8b --stages baseline
python model_bakeoff.py --provider ollama --candidates llama-3.1-8b --stages gepa bootstrap

# A model not in CANDIDATES at all -- either provider
python model_bakeoff.py --provider ollama --model ollama_chat/llama3.1:8b-instruct-q8_0 --label llama-3.1-8b-q8

# Optimizer knobs
python model_bakeoff.py --auto medium                    # gepa's (and miprov2's) budget preset
python model_bakeoff.py --max-metric-calls 40             # exact gepa budget instead of --auto
python model_bakeoff.py --bootstrap-optimizer miprov2     # heavier alternative to bootstrap-fewshot
python model_bakeoff.py --max-bootstrapped-demos 4 --max-labeled-demos 4

python model_bakeoff.py --out results.csv                 # merge into a specific file
```

The full flag reference is always available with `python model_bakeoff.py --help`; the ones most worth knowing up front:

- `--provider {huggingface,ollama}` — where candidates run (default `huggingface`).
- `--candidates LABEL [LABEL ...]` — restrict to specific labels from `CANDIDATES` (default: all of them; `--provider ollama` requires exactly one).
- `--model` / `--label` — an ad hoc pair for a model not in `CANDIDATES` at all; both or neither.
- `--stages {baseline,gepa,bootstrap}` — which stage(s) to run this invocation (default: all three).
- `--bootstrap-optimizer {bootstrap-fewshot,miprov2}` — which optimizer implements the `bootstrap` stage (default `bootstrap-fewshot`, cheaper).
- `--min-baseline-to-optimize` — skip `gepa`/`bootstrap` for a candidate whose zero-shot mean is below this threshold, whether that baseline came from this invocation or a previously recorded row.
- `--out` — the results file to merge into (default `model_bakeoff_results.csv`).

Expect real calls against whichever provider serves each candidate, plus the teacher model's calls for `gepa`/`bootstrap`. Start with `--stages baseline` and one candidate before running more stages or more candidates.


## The results file

Results accumulate in a CSV (`model_bakeoff_results.csv` by default), keyed by `(label, provider, stage)`. Each invocation reads whatever's already there, merges in its own new rows, and writes the whole thing back out sorted by `CANDIDATES`' own order and then by stage — so:

- Running a Hugging Face pass today and an Ollama pass tomorrow, or scoring one Ollama model per day as you cycle through them, all lands in the same file.
- Re-running the same `(label, provider, stage)` combination overwrites just that row; every other row survives untouched.
- Columns: `label`, `provider`, `model` (the resolved model string actually called), `family`, `tier`, `stage`, `n` (held-out example count), `mean`/`min`/`max` (blended score), `field_mean`/`relation_mean`/`vu_mean`, `elapsed_s`, `n_calls`, `total_cost` (blank when the provider doesn't report per-call cost — Ollama never will, since it's local), and `error` (set instead of scores if the candidate was skipped entirely, e.g. a missing API key).

Each optimized `gepa`/`bootstrap` run is also saved to disk as `optimized_<label>_<provider>_<stage>.json`, alongside the CSV row — load one of these into a `dspy.ChainOfThought(SyntaxAnalysis)` instance with `.load(...)` if you want to inspect or reuse the actual optimized program, not just its score.

No results have been recorded yet as of this release — `model_bakeoff_results.csv` does not exist in this repo until you run the script yourself.


## The candidate roster

`CANDIDATES`, near the top of the file, is carried over unchanged from `arsgrammatica`'s own roster — general-purpose open-weight instruct models, spanning roughly 3B to 120B+ parameters across several major open-weight families, plus one reasoning-distilled model (DeepSeek-R1-Distill-Llama-8B) at the same size as its plain counterpart (Llama-3.1-8B), to see whether chain-of-thought distillation specifically helps a multi-hop structural task like this one. None of these models are Latin- or Greek-specific, so the same spread is the right starting point for asking the analogous question about grammatike's task.

| label | family | tier |
|---|---|---|
| `phi-4-mini` | Microsoft Phi | ~4B |
| `llama-3.2-3b` | Meta Llama | ~3B |
| `llama-3.1-8b` | Meta Llama | ~8B |
| `deepseek-r1-distill-llama-8b` | DeepSeek (R1 distill) | ~8B |
| `qwen-8b` | Alibaba Qwen | ~7B |
| `gpt-oss-20b` | OpenAI (open weights) | ~20B |
| `mistral-small-24b` | Mistral AI | ~24B |
| `llama-3.3-70b` | Meta Llama | ~70B |
| `deepseek-r1-distill-llama-70b` | DeepSeek (R1 distill) | ~70B |
| `gpt-oss-120b` | OpenAI (open weights) | ~120B |

Two caveats worth knowing before running these on Ollama specifically: the *bare* tag `deepseek-r1:8b` has been repointed to a Qwen3-based checkpoint rather than the Llama-based distill this entry is actually about — the explicit `-llama-distill` suffix in `ollama_model` is required to get the right model, so double-check against `ollama list`/the library page before a real run. And Ollama's library currently only carries "Mistral Small 3" (2501), while the Hugging Face entry points at the newer 3.2 (2506) — the two providers aren't quite testing the same checkpoint for `mistral-small-24b`, so treat any HF-vs-Ollama comparison for that label with that in mind.

Model availability and naming move fast on both sides; treat this roster as a snapshot worth re-checking against Hugging Face's Inference Providers docs, each model's Hub page, and `ollama list`/the Ollama library before a real run, not a guarantee. The largest candidates (70B/120B-class) are realistic mainly through `--provider huggingface` — running them locally via Ollama needs serious hardware.

To add a new candidate permanently, add a `dict(...)` entry to `CANDIDATES` with at least `label`, `model` (the `huggingface/org/repo` string), and `ollama_model` (the exact Ollama pull tag); for a one-off you don't want to add to the roster at all, use `--model`/`--label` instead (see the Running it examples above).
