"""
Compares SyntaxAnalysis's performance across candidate task models -- by
default, a spread of openly-licensed, open-weight models -- to help decide
which ones (if any) could realistically replace the Claude Opus model this
program was developed against.

Greek analogue of arsgrammatica's model_bakeoff.py -- same structure, CLI,
and provider logic, adapted to import grammatike's own SyntaxAnalysis,
gepa_metric, and gold-example fixtures instead of arsgrammatica's.

WHY THIS SCRIPT EXISTS
-----------------------
"Can model X run this program" turns out to be three separate questions,
answered here as three STAGES per candidate (pick which to run with
--stages):

  1. baseline (zero-shot): does the model do anything useful with the SAME
     instructions (SyntaxAnalysis's docstring) that were written and tuned
     against Opus, with no optimization of its own?
  2. gepa: if GEPA is allowed to rewrite the instructions specifically for
     this model -- reading feedback on the model's OWN attempts and
     proposing better wording -- how much of the gap closes?
  3. bootstrap: if Opus itself is allowed to solve some of the training
     passages (a "teacher"), and the verified-correct traces are attached
     to the candidate's own prompt as worked few-shot examples -- rather
     than just better-worded instructions -- how much closes?

(1) vs. (2)/(3) separates "does this model already get it" from "can it be
brought up to speed"; (2) vs. (3) separates two different ways of doing the
bringing-up: GEPA only ever improves WORDING (the candidate still solves
every example itself, even during optimization), while bootstrap-fewshot
lets Opus actually DO some of the work up front and hands the candidate
finished examples to pattern-match against. A model that's mediocre at (1)
but closes most of the gap at (2) or (3) is a much better candidate than
one that's mediocre at all three -- the latter suggests a real capability
ceiling, not just a prompting mismatch.

TWO PROVIDERS: HUGGING FACE (hosted) AND OLLAMA (local, one at a time)
------------------------------------------------------------------------
--provider huggingface (the default) routes each candidate through Hugging
Face's Inference Providers layer -- see the CANDIDATE MODELS section below
-- and can run several candidates in one process, one after another,
since each is just an API call.

--provider ollama is for models you pull and run locally with `ollama`.
Realistically you can only have ONE model loaded in Ollama at a time, so
this script's whole shape is built around that: it does not loop over
several candidates in one invocation for this provider. Instead:

    1. `ollama pull llama3.1:8b` (or whichever candidate), then make sure
       `ollama serve` is running (or let `ollama run` start it).
    2. `python model_bakeoff.py --provider ollama --candidates llama-3.1-8b`
       -- scores THAT one candidate and MERGES the result into --out
       (default model_bakeoff_results.csv), leaving every other row already
       in that file untouched.
    3. Stop that model, pull/load the next one, repeat step 2 with a
       different --candidates label.

Every invocation is a normal one-shot run, not a long-lived process -- the
CSV file is what accumulates results across separate invocations (and
across days). Re-running the same candidate/provider/stage combination
overwrites just that row; every other row survives. --provider ollama
requires --candidates to name EXACTLY one label (or --model/--label for an
ad hoc one-off, see below) -- this script deliberately refuses to guess
which model you currently have loaded.

HELD-OUT EVALUATION
--------------------
optimize_gepa.py trains (and, since it passes no valset, also does Pareto
tracking) against ALL of tests/fixtures/gold_examples.py's GOLD_EXAMPLES --
a reasonable choice for tuning one model, but it makes cross-model
comparison unreliable: a model's post-optimization score would partly
reflect how well its own optimized prompt/demos fit the very examples it's
judged on. This script instead holds out a fixed slice (HELD_OUT_SLUGS
below) that NO candidate's gepa/bootstrap stage ever trains against --
every candidate optimizes against the same remaining trainset and is
scored against the same untouched held-out set, so scores are actually
comparable across models and across stages. The held-out slice is
deliberately stratified across grammatike's own Greek constructions: a
plain independent clause, a subordinating-conjunction dependent clause, a
relative pronoun (exercising the relatedtoken2 overflow slot), an indirect
statement, a circumstantial genitive absolute, an attributive participle
(Greek's three-way participle split has no Latin analogue), a depth-2
nesting case, and three further relations (apposition, indirect question,
complementary infinitive) -- so a low held-out score can be traced to a
specific construction, not just "worse overall."

SUB-SCORES, NOT JUST THE BLENDED NUMBER
-----------------------------------------
grammatike.gepa_metric.syntax_metric() returns a single blended score,
but also (see that module) the three dimensions it blends: field_score
(tokentype/lemma/verbalunitid), relation_score (the actual dependency
relations -- weighted highest, 0.5, since they're the heart of the
scheme), and vu_score (verbal-expression classification). This script
reports all three per candidate/stage, since a model that nails
field_score but collapses on relation_score is failing at multi-hop
structural reasoning specifically -- a different (and probably less
prompt-fixable) problem than a model that's just generally worse across
the board.

WHAT THIS SCRIPT DOES NOT DO
------------------------------
- It doesn't check malformed-output rate (validate() / the
  find_unanchored_coordinated_verbs() heuristic) -- a model that frequently
  produces referentially-broken output (invented ids, reused 'root', etc.)
  will already show up as a low relation_score/field_score here, but if you
  want the malformed-output rate as its own number, run validate() over
  each candidate's raw predictions yourself.
- It doesn't do real weight-level fine-tuning (dspy.BootstrapFinetune) --
  the "bootstrap" stage attaches Opus-solved examples to the candidate's
  PROMPT, it never touches the candidate's own weights. Fine-tuning needs
  different infrastructure (a locally fine-tunable checkpoint, or a
  provider with a real fine-tuning API) than either provider this script
  supports.
- It doesn't tune the CANDIDATES list to whatever's cheapest or fastest --
  see the CANDIDATE MODELS section below for what's included and why, and
  treat it as a starting point, not a fixed roster.

CANDIDATE MODELS
-----------------
Each entry in CANDIDATES carries TWO possible model identifiers, one per
provider:

  - model         -- a litellm-style "huggingface/<org>/<repo>" string (see
                     https://docs.litellm.ai/docs/providers/huggingface and
                     https://huggingface.co/docs/inference-providers),
                     used when --provider huggingface.
  - ollama_model  -- the exact Ollama library pull tag (see
                     https://ollama.com/library), used when --provider
                     ollama, as "ollama_chat/<ollama_model>".

Model availability and naming move fast on both sides (new generations
supersede old ones every few months, and Ollama's own tag conventions for
a given release don't always match the Hugging Face repo name -- e.g. a
"bare" tag like deepseek-r1:8b can silently repoint to a different base
model between Ollama library updates). Treat CANDIDATES as a snapshot
worth checking against https://huggingface.co/docs/inference-providers,
each model's own HF Hub page, and `ollama list`/https://ollama.com/library
before a real run, not a guarantee. Some Hugging Face entries below may
need an explicit provider suffix ("huggingface/together/org/repo") or a
dedicated Inference Endpoint (set via that candidate's own api_base_env)
instead of the bare "huggingface/" form, depending on how that checkpoint
is currently being served. The largest candidates here (70B/120B-class)
are realistic mainly through --provider huggingface -- running them
locally via Ollama needs serious hardware.

This roster is carried over unchanged from arsgrammatica's own CANDIDATES:
these are general-purpose open-weight instruct models, not Latin-specific
in any way, so the same spread is exactly the right starting point for
asking the analogous question about grammatike's Greek task. (Scoped down
from the Latin file's roster in one respect: this port keeps the full
ten-entry list rather than trimming it, since a shorter list would answer
a narrower question than arsgrammatica's own bakeoff does -- see this
task's own completion notes for why nothing was actually cut here.)

If a model you want to test isn't in CANDIDATES at all (e.g. a specific
Ollama tag/quantization you've already pulled), skip the catalog with
--model/--label -- see USAGE below.

ENVIRONMENT
------------
--provider huggingface needs a Hugging Face access token in .env:

    HUGGINGFACE_API_KEY=hf_...

(a token with Inference Providers access -- see
https://huggingface.co/settings/tokens). Some candidates may specify their
own api_key_env/api_base_env if they need a different credential or a
dedicated endpoint instead.

--provider ollama needs nothing in .env by default -- it talks to
http://localhost:11434 (Ollama's default local address), no API key. Set
OLLAMA_API_BASE in .env (or pass --ollama-api-base) to point at a
different address (e.g. Ollama running on another machine on your
network).

The TEACHER model -- Opus, used by both the "gepa" stage (as GEPA's
reflection_lm, reading feedback and proposing better instructions) and the
"bootstrap" stage (actually solving training passages, whose verified
output becomes the candidate's few-shot demos) -- is kept FIXED across
every candidate and every provider, defaulting to this repo's already-
configured main model (API_BASE/MODEL/API_KEY, same as optimize_gepa.py).
The question these two stages answer is "can Opus lift this model's
score," not "can this model teach/optimize itself" -- and GEPA's own docs
recommend a strong reasoning model for reflection regardless of the task
model being tuned. Override with REFLECTION_MODEL (and
REFLECTION_API_BASE/REFLECTION_API_KEY if they differ), exactly like
optimize_gepa.py.

USAGE
------
Hugging Face, several candidates in one run:
    python model_bakeoff.py --stages baseline                     # cheap first pass, every candidate
    python model_bakeoff.py --stages baseline gepa bootstrap       # everything, every candidate
    python model_bakeoff.py --candidates llama-3.1-8b gpt-oss-20b  # only these two
    python model_bakeoff.py --min-baseline-to-optimize 0.3         # skip gepa/bootstrap below this baseline

Ollama, one candidate per invocation (see TWO PROVIDERS above):
    python model_bakeoff.py --provider ollama --candidates llama-3.1-8b --stages baseline
    python model_bakeoff.py --provider ollama --candidates llama-3.1-8b --stages gepa bootstrap

Ad hoc model not in CANDIDATES at all (either provider):
    python model_bakeoff.py --provider ollama --model ollama_chat/llama3.1:8b-instruct-q8_0 --label llama-3.1-8b-q8

Optimizer knobs:
    python model_bakeoff.py --auto medium                          # gepa's (and miprov2's) budget preset
    python model_bakeoff.py --max-metric-calls 40                  # exact gepa budget instead of --auto
    python model_bakeoff.py --bootstrap-optimizer miprov2          # heavier alternative to bootstrap-fewshot
    python model_bakeoff.py --max-bootstrapped-demos 4 --max-labeled-demos 4

    python model_bakeoff.py --out results.csv

Expect this to make real calls against whichever provider serves each
candidate, plus the teacher model's calls (for gepa/bootstrap stages).
Start with --stages baseline and one candidate before running more stages
or more candidates.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Reuse syntaxer_main.py's own .env-reading helper rather than duplicating it.
sys.path.insert(0, str(Path(__file__).parent))
from syntaxer_main import _env  # noqa: E402

# tests/ isn't an installed package -- add it to sys.path the same way
# pytest and optimize_gepa.py do, so "from fixtures.gold_examples import
# GOLD_EXAMPLES" and "from conftest import tokens_from_canned_answer"
# resolve the same way they do everywhere else in this repo.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from conftest import tokens_from_canned_answer  # noqa: E402
from fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

from grammatike.gepa_metric import syntax_metric
from grammatike.greek_syntax_dspy import SyntaxAnalysis
from grammatike.models import TokenAnalysis, VerbalExpression


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
#
# label            -- short name used in output, --candidates filtering, and
#                      as the CSV's primary key (together with provider/stage)
# model            -- litellm-style "huggingface/org/repo" string, used when
#                      --provider huggingface
# ollama_model     -- exact Ollama library pull tag, used (as
#                      "ollama_chat/<ollama_model>") when --provider ollama
# family / tier    -- for grouping/reading the report, not used functionally
# notes            -- why this one's here / caveats worth knowing before running it
# api_key_env      -- (huggingface only) env var holding this candidate's API
#                      key (default: HUGGINGFACE_API_KEY); override per-
#                      candidate if a given checkpoint needs a different token
# api_base_env     -- (huggingface only) optional env var for a custom
#                      api_base (a specific Inference Providers route, or a
#                      dedicated Inference Endpoint URL)
#
# Chosen to span roughly 3B to 120B+ across several major open-weight
# families, plus one reasoning-distilled model at the same size as its
# plain counterpart (deepseek-r1-distill-llama-8b vs. llama-3.1-8b) to see
# whether chain-of-thought distillation specifically helps on this kind of
# multi-hop structural-reasoning task. Left out for now, but worth adding
# once you've confirmed current availability: newer generations in each
# family that may have shipped since this file was written, and very large
# frontier-scale open-weight MoE models (Llama 4 Maverick, DeepSeek V3/V4-
# class, Kimi-class, GLM-class) -- those are less "smaller model I could
# plausibly self-host or run cheaply" and more "another frontier lab's
# model," a different question than the one this script answers.
CANDIDATES = [
    dict(
        label="phi-4-mini",
        model="huggingface/microsoft/Phi-4-mini-instruct",
        ollama_model="phi4-mini:3.8b",
        family="Microsoft Phi",
        tier="~4B",
        notes="Smallest candidate here -- a floor for 'can a tiny model do this at all'.",
    ),
    dict(
        label="llama-3.2-3b",
        model="huggingface/meta-llama/Llama-3.2-3B-Instruct",
        ollama_model="llama3.2:3b",
        family="Meta Llama",
        tier="~3B",
        notes="Even smaller than phi-4-mini; mainly useful to confirm the floor rather than as a serious candidate.",
    ),
    dict(
        label="llama-3.1-8b",
        model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
        ollama_model="llama3.1:8b",
        family="Meta Llama",
        tier="~8B",
        notes="Widely used small-model baseline; lots of prior art on its structured-output behavior elsewhere.",
    ),
    dict(
        label="deepseek-r1-distill-llama-8b",
        model="huggingface/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ollama_model="deepseek-r1:8b-llama-distill-q4_K_M",
        family="DeepSeek (R1 distill)",
        tier="~8B",
        notes="Same size as llama-3.1-8b -- a direct test of whether reasoning-distillation "
              "specifically helps a multi-hop structural task like this one. NOTE: on Ollama, the "
              "BARE tag 'deepseek-r1:8b' has been repointed to a Qwen3-based checkpoint, not this "
              "Llama-based distill -- the explicit '-llama-distill' suffix above is required to get "
              "the model this label is actually about; double check against `ollama list`/the "
              "library page before running.",
    ),
    dict(
        label="qwen-8b",
        model="huggingface/Qwen/Qwen2.5-7B-Instruct",
        ollama_model="qwen2.5:7b",
        family="Alibaba Qwen",
        tier="~7B",
        notes="A third distinct family at the same rough tier as the two Llama-based 8B entries. "
              "Qwen3 (ollama tag qwen3:8b) has since superseded Qwen2.5 as Ollama's featured series at "
              "this size -- worth trying both if you want the current-generation comparison too.",
    ),
    dict(
        label="gpt-oss-20b",
        model="huggingface/openai/gpt-oss-20b",
        ollama_model="gpt-oss:20b",
        family="OpenAI (open weights)",
        tier="~20B",
        notes="Apache-2.0, native reasoning-effort control -- worth comparing against the "
              "8B reasoning distill above to see whether the jump in size matters more than "
              "the distillation did.",
    ),
    dict(
        label="mistral-small-24b",
        model="huggingface/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        ollama_model="mistral-small:24b-instruct-2501-q4_K_M",
        family="Mistral AI",
        tier="~24B",
        notes="A fourth family, between gpt-oss-20b and the 70B-class entries below. NOTE: Ollama's "
              "library currently only carries 'Mistral Small 3' (2501), not the newer 3.2 (2506) the "
              "HF-hosted entry above points at -- the two providers aren't testing quite the same "
              "checkpoint here; treat any HF-vs-Ollama comparison for this label with that in mind.",
    ),
    dict(
        label="llama-3.3-70b",
        model="huggingface/meta-llama/Llama-3.3-70B-Instruct",
        ollama_model="llama3.3:70b",
        family="Meta Llama",
        tier="~70B",
        notes="The largest dense (non-MoE) open-weight Llama generation available at this "
              "writing; the natural 'how far does more scale get you' data point. Needs serious "
              "local hardware to run via --provider ollama.",
    ),
    dict(
        label="deepseek-r1-distill-llama-70b",
        model="huggingface/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ollama_model="deepseek-r1:70b-llama-distill-q4_K_M",
        family="DeepSeek (R1 distill)",
        tier="~70B",
        notes="Reasoning distillation at 70B -- pairs with deepseek-r1-distill-llama-8b to "
              "separate 'does distillation help' from 'does scale help'. Also realistic locally only "
              "with serious hardware; the bare Ollama tag 'deepseek-r1:70b' currently still points at "
              "this Llama-based distill (unlike the 8B bare tag), but the explicit suffix above is "
              "kept for clarity/future-proofing -- verify before a real run either way.",
    ),
    dict(
        label="gpt-oss-120b",
        model="huggingface/openai/gpt-oss-120b",
        ollama_model="gpt-oss:120b",
        family="OpenAI (open weights)",
        tier="~120B",
        notes="Largest candidate here. If even this doesn't close the gap with Opus on "
              "relation_score specifically, the ceiling is probably multi-hop structural "
              "reasoning depth, not raw parameter count within the 'openly available' range. "
              "Realistic mainly via --provider huggingface.",
    ),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set
# ---------------------------------------------------------------------------
#
# Stratified across grammatike's own Greek constructions -- deliberately
# not a 1:1 mapping of arsgrammatica's own slugs (this package's fixtures
# use different slug names and, for the participle split and the
# attributive-participle construction specifically, cover a distinction
# Latin's own scheme doesn't make at all).

HELD_OUT_SLUGS = [
    "unit_verb_root_ten_thuran_anoixen",                   # baseline: plain independent clause
    "subordinating_conjunction_dependent_kategorei_hos",   # subordinating-conjunction dependent clause
    "relative_pronoun_ho_aner_hon_eidon",                  # relative pronoun + relatedtoken2 overflow
    "indirect_statement_infinitive_ephaske_lychnon",       # indirect statement (infinitive)
    "circumstantial_genitive_absolute_proiontos_de_tou_chronou",  # circumstantial participle / genitive absolute
    "attributive_participle_ho_aner_ho_hybrizon",          # attributive participle -- Greek's 3-way participle split has no Latin analogue
    "depth_two_epei_edei_hemartekenai",                    # depth-2 nesting
    "apposition_demosthenes_ho_rhetor",                    # apposition
    "indirect_question_ouk_oida_tis_elthen",                # indirect question
    "complementary_infinitive_exesti_helesthai",           # complementary infinitive
]


def _example_from(gold_example):
    tokens = tokens_from_canned_answer(gold_example.canned_answer)
    verbalunits = [VerbalExpression(**vu) for vu in gold_example.canned_answer["verbalunits"]]
    tokengraph = [TokenAnalysis(**tok) for tok in gold_example.canned_answer["tokengraph"]]
    return dspy.Example(
        slug=gold_example.slug,
        passage=gold_example.passage,
        tokens=tokens,
        verbalunits=verbalunits,
        tokengraph=tokengraph,
    ).with_inputs("passage", "tokens")


def build_split():
    """Partition GOLD_EXAMPLES into (trainset, heldout) by slug membership
    in HELD_OUT_SLUGS. Every candidate's gepa/bootstrap stage trains only on
    trainset; every candidate/stage is scored only on heldout, so scores are
    comparable across candidates and across stages rather than each
    reflecting how well it memorized its own training slice."""
    held_out = set(HELD_OUT_SLUGS)
    known = {e.slug for e in GOLD_EXAMPLES}
    missing = held_out - known
    if missing:
        raise RuntimeError(
            f"HELD_OUT_SLUGS names slug(s) not found in GOLD_EXAMPLES: {sorted(missing)} "
            "-- gold_examples.py may have been renamed/removed since this list was written."
        )
    trainset, heldout = [], []
    for example in GOLD_EXAMPLES:
        bucket = heldout if example.slug in held_out else trainset
        bucket.append(_example_from(example))
    return trainset, heldout


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

def _resolve_model_string(candidate, provider):
    """The litellm-style model string to actually call for `candidate`
    under `provider` -- an explicit "override_model" (from --model) always
    wins regardless of provider; otherwise it's candidate["model"] for
    huggingface or "ollama_chat/<candidate['ollama_model']>" for ollama."""
    if candidate.get("override_model"):
        return candidate["override_model"]
    if provider == "ollama":
        ollama_model = candidate.get("ollama_model")
        if not ollama_model:
            raise RuntimeError(
                f"{candidate['label']!r} has no ollama_model entry in CANDIDATES -- add one "
                "(check `ollama list`/https://ollama.com/library for the exact tag), or use "
                "--model/--label for a one-off override."
            )
        return f"ollama_chat/{ollama_model}"
    model = candidate.get("model")
    if not model:
        raise RuntimeError(
            f"{candidate['label']!r} has no model entry in CANDIDATES for --provider huggingface."
        )
    return model


def _configure_candidate_lm(candidate, provider, ollama_api_base):
    """Build a dspy.LM for one candidate under one provider.

    ollama: no API key needed (a local, unauthenticated daemon); api_base
    defaults to ollama_api_base (see --ollama-api-base / OLLAMA_API_BASE).

    huggingface: resolves the API key from the env var the candidate names
    (default HUGGINGFACE_API_KEY), and an optional api_base override from
    the env var its api_base_env names, if any -- letting litellm resolve
    the "huggingface/..." model string on its own otherwise.
    """
    model_string = _resolve_model_string(candidate, provider)

    if provider == "ollama":
        return dspy.LM(model=model_string, api_base=ollama_api_base)

    api_key_env = candidate.get("api_key_env", "HUGGINGFACE_API_KEY")
    api_key = _env(api_key_env, api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key: set {api_key_env} in .env (a Hugging Face access token with "
            "Inference Providers access -- see https://huggingface.co/settings/tokens)"
        )
    kwargs = dict(model=model_string, api_key=api_key)
    api_base_env = candidate.get("api_base_env")
    if api_base_env:
        api_base = _env(api_base_env, api_base_env)
        if api_base:
            kwargs["api_base"] = api_base
    return dspy.LM(**kwargs)


def _configure_teacher_lm():
    """The strong model used to help every candidate, in two different
    ways depending on which stage is running (see this file's module
    docstring): as GEPA's reflection_lm for the "gepa" stage, and as the
    program that actually solves training passages for the "bootstrap"
    stage. Fixed across every candidate and provider. Mirrors
    optimize_gepa.py's own _configure_reflection_lm() exactly, defaulting
    to this repo's main configured model (API_BASE/MODEL/API_KEY) unless
    REFLECTION_MODEL (and optionally REFLECTION_API_BASE/
    REFLECTION_API_KEY) override it."""
    reflection_model = _env("REFLECTION_MODEL", "REFLECTION_MODEL", None) or _env("MODEL", "MODEL")
    api_base = _env("REFLECTION_API_BASE", "REFLECTION_API_BASE", None) or _env(
        "API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm"
    )
    api_key = _env("REFLECTION_API_KEY", "REFLECTION_API_KEY", None) or _env("API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key for the teacher model. Set REFLECTION_API_KEY or API_KEY in .env."
        )
    return dspy.LM(model=reflection_model, api_base=api_base, api_key=api_key)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_program(program, examples, lm):
    """Run `program` over every example under `lm` (scoped via
    dspy.context, not a global dspy.configure(), so this never leaks one
    candidate's LM into the next), scoring each with syntax_metric().
    Returns aggregate stats: mean/min/max of the blended score, the mean of
    each of the three sub-scores, total wall-clock seconds, how many LM
    calls this made, and total cost across those calls (None if the
    configured provider doesn't report per-call cost -- Ollama never will,
    since it's local; not every remote provider does either).

    A call that raises outright (a real model can fail a request entirely
    -- rate limit, timeout, output DSPy can't parse into the signature's
    fields at all) is scored as a flat zero across every dimension rather
    than crashing the whole bakeoff; which example failed and why is
    recorded in the returned "problems" dict for the caller to print or
    inspect.
    """
    scores, field_scores, relation_scores, vu_scores = [], [], [], []
    problems = {}
    start = time.perf_counter()
    history_start = len(lm.history)
    with dspy.context(lm=lm):
        for example in examples:
            try:
                pred = program(passage=example.passage, tokens=example.tokens)
                result = syntax_metric(example, pred)
            except Exception as exc:  # noqa: BLE001 -- a live LM call can fail in many ways; that's data, not a bug here
                scores.append(0.0)
                field_scores.append(0.0)
                relation_scores.append(0.0)
                vu_scores.append(0.0)
                problems[example.slug] = f"{type(exc).__name__}: {exc}"
                continue
            scores.append(result.score)
            field_scores.append(result.field_score)
            relation_scores.append(result.relation_score)
            vu_scores.append(result.vu_score)
    elapsed = time.perf_counter() - start

    calls = lm.history[history_start:]
    costs = [c.get("cost") for c in calls if isinstance(c, dict) and c.get("cost") is not None]
    total_cost = sum(costs) if costs else None

    return dict(
        n=len(examples),
        mean=statistics.fmean(scores),
        min=min(scores),
        max=max(scores),
        field_mean=statistics.fmean(field_scores),
        relation_mean=statistics.fmean(relation_scores),
        vu_mean=statistics.fmean(vu_scores),
        elapsed_s=elapsed,
        n_calls=len(calls),
        total_cost=total_cost,
        problems=problems,
    )


def _format_stats_line(stats):
    cost = f", ${stats['total_cost']:.4f}" if stats["total_cost"] is not None else ""
    return (
        f"mean={stats['mean']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}  "
        f"fields={stats['field_mean']:.3f}  relations={stats['relation_mean']:.3f}  "
        f"verbal-expr={stats['vu_mean']:.3f}  ({stats['elapsed_s']:.1f}s, {stats['n_calls']} calls{cost})"
    )


# ---------------------------------------------------------------------------
# Optimizer stages
# ---------------------------------------------------------------------------

def _run_gepa_stage(candidate, task_lm, teacher_lm, trainset, heldout, args):
    """The 'gepa' stage: task_lm solves every example itself (both during
    optimization and at final scoring); teacher_lm only ever reads
    syntax_metric's feedback text and proposes better instructions for
    task_lm to follow. Mirrors optimize_gepa.py's own GEPA setup."""
    optimizer_kwargs = dict(
        metric=syntax_metric,
        reflection_lm=teacher_lm,
        track_stats=True,
        log_dir=str(Path(__file__).parent / "gepa_logs" / f"{candidate['label']}-gepa"),
    )
    if args.max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        optimizer_kwargs["auto"] = args.auto
    gepa = dspy.GEPA(**optimizer_kwargs)

    # A fresh ChainOfThought instance -- NOT the shared module-level
    # `analyze` from greek_syntax_dspy.py -- so optimizing one candidate's
    # prompt never clobbers another's or the shared instance the rest of
    # the package uses.
    student = dspy.ChainOfThought(SyntaxAnalysis)
    with dspy.context(lm=task_lm):
        optimized = gepa.compile(student=student, trainset=trainset)

    return optimized, _score_program(optimized, heldout, task_lm)


def _run_bootstrap_stage(candidate, task_lm, teacher_lm, trainset, heldout, args):
    """The 'bootstrap' stage: teacher_lm (Opus) actually SOLVES some
    training passages; only the ones syntax_metric confirms correct get
    attached to the candidate's own program as few-shot demonstrations
    (task_lm never has to solve a training example itself here, only the
    held-out scoring examples afterward -- a different kind of help than
    gepa's, see module docstring).

    IMPORTANT wiring note (found by testing against DummyLM before trusting
    this): dspy.BootstrapFewShot/MIPROv2 do NOT use whatever LM a `teacher`
    program object happens to have bound via .set_lm() -- they read
    dspy.settings.lm inside a `with dspy.context(**teacher_settings):`
    block instead (see dspy/teleprompt/bootstrap.py's _bootstrap_one_
    example). So the teacher's model is set via the constructor's
    teacher_settings=dict(lm=teacher_lm) kwarg, not by pre-binding a
    separate teacher program -- there's no need to build one at all; the
    optimizer defaults to deep-copying the student as its teacher and runs
    that copy under teacher_settings' LM regardless.
    """
    student = dspy.ChainOfThought(SyntaxAnalysis)

    if args.bootstrap_optimizer == "miprov2":
        # Wired per dspy/teleprompt/mipro_optimizer_v2.py's source: prompt_model
        # is the model that proposes instructions (Opus, same as GEPA's
        # reflection_lm), task_model is what's actually evaluated during the
        # search (the candidate, via `with dspy.context(lm=self.task_model)`
        # internally), and teacher_settings governs the demo-bootstrapping
        # step exactly like BootstrapFewShot's does. NOTE: only the
        # bootstrap-fewshot path above was smoke-tested end to end against
        # DummyLM before shipping this script -- a full MIPROv2 trial loop
        # needs far more canned responses than is practical to hand-build for
        # that kind of check, so this path follows the documented API
        # faithfully but start with --bootstrap-optimizer bootstrap-fewshot
        # to validate your own setup first.
        optimizer = dspy.MIPROv2(
            metric=syntax_metric,
            prompt_model=teacher_lm,
            task_model=task_lm,
            teacher_settings=dict(lm=teacher_lm),
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            auto=args.auto,
            track_stats=True,
            log_dir=str(Path(__file__).parent / "gepa_logs" / f"{candidate['label']}-miprov2"),
        )
        optimized = optimizer.compile(student=student, trainset=trainset)
    else:
        optimizer = dspy.BootstrapFewShot(
            metric=syntax_metric,
            teacher_settings=dict(lm=teacher_lm),
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            max_rounds=1,
        )
        optimized = optimizer.compile(student=student, trainset=trainset)

    return optimized, _score_program(optimized, heldout, task_lm)


# ---------------------------------------------------------------------------
# CSV output (read-merge-write, so separate invocations accumulate results)
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "label", "provider", "model", "family", "tier", "stage",
    "n", "mean", "min", "max", "field_mean", "relation_mean", "vu_mean",
    "elapsed_s", "n_calls", "total_cost", "error",
]

_STAGE_ORDER = {"baseline": 0, "gepa": 1, "bootstrap": 2, "skipped": 3}


def _read_existing_rows(path):
    """Every row already in `path`, keyed by (label, provider, stage) --
    empty if the file doesn't exist yet. This is what makes separate
    invocations (one per Ollama candidate you cycle through, or a
    Hugging Face run days apart) accumulate into one results file instead
    of each overwriting the last."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {
            (row.get("label"), row.get("provider"), row.get("stage")): row
            for row in csv.DictReader(f)
        }


def _merge_and_write_csv(path, new_rows):
    """Merge `new_rows` into whatever's already in `path` (by (label,
    provider, stage) -- re-running the same combination overwrites just
    that row) and rewrite the whole file, ordered by CANDIDATES' own order
    (ad hoc/unknown labels sort last) then by stage."""
    existing = _read_existing_rows(path)
    for row in new_rows:
        key = (row.get("label"), row.get("provider"), row.get("stage"))
        existing[key] = {field: row.get(field) for field in _CSV_FIELDS}

    label_order = {c["label"]: i for i, c in enumerate(CANDIDATES)}

    def sort_key(item):
        (label, _provider, stage), _row = item
        return (label_order.get(label, len(CANDIDATES)), label, _STAGE_ORDER.get(stage, 99))

    ordered = [row for _key, row in sorted(existing.items(), key=sort_key)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)


def _existing_baseline_mean(path, label, provider):
    """Look up a previously-recorded baseline mean for (label, provider) in
    `path`, if any -- lets --stages gepa/bootstrap apply
    --min-baseline-to-optimize even when baseline wasn't run THIS
    invocation (e.g. you ran --stages baseline yesterday for this Ollama
    candidate, and today just want --stages bootstrap). Returns None if
    there's no recorded baseline to check."""
    existing = _read_existing_rows(path)
    row = existing.get((label, provider, "baseline"))
    if row is None or row.get("mean") in (None, ""):
        return None
    return float(row["mean"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SyntaxAnalysis across candidate task models on a held-out gold-example slice."
    )
    parser.add_argument(
        "--provider", choices=["huggingface", "ollama"], default="huggingface",
        help="Where candidates run (default: %(default)s). --provider ollama requires exactly one "
             "candidate per invocation -- see this file's module docstring.",
    )
    parser.add_argument(
        "--candidates", nargs="*", default=None, metavar="LABEL",
        help="Only run candidates with these labels (default: every entry in CANDIDATES; "
             "--provider ollama requires exactly one).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Ad hoc override: a full litellm model string to run instead of anything in "
             "CANDIDATES (e.g. 'ollama_chat/llama3.1:8b-instruct-q8_0'). Requires --label.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Name for the --model ad hoc override, used in output and the results CSV.",
    )
    parser.add_argument(
        "--stages", nargs="+", choices=["baseline", "gepa", "bootstrap"],
        default=["baseline", "gepa", "bootstrap"],
        help="Which stage(s) to run this invocation (default: all three). Run just 'baseline' "
             "for a cheap first pass before spending a gepa/bootstrap budget.",
    )
    parser.add_argument(
        "--bootstrap-optimizer", choices=["bootstrap-fewshot", "miprov2"], default="bootstrap-fewshot",
        help="Which optimizer implements the 'bootstrap' stage (default: %(default)s -- cheaper and "
             "the only one smoke-tested end to end here; miprov2 is heavier and also proposes new "
             "instructions, not just demos).",
    )
    parser.add_argument(
        "--max-bootstrapped-demos", type=int, default=3,
        help="Max Opus-solved training examples attached as demos in the 'bootstrap' stage (default: %(default)s).",
    )
    parser.add_argument(
        "--max-labeled-demos", type=int, default=3,
        help="Max plain gold-labeled examples (no teacher solving needed) attached as demos in the "
             "'bootstrap' stage (default: %(default)s).",
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--auto", choices=["light", "medium", "heavy"], default="light",
        help="Budget preset for the 'gepa' stage (and for 'bootstrap' when --bootstrap-optimizer "
             "miprov2) (default: %(default)s).",
    )
    budget.add_argument(
        "--max-metric-calls", type=int, default=None,
        help="Exact call budget for the 'gepa' stage instead of --auto.",
    )
    parser.add_argument(
        "--min-baseline-to-optimize", type=float, default=0.0,
        help="Skip a candidate's gepa/bootstrap stage (but still report its baseline, if run) if its "
             "zero-shot mean score on the held-out set -- from this invocation, or from a previously "
             "recorded row in --out if baseline wasn't run this time -- is below this threshold. "
             "Default: 0.0 (never skip).",
    )
    parser.add_argument(
        "--ollama-api-base", default=None,
        help="Override Ollama's address (default: OLLAMA_API_BASE in .env, or http://localhost:11434).",
    )
    parser.add_argument(
        "--out", default="model_bakeoff_results.csv",
        help="Results file to merge into (default: %(default)s). Existing rows for other "
             "candidates/providers/stages are preserved; a rerun of the same combination overwrites "
             "just that row.",
    )
    args = parser.parse_args()

    if args.model and not args.label:
        raise SystemExit("--model requires --label (a name for this ad hoc candidate).")
    if args.label and not args.model:
        raise SystemExit("--label without --model has nothing to name -- did you mean --candidates?")

    trainset, heldout = build_split()
    print(
        f"{len(trainset)} training fixtures, {len(heldout)} held-out fixtures "
        "(never used for any candidate's gepa/bootstrap stage).\n"
    )

    if args.model:
        candidates = [dict(
            label=args.label, override_model=args.model,
            family="ad hoc", tier="?", notes="one-off override via --model/--label",
        )]
    else:
        candidates = CANDIDATES
        if args.candidates:
            wanted = set(args.candidates)
            candidates = [c for c in CANDIDATES if c["label"] in wanted]
            unknown = wanted - {c["label"] for c in candidates}
            if unknown:
                raise SystemExit(
                    f"Unknown candidate label(s): {sorted(unknown)} -- see CANDIDATES in this file "
                    "for valid labels, or use --model/--label for a one-off."
                )

    if args.provider == "ollama" and len(candidates) != 1:
        raise SystemExit(
            "--provider ollama needs exactly one candidate per invocation (only one model can "
            f"realistically be loaded in Ollama at a time) -- got {len(candidates)}. Pass "
            "--candidates <one label>, or --model/--label for an ad hoc one."
        )

    ollama_api_base = args.ollama_api_base or _env("OLLAMA_API_BASE", "OLLAMA_API_BASE", "http://localhost:11434")

    needs_teacher = "gepa" in args.stages or "bootstrap" in args.stages
    teacher_lm = _configure_teacher_lm() if needs_teacher else None

    rows = []
    for candidate in candidates:
        model_label = f"{candidate['label']} [{args.provider}]"
        print(f"=== {model_label} ===")
        try:
            task_lm = _configure_candidate_lm(candidate, args.provider, ollama_api_base)
            resolved_model = _resolve_model_string(candidate, args.provider)
        except RuntimeError as exc:
            print(f"  skipped: {exc}\n")
            rows.append(dict(
                label=candidate["label"], provider=args.provider, model=candidate.get("model", ""),
                family=candidate.get("family", ""), tier=candidate.get("tier", ""),
                stage="skipped", error=str(exc),
            ))
            continue
        print(f"  model string: {resolved_model}")

        baseline_mean_for_gating = None

        if "baseline" in args.stages:
            baseline_program = dspy.ChainOfThought(SyntaxAnalysis)
            baseline = _score_program(baseline_program, heldout, task_lm)
            print(f"  baseline (zero-shot): {_format_stats_line(baseline)}")
            for slug, problem in baseline["problems"].items():
                print(f"    {slug}: {problem}")
            rows.append(dict(
                candidate, provider=args.provider, model=resolved_model, stage="baseline",
                **{k: v for k, v in baseline.items() if k != "problems"},
            ))
            baseline_mean_for_gating = baseline["mean"]

        if "gepa" in args.stages or "bootstrap" in args.stages:
            if baseline_mean_for_gating is None:
                baseline_mean_for_gating = _existing_baseline_mean(args.out, candidate["label"], args.provider)
                if baseline_mean_for_gating is None and args.min_baseline_to_optimize > 0.0:
                    print(
                        "  no recorded baseline for this candidate/provider -- proceeding without "
                        "the --min-baseline-to-optimize gate this invocation."
                    )

        gated_out = (
            baseline_mean_for_gating is not None
            and baseline_mean_for_gating < args.min_baseline_to_optimize
        )
        if gated_out and ("gepa" in args.stages or "bootstrap" in args.stages):
            print(
                f"  baseline ({baseline_mean_for_gating:.3f}) below --min-baseline-to-optimize "
                f"({args.min_baseline_to_optimize}) -- skipping gepa/bootstrap for this candidate.\n"
            )
            continue

        if "gepa" in args.stages:
            print("  running gepa stage -- this makes many real LM calls (task + teacher model)...")
            optimized, stats = _run_gepa_stage(candidate, task_lm, teacher_lm, trainset, heldout, args)
            print(f"  after gepa: {_format_stats_line(stats)}")
            for slug, problem in stats["problems"].items():
                print(f"    {slug}: {problem}")
            rows.append(dict(
                candidate, provider=args.provider, model=resolved_model, stage="gepa",
                **{k: v for k, v in stats.items() if k != "problems"},
            ))
            out_path = Path(__file__).parent / f"optimized_{candidate['label']}_{args.provider}_gepa.json"
            optimized.save(str(out_path))
            print(f"  saved optimized program to {out_path.name}")

        if "bootstrap" in args.stages:
            print(
                f"  running bootstrap stage ({args.bootstrap_optimizer}) -- Opus solves training "
                "examples, task model only runs held-out scoring..."
            )
            optimized, stats = _run_bootstrap_stage(candidate, task_lm, teacher_lm, trainset, heldout, args)
            print(f"  after bootstrap: {_format_stats_line(stats)}")
            for slug, problem in stats["problems"].items():
                print(f"    {slug}: {problem}")
            rows.append(dict(
                candidate, provider=args.provider, model=resolved_model, stage="bootstrap",
                **{k: v for k, v in stats.items() if k != "problems"},
            ))
            out_path = Path(__file__).parent / f"optimized_{candidate['label']}_{args.provider}_bootstrap.json"
            optimized.save(str(out_path))
            print(f"  saved optimized program to {out_path.name}")

        print()

    _merge_and_write_csv(args.out, rows)
    print(f"Results merged into {args.out}")


if __name__ == "__main__":
    main()
