"""
Offline tests for grammatike/verbal_units.py's assign_verbal_units(),
compute_subordination_depths(), and find_unanchored_coordinated_verbs().
Greek analogue of arsgrammatica's test_verbal_units.py.

These run against real gold fixtures (fixtures/gold_examples.py), not
synthetic data, since the tricky cases this module has to get right --
a subordinating conjunction or relative pronoun's own outgoing relation
pointing at the OUTER clause, while the token itself belongs to the INNER
clause it introduces; a genitive-absolute noun redirecting to its
circumstantial participle instead of the verb it points at -- only really
show up in genuine sentences. See verbal_units.py's module docstring for
the reasoning, and find_unanchored_coordinated_verbs()'s own docstring for
how it exploits syntax_model.md's three "connecting word" shapes (single
pair, paired correlative, 3+-member series) to name a missing coordinate
verb precisely, rather than the coarser subordination-depth proxy an
earlier version of that relation forced it to use.
"""

import pytest

from grammatike.models import TokenAnalysis
from grammatike.verbal_units import (
    assign_verbal_units,
    compute_subordination_depths,
    find_unanchored_coordinated_verbs,
)
from fixtures.gold_examples import GOLD_EXAMPLES


def _tokengraph(slug):
    example = next(e for e in GOLD_EXAMPLES if e.slug == slug)
    return [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_every_token_gets_an_entry(example):
    """Every token id in the tokengraph -- including punctuation and
    tokens with no relation at all -- must appear as a key, even if its
    value is None."""
    tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
    assignment = assign_verbal_units(tokengraph)
    assert set(assignment.keys()) == {tok.id for tok in tokengraph}


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda e: e.slug)
def test_every_anchor_is_assigned_to_itself(example):
    tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
    assignment = assign_verbal_units(tokengraph)
    for tok in tokengraph:
        if tok.verbalunitid is not None:
            assert assignment[tok.id] == tok.verbalunitid


def test_subordinating_conjunction_belongs_to_the_clause_it_introduces():
    """ἐπειδή in "ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη" is grammatically
    part of the dependent clause it introduces ("ἐπειδὴ δὲ ἦν πρὸς
    ἡμέραν") -- even though ἐπειδή's own relatedtoken1 points at ἧκεν
    (t6), the MAIN clause's verb, per the "subordinating conjunction"
    relation (see syntax_model.md). It must not be pulled into ἧκεν's
    verbal unit."""
    tokengraph = _tokengraph("dependent_verb_epeide_de_en_hos_hekeen")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t0"] == "t2"  # ἐπειδή -> ἦν's unit, not ἧκεν's
    assert assignment["t2"] == "t2"  # ἦν -> itself
    assert assignment["t7"] == "t6"  # ἐκείνη (subject of ἧκεν) stays in the main clause


def test_relative_pronoun_belongs_to_the_clause_it_introduces_not_its_antecedent():
    """ὅν in "ὁ ἀνὴρ ὃν εἶδον ἀπῆλθεν" points back at its antecedent ἀνήρ
    via relatedtoken1 ("relative pronoun"), and at εἶδον via relatedtoken2
    ("direct object") -- but grammatically, ὅν is part of the εἶδον
    clause, not ἀνήρ's. The relative-pronoun link must not win."""
    tokengraph = _tokengraph("relative_pronoun_ho_aner_hon_eidon")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t2"] == "t3"  # ὅν -> εἶδον's unit, not ἀνήρ's
    assert assignment["t1"] == "t4"  # ἀνήρ (subject of ἀπῆλθεν) stays with the main verb


def test_genitive_absolute_noun_belongs_to_its_circumstantial_participle():
    """In "προϊόντος δὲ τοῦ χρόνου ἧκον μὲν ἀπροσδοκήτως ἐκ ἀγροῦ", χρόνου
    is a true genitive absolute (per syntax_model.md, "otherwise
    unconnected syntactically"): syntactically absolute from ἧκον's own
    clause despite pointing straight at it, it takes its verbal unit
    instead from προϊόντος, the circumstantial participle it grammatically
    agrees with -- so χρόνου and προϊόντος end up in the SAME
    (προϊόντος's) verbal unit, not split between προϊόντος and ἧκον."""
    tokengraph = _tokengraph("circumstantial_genitive_absolute_proiontos_de_tou_chronou")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t3"] == "t0"  # χρόνου -> προϊόντος (its circumstantial participle)
    assert assignment["t0"] == "t0"  # προϊόντος -> itself
    assert assignment["t2"] == "t0"  # τοῦ (χρόνου's article) follows it there
    assert assignment["t4"] == "t4"  # ἧκον -> itself
    assert assignment["t5"] == "t4"  # μέν (sentence connector) stays with ἧκον


def test_circumstantial_participle_noun_that_fits_normally_keeps_its_own_clause_role():
    """In "ἐγὼ ἅπαντα ἐπιδείξω τὰ ἐμαυτοῦ πράγματα, οὐδὲν παραλείπων, ἀλλὰ
    λέγων τἀληθῆ", ἐγώ fits into the main clause as ἐπιδείξω's subject --
    an ORDINARY relation, not "genitive absolute" -- so despite being the
    noun BOTH circumstantial participles (παραλείπων, λέγων) agree with,
    it is assigned to ἐπιδείξω's unit, not redirected to either
    participle's."""
    tokengraph = _tokengraph("circumstantial_fits_clause_ego_hapanta_epideixo")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t0"] == "t2"  # ἐγώ -> ἐπιδείξω's own unit (its own subject role)
    assert assignment["t8"] == "t8"  # παραλείπων -> itself (singleton unit)
    assert assignment["t11"] == "t11"  # λέγων -> itself (singleton unit)


def test_unrelated_and_punctuation_tokens_get_none():
    tokengraph = _tokengraph("aside_proton_men_oun_dei")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t2"] is None  # οὖν: bare discourse particle, no relation at all
    assert assignment["t19"] is None  # trailing "."


def test_exclamation_o_resolves_through_the_vocative_it_introduces():
    """ὦ (t4) has relatedtoken1 -> ἄνδρες (t5), relationship1 'exclamation'
    -- per syntax_model.md's own rule that ὦ introducing a vocative takes
    the vocative noun/pronoun as relation1, not the verb directly -- so it
    should resolve through ἄνδρες's own 'vocative' relation to ἔστι (t16)
    exactly like ἄνδρες itself, via the same generic fallback chase any
    two-hop relation uses (see assign_verbal_units's own docstring)."""
    tokengraph = _tokengraph("aside_proton_men_oun_dei")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t5"] == "t16"  # ἄνδρες -> ἔστι, via 'vocative'
    assert assignment["t4"] == "t16"  # ὦ -> ἄνδρες -> ἔστι, via 'exclamation'


def test_cycle_in_relations_does_not_infinite_loop():
    """Two tokens relating only to each other, with no anchor reachable,
    must resolve to None rather than recursing forever."""
    tokengraph = [
        TokenAnalysis(
            id="t0", token="a", tokentype="lexical",
            relatedtoken1="t1", relationship1="adverbial",
        ),
        TokenAnalysis(
            id="t1", token="b", tokentype="lexical",
            relatedtoken1="t0", relationship1="adverbial",
        ),
    ]
    assignment = assign_verbal_units(tokengraph)
    assert assignment == {"t0": None, "t1": None}


# ---------------------------------------------------------------------------
# compute_subordination_depths()
# ---------------------------------------------------------------------------


def test_independent_verb_has_depth_zero():
    """An independent verb's own relatedtoken1 is the 'root' sentinel,
    which compute_subordination_depths() special-cases directly rather
    than chasing a governing relation at all."""
    tokengraph = _tokengraph("unit_verb_root_ten_thuran_anoixen")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t2"] == 0  # ἀνέῳξεν, independent
    assert warnings == []


def test_dependent_verb_via_subordinating_conjunction_has_depth_one():
    tokengraph = _tokengraph("dependent_verb_epeide_de_en_hos_hekeen")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t6"] == 0  # ἧκεν, independent
    assert depths["t2"] == 1  # ἦν, dependent on ἧκεν via ἐπειδή
    assert warnings == []


def test_dependent_verb_via_relative_pronoun_has_depth_one():
    """εἶδον's own relatedtoken1 points at ὅν (t2), which points (via
    relatedtoken1) at its antecedent ἀνήρ (t1), which in turn points at
    ἀπῆλθεν (t4) -- a two-hop chase through two non-anchor intermediaries
    before reaching the governing anchor."""
    tokengraph = _tokengraph("relative_pronoun_ho_aner_hon_eidon")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t4"] == 0  # ἀπῆλθεν, independent
    assert depths["t3"] == 1  # εἶδον, dependent on ἀπῆλθεν
    assert warnings == []


def test_direct_quote_and_aside_have_depth_one():
    tokengraph = _tokengraph("direct_quote_hina_su_ge_ephe")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t5"] == 0  # ἔφη, independent
    assert depths["t7"] == 1  # πειρᾷς, direct quote framed by ἔφη
    assert warnings == []

    tokengraph = _tokengraph("aside_proton_men_oun_dei")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t16"] == 0  # ἔστι, independent
    assert depths["t8"] == 1  # δεῖ, aside interrupting ἔστι
    assert warnings == []


def test_circumstantial_participle_and_genitive_absolute_have_depth_one():
    tokengraph = _tokengraph("circumstantial_genitive_absolute_proiontos_de_tou_chronou")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t4"] == 0  # ἧκον, independent
    assert depths["t0"] == 1  # προϊόντος, genitive absolute
    assert warnings == []

    tokengraph = _tokengraph("circumstantial_fits_clause_ego_hapanta_epideixo")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t2"] == 0  # ἐπιδείξω, independent
    assert depths["t8"] == 1  # παραλείπων, circumstantial participle
    assert depths["t11"] == 1  # λέγων, circumstantial participle
    assert warnings == []


def test_indirect_statement_has_depth_one():
    tokengraph = _tokengraph("indirect_statement_infinitive_ephaske_lychnon")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t0"] == 0  # ἔφασκε, independent
    assert depths["t3"] == 1  # ἀποσβεσθῆναι, indirect statement governed by ἔφασκε
    assert warnings == []

    tokengraph = _tokengraph("indirect_statement_participle_eide_de_basileian")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t0"] == 0  # εἶδε, independent
    assert depths["t4"] == 1  # φεύγουσαν, indirect statement governed by εἶδε
    assert warnings == []


def test_attributive_participle_reaches_depth_one():
    """The one construction with no Latin precedent at all: ὑβρίζων's own
    relatedtoken1 points at ἀνήρ (not itself an anchor), whose own
    relatedtoken1 (relationship1 'subject') reaches τυγχάνει -- the exact
    same generic hop-through-the-noun chase that already handles
    circumstantial participles, needing no dedicated code path."""
    tokengraph = _tokengraph("attributive_participle_ho_aner_ho_hybrizon")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t10"] == 0  # τυγχάνει, independent
    assert depths["t4"] == 1  # ὑβρίζων, attributive participle on ἀνήρ
    assert warnings == []


def test_depth_two_nesting_through_a_dependent_clause():
    """ἡμαρτηκέναι is an indirect statement governed by ᾔδει -- itself a
    dependent verb one level below ἠνιάθη -- so ἡμαρτηκέναι sits two
    levels below the root, not one."""
    tokengraph = _tokengraph("depth_two_epei_edei_hemartekenai")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths["t5"] == 0  # ἠνιάθη, independent
    assert depths["t1"] == 1  # ᾔδει, dependent on ἠνιάθη via ἐπεί
    assert depths["t3"] == 2  # ἡμαρτηκέναι, indirect statement governed by ᾔδει
    assert warnings == []


def test_every_gold_example_anchor_resolves_with_no_warnings():
    """Sanity check across the whole fixture set: every documented
    governing-relation pattern (subordinating conjunction, relative
    pronoun, direct quote, aside, circumstantial/attributive participle,
    genitive absolute, indirect statement) should resolve cleanly, with no
    unresolved anchors and no warnings -- a regression here would mean a
    fixture's relation shape silently stopped being chase-able."""
    for example in GOLD_EXAMPLES:
        tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
        depths, warnings = compute_subordination_depths(tokengraph)
        assert warnings == [], f"{example.slug}: {warnings}"
        assert all(v is not None for v in depths.values()), (
            f"{example.slug}: unresolved depth in {depths}"
        )


def test_cycle_in_relations_leaves_depth_unresolved_with_warning():
    """Two anchors whose own outgoing relations point only at each other
    (no 'root' sentinel, no third party to break the cycle) must resolve
    to None with an explanatory warning, not recurse forever."""
    tokengraph = [
        TokenAnalysis(
            id="t0", token="a", tokentype="lexical", verbalunitid="t0",
            relatedtoken1="t1", relationship1="unit verb",
        ),
        TokenAnalysis(
            id="t1", token="b", tokentype="lexical", verbalunitid="t1",
            relatedtoken1="t0", relationship1="unit verb",
        ),
    ]
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths == {"t0": None, "t1": None}
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# find_unanchored_coordinated_verbs()
# ---------------------------------------------------------------------------


def test_correctly_coordinated_verbs_produce_no_warning():
    """διῃτώμην and the implied repetition of it are both anchors of their
    own verbal unit. μέν and δέ are a paired correlative: μέν's own
    relation1 -> διῃτώμην, δέ's own relation1 -> t8_implied (each its OWN
    clause's verb, chained to each other via relation2) -- both named
    items are anchored, so nothing is flagged."""
    tokengraph = _tokengraph("implied_repetition_ego_men_ano_dietomen")
    assert find_unanchored_coordinated_verbs(tokengraph) == []


def test_connecting_word_pointing_at_a_non_anchor_produces_no_warning():
    """καί in "εἶδον δύο ἄνδρας καὶ γʹ γυναῖκας" joins two nouns
    (ἄνδρας/γυναῖκας), not two verbs -- neither its relatedtoken1 (ἄνδρας)
    nor its relatedtoken2 (γυναῖκας) is a verbal-unit anchor, so this must
    not be mistaken for a broken coordinate-clause pair."""
    tokengraph = _tokengraph("numeral_vs_lexical_eidon_duo_andras")
    assert find_unanchored_coordinated_verbs(tokengraph) == []


def test_sentence_connector_only_sentence_produces_no_warning():
    """A sentence containing only 'sentence connector' relations (no
    'connecting word' at all) has nothing for this check to look at --
    'sentence connector' is deliberately excluded (see this function's own
    docstring), so this must not crash or flag anything."""
    tokengraph = _tokengraph("dependent_verb_epeide_de_en_hos_hekeen")
    assert find_unanchored_coordinated_verbs(tokengraph) == []


def test_flags_a_coordinated_verb_that_lost_its_own_anchor():
    """Simulates the real live-LM mistake this function exists to catch:
    take the correct implied-repetition fixture and strip the second
    clause's own anchor (verbalunitid, and its 'root'/'unit verb'
    relation) -- exactly what a live model might produce for an elided
    verb it failed to record as 'implied repetition' -- and confirm the
    resulting asymmetry (διῃτώμην still anchored, its coordinate partner
    no longer) is flagged BY NAME: δέ's own relation1 (t8_implied) is what
    lost its anchor, so the warning names t8_implied directly -- the
    precision the old, pre-relation2 version of this check couldn't
    achieve (it could only point back at the series' first, already-fine
    member, t3)."""
    tokengraph = _tokengraph("implied_repetition_ego_men_ano_dietomen")
    for tok in tokengraph:
        if tok.id == "t8_implied":
            tok.verbalunitid = None
            tok.relatedtoken1 = None
            tok.relationship1 = None

    warnings = find_unanchored_coordinated_verbs(tokengraph)
    assert len(warnings) == 1
    assert "t8_implied" in warnings[0]  # the specific token that lost its anchor
    assert "t6" in warnings[0]  # δέ, the connecting word naming it


def test_flags_a_lone_connecting_word_missing_its_second_member_entirely():
    """A single connecting word whose relation1 already points at a
    recognized anchor, but with relation2 entirely unset and no
    correlative partner naming it either, looks like one half of a
    coordinated pair whose other half was never recorded at all -- a
    distinct, cheaper-to-detect failure from the "recorded but unanchored"
    case above."""
    tokengraph = _tokengraph("circumstantial_fits_clause_ego_hapanta_epideixo")
    for tok in tokengraph:
        if tok.id == "t10":  # ἀλλά
            tok.relatedtoken2 = None
            tok.relationship2 = None

    warnings = find_unanchored_coordinated_verbs(tokengraph)
    assert len(warnings) == 1
    assert "t10" in warnings[0]


def test_paired_connecting_words_resolve_to_their_own_respective_clauses():
    """μέν and δέ each belong to the clause they themselves introduce --
    μέν to διῃτώμην's unit, δέ to t8_implied's unit -- not both to the
    first clause the way an earlier version of syntax_model.md's
    'connecting word' relation (before relation2 named the correlative
    partner) would have resolved them."""
    tokengraph = _tokengraph("implied_repetition_ego_men_ano_dietomen")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t1"] == "t3"  # μέν -> διῃτώμην's own unit
    assert assignment["t6"] == "t8_implied"  # δέ -> t8_implied's own unit
