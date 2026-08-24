"""
grammatike: DSPy program analyzing the syntax of Ancient Greek passages, per
syntax_model.md. Greek analogue of arsgrammatica's latin_syntax_dspy.py.

This module covers only the analysis stage:
  1. SyntaxAnalysis -- a dspy.Signature that takes a passage plus its
     pre-segmented token list and produces `verbalunits` and `tokengraph`,
     using the ids handed to it.
  2. validate()   -- a light sanity check that every id the LM refers to in
     its output actually exists in the input token list, so malformed
     output is easy to spot.

Tokens are not produced here -- they come from a deterministic,
citation-aware segmentation stage (segmentation.py, mirroring
arsgrammatica's own pipeline split). See pipeline.py for the module that
ties the two stages together, including its analyze_passage() convenience
wrapper.

Run this file directly for a quick smoke test against the configured LM:
    python greek_syntax_dspy.py

For tests that don't need network access, see the `tests/` directory,
which drives this signature with dspy's DummyLM.

---------------------------------------------------------------------------
Notes on places this port had to extrapolate beyond syntax_model.md
---------------------------------------------------------------------------
syntax_model.md documents the Greek scheme with worked examples, but a few
corners are stated only as a general rule (or not stated at all) where the
Latin reference implementation had to cover an analogous case. Each is
flagged here, and again with a `# TODO` comment at the point in
SyntaxAnalysis's docstring where it appears, rather than silently invented:

  - 'implied eimi' is documented with exactly one worked example (an
    implied INFINITIVE of εἰμί inside indirect statement). The general
    phrasing ("elided εἰμί in predicate expressions") is broader than
    that one example, so this port also covers (a) a bare predicate
    sentence with no governing verb of speech/thought at all (implied
    FINITE εἰμί), by direct analogy with arsgrammatica's 'implied sum'
    bare-predicate case, and (b) an omitted conjugated εἰμί in a compound
    perfect-system form, by analogy with VerbalExpression's own docstring
    mentioning that construction. Neither (a) nor (b) has a worked example
    in syntax_model.md itself.
  - The RelationLabel comment block in models.py lists "attributive
    participle" (relation1 = the noun an attributive-participle verbal
    expression agrees with) AND separately glosses "attributive" as also
    covering "participle-in-attributive-position". Since syntax_model.md's
    own worked example (ὑβρίζων/ἀνήρ) uses "attributive participle" for
    exactly this relation, and Greek's scheme (unlike Latin's) makes EVERY
    attributive participle its own verbal expression, this port treats
    "attributive participle" as the relation an attributive participle's
    OWN verbal expression uses toward its noun, and reserves "attributive"
    for ordinary adjectives and prepositional phrases modifying a noun.
  - An infinitive functioning as an ordinary noun (subject/object), rather
    than anchoring indirect statement or completing a governing verb, is
    not discussed in syntax_model.md at all, but is common in Greek (e.g.
    an articular infinitive) and directly analogous to arsgrammatica's
    equivalent section; this port keeps that guidance, generalized from
    Latin's, flagged as an extrapolation.
  - καί's well-known double life as connective ("and") versus adverb
    ("also", "even") is not addressed in syntax_model.md, unlike Latin's
    explicit discussion of 'et'. This port carries over guidance for the
    adverbial reading, flagged as an extrapolation from the general
    'adverbial' relation rather than sanctioned by syntax_model.md.
  - syntax_model.md's own text for "attributive to a noun" prepositional
    phrases (Token-level table of dependencies) still carries the Latin
    example "pugna ad Cannas" verbatim (apparently never translated when
    the document was adapted from arsgrammatica's). This port substitutes
    a constructed Greek example instead, flagged where it appears.
  - syntax_model.md states the general rule for indirect questions only
    implicitly, via "a subordinating conjunction or a relative OR
    INTERROGATIVE pronoun" (no worked example). This port supplies a
    constructed example, flagged where it appears, following the same
    convention Latin used for its own (also worked) indirect-question
    example.
  - A *supplementary* participle (ὤν in ὁ ἀνὴρ ἐχθρὸς ὢν ἡμῖν τυγχάνει) is
    explicitly NOT a verbal expression, but syntax_model.md does not say
    what relation, if any, it takes toward its governing verb -- no
    RelationLabel value is dedicated to "supplementary participle". This
    port leaves relatedtoken1/relationship1 unset for such a participle
    itself (no label fits), while still letting it carry its own
    predicate/object/adverbial complements exactly as a linking verb
    would -- see the docstring below for the reasoning.
"""

from typing import List

import dspy

from .models import IMPLIED_TOKENTYPES, Token, VerbalExpression, TokenAnalysis


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class SyntaxAnalysis(dspy.Signature):
    """Analyze the syntax of a passage of Ancient Greek according to a
    two-part scheme:

    (1) a list of verbal expressions. Three constructions count as a
        verbal expression: finite verbs, infinitives, and participles --
        but for participles, only some of them (see below).

        - A finite verb (including a compound perfect-system form made of
          a participle plus a conjugated form of εἰμί, e.g. ὁ νόμος
          γεγραμμένος ἐστίν) is always a verbal expression. Classify its
          syntactic type as 'independent' (main/principal), 'dependent'
          (subordinate, introduced by a subordinating word), 'direct
          quote' (occurring in directly quoted speech framed by another
          verb, e.g. νόμιζε in '"εὐφίλητε" ἔφη "μηδεμιᾷ πολυπραγμοσύνῃ
          προσεληλυθέναι με νόμιζε πρὸς σέ."'), or 'aside' (a verbal
          expression that interrupts the surrounding syntax, e.g. δεῖ in
          'πρῶτον μὲν οὖν, ὦ ἄνδρες, (δεῖ γὰρ καὶ ταῦθ᾽ ὑμῖν διηγήσασθαι)
          οἰκίδιον ἔστι μοι διπλοῦν' interrupting the independent verbal
          expression ἔστι). Example of independent vs. dependent: in
          'ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν ἐκείνη', ἧκεν is 'independent'
          and ἦν is 'dependent' (introduced by the subordinating
          conjunction ἐπειδή).
        - An infinitive is a verbal expression only when part of an
          indirect statement; its syntactic type is always 'indirect
          statement'. Example: in 'ἔφασκε τὸν λύχνον ἀποσβεσθῆναι', ἔφασκε
          is independent and ἀποσβεσθῆναι anchors the indirect-statement
          verbal expression. In a compound perfect-system form (participle
          + a conjugated form of εἰμί), the form of εἰμί anchors the
          verbal expression, same as any other compound form.
        - A participle constitutes a verbal expression in THREE cases,
          each with its own dedicated syntactic_type -- unlike Latin,
          which uses a single 'dependent' value for every predicate-sense
          participle, Greek's scheme gives each its own name:
            - 'indirect statement': a participle expressing indirect
              speech after a verb of perception or thinking. Example: in
              'εἶδε δὲ τὴν βασίλειαν φεύγουσαν', εἶδε is independent and
              φεύγουσαν (not an infinitive here) anchors the
              indirect-statement verbal expression.
            - 'attributive': a participle in attributive position.
              Example: in 'ὁ γὰρ ἀνὴρ ὁ ὑβρίζων εἰς σὲ ἐχθρὸς ὢν ἡμῖν
              τυγχάνει', the repeated article puts ὑβρίζων in attributive
              position with ἀνήρ, so ὑβρίζων anchors an 'attributive'
              verbal expression. In Greek, UNLIKE Latin, every attributive
              participle counts as its own verbal expression -- there is
              no purely-adjectival, non-verbal-expression reading for an
              attributive participle the way Latin's "consentiens laus"
              was not a verbal expression at all.
            - 'circumstantial': a participle in circumstantial position
              (including one forming a genitive absolute). Example: in
              'χρόνου μεταξὺ διαγενομένου, προσέρχεταί μοί τις πρεσβῦτις
              ἄνθρωπος', προσέρχεταί is independent and διαγενομένου
              anchors a 'circumstantial' verbal expression.
          By contrast, a *supplementary* participle -- one that completes
          the sense of its governing verb as a single predicate idea
          (e.g. with τυγχάνω, λανθάνω, φαίνομαι, παύομαι, or the like),
          rather than standing attributively with a noun or
          circumstantially/adverbially to the clause -- is explicitly NOT
          a verbal expression and gets no `verbalunits` entry at all.
          Example: in that same sentence 'ὁ γὰρ ἀνὴρ ὁ ὑβρίζων εἰς σὲ
          ἐχθρὸς ὢν ἡμῖν τυγχάνει', ὤν supplements τυγχάνει (there is a
          single independent verbal expression, anchored to τυγχάνει) and
          is NOT its own verbal expression. Do not over-generate
          verbal-expression entries for participles: check first whether a
          participle is genuinely attributive (repeated article, or
          agreeing with a noun as its ordinary modifier), genuinely
          circumstantial (an adverbial predication about a noun, loosely
          attached to the clause), or genuinely reporting indirect
          perception -- and only then give it a `verbalunits` entry; a
          participle that instead completes one predicate idea together
          with a governing verb like τυγχάνω does not.
          # TODO: syntax_model.md does not name a RelationLabel for a
          # supplementary participle's own relation to its governing verb
          # (no "supplementary" or "complementary participle" value
          # exists). This port leaves such a participle's own
          # relatedtoken1/relationship1 unset -- no documented label
          # fits -- while still letting it take its own predicate/
          # object/adverbial complements exactly as a linking verb would
          # (e.g. ἐχθρός, the predicate adjective of ὤν in the example
          # above, still relates to ὤν with relationship1 'predicate',
          # exactly as it would to a finite linking verb).

        Classify each verbal expression's semantic type too (transitive
        active / transitive passive / intransitive / linking verb).
        Examples: προσεῖχον in 'προσεῖχον τὸν νοῦν' is transitive active;
        διαφθείρεται in 'ἡ ἐμὴ γυνὴ ὑπὸ τούτου τοῦ ἀνθρώπου διαφθείρεται'
        is transitive passive; εἰσῄει in 'πάντα μου εἰς τὴν γνώμην
        εἰσῄει' is intransitive; ἦ in 'μεστὸς ἦ ὑποψίας' is a linking
        verb.

    (2) a token-by-token dependency graph. For each token, record up to two
        relations to other tokens (by id), using only these relation
        labels:

        - unit verb (independent): every INDEPENDENT verb has relatedtoken1
          = the special sentinel string 'root' -- never an actual token id;
          no real token may be assigned the id 'root' -- and relationship1
          = 'unit verb'. Example: in 'τὴν θύραν ἀνέῳξεν', ἀνέῳξεν has
          relatedtoken1 'root', relationship1 'unit verb'.
        - unit verb (dependent) / subordinating conjunction / relative
          pronoun: the verb of a DEPENDENT clause has relatedtoken1 -> the
          id of its subordinating conjunction or relative/interrogative
          pronoun, relationship1 = 'unit verb'. That conjunction or pronoun
          in turn has relatedtoken1 -> the id of the verb of the clause it
          is subordinate to, with relationship1 = 'subordinating
          conjunction' for a conjunction, or relatedtoken1 -> its
          antecedent's id with relationship1 = 'relative pronoun' for a
          relative pronoun. Example: in 'ἐπειδὴ δὲ ἦν πρὸς ἡμέραν, ἧκεν
          ἐκείνη', ἐπειδή has relatedtoken1 -> ἧκεν, relationship1
          'subordinating conjunction', and ἦν has relatedtoken1 ->
          ἐπειδή, relationship1 'unit verb'. Another example, with a
          conjunction: in 'κατηγόρει ὡς μετὰ τὴν ἐκφορὰν αὐτῇ προσίοι',
          ὡς has relatedtoken1 -> κατηγόρει, relationship1 'subordinating
          conjunction', and προσίοι has relatedtoken1 -> ὡς, relationship1
          'unit verb'. Indirect questions are treated as a kind of
          dependent clause: an interrogative pronoun introducing one is
          treated the same way as a subordinating conjunction -- it has
          relatedtoken1 -> the id of the verb it introduces, relationship1
          = 'subordinating conjunction' (no separate label for this case)
          -- while the dependent verb itself has relatedtoken1 -> the
          interrogative word's id, relationship1 = 'unit verb', exactly
          like any other dependent clause.
          # TODO: syntax_model.md states this indirect-question rule only
          # in passing ("a subordinating conjunction or a relative or
          # interrogative pronoun") with no worked example; the following
          # is constructed by analogy, not quoted from syntax_model.md:
          # in 'οὐκ οἶδα τίς ἦλθεν' ("I don't know who came"), τίς has
          # relatedtoken1 -> οἶδα, relationship1 'subordinating
          # conjunction', and ἦλθεν has relatedtoken1 -> τίς, relationship1
          # 'unit verb'.
        - relative pronoun (second relation): a relative pronoun ALSO
          relates to its own function inside the relative clause, using
          relatedtoken2/relationship2 (since relatedtoken1/relationship1 is
          already used for the antecedent link) -- the ordinary relation
          it would have if it were any other noun/pronoun in that clause
          (e.g. 'direct object', 'subject', a case relation, etc). Example:
          in 'οὐκ ἐγώ σε ἀποκτενῶ, ἀλλ᾽ ὁ τῆς πόλεως νόμος, ὃν σὺ περὶ
          ἐλάττονος τῶν ἡδονῶν ἐποιήσω', ὅν has relatedtoken1 -> νόμος
          (its antecedent), relationship1 'relative pronoun', AND
          relatedtoken2 -> ἐποιήσω, relationship2 'direct object'.
        - indirect statement (governing verb): an infinitive OR participle
          anchoring an indirect-statement verbal expression ALSO has
          relatedtoken1 -> the id of the verb that governs the indirect
          statement (the verb of saying/thinking/perceiving it depends
          on), relationship1 = 'indirect statement' -- matching its own
          syntactic type, the same convention 'direct quote' and 'aside'
          verbal expressions use below. There's no separate
          subordinating-word token to point at first, so the infinitive or
          participle points directly at its governing verb, rather than
          via a conjunction/pronoun intermediary the way a dependent
          finite verb's 'unit verb' relation does. Examples: in
          'ἔφασκε τὸν λύχνον ἀποσβεσθῆναι', ἀποσβεσθῆναι has relatedtoken1
          -> ἔφασκε, relationship1 'indirect statement'; in 'εἶδε δὲ τὴν
          βασίλειαν φεύγουσαν', φεύγουσαν has relatedtoken1 -> εἶδε,
          relationship1 'indirect statement'. In a compound perfect-system
          form, this relation belongs on the conjugated form of εἰμί that
          anchors the verbal expression, same as any other relation into
          it.
        - complementary infinitive: an infinitive that completes the sense
          of a governing verb like βούλομαι, δεῖ, or ἐθέλω (rather than
          reporting indirect speech) has relatedtoken1 -> the id of that
          governing verb, relationship1 = 'complementary infinitive'.
          Unlike an indirect-statement infinitive, this does NOT make the
          infinitive its own verbal expression -- it gets no `verbalunits`
          entry of its own; the governing verb is still the only verbal
          expression here. Example: in 'ἔξεστι ἑλέσθαι', ἑλέσθαι has
          relatedtoken1 -> ἔξεστι, relationship1 'complementary
          infinitive'.
        - modal particle: the particle ἄν has relatedtoken1 -> the id of
          the verb of ITS OWN verbal unit (not some other unit's verb),
          relationship1 = 'modal particle'. Example: in 'εἰ τὴν αὐτὴν
          γνώμην περὶ τῶν ἄλλων ἔχοιτε, οὐκ ἂν εἴη, ὅστις οὐκ ἐπὶ τοῖς
          γεγενημένοις ἀγανακτοίη' (two dependent verbal expressions plus
          one independent verbal expression anchored to εἴη), ἂν has
          relatedtoken1 -> εἴη, relationship1 'modal particle' -- εἴη is
          ἂν's own verbal unit's verb, the same verb οὐκ (adverbial) also
          relates to.
        - infinitive used as a noun: an infinitive can also function as an
          ordinary noun -- most often a verb's subject or object -- rather
          than anchoring an indirect statement or completing another verb.
          Treat it exactly like any other noun in that role: relatedtoken1
          -> the verb it's the subject/object of, relationship1 =
          'subject' or 'direct object' as appropriate (no dedicated label,
          and again no `verbalunits` entry of its own). If the infinitive
          carries a definite article (an articular infinitive, e.g. τὸ
          ζῆν), that article relates to the infinitive exactly as it would
          to a substantivized adjective or adverb: relatedtoken1 -> the
          infinitive's id, relationship1 'article'. Like any verbal form,
          an infinitive used this way can still take its own object or
          adverb, related to it the same way they'd relate to a finite
          verb.
          # TODO: syntax_model.md does not discuss this construction at
          # all; it is carried over from arsgrammatica's equivalent
          # section by direct analogy, since substantival infinitives
          # (often articular) are common in Greek too. No example here is
          # quoted from syntax_model.md.
        - sentence connector: true asyndeton is rare at the root level of
          a sentence -- there is normally a connecting word expressing the
          relation of the sentence to its predecessor. This connecting
          word has relatedtoken1 -> the verb of THIS sentence (not the
          previous one), relationship1 = 'sentence connector'. Example: in
          'ταύτην γὰρ ἐμαυτῷ μόνην ἡγοῦμαι σωτηρίαν', γάρ has relatedtoken1
          -> ἡγοῦμαι, relationship1 'sentence connector'. The particle μέν
          begins a list of items, continued by δέ -- ordinarily WITHIN one
          sentence (see 'connecting word' below), but when the items are
          instead split across distinct, separately terminated sentences,
          μέν or δέ is a 'sentence connector' too, exactly like γάρ, rather
          than a 'connecting word': it has relatedtoken1 -> the verb of
          ITS OWN sentence, relationship1 'sentence connector', with no
          relation at all to the other sentence's verb (a sentence
          connector never records a cross-sentence link -- only "this
          sentence's own verb"). Examples: in the complete, terminated
          sentence 'περὶ μὲν οὖν τοῦ μεγέθους τῆς ζημίας ἅπαντας ὑμᾶς
          νομίζω τὴν αὐτὴν διάνοιαν ἔχειν, καὶ οὐδένα οὕτως ὀλιγώρως
          διακεῖσθαι, ὅστις οἴεται δεῖν συγγνώμης τυγχάνειν ἢ μικρᾶς
          ζημίας ἀξίους ἡγεῖται τοὺς τῶν τοιούτων ἔργων αἰτίους.', μέν has
          relatedtoken1 -> νομίζω, relationship1 'sentence connector'; in
          the following, separate sentence 'ἡγοῦμαι δέ, ὦ ἄνδρες, τοῦτό με
          δεῖν ἐπιδεῖξαι.', δέ has relatedtoken1 -> ἡγοῦμαι, relationship1
          'sentence connector'.
        - connecting word: when a connecting word (coordinating conjunction
          or connecting particle, e.g. καί, ἀλλά, τε, μέν, δέ, οὔτε) joins
          a pair or series of nouns, adjectives, adverbs, or whole clauses
          WITHIN a sentence (as opposed to linking one sentence to the
          previous one -- see 'sentence connector' above), it uses
          relationship1 = 'connecting word', in one of three shapes:
            - a SINGLE connecting word joining a pair: relatedtoken1 -> the
              id of the FIRST connected item, relatedtoken2 -> the id of
              the SECOND connected item, relationship2 = 'connecting word'
              too (both fields on the one connecting-word token). Examples:
              in 'ἐπιτηρῶν γὰρ τὴν θεράπαιναν τὴν εἰς τὴν ἀγορὰν
              βαδίζουσαν καὶ λόγους προσφέρων ἀπώλεσεν αὐτήν', καί joins
              the participles ἐπιτηρῶν and προσφέρων: relatedtoken1 ->
              ἐπιτηρῶν, relatedtoken2 -> προσφέρων. In 'ἐγὼ τοίνυν ἐξ
              ἀρχῆς ὑμῖν ἅπαντα ἐπιδείξω τὰ ἐμαυτοῦ πράγματα, οὐδὲν
              παραλείπων, ἀλλὰ λέγων τἀληθῆ', ἀλλά joins the participles
              παραλείπων and λέγων the same way. In 'οἰκίδιον ἔστι μοι
              διπλοῦν, ἴσα ἔχον τὰ ἄνω τοῖς κάτω κατὰ τὴν γυναικωνῖτιν καὶ
              κατὰ τὴν ἀνδρωνῖτιν', καί joins the two prepositional
              phrases' own prepositions (the first and second κατά).
            - a PAIRED correlative (e.g. postpositive τε...καί, or a
              repeated καὶ...καί, or -- within a single sentence -- μέν
              continued by δέ): each of the two connecting words has
              relatedtoken1 -> ITS OWN adjacent connected item (not "the
              first item" generically -- whichever item that particular
              connector itself sits next to), and relatedtoken2 -> the id
              of the OTHER connecting word (not another connected item).
              Example: in 'ἐφύλαττόν τε καὶ προσεῖχον τὸν νοῦν', τε and
              καί join the verbal expressions ἐφύλαττον and προσεῖχον: τε
              has relatedtoken1 -> ἐφύλαττον, relatedtoken2 -> καί
              (relationship2 'connecting word' too); καί has relatedtoken1
              -> προσεῖχον, relatedtoken2 -> τε. Example (repeated καί):
              in 'περὶ τούτου γὰρ μόνου τοῦ ἀδικήματος καὶ ἐν δημοκρατίᾳ
              καὶ ὀλιγαρχίᾳ ἡ αὐτὴ τιμωρία τοῖς ἀσθενεστάτοις πρὸς τοὺς τὰ
              μέγιστα δυναμένους ἀποδέδοται', the first καί has
              relatedtoken1 -> δημοκρατίᾳ, relatedtoken2 -> the second
              καί; the second καί has relatedtoken1 -> ὀλιγαρχίᾳ,
              relatedtoken2 -> the first καί. Example (μέν...δέ within one
              sentence, not split across sentences -- contrast 'sentence
              connector' above): in 'ἐγὼ μὲν ἄνω διῃτώμην, αἱ δὲ γυναῖκες
              κάτω', μέν has relatedtoken1 -> διῃτώμην (its own,
              first clause's verb), relatedtoken2 -> δέ; δέ has
              relatedtoken1 -> the second clause's own verb (here the
              implied-repetition token standing in for the elided verb --
              see 'implied repetition' below), relatedtoken2 -> μέν.
            - a SERIES of 3 or more connected items: every connecting word
              still has relatedtoken1 -> ITS OWN adjacent item, same as the
              paired case; relatedtoken2 chains the series together --
              the FIRST connecting word's relatedtoken2 -> the SECOND
              connecting word's id (forward), and every LATER connecting
              word's relatedtoken2 -> the id of the connecting word
              immediately BEFORE it (backward) -- so the whole series is
              still traceable by following relatedtoken2 links between the
              connecting-word tokens, even though no single token names
              every member. Example: in 'οὔτε γὰρ συκοφαντῶν γραφάς με
              ἐγράψατο, οὔτε ἐκβάλλειν ἐκ τῆς πόλεως ἐπεχείρησεν, οὔτε
              ἰδίας δίκας ἐδικάζετο.', the first οὔτε has relatedtoken1 ->
              ἐγράψατο, relatedtoken2 -> the second οὔτε; the second οὔτε
              has relatedtoken1 -> ἐπεχείρησεν, relatedtoken2 -> the first
              οὔτε; the third οὔτε has relatedtoken1 -> ἐδικάζετο,
              relatedtoken2 -> the second οὔτε. The same pattern can occur
              with μέν starting a series and δέ continuing it.
          # TODO: syntax_model.md does not discuss καί's double life as
          # connective ("and") vs. adverb ("also"/"even"), unlike Latin's
          # explicit treatment of 'et'. By analogy: when καί modifies a
          # single word rather than joining two, treat it like any other
          # adverb instead -- relatedtoken1 -> the verb (or nearest token)
          # it emphasizes, relationship1 'adverbial', not 'connecting
          # word'. This guidance is an extrapolation, not sanctioned by a
          # syntax_model.md example.
        - direct quote / aside: a verbal expression of syntactic type
          'direct quote' or 'aside' has relatedtoken1 -> the id of the verb
          of the clause it interrupts or is framed by, relationship1 =
          'direct quote' or 'aside' respectively (matching its syntactic
          type). Examples above under (1).
        - circumstantial participle / genitive absolute: a circumstantial
          participial verbal expression's own relatedtoken1 -> the id of
          the noun or pronoun it agrees with, relationship1 =
          'circumstantial participle'. That noun in turn: if it also fits
          a normal role in the surrounding clause (e.g. it's already the
          main verb's subject), it takes THAT normal relation instead
          (nothing extra to add). Example: in 'ἐγὼ ἅπαντα ἐπιδείξω τὰ
          ἐμαυτοῦ πράγματα, οὐδὲν παραλείπων, ἀλλὰ λέγων τἀληθῆ', both
          παραλείπων and λέγων have relatedtoken1 -> ἐγώ, relationship1
          'circumstantial participle', and ἐγώ (already the subject of
          ἐπιδείξω) has relatedtoken1 -> ἐπιδείξω, relationship1
          'subject' -- nothing further added for the participles. If the
          noun is a GENITIVE with no other syntactic connection to the
          sentence (a genitive absolute), it instead has relatedtoken1 ->
          the id of the main verb, relationship1 = 'genitive absolute'.
          Example: in 'προϊόντος δὲ τοῦ χρόνου ἧκον μὲν ἀπροσδοκήτως ἐξ
          ἀγροῦ', προϊόντος has relatedtoken1 -> χρόνου, relationship1
          'circumstantial participle', and χρόνου in turn has
          relatedtoken1 -> ἧκον, relationship1 'genitive absolute'.
        - attributive participle: an attributive participial verbal
          expression's own relatedtoken1 -> the id of the noun or pronoun
          it agrees with, relationship1 = 'attributive participle'.
          Example: in 'ὁ γὰρ ἀνὴρ ὁ ὑβρίζων εἰς σὲ ἐχθρὸς ὢν ἡμῖν
          τυγχάνει', ὑβρίζων has relatedtoken1 -> ἀνήρ, relationship1
          'attributive participle'.
          # TODO: models.py's RelationLabel comment block also glosses
          # "attributive" itself as covering "participle-in-attributive-
          # position", alongside the dedicated "attributive participle"
          # value documented above and in syntax_model.md's own worked
          # example. Since Greek (unlike Latin) makes EVERY attributive
          # participle its own verbal expression, this port resolves the
          # overlap by always using 'attributive participle' for an
          # attributive participle's relation to its noun, and reserving
          # plain 'attributive' for ordinary adjectives and prepositional
          # phrases (see below) -- flagged here as an unresolved tension
          # in the fixed contract, not something syntax_model.md itself
          # disambiguates.
        - auxiliary: in a compound perfect-system verb form (participle +
          a conjugated form of εἰμί), the form of εἰμί anchors the verbal
          expression and is the target of every relation into it (subject,
          direct object, agent, etc); the participle itself has
          relatedtoken1 -> the id of that form of εἰμί, relationship1 =
          'auxiliary'. Example: in 'ὁ νόμος γεγραμμένος ἐστίν', γεγραμμένος
          has relatedtoken1 -> ἐστίν, relationship1 'auxiliary'.
        - agent: the preposition ὑπό introducing the agent of a passive
          verb has relatedtoken1 -> the passive verb's id (the id of the
          form of εἰμί, for a compound form), relationship1 = 'agent'. The
          noun/pronoun governed by that ὑπό has relatedtoken1 -> the id of
          ὑπό, relationship1 = 'object of preposition'. Example: in 'ἡ ἐμὴ
          γυνὴ ὑπὸ τούτου τοῦ ἀνθρώπου διαφθείρεται', ὑπό has relatedtoken1
          -> διαφθείρεται, relationship1 'agent', and ἀνθρώπου has
          relatedtoken1 -> ὑπό, relationship1 'object of preposition'.
        - subject / direct object / predicate: a noun or pronoun serving
          as subject or direct object has relatedtoken1 -> the id of the
          verb (the id of the form of εἰμί, for a compound form),
          relationship1 = 'subject' or 'direct object'. This applies to
          the accusative subject of an infinitive in indirect statement
          too. Examples: in 'ἐμοίχευεν Ἐρατοσθένης τὴν γυναῖκα τὴν ἐμήν',
          Ἐρατοσθένης has relatedtoken1 -> ἐμοίχευεν, relationship1
          'subject', and γυναῖκα has relatedtoken1 -> ἐμοίχευεν,
          relationship1 'direct object'. A noun, pronoun, or adjective
          serving as the predicate complement of a LINKING verb uses
          relationship1 = 'predicate' instead, same relatedtoken1 target.
          Example: in 'ᾤμην τὴν ἐμαυτοῦ γυναῖκα πασῶν σωφρονεστάτην εἶναι
          τῶν ἐν τῇ πόλει', εἶναι anchors an 'indirect statement'/'linking
          verb' verbal expression governed by ᾤμην; γυναῖκα is its subject
          (relatedtoken1 -> εἶναι, relationship1 'subject') and
          σωφρονεστάτην is its predicate adjective (relatedtoken1 ->
          εἶναι, relationship1 'predicate'). If the token is a relative
          pronoun already using relatedtoken1/relationship1 for its
          antecedent link, put this relation in relatedtoken2/
          relationship2 instead (see 'relative pronoun' above).
        - article: the definite article relates to the noun (or
          substantivized adjective, adverb, or infinitive) it accompanies:
          relatedtoken1 -> that word's id, relationship1 = 'article'.
          Example: in 'τῷ χρόνῳ πεισθείη', τῷ has relatedtoken1 -> χρόνῳ,
          relationship1 'article'. When an adjective is in attributive
          position with a REPEATED article (article-noun-article-
          adjective), the second article instead has relatedtoken1 -> the
          id of that adjective, relationship1 = 'article' -- and the
          adjective itself still gets the ordinary 'attributive' relation
          to the noun (see below). Example: in 'εἵλου τοιοῦτον ἁμάρτημα
          ἐξαμαρτάνειν εἰς τὴν γυναῖκα τὴν ἐμήν', the first τήν has
          relatedtoken1 -> γυναῖκα, relationship1 'article'; the second
          τήν has relatedtoken1 -> ἐμήν, relationship1 'article'; and ἐμήν
          has relatedtoken1 -> γυναῖκα, relationship1 'attributive'. (If
          the adjective instead stood between article and noun with no
          second article, e.g. 'τὴν ἐμὴν γυναῖκα', the single τήν and ἐμήν
          keep exactly those same relations -- there is simply no second
          article token to add.)
        - attributive: an adjective in attributive position, a participle
          in attributive position AS AN ORDINARY MODIFIER of a noun
          distinct from its own verbal-expression relation (see the
          'attributive participle' TODO above), or a prepositional phrase
          modifying a noun, has relatedtoken1 -> the noun's id,
          relationship1 = 'attributive'. Example (adjective): ἐμήν in the
          εἵλου example above. Example (prepositional phrase):
          # TODO: syntax_model.md's own worked example here ("attributive
          # to a noun: in the phrase pugna ad Cannas...") is still the
          # untranslated Latin example, apparently left over when the
          # document was adapted for Greek. Substituting a constructed
          # Greek example instead: in 'ἡ μάχη ἡ ἐν Μαραθῶνι', ἐν has
          # relatedtoken1 -> μάχη, relationship1 'attributive', and
          # Μαραθῶνι has relatedtoken1 -> ἐν, relationship1 'object of
          # preposition'. An adjective used as a substantive (standing in
          # for a noun) is treated as a noun/pronoun instead, not as
          # attributive. Example: in 'ἐκείνη μὲν ἀπηλλάγη', ἐκείνη has
          # relatedtoken1 -> ἀπηλλάγη, relationship1 'subject' (not
          # 'attributive' or 'demonstrative' -- it stands for a noun
          # here, it does not modify one).
        - demonstrative: a demonstrative pronoun modifying a noun -- unlike
          an ordinary adjective, NOT in attributive position -- has
          relatedtoken1 -> the noun's id, relationship1 = 'demonstrative'.
          Example: in 'ταύτην ἔλαβον τὴν δίκην', ταύτην has relatedtoken1
          -> δίκην, relationship1 'demonstrative'.
        - adverbial (bare adverb): an adverb modifying a verb has
          relatedtoken1 -> the verb's id, relationship1 = 'adverbial'.
          Example: in 'διαρρήδην εἴρηται', διαρρήδην has relatedtoken1 ->
          εἴρηται, relationship1 'adverbial'. An adverb can also stand in
          attributive position modifying a noun -- same relationship1
          value either way, just a noun instead of a verb on the other
          end. Example: in 'δᾷδας λαβόντες ἐκ τοῦ ἐγγύτατα καπηλείου
          εἰσερχόμεθα', ἐγγύτατα has relatedtoken1 -> καπηλείου,
          relationship1 'adverbial'.
        - prepositional phrases: the preposition has relatedtoken1 -> the
          id of the verb (adverbial) or noun (attributive) it modifies,
          relationship1 = 'adverbial' or 'attributive'. The noun/pronoun
          it governs has relatedtoken1 -> the id of the preposition,
          relationship1 = 'object of preposition' (or relatedtoken2/
          relationship2 if relatedtoken1 is already used for a
          relative-pronoun link). Example: in 'γυναῖκα ἠγαγόμην εἰς τὴν
          οἰκίαν', εἰς has relatedtoken1 -> ἠγαγόμην, relationship1
          'adverbial', and οἰκίαν has relatedtoken1 -> εἰς, relationship1
          'object of preposition'.
        - genitive: a noun or pronoun in the genitive that modifies
          ANOTHER NOUN -- and isn't already covered by a more specific
          relation above (object of preposition, genitive absolute, etc)
          -- has relatedtoken1 -> the id of that noun, relationship1 =
          'genitive'. This is purely a syntactic (case-function) label,
          not a semantic one -- don't distinguish e.g. possessive vs.
          partitive genitive. Example: in 'ᾤχετο εἰς τὸ ἱερὸν μετὰ τῆς
          μητρὸς τῆς ἐκείνου', ἐκείνου has relatedtoken1 -> μητρός,
          relationship1 'genitive'.
        - dative / accusative: a noun in the dative or accusative case
          that depends on a verb or another noun -- and isn't already
          covered by a more specific relation above (subject, direct
          object, object of preposition, etc) -- has relatedtoken1 -> the
          id of the verb or noun it depends on, relationship1 = the
          matching case name ('dative' or 'accusative'). Example (dative,
          linked to a verb): in 'οὔτε ἔχθρα ἐμοὶ καὶ ἐκείνῳ οὐδεμία ἦν
          πλὴν ταύτης, οὔτε χρημάτων ἕνεκα ἔπραξα ταῦτα', both ἐμοί and
          ἐκείνῳ have relatedtoken1 -> ἦν, relationship1 'dative'. Example
          (accusative of extent of time, linked to a verb): in 'ταῦτα
          πολὺν χρόνον οὕτως ἐγίγνετο', χρόνον has relatedtoken1 ->
          ἐγίγνετο, relationship1 'accusative'. Note that Greek has NO
          ablative case and NO dedicated relation label for one -- a
          Latin-scheme 'ablative' relation simply does not arise here.
        - vocative: a noun in the vocative case (direct address) has
          relatedtoken1 -> the id of the verb of the clause it's addressed
          within, relationship1 = 'vocative'. Unlike 'genitive'/'dative'/
          'accusative' above, a vocative relates to a verb only, never to
          another noun. Example: in 'ἐγὼ μὲν οὖν, ὦ ἄνδρες, οὐκ ἰδίαν ὑπὲρ
          ἐμαυτοῦ νομίζω ταύτην γενέσθαι τὴν τιμωρίαν', ἄνδρες has
          relatedtoken1 -> νομίζω, relationship1 'vocative'.
        - apposition: when one noun stands in apposition to another, the
          appositive has relatedtoken1 -> the id of the first (the noun it
          restates or further identifies), relationship1 = 'apposition'. A
          genitive depending on either noun still gets its own ordinary
          'genitive' relation, pointing at whichever noun it actually
          depends on -- apposition doesn't change that.
          # TODO: syntax_model.md gives only the general definition here,
          # no worked Greek example. Constructed illustration: in
          # 'Δημοσθένης ὁ ῥήτωρ ἦλθεν', ῥήτωρ has relatedtoken1 ->
          # Δημοσθένης, relationship1 'apposition' (and ὁ has
          # relatedtoken1 -> ῥήτωρ, relationship1 'article').
        - exclamation: an exclamatory word has relatedtoken1 -> the id of
          the verb of its own verbal unit, relationship1 = 'exclamation' --
          EXCEPT the frequent exclamatory particle ὦ introducing a
          vocative, which instead has relatedtoken1 -> the id of the
          vocative noun/pronoun it introduces (not the verb directly).
          Example: in 'ἐγὼ μὲν οὖν, ὦ ἄνδρες, οὐκ ἰδίαν ὑπὲρ ἐμαυτοῦ
          νομίζω ταύτην γενέσθαι τὴν τιμωρίαν, ἀλλ' ὑπὲρ τῆς πόλεως
          ἁπάσης', ὦ has relatedtoken1 -> ἄνδρες (the vocative it
          introduces), relationship1 'exclamation' -- NOT relatedtoken1 ->
          νομίζω directly, even though ἄνδρες's own relatedtoken1 does
          point to νομίζω (relationship1 'vocative'). This same pattern
          applies wherever ὦ introduces a vocative elsewhere in a passage,
          e.g. 'πρῶτον μὲν οὖν, ὦ ἄνδρες, ...': ὦ -> ἄνδρες, 'exclamation'.

        Only assign relations described above. Leave relatedtoken/
        relationship fields unset for tokens with no relation of these
        kinds -- not every token will have one (e.g. a bare accusative of
        respect not covered above). Use
        only the token ids given in the input `tokens` list, the sentinel
        'root', or a NEW id you create for an implied token (see below), in
        your output; never invent an id for anything else.

    (3) implied/elided tokens. `grammatike` recognizes two DIFFERENT
        situations where a verbal expression exists grammatically but has
        no surface realization in the passage at all -- rather than skip
        these, add a NEW entry to `tokengraph` (and a matching new entry to
        `verbalunits`, since an implied token always anchors its own
        verbal expression) with: a brand-new id, not used by any entry in
        `tokens` or elsewhere in your own output (see the naming rule
        below); the matching tokentype below; and no `token` value (leave
        it unset/None) -- these go together, and 'implied eimi' /
        'implied repetition' are the ONLY two tokentype values whose id
        isn't one of `tokens`' own ids and whose `token` is empty.

        - tokentype 'implied eimi': an elided form of εἰμί ('to be') in a
          predicate expression. The documented case is an implied
          INFINITIVE of εἰμί inside indirect statement: the implied token
          anchors a verbal expression classified 'indirect statement' and
          'linking verb', relates to its governing verb of thinking/saying
          via relatedtoken1/relationship1 = 'indirect statement' exactly
          as a written-out infinitive would, and the subject/predicate of
          the predication relate to it as 'subject'/'predicate' exactly as
          they would to any linking verb. Example: 'ταύτην τὴν ὕβριν
          ἅπαντες ἄνθρωποι δεινοτάτην ἡγοῦνται' has an independent verbal
          expression ἡγοῦνται governing an implied infinitive of εἰμί
          (syntactic type 'indirect statement', semantic type 'linking
          verb') whose relatedtoken1 -> ἡγοῦνται, relationship1 'indirect
          statement'; 'ταύτην τὴν ὕβριν' relates to it as 'subject' and
          δεινοτάτην as 'predicate'.
          # TODO: the two sub-cases below extrapolate from that one
          # documented example and from the general phrasing "elided εἰμί
          # in predicate expressions" -- syntax_model.md gives no worked
          # example for either:
            - a bare predicate construction with NO governing verb at all
              (subject + predicate noun/adjective, nothing else): the
              implied token anchors a verbal expression classified
              'independent' (or 'dependent', if the elided-εἰμί clause is
              itself subordinate) and 'linking verb'; subject and
              predicate relate to it exactly as they would to any linking
              verb.
            - an omitted conjugated εἰμί in a compound perfect-system form
              (the participle left standing alone for its auxiliary): the
              implied token stands in for the omitted form of εἰμί --
              everything that would normally relate to that auxiliary
              (subject, the participle's own 'auxiliary' relation, etc.)
              relates to the implied token instead, exactly as if the
              auxiliary had been written out.
        - tokentype 'implied repetition': a verb elided from a later
          verbal expression in a coordinated series because it repeats the
          verb of an earlier one. Add ONE implied token per omitted
          repeated verb, repeating that verb's OWN syntactic_type and
          semantic_type exactly (whatever those happen to be in context --
          not necessarily 'independent'/'intransitive'), and give whatever
          would relate to the omitted verb (subject, adverbial, a
          connecting word, etc.) its normal relation into the implied
          token instead, exactly as if the verb had been repeated.
          Example: 'ἐγὼ μὲν ἄνω διῃτώμην, αἱ δὲ γυναῖκες κάτω' has an
          explicit verbal expression διῃτώμην ('independent'/
          'intransitive') with subject ἐγώ, and a second, implied verbal
          expression (tokentype 'implied repetition') repeating
          διῃτώμην's own 'independent'/'intransitive' classification, with
          subject γυναῖκες and adverbial κάτω relating to the implied
          token instead of to διῃτώμην.

        Naming an implied token's id (both tokentypes): append '_implied'
        to the id of the LAST real token in `tokens` that precedes where
        the elided word would have stood (or, if the elided word would
        come before every real token in the sentence, the FIRST real
        token's id instead). If more than one implied token is ever
        needed at the same position, append '2', '3', ... after '_implied'
        to keep them unique (e.g. 't5_implied', 't5_implied2'). Place the
        new `tokengraph` entry at the list position where the elided word
        would have appeared, among the tokens of its own clause -- this
        keeps it grouped with the rest of its verbal expression for
        anything that reads `tokengraph` in order.
    """

    passage: str = dspy.InputField(desc="The Ancient Greek passage to analyze, exactly as written.")
    tokens: List[Token] = dspy.InputField(
        desc="Pre-segmented tokens of the passage, in order, with fixed ids. Reference these ids in your output; do not create new ones."
    )
    verbalunits: List[VerbalExpression] = dspy.OutputField(
        desc="One entry per verbal expression (finite verb; infinitive or participle used in indirect speech; attributive participle; or circumstantial participle) in the passage."
    )
    tokengraph: List[TokenAnalysis] = dspy.OutputField(
        desc=(
            "One entry per token in `tokens`, in the same order, with its "
            "type and any relations -- PLUS one additional entry for each "
            "implied/elided token you add (see this signature's docstring), "
            "positioned where that token's clause falls in reading order."
        )
    )


analyze = dspy.ChainOfThought(SyntaxAnalysis)


# ---------------------------------------------------------------------------
# Runner + validation
# ---------------------------------------------------------------------------

def validate(tokens: List[Token], result) -> List[str]:
    """Check that every id the LM produced actually exists among `tokens`
    -- OR is a legitimately new implied token (tokentype in
    IMPLIED_TOKENTYPES -- 'implied eimi' or 'implied repetition'; see
    SyntaxAnalysis's docstring) -- and that implied tokens themselves are
    well-formed. Returns a list of human-readable problem descriptions
    (empty if clean).

    'root' is a special sentinel value for an independent verb's own
    relatedtoken1 (see SyntaxAnalysis's docstring) -- it is never treated as
    an unknown id, but syntax_model.md also requires that no actual token
    ever be assigned the id 'root', so that's checked here too.

    Implied tokens get their own, narrower checks: a tokengraph entry
    claiming an IMPLIED_TOKENTYPES value must use a genuinely NEW id (not
    one already in `tokens`) and must leave `token` unset (None) -- getting
    either wrong is exactly the kind of malformed output this function
    exists to catch, not a legitimate implied token. A non-implied entry,
    conversely, must use one of `tokens`' own ids and must NOT have
    `token=None` -- only 'implied eimi'/'implied repetition' may omit real
    surface text."""
    valid_ids = {t.id for t in tokens}
    problems = []

    if "root" in valid_ids:
        problems.append(
            "token id 'root' is reserved as the sentinel relatedtoken1 "
            "value for independent verbs and must not be assigned to an "
            "actual token"
        )

    implied_ids = {tok.id for tok in result.tokengraph if tok.tokentype in IMPLIED_TOKENTYPES}
    known_ids = valid_ids | implied_ids

    for tok in result.tokengraph:
        if tok.tokentype in IMPLIED_TOKENTYPES:
            if tok.id in valid_ids:
                problems.append(
                    f"tokengraph entry {tok.id!r} is tokentype={tok.tokentype!r} but "
                    "reuses an id already in the input `tokens` list -- an "
                    "implied token must use a new id"
                )
            if tok.token is not None:
                problems.append(
                    f"tokengraph entry {tok.id!r} is tokentype={tok.tokentype!r} but "
                    f"has a non-None token value {tok.token!r} -- an implied "
                    "token's text must be left unset"
                )
        else:
            if tok.id not in valid_ids:
                problems.append(f"tokengraph entry has unknown id {tok.id!r}")
            if tok.token is None:
                problems.append(
                    f"tokengraph entry {tok.id!r} has token=None but "
                    f"tokentype={tok.tokentype!r} -- only 'implied eimi'/"
                    "'implied repetition' may omit surface text"
                )
        for field in ("relatedtoken1", "relatedtoken2"):
            val = getattr(tok, field)
            if val is not None and val != "root" and val not in known_ids:
                problems.append(f"token {tok.id!r} {field}={val!r} is not a known token id")

    for vu in result.verbalunits:
        if vu.id not in known_ids:
            problems.append(f"verbal expression id {vu.id!r} is not a known token id")

    return problems


def print_analysis(tokens: List[Token], result):
    print("Tokens:")
    for t in tokens:
        print(f"  {t.id:>4}  {t.text}")

    print("\nVerbal expressions:")
    for vu in result.verbalunits:
        print(f"  id={vu.id}  syntactic_type={vu.syntactic_type}  semantic_type={vu.semantic_type}")

    print("\nToken graph:")
    for tok in result.tokengraph:
        rels = []
        if tok.relationship1:
            rels.append(f"{tok.relationship1} -> {tok.relatedtoken1}")
        if tok.relationship2:
            rels.append(f"{tok.relationship2} -> {tok.relatedtoken2}")
        rel_str = "; ".join(rels) if rels else "-"
        vu_str = f" [verbal unit {tok.verbalunitid}]" if tok.verbalunitid else ""
        token_str = tok.token if tok.token is not None else f"({tok.tokentype})"
        print(f"  {tok.id:>4}  {token_str:<15} type={tok.tokentype:<11} lemma={tok.lemma or '-':<15} {rel_str}{vu_str}")


if __name__ == "__main__":
    # Quick smoke test against the configured LM, mirroring arsgrammatica's
    # latin_syntax_dspy.py __main__ block. Requires dspy to already be
    # configured with a language model (dspy.configure(lm=...)) by whatever
    # environment runs this file -- this module itself takes no position on
    # which LM or settings to use.
    from .models import Token as _Token

    demo_tokens = [
        _Token(id="t1", text="τὴν"),
        _Token(id="t2", text="θύραν"),
        _Token(id="t3", text="ἀνέῳξεν"),
        _Token(id="t4", text="."),
    ]
    demo_passage = "τὴν θύραν ἀνέῳξεν."
    demo_result = analyze(passage=demo_passage, tokens=demo_tokens)
    print_analysis(demo_tokens, demo_result)
    problems = validate(demo_tokens, demo_result)
    if problems:
        print("\nValidation problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nValidation: OK")
