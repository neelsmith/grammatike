# Overview

This repository hosts a python package leveraging language models with `dspy` to analyze the syntax of passages of ancient Greek. The unique analytical scheme is specific to ancient Greek, and is documented here.

## Principles

1. Syntactic analysis must be consistent with morphological analysis.
2. Analysis should be expressed in syntactic rather than semantic terms.


## Basic model

SVO+.

Citable passages. Canonically citable or scratch pad level (eg, homework).


## Implementation


In this scheme, analysis of a passage of Latin is expressed in two related structures:

- a list of verbal expressions, generally corresponding to clauses in an English translation
- a token-level table capturing principal relations in a dependency graph

Sentences are simply sequences of tokens that segment a text in syntactically coherent units.



### Table of verbal expressions

"Verbal expressions" are subject-verb ideas that most frequently correspond to clauses in an English translation. (Of course in Latin the subject may be implicit where that is not possible in English.) This scheme identifies three possible constructions as verbal expressions: finite verbs, infinitives and participles.

1. *Every finite verb* constitutes a verbal expression. Latin finite verbs include the compound forms of the perfect and pluperfect tenses (composed of a past participle plus a form of *sum*) as well as conjugated verbs forms identifiable by tense-mood-voice-person-number. 
2. *Infinitives* constitute a verbal expression when they are part of an expression in indirect speech.
3. *Participles* constitute a verbal expression when they have a *predicate* sense rather than purely *attributive* sense. 

In this scheme, verbal expressions are classified according to:

1. their *syntactic type*. The possibilities for each construction are:
    - for finite verbs:
        - *independent* (also called "main" or "principal") verbs. These are syntactically independent finite verbs: their clause is syntacitcally coherent by itself.
        - *dependent* ("subordinate" or "secondary") verbs. These are finite verbs that are introduced by a subordinating word (such as subordinating conjunction or relative pronoun). They cannot appear without an explicit or implicit governing (superior) clause. Examples: in the sentence *principes Albanorum in patres, ut ea quoque pars rei publicae cresceret, legit* the verb *legit* is classified as an `independent` verbal expression, and *cresceret* is classified as `dependent` (introduced by the subordinating conjunction *ut*).
        - *direct quote*: used for verbal expressions in directly quoted speech. For example, in the sentence *Tuum est' inquit, 'Servi regnum'*, the token *inquit* is a verbal unit classified syntactically as an `independent` clause, while the verb *est* occurs in directly quoted speech and is classifed as `direct quote`.
        - *aside*: classification for a verbal unit that injects a statement by interrupting the syntactic flow of the surrounding text. Example: in the sentence *equidem — pace dixerim deum—eos nos iam populi Romani beneficio esse spero*, the verb *spero* is classified as `independent`, the infinitive *esse* is part of an indirect statement, and the entire phrase *pace dixerim deum* is an aside, anchored by the finite verbal expression *dixerim* of type `aside`.
    - for infinitives: *indirect statement* when they are part of an expression in indirect speech. Example: in the sentence *facturum enim se fuisse dixit*, the verb *dixit* is an independent verbal expression, and *facturum fuisse* is the compound verb form for the future infinitive. The verbal unit wil be anchored to the infinitive *fuisse* of syntactic type `indirect statement`.
    - for participles: For example, in the sentence *gloria est consentiens laus bonorum*, the participle *consentiens* has an attributive sense with *laus*, "universal praise": this is *not* a verbal expression. But in the sentence *Anco regnante Lucumo, vir inpiger ac divitiis potens, Romam commigravit* the participle *regnante* has a predicate sense with *Anco* "while Ancus was reigning..."  When it is uncertain whether to take the relation of a participle either as attribute or a circumstantial participle **prefer the treatment as circumstantial participle**. For example, in the sentence *ille moriens, cum sciret sagittas hydrae Lernaeae felle tinctas quantam uim haberent ueneni, sanguinem suum exceptum Deianirae dedit*, the participle *moriens* agrees with *ille* and the participle *tinctas* agrees with *sagittas*. Treat both as circumstantial participles definiing a verbal expression.
2. by their *semantic type* ,as *transitive active*, *transitive passive*, *intransitive* or a *linking verb*. In the sentence *principes Albanorum in patres, ut ea quoque pars rei publicae cresceret, legit*m the verb *legit* is *transitive active*; and *cresceret* is *intransitive*. In *urbs a Romulo condita est*, the compound verb *condita est* is *transitive passive*.  In the sentence *Etruria erat vicina*, the verb *erat* is a *linking verb*.


## Token-level table of dependencies

### Tokenization

The textual content of Latin passages with citation references may be analyzed; the analyzing program will keep track of the citation. 

The text of the passage must be tokenized, and each token classified as one of:

-  a *punctuation* token. Any Unicode punctuation character, in editions of Latin texts most commonly including period, comma, question mark, semicolon, colon, parentheses and brackets of various kinds, single or double quotation marks, dashes or hyphens.  Example: "." in the phrase *arma virumque cano*
-  an *enclitic* token. Example: the enclitic *que* in the phrase *arma virumque cano.* Tokenization of enclitics must consider the context. Example: in the phrase, *aequa ratione imperat*, the string *ratione* is a single lexical token (noun in the ablative singular); in the phrase *ratione docet?*, the string *ratione* represents the enclitic token *ne* (question words) with the lexical token *ratio* (noun in the nominative singular). Tokenization must also recognize the small number of frequently occurring words that have incorporated an original historic enclitic into a single lexical item such as *quisque* (and its compounds), or *plerusque*. Example: forms such as *quisque*, *cuique* and *quemque* must all be treated a single lexical token.
-  a *lexical* token. Example: the tokens *arma*, *virum* and *cano* in the phrase *arma virumque cano.*
- a *praenomen*, including its punctuating period. Example: *M.* in the phrase *M. Agrippa L. f. cos. tertium fecit*
- other *abbreviation*, including its period. Example: *f.* and *cos.* in the the phrase *M. Agrippa L. f. cos. tertium fecit*
- a *numeral* written numerically. Example: *XXV* in the phrase *hiberna aberant ab eo milia passuum XXV*




### Syntactic relations among tokens

In the first phase of implementing our syntax model, we will record the following set of relations among tokens.


#### Verbs and their principal construction

- verb of an independent clause: the `relation1` of independent verbs has the special value `root` which must not be used as identifier for any token. Its 'relationship1` value is `unit verb`. Example: in *arma virumque cano*, *cano* is an independent verb with `relation1` value `root`, and `relationship1` value `unit verb`.

- verbs in direct quotes: the `relation1` will be the ID of the verb of the governing verbal expression, with a value of `direct quote` for `relationship1`. Example of direct quote: in *'hanc ego aram' inquit 'Pudicitiae plebeiae dedico'*, the verbal unit anchored to `dedico` is direct speech subordinate to *inquit*. The token *dedico* will therefore have the id of *inquit* for its `relation1`, with `direct quote` as its `relationship1`.

- verbs in asides: the `relation1` will be the ID of the verb of the governing verbal expression, with a value of  `aside` for `relationship1`. Example: in the sentence *Sp. Nautius — Octavium Maecium quidam eum tradunt — cum auxiliaribus cohortibus erat.* the verbal expression anchored to *tradunt* is an aside, interrupting the verbal expression with *erat*. The token *tradunt* will have the ID of *erat* for `relation1` with a value of  `aside` for `relationship1`. 


- infinitives in indirect statement: the 'relation1` will the the ID of the verb of the governing verbal expression, with the value `indirect statement` for `relationship1`. Example: in *id philtrum esse dixit*, the token *esse* will have the ID of *dixit* for its `relation1` with `indirect statement` for `relationship1`. 

- multi-word compound verb forms in the perfect passive of future infinitive: the conjugated form of *sum* will be taken as the verb of the verbal unit. The associated participle will relate to the form of *sum* as its `auxiliary`. Examples: in *urbs condita est* with token ids `t1`, `t2` and `t3`, the participle *condita* has for its `relation1` the value `t3` (*est*), and for `relationship1`, *auxiliary*. In the sentence *facturum enim se fuisse dixit*, the token *facturum* will have the ID of *fuisse* as its `relation1` value, with a `relationship1` value of `auxiliary`.

- verb of a dependent clause: the verb of a dependent clause must be related to a subordinating word, either a subordinating conjunction or a relative pronoun. *relation1* will be the ID of the conjunction of pronoun, and the value of *relationship1* will be *unit verb*.  In the sentence  *Hercules cum gregem perlustrasset, pergit ad proximam speluncam*, the verb *perlustrasset* is releated to the subordinating conjunction *cum* with the value of `unit verb` for `relationship1`. Indirect questions are treated as a type of dependent clause. Example: in *Theseus audit quanta calamitate ciuitas afficeretur.*, the interrogative pronoun *quanta* introduces the dependent clause, anchored to the verb *afficeretur*, so *afficeretur* will have the id of *quanta* for its `relation1` with `unit verb` as the value of `relationship1`.

- agent of passive verbs: if a passive verb includes an expression for agent using *a* or *ab* plus a nominal expression in the ablative, *a* or *ab* should have the passive verb token as *relation1* and *agent* as the value of *relationship1*. The noun or pronoun constructed with *a/ab* should have the id of *a/ab* as its *relation1* and *object of preposition* as its*relationship1* value. Example: if *urbs a Romulo condita est* is tokenized with the IDs `t1`, `t2`...`t5`, then `t2` (*a*) will have a `relation1` of `t5` (*est*), and `relationship1` of `agent`. The token *Romulo* will be related to `t2` as a normal `object of preposition` (see below).


- verbal units with participles: when a verbal unit is anchored to a participle, the participle has as its `relation1` the id of the noun or pronoun it agrees with, with the `relationship1` value `circumstantial participle`. If the governing noun fits syntactically into the superior verbal unit, it takes its construction  as usual (see more below). For example, in the sentence *eum advenientem laeti omnes accepere*, the participle *advenientem* has the id of *eum* for `relation1` and has the `relationship1` value `circumstantial participle`. The token *eum* in turn is the direct object of the independent verb *accepere*. If, however, the noun is an ablative that is otherwise unconnected syntactically to the sentence, it has as `relation1` the ID of the verb, and has for `relationship1` the value `ablative absolute`. Example: in the sentence *Anco regnante Lucumo Romam commigravit*, the participle *regnante* has the ID of *Anco* as its `relation1` with the `relationship1` value `circumstantial participle`. *Anco* in turn has the ID of the verb *commigravit* as its `relation1`, and has the `relationship1` value `ablative absolute`.


- subordinating conjunctions: *relation1* will be the ID of the verb in their governing (superior clause), and the *relationship1* will be *subordinating conjunction*. In the sentence  *Hercules cum gregem perlustrasset, pergit ad proximam speluncam*, the  conjunction *cum* is releated as a subordinating conjunction to the main verb *pergit*.  Here's a partial extract of the relations resulting from this sentence:

| ID | token | relation1 | relationship1 |
| --- | --- | --- | --- |
| cum | t2 | t5 | subordinating conjunction |
| perlustrasset | t4 |  t2 | unit verb |
| pergit | t5 | | |


- relative pronouns: *relation1* will be the ID of its antecedent, and *relationship1* will be *relative pronoun*. Example: here is a partial extract from an analysis of the sentence *Latini, cum quibus ictum foedus erat, sustulerant animos.*


| ID | token | relation1 | relationship1 | relation2 | relationship2 |
| --- | --- | --- | --- | --- | --- |
| Latini | t1 | t9 | subject | | |
| cum | t3 | | | | | 
| quibus | t4 | t1 | relative pronoun | t3 | object of preposition |
| ictum | t5 | t7 | auxiliary | | |
| erat | t7 | t4 | unit verb | | |
| sustulerant | t9 | | | 
| animos | t10 | t9 | direct object | | 


- coordinating conjunctions: when coordinating conjunctions join pairs of adjectives, nouns or prepositional phrases, they use the IDs of the nouns, adjectives or prepositions of the prepositional phrases for `relation1` and `relation2`, and `coordinating conjunction` for both `relationship1` and `relationship2`. Example: in *arma virumque cano*, the conjunction *que* will have the ids of *arma* and *virum* for`relation1` and `relation2` and `coordinating conjunction` for both `relationship1` and `relationship2`. When two verbal expressions are coordinated, the conjunction will use the IDs of the two verbs. Example: the sentence *suo tempore peperit Chrysen iuniorem et dixit se ab Apolline concepisse* has two verbal expressions joined by *et*. The conjunction *et* will have the IDs of *peperit* and *dixit* for`relation1` and `relation2` and `coordinating conjunction` for both `relationship1` and `relationship2`. Note that (unlike when joining nouns, adjectives or prepositional phrase) in such instances, the word order of the conjunction will be dictated by the opening of the new verbal expression, and may not always be close in word order to the verb token itself. Example: in *ille fidem suam infirmare noluit, Hermionenque ab Oreste adduxit*, the two verbal expressions with *noluit* and *adduxit* are joined by *que*, even though the enclitic *que* is physically attached to *Hermionen*, the direct object of *adduxit*. Here, *que* will have the IDs of *noluit* and *adduxit* for`relation1` and `relation2` and `coordinating conjunction` for both `relationship1` and `relationship2`. Sometimes a coordinating conjunction introduces a verbal expression to begin a new sentence; although there may be an implicit connection to a preceding sentence, we only mark the conjunction as related to the explicit verb. Example: in *sed re cognita, iussu Cereris Triptolemo regnum dedit*, the conjunction *sed* introduces the entire verbal expression with *dedit*, but we do not mark any implied relation to a preceding expression. *sed* will have the ID of *dedit* for `relation1`, with `relationship1` as *coordinating conjunction*. A further important detail to note is that the word *et* can be used as a conjunction or adverbially ("even", "also")! Example: in *Tu quoque, Brute, fili mi, et tu?* the token *et* is functioning as an adverb, not a conjunction and will have the last lexical token *tu* as its `relation1`, with the `relationship1` value *adverbial*.






- noun or pronoun serving as the subject of a verbal expression: *relation1* will be the id of the token of the verb. If it is a compound verb form in the perfect passive system, this should be the id of the form of *sum*. The value of *relation1ship* will be *subject*.

- noun or pronoun functioning as direct object of a verbal expression: *relation1* will be the id of the token of the verb. If it is a compound verb form in the perfect passive system, this should be the id of the form of *sum*. The value of *relation1ship* will be *direct object*.

- noun or pronoun functioning as the predicate of a linking verb: *relation1* will be the id of the token of the verb. If it is a compound verb form in the perfect passive system, this should be the id of the form of *sum*. The value of *relationship1* will be *predicate*. Example: In the sentence *Lucumo Demarati Corinthii filius erat*, *Lucumo* is the subject of the linking verb *erat*, and *filius* is the predicate.


- complementary infinitive: with verbs like *volo*, *incipio*, *audeo* or *licet*, *decet* (among others), infinitives essentially complete the idea of the principal verb. In this use, the infinitive token will use the id of the main verb as `relation1` with a value of `complementary infinitive` for `relationship1`. Example: in the sentence *Amphion autem cum templum Apollinis expugnare vellet, ab Apolline sagittis est interfectus.*, the infinitive *expugnare* stands in a complementary relation to *vellet*, and will have the ID of *vellet* as `relation1` with `complementary infinitive` for `relationship1`.

- other infinitive constructions: the infinitive may also function as a noun, and in these constructions is recorded in the same way as any other. Example: in *dolere malum est*, the infinitive *dolere* is the subject of the verb *est*. It will therefore have the id of *est* as its `relation1` and a value of `subject` for `relationship1`. As in all infinitive constructions, the infinitive, like any other verbal form, may have objects; if a subject is explicitly given, it will be in the accusative.

#### Adjectives, adverbs, prepositional phrases


- adjectives: if an adjective is used as a substantive, it is treated as a noun or pronoun. When it describes a noun, it has the noun's id as its `relation1`, and `adjectival` as its `relationship1`. In the sentence *Lucumo superfuit patri bonorum omnium heres*, the adjective *omnium* will have the id of *bonorum* as its `relation1` and the `relationship1` value will be `adjectival`. The token `bonorum` will be treated as a noun (see below).

- adverbs: adverbs have the id of the verb they modify as `relation1`, with `relationship1` value `adverbial`. In the sentence *ad Ianiculum forte ventum erat*, the adverb *forte* will take the id of `erat` for `relation1` with `relationship1` value `adverbial`.  `ventum` will also be related to `erat` but with `relationship1` value `auxiliary`.

- prepositional phrases: prepositional phrases either stand in an adverbial relation to a verbal expression or in an attributive relation to a nominal expression. 
   - attributive to a noun: in the phrase *pugna ad Cannas*, "the battle near Cannae," the prepositional phrase *ad Cannas* is attributive. *ad* will have as *relation1* the id of *pugna*, and its *relationship1* will be *attributive*. (*Cannas* will be the object of the preposition, see below.)
   - adverbial related to a verb: in the sentence *statua Atti in comitio in gradibus ipsis ad laevam curiae fuit*, the three prepositional phrases *in comitio*, *in gradibus ipsis* and *ad laevam curiae* will each each have for `relation1` the id of the verb *fuit*, with the `relationship1` value `adverbial`.


- gerundives: gerundives are adjective forms and can be related to a noun just like any other adjective. Example: in the sentence, *Metapontus exiit ad Dianam Metapontinam ad sacrum faciendum*, the gerundive *faciendum* is in agreement with the noun *sacrum*. It till have the ID of *sacrum* for `relation1` with a value of `adjectival` as its `relationship1`.

Example: in the phrase *ars bene disserendi*, the gerund *disserendi* is a gentive noun related to *ars*. It will have the id of *ars* as `relation1` and `genitive` as the value of `relationship1`. As verbal forms, gerunds can have objects or take adverbs. *bene* in the precediong example is an adverb that will stand in relation to *disserendi* in an `adverbial` relationship, so will have the id of `disserendi` for `relaion1`, and `adverbial` as the value of `relationship1`.


### Noun relations

Traditional grammatical analyses typically conflate syntax and semantics: this model hews narrowly to syntax. In describing the syntactic role of nouns that are not directly tied to verbs as subject, predicate or object, or to prepositions as their objects, we prefer to identify their relationship to other tokens in terms of case function.

- *genitive*: when a noun in the genitive depends on another noun, we cateogorize its relationship type as *genitive* (without semantic distinctions such as "possessive" or "partitive"). Example: in the sentence *hic filius erat regis*, the token *regis* is in the genitive, and relates to *filius*. We use the ID of *filius* as its `relation1`, with `genitive` as its value for `relationship1`.
- *dative*: dative relationships can be linked to verbs or nouns. In the sentence *audeat deinde talia alius, nisi in hunc insigne iam documentum mortalibus dedero*, the dative noun *mortalibus* relates to the verb *dedero*. *mortalibus* will have the id of *dedero* as its `relation1` with a value of `dative` for `relationship1`.
- *ablative*: ablative relationships can be linked to verbs or nouns. In the sentence *omnia ferro flammaque miscet*, the two ablative tokens *ferro* and *flamma* both relate to the verb token *miscet* and will have the ID of *miscet* as their `relation1` with the value `ablative` for `relationship1`.
- apposition: when one noun stands in apposition to another, the appositve takes the id of the first noun as the value for `relation1`, and has `apposition` as the value for `relationship1`. Example: in the sentence *Neptunus et Aegeus Pandionis filius in fano Mineruae cum Aethra Pitthei filia una nocte concubuerunt*, the token *filius* is in apposition to *Aegeus* and takes the ID of *Aegeus* as `relation1` with `apposition` for `relationship1`. The genitive *Pandionis* depends on *filius* and will have the ID of *filius* for `relation1`, with `genitive` as `relationship1`. Similarly, *filia* is in apposition to *Aethra* and will have the id of *Aethra* for `relation1`, with `apposition` as `relationship1`. THe genitie *Pitthei* depends on *filia* and  will have the ID of *filia* for `relation1`, with `genitive` as `relationship1`. 



- gerunds: gerunds are noun forms, used in the oblique cases (where the infinitive could be used in the nominative). They can be related to the syntax of an expression like any other noun. Example: in the phrase *ars bene disserendi*, the gerund *disserendi* is a gentive noun related to *ars*. It will have the id of *ars* as `relation1` and `genitive` as the value of `relationship1`. As verbal forms, gerunds can have objects or take adverbs. *bene* in the precediong example is an adverb that will stand in relation to *disserendi* in an `adverbial` relationship, so will have the id of `disserendi` for `relaion1`, and `adverbial` as the value of `relationship1`.


### TBA

- gerunds and gerundives

