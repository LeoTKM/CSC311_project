# v2 plan: operations on the four short-answer questions

Scope: this file records only what will be done to the free-text answers Q8-Q11
(`food`, `colour`, `song`, `feeling`). Structured columns, payment parsing, model and
selection of the forest are out of scope here.

**Data basis.** Every number in this file is measured on `data/splits/train.csv` alone:
606 answers from 202 respondents. The scripts in `analysis/` read that file and nothing
else. The other two splits are excluded from every decision recorded here.

Status: **all four fields decided.**

---

## 1. Tokenisation

One rule set, applied identically when a vocabulary is fitted and when a single new
answer is scored at inference.

Step 0, before anything below: the reply goes through `clean()`, so `n/a`, `na`, `null`
and friends stay empty and the missing-value flags keep their meaning.

| # | Step | Example |
|---|---|---|
| 1 | Unicode NFKC | full-width letters and commas become half-width |
| 2 | delete apostrophes | `don't` becomes `dont` |
| 3 | lowercase | `Coca-Cola` becomes `coca-cola` |
| 4 | every character outside the class `a-z 0-9` becomes a space | `coca-cola` becomes `coca cola` |
| 5 | split on spaces | `coca cola` becomes `coca`, `cola` |
| 6 | drop tokens shorter than 2 characters | `a` dropped |
| 7 | drop tokens containing no letter | `2019` dropped |

The apostrophes deleted in step 2 are the ASCII one and its typographic variants.

The hyphen is a separator: `hot-dog` and `hot dog` produce the same tokens.
On the training split the three policies give the same vocabulary at `min_df` = 3
(119 tokens for food), the same held-out out-of-vocabulary rate (17.5 to 17.6%), and the
same all-unseen rate (2.5%). Only one hyphenated compound survives `min_df` = 3, the
brand `coca-cola` (8 rows); `hot-dogs` has 2 rows and dies anyway. Splitting is chosen so
that an unseen compound degrades into two known words instead of vanishing.

Anonymisation placeholders such as `[this sport]` or `[a competition]` are kept. Step 4
turns them into ordinary tokens (`this`, `sport`, `competition`). They are a
deterministic product of the survey instrument, identical when fitting and when scoring,
so they are consistent features rather than noise. On the training split 8 food replies,
6 colour replies, 42 song replies and 8 feeling replies contain one.

Call: `text.tokens(value, hyphen='space', min_len=2, require_letter=True)`.

---

## 2. Encoding of the open-vocabulary fields: food, song, feeling

- One vocabulary per field, fitted on that field alone.
- Candidate pool: tokens with document frequency >= 3 in the fitting rows. On the
  training split that leaves 119 food tokens, about 115 song tokens and about 150
  feeling tokens.
- Terms ranked by chi-square against the 3-class label, from the 2x3 contingency table.
- The top **K** terms of the field are kept; K is set per field.
- The vocabulary is then frozen.
- A kept term is encoded as binary presence, 1 or 0.
- No idf weighting. No stopword list. No stemming, except the standard form that
  section 3 applies to `feeling`.
- No n-grams: on the song answers every bigram is at most as frequent as its strongest
  single word, so phrases add no row coverage.
- Inside cross-validation the pool, the ranking and K are applied per fold, using that
  fold's training rows only.

Why chi-square rather than document frequency: at the same cap, the terms picked by class
association carry more of it. Mean Cramer's V of the capped set, document frequency first
versus association first, on the training split:

| Field | cap | by document frequency | by association | overlap |
|---|--:|--:|--:|--:|
| food | 50 | 0.104 | 0.119 | 35 / 50 |
| colour | 20 | 0.161 | 0.171 | 16 / 20 |
| song | 50 | 0.178 | 0.203 | 31 / 50 |
| feeling | 50 | 0.080 | 0.102 | 25 / 50 |

Binary presence rather than counts: a reply repeats one of its own tokens in 39 food rows,
12 colour rows, 109 song rows and 43 feeling rows out of 606, so counts differ from
presence almost only where the repetition is a title (`waka waka`).

---

## 3. Standard form, applied to the feeling tokens only

The same emotion is written with different endings (`excited`, `excitement`, `exciting`).
Before the vocabulary is counted, every feeling token is folded onto one standard form by
a rule, so the derivations of one emotion become one column instead of several thin ones.

Applied in this order:

**Step 1, irregular table.** Six groups of words that the rule cannot line up are mapped
directly. Seventeen words, six standard forms.

| Standard form | Words |
|---|---|
| `energetic` | energetic, energized, energize, energy, energetized, energies |
| `anxious` | anxious, anxiety |
| `happy` | happy, happiness |
| `sad` | sad, sadness |
| `tense` | tense, tension |
| `relaxe` | relaxed, relaxing, relaxation, relax |

**Step 2, suffix chain.** Up to three passes. On each pass the first matching suffix is
stripped, longest suffix first, and only if at least 3 characters remain:

```
ations ation ments ment ness dom ions ion ances ance ences ence ities ity ings ing edly ed ly es
```

**Step 3, restore a silent e.** If the remaining string is at least 4 characters, ends in
a consonant, and that consonant is preceded by a vowel that is itself preceded by a
consonant, append `e`.

Worked examples:

| Input | After step 1 | After step 2 | After step 3 |
|---|---|---|---|
| `excited` | - | `excit` | `excite` |
| `excitement` | - | `excit` | `excite` |
| `exciting` | - | `excit` | `excite` |
| `bored` | - | `bor` | `bor` |
| `boring` | - | `bor` | `bor` |
| `confused` | - | `confus` | `confuse` |
| `confusion` | - | `confus` | `confuse` |
| `interested` | - | `interest` | `interest` |
| `interesting` | - | `interest` | `interest` |
| `nervousness` | - | `nervous` | `nervous` |
| `energized` | `energetic` | - | - |
| `relaxation` | `relaxe` | - | - |

Opposites are never merged: `uninterested` carries no stripped suffix of its own, so it
stays a separate token.

The standard form is **not** applied to `food`, `colour` or `song`.

---

## 4. Encoding of the closed-vocabulary field: colour

Colour answers come from a closed concept, so `colour` is not treated as open text.
It is encoded as a fixed multi-hot vector over a colour lexicon.

- The reply is tokenised by section 1, then every token is looked up in the lexicon.
- A column is 1 when any word of its token group appears in the reply, 0 otherwise.
- The lexicon has 17 columns and is **fixed**: it is not fitted, it does not use the
  label, and the column set is identical in every fold and at inference.
- Spelling variants are merged into one column.
- A colour outside the lexicon leaves every column at 0 and is recorded by
  `colour_no_terms`.
- No `min_df`, no chi-square ranking, no K and no standard form apply to this field.

| Column | Tokens that trigger it |
|---|---|
| `blue` | blue, blues |
| `red` | red |
| `orange` | orange |
| `green` | green |
| `white` | white |
| `black` | black, balck |
| `brown` | brown |
| `yellow` | yellow, yellowish |
| `gold` | gold, golden |
| `purple` | purple |
| `navy` | navy |
| `gray` | gray, grey |
| `beige` | beige |
| `maroon` | maroon |
| `burgundy` | burgundy |
| `cream` | cream |
| `tan` | tan |

The 17 columns are kept in full even though several of them are thin. On the training
split 14 of the colours occur, `navy`, `gray`, `beige` and `maroon` on 2 to 3 rows, and
`burgundy`, `cream` and `tan` on at most one row each. The fixed schema is the point: a
colour that exists in the lexicon always has a column to land in, whatever the fitting
rows happened to contain. A curated list of 15 colours covers 96% of the training
replies, which is why this field can be closed at all.

---

## 5. Columns added per field, independent of the vocabulary

| Column | Definition |
|---|---|
| `<field>_non_answer` | 1 if the reply is empty or normalises to a non-answer (`none`, `nothing`, `unsure`, `n/a`, ...) |
| `<field>_no_terms` | 1 if the reply is not a non-answer but none of its tokens is in the fitted vocabulary, or none of its tokens is in the colour lexicon |
| `<field>_item_count` | number of tokens the reply produces, counted before any vocabulary or lexicon lookup |

All three are computed from the raw token list, so a reply built entirely from unseen
words still leaves a trace in `_no_terms` and `_item_count`.

How often a food reply lands on nothing, measured on held-out rows of the training split:

| measure | value |
|---|--:|
| content tokens outside the fitting vocabulary | 15.2% |
| replies containing at least one unseen word | 24.9% |
| replies whose every word is unseen | 2.4% |
| replies with no selected term at K = 30 | 19.0% |

A token that appears in no fitting row is indistinguishable, for the model, from a token
that appears once and is dropped by `min_df`. That is why the three columns above are
computed before the vocabulary lookup.

---

## 6. Unseen words at scoring time

- Open fields: the vocabulary is frozen after fitting. A token outside it contributes 0
  to every text column, and the reply falls into `<field>_no_terms`.
- Colour: a token outside the lexicon contributes 0 to all 17 columns, and the reply
  falls into `colour_no_terms`.

No unknown bucket column, and no vocabulary expansion when scoring.

---

## 7. Per-field settings

| Field | Question | Encoding | Standard form | Term columns | Auxiliary | Total |
|---|---|---|---|--:|--:|--:|
| `food` | What foods and drinks do you associate with watching this sport in the place or culture you know best? | chi-square selection, K = 30, min_df = 3 | no | 30 | 3 | 33 |
| `colour` | What colour do you associate with this sport? | fixed 17-column lexicon multi-hot | at the lexicon | 17 | 3 | 20 |
| `song` | Which song do you associate with this sport? | chi-square selection, K = 30, min_df = 3 | no | 30 | 3 | 33 |
| `feeling` | How do you feel when you watch this sport? | chi-square selection, K = 60, min_df = 3 | yes | 60 | 3 | 63 |
| | | | | **137** | **12** | **149** |

### How K was chosen

Rows needed for a selected-term coverage of 80 / 90 / 95% of the answered training
replies, and mean Cramer's V at ranks 5 / 10 / 20 / 30 / 50:

| Field | terms at df >= 5 | K for 80 / 90 / 95% | V at 5 / 10 / 20 / 30 / 50 |
|---|---|--:|---|
| food | 83 | 20 / 49 / 76 | 0.240 / 0.153 / 0.107 / 0.092 / 0.054 |
| colour | 24 | 8 / 8 / 11 | 0.189 / 0.106 / 0.033 / - / - |
| song | 84 | - / - / - | 0.426 / 0.323 / 0.153 / 0.141 / 0.093 |
| feeling | 101 | 52 / - / - | 0.164 / 0.115 / 0.089 / 0.082 / 0.064 |

`colour` is closed: eight terms already cover 90% of the replies and V collapses after
them, so a curated list replaces selection entirely. `song` and `feeling` never reach 90%
coverage even with the whole pool, because the long tail of titles and sentences shares no
tokens with anything else; their K is set by where the block's own accuracy stops moving,
not by coverage.

Splitting a field into semantic sub-vocabularies was tested and rejected: on the training
split every group scores below its own strongest member.

| Field | group | rows | group V | best single term V |
|---|---|--:|--:|--:|
| food | drinks | 396 (65%) | 0.236 | 0.321 (beer) |
| food | fast food | 387 (64%) | 0.154 | 0.346 (hot) |
| food | snacks | 102 (17%) | 0.151 | 0.245 (peanuts) |
| colour | primary | 285 (47%) | 0.265 | 0.454 (blue) |
| colour | secondary | 223 (37%) | 0.471 | 0.641 (orange) |
| colour | neutral | 151 (25%) | 0.187 | 0.279 (white) |

---

## 8. Food (Q8) - decided

- Question: *What foods and drinks do you associate with watching this sport in the
  place or culture you know best?*
- Tokenisation: section 1, hyphen as separator, no standard form.
- Encoding: section 2, K = 30 terms, binary presence.
- Candidate pool `min_df` = 3, which leaves 119 tokens on the training split.
- No plural folding.
- Auxiliary columns: `food_non_answer`, `food_no_terms`, `food_item_count`.
- Total: 33 columns.

**No stopword list.** On the training split three function words rank inside the top 30
by chi-square:

| term | df | V |
|---|--:|--:|
| `with` | 22 | 0.107 |
| `of` | 12 | 0.101 |
| `this` | 19 | 0.094 |

They are kept. On the training split, removing them lowers the mean V of the selected 30
from 0.1524 to 0.1517, so by this data they are weak but real features rather than slots
wasted on noise. `as` (df 5, V 0.129) enters the same way. The three are driven by how the
reply was phrased rather than by what was named; that is the price of ranking by class
association and taking the top K without a filter.

---

## 9. Colour (Q9) - decided

- Question: *What colour do you associate with this sport?*
- Tokenisation: section 1, hyphen as separator.
- Encoding: section 4, the fixed 17-column lexicon multi-hot vector.
- Binary presence per colour, no counts.
- All 17 columns are always emitted.
- Auxiliary columns: `colour_non_answer`, `colour_no_terms`, `colour_item_count`.
- Total: 20 columns.

The lexicon was compared against chi-square selection on the training split:

| Encoding of the colour answer | columns | forest accuracy | replies landing on nothing |
|---|--:|--:|--:|
| fixed lexicon, binary | 17 | 0.7498 | 4.0% |
| fixed lexicon plus item count | 17 | 0.7503 | 1.5% |
| chi-square top 10, ranked per fold | 10 | 0.7447 | 4.9% |
| lexicon with token counts | 17 | 0.7498 | 4.0% |

The 0.5-point difference is inside the noise, but the lexicon also removes the label from
the block entirely and keeps the column set identical at every fold and at inference.
Counts and presence give the identical matrix here, because no reply names the same
colour twice.

---

## 10. Song (Q10) - decided

- Question: *Which song do you associate with this sport?*
- Tokenisation: section 1, hyphen as separator, no standard form.
- Encoding: section 2, K = 30 terms, binary presence.
- Candidate pool `min_df` = 3, which leaves about 115 tokens on the training split.
- No stopword list. The most discriminative tokens are title fragments: `the`, `to`,
  `me`, `for`, `this` all come from the two dominant titles, so a list would remove the
  signal.
- No n-grams.
- The refusal forms the pipeline already blanks (`n/a`, `na`, `null`, `-`, `[]`) stay as
  they are. The set is **not** widened: blanking `none`, `no idea`, `no particular song`
  and similar replies measured slightly worse than keeping them, because saying "no song"
  is itself class-informative.
- Replies whose title is outside the popular ones are accepted as noise. On the training
  split 574 answered rows contain 325 distinct normalised titles, and the long tail shares
  no tokens with anything else.
- Auxiliary columns: `song_non_answer`, `song_no_terms`, `song_item_count`.
- Total: 33 columns.

The block is weak on its own and flat in K: forest accuracy stays between 0.630 and 0.635
for K from 20 to the whole pool of about 120 terms, so K = 30 is chosen for compactness
rather than accuracy. One title occupies several columns (`take`, `me`, `out`, `to`,
`the`, `ball`, `game`), which is a cost of the token representation and the price of
matching every written variant of the same title.

---

## 11. Feeling (Q11) - decided

- Question: *How do you feel when you watch this sport?*
- Tokenisation: section 1, hyphen as separator, then the standard form of section 3.
- Encoding: section 2, K = 60 terms, binary presence.
- Candidate pool `min_df` = 3, counted after the standard form is applied.
- No stopword list. `and` alone reaches Cramer's V = 0.158, higher than most emotion
  words, because it marks replies that name two emotions.
- No flat-reply flag. Its association collapses to V = 0.052 once replies that say only
  `bored` are excluded.
- Auxiliary columns: `feeling_non_answer`, `feeling_no_terms`, `feeling_item_count`.
  The reply-length column carries no signal here (V about 0.08) but is kept so that all
  four fields have the same three auxiliary columns.
- Total: 63 columns.

The block's ceiling is structural: `excited` (V 0.310) is equally common in Soccer and
Basketball replies, and `bored` (V 0.267) is mostly a Baseball marker, so this field
separates Baseball from the other two rather than all three from each other. The standard
form is what makes the block usable at all: forest accuracy rises from 0.4774 to 0.4927 at
K = 60 once the derivations are folded together.

---

## Pending

- Whether the text block is L2 row-normalised before it reaches the model.
