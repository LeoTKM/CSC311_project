# v3: a random forest on 173 columns of the CSC311 survey

The task is to predict which of three sports - Soccer, Baseball, Basketball - a survey
response is about. v3 is a feature-design experiment: it keeps the forest and the
cross-validation protocol of the two earlier models and changes only how the four
free-text answers are turned into columns.

Everything in this directory is fitted and scored on `data/splits/train.csv` alone:

    606 rows, 202 respondents, three rows per respondent

The splits are grouped by respondent, so one person never straddles two folds.

## Which splits are read

The test split is named nowhere in this directory, and `smoke_check.py` asserts that.

The validation split is read by exactly one stage, `validation`, and that stage is not
part of `--stage all`. It fits the 606 training rows and scores the 132 validation rows;
nothing from the validation split ever reaches a vocabulary, an imputer or a tree. It
exists so that v3 has a number standing next to the ones the two earlier models reported
on the same 132 rows and 44 respondents, and it is kept out of `all` because every reading
of the validation split is one that stops being available for later design decisions.

There is no stage that refits on train plus validation.

## The feature matrix

    26  structured   ease, strategy, injury, the 21 multi-hot options,
                     payment_amount (mean filled, log1p), payment_missing
     6  flags        q1_missing, q2_missing, q3_missing, q5_missing, q6_missing,
                     q7_missing
    53  food         50 most frequent terms, food_non_answer, food_no_terms,
                     food_item_count
    20  colour       the 17 fixed lexicon colours, colour_non_answer, colour_no_terms,
                     colour_item_count
    53  song         50 most frequent terms, song_non_answer, song_no_terms,
                     song_item_count
    15  feeling      12 most frequent terms, feeling_non_answer, feeling_no_terms,
                     feeling_item_count
    --
   173  total

Q4 - "Individual brilliance can outweigh teamwork in this sport" - is dropped along with
its missing flag, so the structured block has 32 columns rather than 34. Q4 has eight
blank answers in the training split.

## The four short-answer fields

The answers are tokenised the same way everywhere:

    half-width (NFKC) -> delete apostrophes -> lowercase
    -> every character outside [a-z0-9] becomes a space -> split
    -> drop tokens shorter than 2 -> drop tokens with no letter

so `hot-dog` and `hot dog` agree, `50 cent` keeps `cent`, and `nba2k` survives. A token
is present or absent; there are no term weights and no suffix stripping.

**Term selection ranks by frequency, not by class relevance.** For each field, the
candidate pool is the terms appearing in at least two training answers, ordered by total
occurrences, then by number of answers containing the term, then alphabetically; the
first K are kept. The labels are not an input. This is the decision that most clearly
separates v3 from a chi-square selection. Measured on the training split, at a 50-term
budget, frequency ranking leaves these numbers of rows with no active column at all:

    field      frequency       chi-square
    food        40 (6.6%)       93 (15.3%)
    song       162 (26.7%)     211 (34.8%)
    feeling    115 (19.0%)     185 (30.5%)

A row whose block is entirely zero gives the forest nothing to work with, and song
titles are the worst case because most of them contain `the`, `you` or `we` - frequent
terms that carry no class information but do keep the answer recognisable, and that
chi-square removes first.

**The colour answer uses a fixed lexicon** of 17 concepts with no fitted state, so a fold
cannot change the colour columns. Variants and typos merge: `grey`/`gray`,
`black`/`balck`, `blues`, `yellowish`, `golden`.

**Each text field carries three auxiliary columns.**

    <field>_non_answer   blank, or a reply in the refusal set (none, nothing, n/a, ...)
    <field>_no_terms     answered, but not one word landed in the kept vocabulary
    <field>_item_count   how many tokens the answer has

`_no_terms` is the column the earlier models did not have: an answer we cannot recognise
and an answer that is genuinely all-negative looked identical to the forest before. It is
also why the value block is 0/1: the answer-breadth information that a TF-IDF row norm
would smear across every column instead lives in one explicit `_item_count` column.

## Stages

| stage | what it does | splits |
|---|---|---|
| `cv` | one fixed setting, 15 grouped folds (3 fold seeds x 5 splits) | train |
| `ablation` | eight field sets x a four-configuration grid, to price each field | train |
| `search` | the 150-configuration grid below, 15 grouped folds, one-standard-error choice | train |
| `diagnostics` | tree count, learning curve, permutation importance by block and column | train |
| `validation` | fit 606 rows, score 132 rows, once; opt-in by stage name | train, val |

The grid is the same one both earlier models used, so the numbers are comparable:

    max_depth        4, 6, 8, 12, 16, None
    min_samples_leaf 1, 2, 3, 5, 8
    max_features     sqrt, log2, 0.3, 0.5, 1.0

Selection takes the best cross-validated accuracy, then the simplest configuration within
one standard error of it, where simplicity is shallower first, then larger leaves, then
fewer features. "One standard error" is the spread divided by the square root of the
number of fold observations, so it is 0.0205 / sqrt(15) = 0.0053 here.

## Running it

    python run.py --stage all
    python run.py --stage cv --dry-run
    python run.py --stage diagnostics --source random_forest_deliverables/v3_forest
    python run.py --stage validation
    python smoke_check.py

`--n-estimators` defaults to 300 and `--dry-run` drops to one fold seed and a
four-configuration grid. Artifacts land in `random_forest_deliverables/v3_forest/` unless `--out` says
otherwise. Every JSON written there records `test_accessed: false`, which splits it read,
the sha256 of the five source files and of each split it read, and the library versions.

## Reading the result

Cross-validation at `max_depth=8, min_samples_leaf=2, max_features=log2`, which is the
`cv` default and the two earlier models' chosen shape:

    accuracy        0.9065 +- 0.0207       macro F1  0.9062
    log loss        0.5402                 in-fold train accuracy  0.9380
    per fold seed   0.9077 / 0.9093 / 0.9024

The search over 150 configurations:

    best     0.9098   max_depth=12, min_samples_leaf=1, max_features=log2
    band     0.0053, threshold 0.9045, 12 of 150 configurations inside
    chosen   0.9054   max_depth=8,  min_samples_leaf=2, max_features=sqrt

The chosen configuration is not the best one. No configuration shallower than depth 8
reaches the band at all - the best depth-4 point is 0.9000 and the best depth-6 point is
0.9032 - so the choice is made among the three depth-8 configurations that are inside it,
and the simplest of those is `sqrt` with `min_samples_leaf=2`. `sqrt` on 173 columns
draws 13 features per split against 7 for `log2`, and the extra draw is enough to put the
shallower tree below two faster ones, both still inside the same band.

Diagnostics at the chosen configuration:

    trees     100     200     300     500     800
    accuracy  0.9033  0.9032  0.9054  0.9043  0.9060
    log loss  0.4622  0.4594  0.4590  0.4579  0.4562

    fraction   rows  respondents  accuracy
    0.25        120           40    0.8867
    0.50        242           81    0.8994
    0.75        365          122    0.8989
    1.00        485          162    0.9071

The tree count is flat between 200 and 800, and the learning curve is still climbing at
the full fitting set, which means the 606 rows - not the column count - are the
constraint.

Permutation importance, summed inside each fold over that fold's own columns and then
averaged over the 15 folds:

    block            columns/fold     sum    share
    structured               26.0  0.1154    43.4%
    colour                   20.0  0.0858    32.3%
    song                     53.0  0.0468    17.6%
    food                     53.0  0.0145     5.5%
    feeling                  15.0  0.0032     1.2%
    missing flags             6.0  0.0000     0.0%

    colour_green        +0.04611      music_hiphop_rap    +0.01298
    song_waka           +0.03578      season_winter       +0.00920
    location_indoor     +0.03459      ease                +0.00788
    music_latin         +0.02377      injury              +0.00736
    colour_orange       +0.02352      food_beer           +0.00698
    colour_blue         +0.01436      music_country       +0.00657

Feeling earns 1.2% of the importance with 15 columns. That is the reason its budget is 12
terms rather than the 50 to 60 the earlier models carried: at that size the block stops
costing accuracy, but it still does not earn its columns.

Block ablation, one fold seed and four configurations per field set, so the numbers sit at
a different tree shape than the chosen one and are read as differences rather than as
accuracies:

    variant              fields                    cols   best acc   mean acc
    all                  food,colour,song,feeling   173     0.9333     0.9006
    no_food              colour,song,feeling        120     0.9333     0.8953
    no_colour            food,song,feeling          153     0.9167     0.8601
    no_song              food,colour,feeling        120     0.9187     0.8790
    no_feeling           food,colour,song           158     0.9333     0.8982
    no_song_no_feeling   food,colour                105     0.9187     0.8732
    colour_only          colour                      52     0.9083     0.8701
    none                 -                           32     0.8500     0.8084

In mean accuracy each field is worth: colour +0.0405, song +0.0216, food +0.0053,
feeling +0.0024. The 32 structured columns reach 0.8084 on their own and the colour
lexicon alone reaches 0.8701.

## The validation reading

Fitted on the 606 training rows at the chosen configuration, scored once on the 132
validation rows (44 respondents, three rows each):

    accuracy        0.9091          macro F1  0.9083
    log loss        0.4021          in-fold train accuracy  0.9406   (gap +0.0315)
    bootstrap       accuracy [0.8561, 0.9545]   macro F1 [0.8543, 0.9544]
    respondents with all three correct   33 of 44

    precision / recall / F1
    Soccer       0.9487 / 0.8409 / 0.8916
    Baseball     0.9286 / 0.8864 / 0.9070
    Basketball   0.8627 / 1.0000 / 0.9263

    confusion, rows are actual
                 Soccer   Baseball  Basketball
    Soccer           37          3           4
    Baseball          2         39           3
    Basketball        0          0          44

Twelve of the 132 rows are wrong. The 208-column model also gets twelve wrong, but not the
same twelve: nine rows are missed by both, three only by it and three only by v3. The
183-column model gets ten wrong and shares nine of v3's twelve. **The cross-validated
advantage does not survive the held-out read.** That is the expected outcome for a
difference of 0.0011 on 606 rows against a one-standard-error band of 0.0053, and it is
recorded here rather than smoothed over: on this split v3 ties one earlier model and loses
to the other.

Two things do come out of the reading. The log loss is 0.4021 against 0.4953 for the
183-column model, so the same count of errors is spread over much sharper probabilities -
v3's mistakes are narrow, not confident. And Soccer recall is 0.8409 for the third model
in a row, with all 44 Soccer responses making up 7 of the 12 errors (3 called Baseball, 4
called Basketball). That is now a three-way confirmation that Soccer recall is a property
of the survey rather than of any one feature design.

## Files

    text.py          the tokenizer
    encode.py        survey parsing, the refusal set, the colour lexicon, column lists
    features.py      the 173-column matrix and its fold-local fitting
    selection.py     grouped folds, the grid, one-standard-error selection
    run.py           the five stages
    smoke_check.py   column counts, lexicon and auxiliary-column cases, fold integrity

`encode.py` is a copy of the same file in `train/random_forest_v2/`, which is itself a
copy of `data_preprocessing/encode_all.py`, so that editing one model cannot silently
change another.
