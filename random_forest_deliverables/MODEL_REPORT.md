# Random forest model report

What this is: a description of the random forest we built for the CSC311 survey task, what
it is made of, how its configuration was chosen, and how it performs. It describes the model
as an object.

What this is not: `README.md` in this directory describes how to run the pipeline (stages,
flags, artifacts). Neither document is the course report.

The model described here is the one saved at
`random_forest_deliverables/forest_final_v1/final_model.joblib`, sha256
`c08f5777706ad9a87740e0f6d732a80ac1a711d02a44d0983747a54fab765772`, 1,089,420 bytes.

---

## 1. The task and the data

Classify a survey response as Soccer, Baseball or Basketball.

The raw dataset holds 870 responses from 290 respondents. Every respondent answered the same
twelve questions three times, once per sport, so each contributes exactly three rows and the
three classes are equal by construction. Grouping is therefore not optional: the three rows of
one respondent share writing style, vocabulary and personal taste, and any row-level split
would leak a person across training and evaluation.

Splits were made once, before any modelling, with seed 311 and grouped by respondent:

    train        606 responses / 202 respondents
    validation   132 responses /  44 respondents
    test         132 responses /  44 respondents

Each split holds exactly one row per sport per respondent, so every split is class-balanced.

The model under discussion is fitted on train plus validation: **738 responses from 246
respondents**. Validation was merged in only after the configuration was fixed and after
validation had already been used for its one model-selection reading.

## 2. The twelve questions

    Q1   ease of starting, 0-10 numeric
    Q2   strategy matters more than athleticism, 1-5
    Q3   injury risk, 1-5
    Q4   individual brilliance can outweigh teamwork, 1-5
    Q5   where it is played, multi-select, 7 options
    Q6   season, multi-select, 4 options
    Q7   music associated with it, multi-select, 10 options
    Q8   food and drink association, free text
    Q9   colour association, free text
    Q10  song association, free text
    Q11  feeling when watching, free text
    Q12  willingness to pay, free text with numbers

## 3. The feature space: 208 columns

| Block | Columns |
|---|--:|
| TF-IDF text: food 50, colour 20, song 50, feeling 50 | 170 |
| Text-missing flags, one per text field | 4 |
| Q5-Q7 multi-hot indicators (7 + 4 + 10) | 21 |
| Question-level missing flags, `q1_missing` ... `q7_missing` | 7 |
| Q2-Q4 agreement items, mapped to 0 / 0.25 / 0.5 / 0.75 / 1 | 3 |
| Q12 `payment_amount` (mean-filled, then `log1p`) and `payment_missing` | 2 |
| Q1 ease, divided by 10 | 1 |
| **Total** | **208** |

Text handling: each field gets its own TF-IDF encoder, lowercase, token pattern
`(?u)\b[a-zA-Z]{2,}\b`, unigrams, `min_df=2`, `sublinear_tf`, `smooth_idf`, `norm='l2'`, no
stopword list, per-field caps of 50 / 20 / 50 / 50. An empty answer becomes an all-zero block
plus its missing flag.

Structured handling: Q1-Q4 are median-imputed, multi-hot columns are mode-imputed with ties
resolved to zero, payment is filled with the training mean (256.92) and then `log1p`.
**Nothing is standardised.** A forest is invariant to monotone rescaling of any single column,
so scaling would change no split and would only complicate the export.

Every fitted quantity - the four vocabularies, their IDF weights, the imputation values - comes
from training rows only. A held-out row is transformed without refitting.

Two conventions that matter to anyone reading the artifact:

- **Class order.** scikit-learn stores `['Baseball', 'Basketball', 'Soccer']`, alphabetical.
  Every consumer must map by name, never by column position.
- **`log1p` on payment does not change this model.** With depth capped at 6 the transform
  alters no split and no prediction; it is kept so the pipeline is shared with the other
  families, which do need it.

## 4. The model

    algorithm        random forest, 300 CART trees, Gini impurity, bootstrap sampling
    aggregation      class probabilities averaged across trees, then argmax
    max_depth        6
    min_samples_leaf 3
    max_features     log2, that is 7 of the 208 columns offered at each split
    random_state     311
    n_jobs           1

Fixed by construction rather than searched, and why:

| Setting | Value | Reason |
|---|---|---|
| `n_estimators` | 300 | a tree-count curve is flat from 200 upward (section 7) |
| `class_weight` | None | the classes are exactly balanced, 246 responses each |
| `oob_score` | False | a row-level bootstrap does not keep one respondent's three rows together, so out-of-bag scoring would not be respondent-disjoint |
| `criterion` | gini | not varied; no reason to expect a difference at this scale |
| `n_jobs` | 1 | single-threaded so timings are reproducible |

**Measured size.** 300 trees, **10,498 nodes**, 5,399 leaves, every tree exactly depth 6,
about 35 nodes per tree.

## 5. How this configuration was chosen

Selection happened in two stages, both scored on the 606 training responses only.

**Stage one, the family comparison.** Three families were tuned independently, to convergence,
before any of them was compared - tuning only the apparent winner would have understated the
others.

| Family | Settings searched | Train acc. | CV acc. | Val. acc. |
|---|--:|--:|--:|--:|
| Softmax regression, `C` in 0.001 ... 100 | 6 | 97.03 | 90.43 +- 1.60 | 90.91 |
| **Random forest**, depth x leaf x `max_features` | 24 | 99.34 | 91.27 +- 2.14 | **93.18** |
| MLP, 16/32/64 hidden x alpha 0.01 ... 10 | 12 | 96.20 | 90.10 +- 2.78 | 90.91 |

Plus one boundary extension per family, giving 44 configurations and 220 fits. The forest led
on validation by three responses out of 132, which is not a decisive margin; it was taken
forward because it also has the structure the exploration predicted would matter - thresholds
on individual columns.

**Stage two, the hyperparameter refinement.** A 150-configuration grid over depth
{4, 6, 8, 12, 16, unlimited} x leaf size {1, 2, 3, 5, 8} x `max_features`
{sqrt, log2, 0.3, 0.5, 1.0}, evaluated on fifteen grouped folds (three fold seeds x five
splits, 2,250 fits, 1,600 s).

Selection rule: highest mean cross-validated accuracy, then the one-standard-error band, then
the simplest configuration inside it, where simplicity means a shallower tree, then a larger
leaf, then fewer candidate split features.

    best point       max_depth=None, max_features=sqrt, min_samples_leaf=1
                     CV 0.9082, in-fold train accuracy 0.9934
    1-SE band        width 0.005196 = 0.0201 / sqrt(15)
                     threshold 0.902961, 20 of 150 configurations inside
    chosen           max_depth=6, max_features=log2, min_samples_leaf=3
                     CV 0.9043, in-fold train accuracy 0.9260

The chosen model gives up 0.0039 of cross-validated accuracy against the best point. What it
buys is the difference between 99.34% and 92.60% in-fold training accuracy, and a model 6.8
times smaller in nodes. Section 6 shows that the trade was the right way round: on held-out
data the deeper configuration was not better.

## 6. Measured performance

| Reading | Protocol | Accuracy | Macro F1 | 95% respondent-bootstrap CI |
|---|---|--:|--:|---|
| Cross-validation | 15 grouped folds, 606 rows | 0.9043 +- 0.0230 | 0.9038 | fold spread 0.867 - 0.933 |
| Validation | fit 606, score 132 | 0.9091 (120/132) | 0.9088 | [0.8561, 0.9545] |
| **Test** | fit 738, score 132, read once | **0.9015 (119/132)** | 0.9022 | [0.8485, 0.9470] |

Cross-validated log loss is 0.6296. The in-fold training accuracy across folds is 0.9260, so
the training-to-cross-validation gap is **0.0217**. That is the smallest of the four forest
models we built: the others, which differ in depth and in feature space, sit at 0.0371, 0.0385
and 0.0852.

The three readings agree to within 0.008. Test is 0.0028 below cross-validation and 0.0076
below validation, both far inside a standard error of 0.026 on 132 rows, and the two bootstrap
intervals overlap almost completely. There is no sign that the model degrades out of sample.

Per class:

    validation (120/132)            test (119/132)
                 P       R      F1              P       R      F1
    Soccer     .9737   .8409   .9024          .8864   .8864   .8864
    Baseball   .9091   .9091   .9091          .8511   .9091   .8791
    Basketball .8600   .9773   .9149          .9756   .9091   .9412

Thirty-three of 44 validation respondents are fully correct, and 32 of 44 on test.

## 7. Diagnostics

These are **five folds on fold seed 311**, not the fifteen folds used for the headline number.
Fold seed 311 is the most favourable of the three, which is why the figures below run about
0.002 above the headline.

Tree count, at the chosen settings:

    trees     100      200      300      500      800
    accuracy  0.8977   0.8995   0.9061   0.9078   0.9078
    log loss  0.6298   0.6298   0.6302   0.6320   0.6337

Accuracy rises from 200 to 300 trees and then flattens, and log loss begins to worsen from 300
onward, so 300 is a reasonable operating point rather than a tuned one.

Learning curve, subsampling respondents inside each fitting fold:

    fraction of respondents   25%     40%     55%     70%     85%    100%
    accuracy                 0.8879  0.8912  0.8913  0.8978  0.9044  0.9061

Still rising at the full 483-row fitting folds. **The 606 training responses, not the model, are
the binding constraint** - which is the main reason to expect no further gain from feature or
hyperparameter work.

Permutation importance, summed per fold over that fold's own columns, then averaged:

| Block | Columns | Permutation importance |
|---|--:|--:|
| text: colour | 20 | 0.0695 |
| Q7 multi-select (music) | 10 | 0.0403 |
| text: song | 50 | 0.0366 |
| Q5 multi-select (venue) | 7 | 0.0267 |
| text: food | 50 | 0.0233 |
| Q12 payment | 2 | 0.0100 |
| text: feeling | 50 | 0.0094 |
| Q6 multi-select (season) | 4 | 0.0091 |
| Q3 injury | 1 | 0.0047 |
| Q4 individual | 1 | 0.0046 |
| Q2 strategy | 1 | 0.0027 |
| Q1 ease | 1 | 0.0023 |
| missing flags | 7 | 0.0000 |

Two things stand out. The colour block is the single most valuable group despite holding only
20 columns, and 50 feeling columns earn less than 4 season columns. The seven missingness flags
earn exactly nothing.

## 8. What each text field is worth

Six field sets, four configurations each (`ABLATION_GRID`: depth 4 or unlimited x leaf 1 or 5 x
sqrt), five folds on seed 311. Because these sit at a different tree shape than the chosen
configuration, read them as differences, not as accuracies.

| Field set | Columns | Mean of 4 configs | Best of 4 |
|---|--:|--:|--:|
| food, colour, song, feeling | 208 | 0.9015 | 0.9333 |
| food, colour, song (no feeling) | 157 | 0.9008 | 0.9333 |
| food, colour (no song, no feeling) | 106 | 0.8820 | 0.9350 |
| food, colour, feeling (no song) | 157 | 0.8778 | 0.9187 |
| food only | 85 | 0.8285 | 0.8537 |
| structured only, no text | 34 | 0.8207 | 0.8667 |

Reading the mean column: colour is indispensable, song is worth about 2.2 points, food about
0.5, feeling about 0.1. Removing both song and feeling reaches the same best accuracy as
keeping everything, while removing song alone clearly hurts - the two fields overlap, and the
forest can substitute one for the other.

## 9. Where the errors are

Validation, rows are true:

                 Soccer   Baseball  Basketball
    Soccer           37          3           4
    Baseball          1         40           3
    Basketball        0          1          43

Test, rows are true:

                 Soccer   Baseball  Basketball
    Soccer           39          5           0
    Baseball          3         40           1
    Basketball        2          2          40

Total errors are the same on both splits, 12 and 13, but they are not the same errors, which is
the expected behaviour for 132-row samples.

The consistent finding is a **one-way leak away from Soccer**. Soccer recall is the weakest
class in every reading (0.8409 on validation, 0.8864 on test), and uncertain rows move toward
the other two. On validation the largest single pair is Soccer called Basketball (4 rows); on
test that pair is empty and Soccer loses 5 rows to Baseball instead, while Basketball's recall
falls from 43/44 to 40/44 and gives those rows back.

The mechanism is visible in the feature space. Soccer's strongest markers are narrow:
`green` appears in 99 of 202 Soccer training responses, `waka` in 89, `beer` in about half.
Basketball's markers are near class-specific instead - the indoor-venue indicator is selected
by 97% of Basketball responses, winter by 57%, hip-hop/rap by 83%. A Soccer respondent whose
answers fall outside the narrow markers has little left to identify them by, and the forest
sends them to whichever other class their remaining columns resemble.

This is a property of the survey, not of this feature space. Both alternative text pipelines we
built score the same 37/44 Soccer recall on validation, and the deeper configuration - which
reaches 39/44 - gets there by fitting the training data far harder, not by finding a new signal.

## 10. Size and inference cost

Export footprint: three integer arrays (left child, right child, split feature), one threshold
array and one class-probability table, 10,498 entries each. Roughly 0.5 MB, against a 10 MB
limit.

A hand-written NumPy traversal of those arrays - no scikit-learn at inference - reproduces the
library's predictions on all 132 held-out responses exactly, and scores them in **about
140 ms** including feature construction: 36 ms for the tree descent, 103 ms for preprocessing.
Sixty predictions therefore take about 0.06 s, against a one-minute requirement. Timings move
with machine load - the same measurement took 176 ms while other work was running - so treat
the figure as an order of magnitude, not a benchmark.

Cost scales with tree depth, not with row count. Depth 6 with 10,498 nodes is cheap; the deeper
configuration we rejected has 71,342 nodes at depth 31 and its traversal takes about three
times as long, which is another reason it is not the model of record here.

## 11. Tried and not adopted

Two alternative text designs were built and cross-validated over the same fifteen folds, holding
the family fixed:

| Design | Columns | CV accuracy | Val. accuracy |
|---|--:|--:|--:|
| adopted: TF-IDF, terms by frequency cap, 208 columns | 208 | 0.9043 | 0.9091 |
| chi-square term selection, 0/1 presence, fixed 17-colour lexicon | 183 | 0.9043 | 0.9242 |
| frequency term selection, Q4 dropped, smaller feeling block | 173 | 0.9054 | 0.9091 |

All three sit within one standard error of each other, so none was adopted. The exercise is
worth recording because it establishes that accuracy on this task is insensitive to the text
representation once the colour and music columns are present.

Also rejected: the unlimited-depth configuration from the grid (section 5), on the grounds of a
99.34% in-fold training accuracy and a 2.8x larger export.

## 12. Artifacts

| Path | Content |
|---|---|
| `random_forest_deliverables/forest_search_v1/` | 150-configuration CV and the 1-SE decision |
| `random_forest_deliverables/forest_ablation_v1/` | the six field sets |
| `random_forest_deliverables/forest_diagnostics_v1/` | tree count, learning curve, permutation importance |
| `random_forest_deliverables/forest_validation_v1/` | validation predictions, metrics, confusion |
| `random_forest_deliverables/forest_final_v1/` | **the model**: `final_model.joblib`, `final_manifest.json`, and the one-off test evaluation |
| `random_forest_deliverables/V1_ARCHIVE.md` | frozen model card with every sha256 |
| `random_forest_deliverables/experiment_log.md` | the pre-registration and the test result |
| `random_forest_deliverables/inference_time.json` | the timing measurement in section 10 |

Reproducing the fitted model from the working tree requires the source hashes in
`V1_ARCHIVE.md` to still match; the serialized `final_model.joblib` keeps working either way,
because it carries its own vectorizers, IDF weights and imputation values.

## 13. Known limitations

1. **The sample is small.** 246 respondents in the final fit. The learning curve is still
   rising, so every number here is a statement about 246 people, not about the population.
2. **132 rows resolve nothing.** A standard error of 0.026 on the held-out splits means
   differences under about three or four responses cannot be detected. The comparison in
   section 11, and the choice in section 5, should be read with that in mind.
3. **Soccer is under-predicted and we did not fix it.** Recall 0.84-0.89 against 0.91 for the
   other two. The three feature designs we tried all fail the same way.
4. **The vocabulary is cohort-specific.** `waka` and `green` are cultural markers of this
   particular student sample. On the course's hidden test set, compiled from TA and instructor
   responses, those markers may be rarer and performance may drop.
5. **The test split has been read once.** 0.9015 is this model's single independent estimate.
   Any later change to the model invalidates it.
