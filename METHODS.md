# Methods and recorded findings

## Cohort and outcome

The two supplied handoffs each contain 2,155 unique subjects, with 29 and 31
non-overlapping predictors. Original inputs were preserved and their SHA-256
hashes recorded locally. This is baseline prediction, not repeated visit prediction.

Use the original ADNIMERGE baseline date as the index. ADNIMERGE dates are parsed
explicitly as day-month-year; newer source tables and the shared list use ISO dates.
Include CN, SMC, EMCI, and LMCI; exclude AD at baseline and dementia documented
at or before the index. DXSUM's harmonized diagnosis 3 means dementia of any cause,
not necessarily Alzheimer's disease. See the
[ADNI diagnosis documentation](https://adni.loni.usc.edu/quick-start-guide-asset/diagnostic.html).

Positive means an observed dementia diagnosis strictly after index and on/before
the fifth calendar anniversary. A later reversion does not erase that event.
Negative requires no observed within-window dementia plus a valid diagnosis on
or after the anniversary. This is an observation-based endpoint, not continuous
surveillance. Identical repeated diagnoses are harmless; differing diagnoses on
the same date and undated dementia records quarantine the subject.

| Cohort disposition | Subjects |
|---|---:|
| Five-year nonconverters | 732 |
| Incident dementia | 324 |
| Insufficient follow-up | 698 |
| Dementia at baseline | 367 |
| Unknown baseline diagnosis | 23 |
| Undated dementia | 8 |
| Conflicting same-day diagnosis | 2 |
| Additional dementia at/before index | 1 |
| Total | 2,155 |

The old labels are retained only for reconciliation. They included 363 baseline-AD
subjects among positives and short-follow-up subjects among negatives.

## Predictor preparation

The matrix has 60 source predictors plus three laboratory bound indicators.
Repeated ADNIMERGE fields use `_bl` versions. No diagnosis, RID, date, phase,
follow-up measurement, or outcome field is passed to the classifier.

NEUROBAT values must be baseline or screening assessments within 90 days before
index, including index day. Prefer screening for Logical Memory and baseline for
other assigned scores; fall back only among eligible records. Same-priority,
same-day conflicting values become missing. Per-feature dates and decisions stay
in the local provenance table. This corrected 30–120 cells per neuropsychology
column relative to the supplied B table.

For each history condition, any documented pre-index positive becomes 1;
documented negative becomes 0; absent usable history remains missing. There were
417 subjects without eligible history records, whose zeros were restored to
missing for each assigned condition. `MH3HEAD` denotes head/eyes/ears/nose/throat
history and must not be interpreted as a specific traumatic-brain-injury flag.

Laboratory bounds are represented by their stated boundary plus a signed bound
indicator. Negative RAVLT change scores are retained because they can be valid.
Plasma is excluded from the primary dataset because later blood measurements
cannot be used in original-baseline prediction.

The earlier audit's 3,066 total minus 34 Tier 1 minus 33 Tier 2 equals 2,999,
not its reported 2,943 deferred fields. The assigned primary matrix uses 60 of
the 67 candidates; the seven plasma candidates are deferred.

## Development and calibration

Fixed, outcome-stratified subject partitions, seed 42:

| Partition | Subjects | Converters | Nonconverters |
|---|---:|---:|---:|
| Development | 633 | 194 | 439 |
| Calibration/threshold | 211 | 65 | 146 |
| Final validation | 212 | 65 | 147 |

Five fixed development folds compare clinical (14), core (28), and expanded (60)
source-feature panels. Bounds are added when their CSF field is in the panel.
Constant and >80%-missing columns are removed on the training fold only.
Numeric imputation, scaling, categorical encoding, and missingness indicators are
also fitted inside each training fold. Unseen categories are supported.

Compare majority class, logistic regression, random forest, XGBoost, and CatBoost.
Eight sampled parameter configurations per model/panel are evaluated, below the
30-configuration cap. Class weighting is a candidate hyperparameter; balanced
weights for logistic/forest/CatBoost are computed within each fitted training
sample, and XGBoost compares fixed positive weights of 1 and 2.

Within each development fold, reserve 25% of its training subjects to select an
F1 threshold. Fit on the remaining inner subjects, select that threshold, refit on
the full outer training fold, then score the untouched outer fold. Threshold
selection never uses outer validation labels. Model selection uses mean F1,
then recall, then simpler model. Search results are development estimates and
can be optimistic after selecting among configurations.

The three strongest development candidates additionally compare scaled KNN
imputation and MRI/ICV normalization. Across 109 recorded comparisons, the winner
was balanced logistic regression with C=0.01, core features, and KNN imputation.
Its development mean F1 was 0.8721 (fold standard deviation 0.0442).

Fit the chosen pipeline on development only. Fit a sigmoid calibrator on the
reserved calibration partition, and choose maximum-F1 threshold 0.5130783821
there, resolving F1 ties by recall. Calibration-set results are fitted estimates,
not independent validation. The final panel retains 28 source fields and two
nonconstant bound indicators before encoding and missingness expansion.

## Frozen validation and interpretation

The final pipeline was frozen before validation. Accuracy is 89.62%, precision,
recall, and F1 are each 83.08%; ROC AUC is 0.9667, average precision 0.9250,
and Brier score 0.06691. The confusion matrix is [[136,11],[11,54]], with actual
class as rows and predicted class as columns, ordered [nonconverter, converter].
Threshold 0.5 happens to yield the same validation decisions as the selected
threshold; it was not adopted after viewing results.

The 2,000 subject-bootstrap intervals keep class counts fixed and hold model and
threshold fixed. They measure held-out sampling uncertainty, not variability from
retraining or alternative cohort definitions.

All 65 validation converters are in the MCI subgroup. MCI validation accuracy
is 83.2%, precision 84.38%, recall 83.08%, F1 83.72%, and ROC AUC 0.9277
(n=125). CN/SMC has n=87 and no converters; its sensitivity cannot be estimated.
ADNI3 validation likewise has no converters; ADNIGO has only two. Zero-event
groups use zero-division conventions for positive-class scores in the JSON and
must not be interpreted as estimates of positive-case performance.

ADNI recruitment, incomplete follow-up exclusions, phase differences, missing
biomarkers, and the small validation sample constrain generalization. Source
ADNIMERGE `_bl` values provide merged-table baseline provenance; original
acquisition timestamps are not available for every measurement in this snapshot.
There is no independent external-cohort validation or clinical deployment claim.

## Verification and outputs

Synthetic tests cover label dates, baseline exclusion, undated/conflicting visits,
history aggregation, temporal filtering, lab bounds, handoff contracts, leakage-safe
preprocessing, saved-model predictions, subject splits, and an end-to-end run.
The fresh-process reproduction rebuilds the matrix and selected model separately;
its summary is stored locally in `reproduction_check.json`.

Public outputs contain aggregate counts, experiment summaries, the registry, and
plots. All subject records, splits, provenance, predictions, and trained artifacts
remain in ignored local storage.
