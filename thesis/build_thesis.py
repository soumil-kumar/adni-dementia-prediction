"""Author the implementation-centred thesis from saved evidence; no fitting."""
from pathlib import Path
import json, shutil
root=Path(__file__).resolve().parent; repo=root.parent
read=lambda name:json.loads((repo/name).read_text())
results=read('results/validation_results.json'); records=read('results/development_comparison.json')['records']
audit=read('results/data_audit.json'); versions=read('results/run_summary.json')['versions']
for name in ['validation_plots.png','missingness.png','calibration.png','coefficients.png']:
    shutil.copyfile(repo/'report/figures'/name,root/'figures'/name)
def table(head,rows,spec=None):
    return ('\\begin{center}\\small\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{'+(spec or 'l'+'r'*(len(head)-1))+'}\\toprule\n'+
        ' & '.join(head)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n')
parts=[r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,xcolor}
\usepackage[colorlinks=true,linkcolor=blue!40!black,urlcolor=blue!40!black,citecolor=blue!40!black]{hyperref}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small Baseline ADNI dementia prediction}\fancyhead[R]{\small B.Tech. project thesis}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{3em}
\newcommand{\code}[1]{\texttt{\detokenize{#1}}}
\newcommand{\fig}[3]{\begin{figure}[!ht]\centering\includegraphics[width=#2\linewidth]{figures/#1}\caption{#3}\end{figure}}
\setcounter{tocdepth}{1}
\begin{document}
\begin{titlepage}\centering\vspace*{25mm}
{\Large Bachelor of Technology Project\par}\vspace{17mm}
{\Huge\bfseries Predicting Five-Year\par Dementia Risk from\par Baseline ADNI Data\par}\vspace{12mm}
{\Large A reproducible tabular machine-learning study\par}\vspace{16mm}
{\large Technical thesis\par 2 November 2026\par}\vfill
\begin{minipage}{.87\textwidth}
\textbf{Study focus}\par
From source-table reconciliation and incident-outcome construction to calibrated prediction and untouched subject-level validation.

\textbf{Implemented outcome}\par
Any incident dementia within five calendar years among participants without dementia at their original baseline.

\textbf{Code and aggregate evidence}\par
\url{https://github.com/soumil-kumar/adni-dementia-prediction}
\end{minipage}\vspace{12mm}
\end{titlepage}\setcounter{page}{2}
''']
def page(text):parts.append(text+'\n\\clearpage\n')
page(r'''\section*{Abstract}\addcontentsline{toc}{section}{Abstract}
This Bachelor of Technology project investigates whether baseline tabular measurements can predict observed dementia conversion within five years in the Alzheimer's Disease Neuroimaging Initiative (ADNI). Its main contribution is an end-to-end, reproducible machine-learning workflow in which temporal availability, outcome construction, missing-data treatment, and evaluation isolation are explicit. The implemented target is any dementia, rather than a separately adjudicated Alzheimer-specific outcome.

The project began with 17 inventoried source tables, a domain-guided feature audit, and two independently extracted feature handoffs. The handoffs were validated and merged into 60 source predictors for 2,155 original-baseline participants. Date-based neuropsychological corrections, restoration of unavailable medical history to missing, and explicit parsing of laboratory bounds addressed material data-quality problems. Longitudinal diagnosis records were used only to construct the target. After excluding baseline dementia, inadequate follow-up, and ambiguous records, the binary cohort contained 1,056 participants: 324 converters and 732 observed five-year nonconverters.

A fixed outcome-stratified split assigned 633 participants to development, 211 to calibration and threshold selection, and 212 to final validation. Five-fold development comparisons evaluated logistic regression, random forest, XGBoost, and CatBoost across clinical, core, and expanded feature panels. All learned preprocessing was fitted inside training folds. Inner training holdouts selected decision thresholds without using outer-fold outcomes. Across 109 recorded comparisons, regularized logistic regression with balanced class weights, core features, and scaled distance-weighted K-nearest-neighbor imputation achieved the best mean development $F_1$.

The selected pipeline was fitted on development data and sigmoid-calibrated on the reserved calibration partition. A probability threshold of 0.513078 was fixed before opening final validation. Accuracy was 89.62\%, precision 83.08\%, recall 83.08\%, and $F_1=0.8308$. ROC AUC was 0.9667, average precision 0.9250, and Brier score 0.06691. The confusion matrix contained 136 true negatives, 11 false positives, 11 false negatives, and 54 true positives. Bootstrap intervals quantify uncertainty conditional on the frozen model and held-out class counts.

The software provides preparation, training, evaluation, prediction, and reproduction commands. Thirteen synthetic tests and repository CI passed; a separate fresh-process run recovered identical data and splits and reproduced probabilities within $10^{-12}$. All source inputs were preserved, and patient-level artifacts remain local.

The findings support a reproducible ADNI research result, not a general-population clinical risk guarantee. Important limitations include selection induced by incomplete follow-up, uncertain acquisition dates for some merged baseline fields, correlated predictors, a small final validation sample, and no converters in its CN/SMC subgroup. These constraints guide a focused path toward external and temporal validation.

\textbf{Keywords:} dementia prognosis; ADNI; baseline biomarkers; temporal leakage; missing data; calibration; reproducibility.
''')
page(r'''\tableofcontents
\vfill
\textbf{Report conventions}\par
``Development'' refers to model comparison; ``calibration'' to fitting the probability mapping and operating threshold; ``final validation'' to the untouched evaluation partition. Development cross-validation results are never substituted for final validation results. Predictor counts distinguish source measurements from derived indicators and encoded model inputs.

\textbf{Abbreviations}\par
AD: Alzheimer's disease; CN: cognitively normal; SMC: significant memory concern; MCI: mild cognitive impairment; EMCI/LMCI: early/late MCI; CSF: cerebrospinal fluid; MRI: magnetic resonance imaging; PET: positron emission tomography; RID: participant roster identifier; ICV: intracranial volume; KNN: K-nearest neighbors; ROC: receiver operating characteristic; AP: average precision; ECE: expected calibration error.
''')
page(r'''\section{Research problem and project contribution}
\subsection{Prediction rather than recognition of existing disease}
Dementia prognosis concerns a future clinical event. Identifying a participant who already has dementia is a different task, even if the same cognitive tests and biomarkers are useful. A valid five-year model therefore needs an explicit index date, baseline eligibility, an observation window, and a definition of the evidence required for a negative outcome. Without these choices, high apparent accuracy may reflect recognition of existing disease or incomplete follow-up rather than forecasting.

ADNI provides repeated clinical assessments, demographics, genetic information, fluid biomarkers, and imaging-derived measures. These sources make multimodal prognosis feasible, but also create opportunities for temporal leakage and duplicate observations. The central research question is: \emph{can information available at an originally non-demented participant's baseline discriminate observed conversion to dementia within five calendar years?}
\subsection{Operational scope}
The index is each participant's original ADNIMERGE baseline. Only tabular features are modeled. Numeric MRI volumes and PET composite measurements are included, but raw images, image segmentation, convolutional networks, and image-processing claims are outside the implemented pipeline. There is one prediction record per participant, not a series of rolling visit-level predictions.

The original motivation concerned Alzheimer's disease and neurocognitive decline. However, the harmonized diagnosis field used here establishes dementia status, not necessarily Alzheimer etiology \cite{adni}. Consequently, the report consistently names the actual endpoint as \emph{any incident dementia}. This distinction changes what can be concluded from the model.
\subsection{Objectives and contributions}
The objectives are to construct a defensible labeled cohort; reconcile independently extracted predictors; compare suitable tabular classifiers without information leakage; produce calibrated probabilities and a documented decision rule; and deliver software that another collaborator can reproduce.

The main contributions are methodological and engineering as well as predictive. They include correcting baseline-disease contamination in existing labels, restoring missing history rather than interpreting it as absence, enforcing date-safe neuropsychological selection, keeping preprocessing inside validation boundaries, and packaging the frozen estimator with its calibration and threshold. A successful project requires credible generalization evidence, not merely the largest observed score.

The selected optimization criterion is $F_1$, balancing positive-case precision and recall. Accuracy remains a required output, but class imbalance makes it an inadequate selection criterion on its own. The final held-out result is the project's performance estimate; no minimum performance was assumed in advance.
''')
page(r'''\section{Study architecture and data provenance}
\subsection{End-to-end information flow}
\begin{center}\fbox{\begin{minipage}{.88\linewidth}\centering
Immutable ADNI tables + A/B feature handoffs\\$\downarrow$\\
Subject/date validation + feature corrections + diagnosis trajectories\\$\downarrow$\\
One-row-per-subject matrix + eligibility and label audit\\$\downarrow$\\
Development $\mid$ Calibration $\mid$ Final validation\\$\downarrow$\\
Fold-local model selection $\rightarrow$ frozen calibrated predictor\\$\downarrow$\\
Held-out metrics + local prediction + reproducibility checks
\end{minipage}}\end{center}
The raw files and original handoffs are read-only inputs to the implementation. Generated subject tables, provenance, split IDs, individual predictions, and model binaries are written to an ignored local directory. Public outputs contain code, configuration, synthetic tests, and aggregates. This separation makes experiments reviewable without redistributing participant records.
\subsection{Sources actually used}
'''+table(['Input','Role'],[(r'\code{person_A_features.csv}','29 supplied demographic/genetic, CSF, cognitive, MRI and PET fields'),(r'\code{person_B_features.csv}','31 supplied ECog, neuropsychological and history fields'),('Shared baseline list','RID and original baseline date'),('ADNIMERGE','Baseline identity, date, diagnosis and enrollment phase'),('DXSUM','Longitudinal diagnosis target construction only'),('NEUROBAT / MEDHIST','Dated corrections to supplied features'),('Old outcome labels','Reconciliation audit only; not training truth')],'p{.36\\linewidth}p{.55\\linewidth}')+r'''The local inventory comprises 17 source CSV tables, including TADPOLE subsets, dictionaries, MRI extracts, and PACC helpers. The main ADNIMERGE table has 13,915 rows and 113 columns; DXSUM has 16,423 rows and 41 columns. Their sizes describe raw snapshots, not the final training sample.

The data resources cover successive ADNI phases. The eligible binary cohort in this run contains ADNI1, ADNI-GO, ADNI2, and ADNI3 index records; the presence of newer downloaded resources does not imply ADNI4 representation in training. TADPOLE-distributed resources are acknowledged as source material, not independent external validation \cite{tadpole}.
''')
page(r'''\section{Constructing the incident-dementia cohort}
\subsection{Index records and observation window}
Let $t_i$ be participant $i$'s original baseline date, and let $h_i$ be its fifth calendar anniversary. Filtering ADNIMERGE to \code{VISCODE == bl} yields 2,155 unique RIDs. Their dates must agree exactly with the shared subject list. ADNIMERGE dates are parsed as day--month--year, while newer tables use ISO year--month--day strings. Explicit formats avoid silent swaps of day and month.

Baseline CN, SMC, EMCI, and LMCI records are eligible for incident prediction. Baseline AD, missing or unrecognized baseline diagnoses, and any dementia recorded at or before index are excluded. Baseline diagnosis is used for eligibility and subgroup reporting, never as a model input. The eligibility declaration accepted by the prediction interface follows the same principle.
\subsection{Trajectory-to-label rules}
Use valid DXSUM diagnoses 1 (unimpaired), 2 (MCI), and 3 (dementia), ordered by examination date \cite{adni}. The binary outcome is
\begin{equation}
y_i=\begin{cases}
1, & \exists\,d\in(t_i,h_i]\text{ with diagnosis }3,\\
0, & \text{no observed within-window event and valid diagnosis at/after }h_i,\\
\mathrm{unknown}, & \text{otherwise.}
\end{cases}
\end{equation}
An event exactly on the fifth anniversary is positive; one the following day is outside the outcome window. A subsequent reversion to MCI or CN does not erase an already observed dementia event. Calendar offsets correctly handle leap-day index dates.

Agreeing duplicate diagnoses on the same date do not change the outcome. Conflicting valid same-day diagnoses quarantine the participant, as does an undated dementia record. Each participant receives one recorded disposition according to explicit exclusion precedence. Missing labels are not imputed, assigned zero, or used to train the classifier.

A valid observation after five years supports a negative label only in conjunction with no observed earlier dementia event. It does not prove continuous disease-free observation. A first diagnosis after the window may represent later conversion or delayed detection; visits provide interval-observed clinical evidence, not exact biological onset times.
\subsection{Why rebuilding labels mattered}
The supplied label table contained 694 positives, but 363 of those participants already had baseline AD. A preliminary reconstruction also identified 22 old negatives without enough follow-up. Training directly on these labels would alter the task and inflate apparent prognostic performance. The corrected output preserves each index date, event date, last diagnosis date, and exclusion reason so the decision can be traced.
''')
page(r'''\subsection{Cohort flow and observed population}
'''+table(['Disposition','Participants'],[(k.replace('_',' '),v) for k,v in audit['cohort_flow'].items()],'p{.70\\linewidth}r')+r'''The known-outcome sample contains 1,056 participants, 49.0\% of the original baseline pool. Its event fraction is $324/1056=30.68\%$. Excluding short-follow-up subjects is a major population restriction rather than a routine missing-value operation.
'''+table(['Baseline group','Nonconverter','Converter','Total'],[('CN',299,6,305),('SMC',162,2,164),('EMCI',146,42,188),('LMCI',125,274,399)])+table(['Baseline characteristic','Nonconverter median','Converter median'],[('Age (years)','71.8','74.4'),('Education (years)','16','16'),('MMSE','29','27'),('CDR sum of boxes','0','2')],'p{.40\\linewidth}rr')+r'''These are descriptive full-cohort summaries, not adjusted associations or inputs to model selection. The cohort has 492 female and 564 male participants. Phase composition is ADNI1 393, ADNI2 353, ADNI3 225, and ADNI-GO 85. Differences in disease stage and phase can influence apparent pooled discrimination.

Most observed converters enter with MCI, particularly LMCI. Therefore, pooled performance must not be read as evidence that the model detects preclinical conversion equally well in CN/SMC. The final validation subgroup analysis makes this restriction especially visible. Descriptive profiling does not justify adding baseline diagnosis to the predictors after the study has been defined.
''')
page(r'''\section{Building a trustworthy predictor matrix}
\subsection{Handoff contract and relational integrity}
Person A provides 29 predictors and Person B 31. Both handoffs contain the same 2,155 unique RIDs, with no shared predictor names. Validation checks exact columns, nonmissing unique identifiers, and subject-set equality. A one-to-one join produces 60 source features; label and audit metadata are joined separately. Retaining rows with unavailable predictors avoids silently selecting complete subjects during the merge.

Many source tables contain repeated visits or repeated conditions. A RID-only join before reducing each source to one eligible record can multiply observations: if a subject has $a$ rows in one source and $b$ in another, their join can produce $ab$ rows. Temporal selection, condition aggregation, and merge-cardinality assertions address this failure before fitting any estimator.
\subsection{Temporal feature corrections}
Repeated ADNIMERGE measures use their \code{_bl} columns from the original baseline record. Follow-up values at month 6, 12, or later cannot be substituted simply because they are more complete. Their inclusion would let the model observe part of the progression it is supposed to predict.

NEUROBAT accepts translated screening or baseline records dated on index day or within the preceding 90 days. Logical Memory prefers screening; other selected tests prefer baseline. Eligible alternatives are ordered deterministically by preferred visit and recency. Conflicting same-priority values on the same date become missing. Compared with the supplied B handoff, 30--120 cells per neuropsychological feature changed. Per-feature selected dates and conflict decisions are retained locally.

For each selected MEDHIST condition, any explicit pre-index positive yields one; documented eligible absence yields zero; no usable pre-index history yields missing. The original B handoff had treated unavailable history as zero. Restoring 417 subjects to missing for each assigned history field prevents lack of documentation from masquerading as absence of disease. \code{MH3HEAD} denotes head/eyes/ears/nose/throat history, not a specific head-trauma diagnosis.
\subsection{Laboratory and numeric handling}
Strings such as \code{>1700} are represented by the stated boundary plus a signed bound indicator ($-1$, $0$, or $1$). They are not silently recoded as exact laboratory measurements. Three indicators are generated for CSF assays. Field-specific sentinel handling preserves valid negative RAVLT change scores; a universal ``negative means missing'' rule would corrupt them.
''')
page(r'''\subsection{Feature panels, missingness, and deferred sources}
'''+table(['Panel','Source fields','Composition'],[('Clinical',14,'Age, sex, education, APOE4 and ten cognitive measures'),('Core',28,'Clinical + three CSF + seven MRI + FDG + three history flags'),('Expanded',60,'Core + supplementary demographics, ECog, NEUROBAT, history and AV45')],'p{.15\\linewidth}rp{.64\\linewidth}')+r'''The earlier audit proposed 34 Tier 1 and 33 Tier 2 candidates. Seven plasma features were deferred because an earliest available blood draw is not necessarily pre-index. The implemented panels are therefore 14/28/60, not fitted 34/67-feature experiments. Source fields and model dimensions must also be distinguished: the final core estimator retains 28 source fields, two nonconstant bound flags, and 60 transformed inputs after encoding and missing-indicator expansion.
\fig{missingness.png}{.88}{Missing fractions for the 25 least complete source predictors in the eligible known-outcome cohort. This descriptive plot does not determine validation-time feature removal.}
The two least ambiguous choices for redundant data were to use selected composite measures rather than granular ROI duplicates, and to avoid reintroducing challenge targets or item-level totals already represented elsewhere. PACC helpers, redundant MRI extracts, and free-text history were not needed by the implemented pipeline. Plasma feasibility and unstructured text require separate studies.

The audit also contains a bookkeeping discrepancy: $3066-34-33=2999$ fields lie outside the candidate set, whereas its Tier 3 heading states 2,943. This is documented rather than treated as a verified exhaustive partition. Candidate selection is a prior specification; it does not prove predictive utility or eliminate overfitting. Only development comparisons and independent validation address those questions.
''')
page(r'''\section{Experimental design and leakage controls}
\subsection{Three partitions with distinct roles}
The binary cohort is sorted by RID and split with seed 42 and outcome stratification. First reserve 20\% for final validation, then reserve one quarter of the remainder for calibration. Integer rounding yields the following counts.
'''+table(['Partition','Subjects','Converters','Nonconverters'],[('Development',633,194,439),('Calibration / threshold',211,65,146),('Final validation',212,65,147)])+r'''Development supports preprocessing choices, hyperparameters, panel selection, and threshold experiments. Calibration supports the final probability mapping and operating threshold. Final validation estimates performance after these choices have been frozen. No scaling, imputation, or missingness-based field removal is fitted on calibration or final-validation records.
\subsection{Five-fold development comparison}
StratifiedKFold creates five fixed development folds. With one row per RID, stratified row splitting is also subject splitting; explicit checks establish RID disjointness. A grouped splitter would be necessary if repeated visits were introduced. The experiment does not claim that GroupKFold was executed when it was not.

Within each outer training fold, a seeded stratified 25\% holdout is used to select the threshold. Fit a candidate on the remaining inner records, maximize $F_1$ on that inner holdout, refit the candidate on all outer-training records, and score the untouched outer fold at the chosen threshold. This protects outer-fold outcomes from threshold optimization, although refitting can change the score scale. Final calibration/threshold fitting handles the final deployed scale separately.

All learned transformations belong to the candidate pipeline and are refitted for each inner or outer training sample. The same outer folds permit paired comparison across panels and estimators. Selection uses mean fold $F_1$, then recall, then estimator simplicity. This is development cross-validation with inner threshold holdouts, not a fully nested hyperparameter evaluation. Searching configurations can make the selected mean optimistic; the final held-out partition remains essential.
\subsection{Leakage checklist implemented in code}
Diagnostic fields and follow-up dates are kept outside the feature allowlist; source records are date-filtered before merging; repeated sources are reduced before one-to-one joins; fitted preprocessing sees training records only; and final-validation opening writes a marker that blocks subsequent preparation and training in that experiment directory. These controls address complementary temporal, relational, and statistical risks.
''')
page(r'''\section{Preprocessing and model development}
\subsection{Training-dependent feature and missing-data treatment}
Within a fitted training sample, remove source/derived fields with more than 80\% missingness and fields with no observed variation. The retained field list is stored and applied unchanged to later records. A feature that happens to become complete or variable in final validation is not reinstated. Missing values in eligible core predictors do not automatically remove participants after cohort definition.

The median comparator imputes numeric fields with training medians, adds indicators, and standardizes them. The selected alternative first standardizes observed numeric values using training means and variances, then performs distance-weighted KNN imputation with $k=5$. Missing indicators are appended separately. Categorical missing values receive a dedicated category; one-hot encoding ignores previously unseen categories safely.

For two partially observed standardized vectors, the distance is based on their jointly observed coordinates and adjusted for the number of coordinates available. Informally, a missing value is estimated as
\begin{equation}\widehat z_{ij}=\frac{\sum_{r\in\mathcal N_{ij}}w_{ir}z_{rj}}{\sum_{r\in\mathcal N_{ij}}w_{ir}},\qquad w_{ir}\propto d(i,r)^{-1},\end{equation}
where neighbors come from the fitted training set. KNN on unscaled volumes and cognitive scores would weight units rather than similarity. Calibration and validation records never become fitted neighbors for each other.
\subsection{Candidate estimator families}
Logistic regression uses the transformed vector $z$ to estimate
\begin{equation}q(x)=\sigma(\beta_0+z^\top\beta),\qquad \sigma(u)=\frac{1}{1+e^{-u}}.\end{equation}
Training minimizes weighted log loss with $L_2$ regularization; smaller $C$ implies stronger shrinkage. Balanced weights are computed from the relevant fitted training sample. Regularization is particularly useful when cognitive measures and imaging features are correlated and only 194 development converters are available.

Random forest averages randomized decision trees and controls complexity using depth, leaf size, and feature subsampling. XGBoost fits a regularized additive tree model by sequential boosting \cite{xgb}. CatBoost provides another boosted-tree comparator. It uses this study's common numeric/one-hot preprocessing, so its evaluation is not a specialized native-categorical treatment. A majority-class predictor anchors interpretation when accuracy alone looks deceptively high.
''')
page(r'''\subsection{Search budget and selected configuration}
Eight sampled configurations per model/panel were compared, below the configured cap of 30. The saved record contains seven initial baselines, 96 sampled comparisons, and six preprocessing alternatives: 109 records in total. This count includes repeated or closely related settings and is not a count of 109 fundamentally different algorithms.
'''+table(['Family','Searched dimensions'],[('Logistic','Regularization C and none/balanced class weights'),('Forest','Depth, minimum leaf size, feature fraction and class weights'),('XGBoost','Tree count/depth, learning rate, row/column sampling, regularization, minimum child weight and positive weight'),('CatBoost','Iterations, depth, learning rate, leaf regularization and class weights')],'p{.20\\linewidth}p{.71\\linewidth}')+r'''The three strongest development candidates also compared KNN imputation and MRI normalization by ICV. The selected configuration was logistic regression with $C=0.01$, balanced weights, the core panel, and KNN imputation. MRI/ICV normalization was not selected. Logistic regression used a maximum of 3,000 iterations and seed 42.
\subsection{Calibrating the final risk score}
Refit the chosen pipeline on all 633 development subjects. Clip its raw probability $q$ to $[10^{-6},1-10^{-6}]$ and form $u=\log(q/(1-q))$. Fit the reserved calibration subjects with
\begin{equation}p=\sigma(au+b),\qquad a\approx1.4433,\quad b\approx-0.7855.\end{equation}
This sigmoid mapping is fitted using logistic regression with $C=10^6$; a negative fitted slope triggers an error. Calibration improves empirical probability alignment but cannot certify a population-wide ``true risk'' \cite{calibration}. Class weighting during classifier fitting makes a separate probability check especially useful.

Choose the threshold on calibrated calibration-set probabilities by maximizing
\begin{equation}F_1=\frac{2TP}{2TP+FP+FN},\qquad \widehat y=\mathbf1[p\geq\tau].\end{equation}
Ties favor greater recall. The resulting $\tau=0.5130783821$ is saved with the estimator. The probability mapping and threshold use the same reserved partition, so its reported fit statistics are not independent performance evidence. No model is refitted using final-validation outcomes.

The saved artifact contains the fitted feature selection, preprocessing, classifier, calibrator, threshold, selected specification, and version. Thus a command-line prediction applies the complete training-time transformation rather than rebuilding preprocessing from the incoming record.
''')
best=[]
for panel in ['clinical','core','expanded']:
 for model in ['logistic','forest','xgboost','catboost']:
    win=max([r for r in records if r['spec']['panel']==panel and r['spec']['model']==model],key=lambda r:r['mean']['f1'])
    best.append((panel,model,*[f"{win['mean'][k]:.3f}" for k in ['f1','precision','recall']]))
page(r'''\section{Results and interpretation}
\subsection{Development comparisons}
'''+table(['Panel','Estimator','Mean $F_1$','Precision','Recall'],best,'llrrr')+r'''Each row is the highest recorded development $F_1$ for that estimator/panel, including its eligible preprocessing alternatives. These are selected development maxima, not independent held-out comparisons. The expanded panel is not uniformly better despite its additional features: extra measurements can add missingness, correlation, and estimation variance as well as signal.

For the winning core logistic model, mean fold accuracy is 0.9179, precision 0.8411, recall 0.9074, $F_1=0.8721$, and ROC AUC 0.9675. Fold standard deviation of $F_1$ is 0.0442. These values summarize development-fold behavior; they are not confidence intervals for the final model.
'''+table(['Matched core logistic specification','Mean development $F_1$'],[('Median imputation; raw MRI','0.86705'),('Scaled KNN imputation; raw MRI','0.87213'),('Median imputation; MRI/ICV','0.86712')],'p{.70\\linewidth}r')+r'''The matched comparison fixes $C=0.01$ and balanced weights. Its approximately 0.0051 KNN advantage is modest, not proof of universal imputation superiority. Only three strongest development candidates received these alternatives; this is a bounded, recorded comparison rather than an exhaustive factorial experiment.

The clinical panel contains genotype and core cognition, while the core panel adds fluid and imaging information. Because different measurements can be imputed and participation is selective, the result should not be presented as a clean clinical cost-effectiveness experiment or as evidence for a plasma assay. It establishes performance for the stated panels on the common eligible population under this pipeline.
''')
metrics=[]
for k,label in [('accuracy','Accuracy'),('precision','Precision'),('recall','Recall / sensitivity'),('f1','$F_1$'),('specificity','Specificity'),('balanced_accuracy','Balanced accuracy'),('roc_auc','ROC AUC'),('average_precision','Average precision'),('brier','Brier score')]:
    lo,hi=results['confidence_intervals_95'][k];metrics.append((label,f"{results['primary'][k]:.4f}",f'{lo:.4f}--{hi:.4f}'))
page(r'''\subsection{Untouched final-validation performance}
After freezing the artifact and threshold, final validation evaluates 212 participants with 65 converters and 147 nonconverters. The results below are the primary performance evidence.
'''+table(['Metric','Estimate',r'95\% bootstrap interval'],metrics)+table(['Actual class','Predicted nonconverter','Predicted converter'],[('Nonconverter',136,11),('Converter',11,54)],'lrr')+r'''Precision is $54/(54+11)$, recall is $54/(54+11)$, specificity is $136/(136+11)$, and accuracy is $190/212$. Precision and recall coincide because FP and FN counts are equal; this is a property of this sample, not a design guarantee. The model misses 11 converters and flags 11 nonconverters at the frozen threshold.

The majority-class comparator yields 69.34\% accuracy and zero recall. The selected model therefore improves positive-case detection, not just overall accuracy. Threshold 0.5 happens to produce the same final-validation decisions as the calibration-selected threshold. It was not adopted after inspecting validation.

The intervals use 2,000 subject bootstrap samples, resampling within each observed class with seed 42. They hold the estimator and threshold fixed and condition on held-out class counts. They measure sampling uncertainty in this validation sample, not variability across retraining, cohort redefinition, or external populations.

ROC AUC measures ranking over thresholds; average precision summarizes the precision--recall staircase and is not trapezoidal PR AUC. AP should be interpreted against the positive fraction of 0.3066. Final $F_1$ (0.8308) is below the selected development mean (0.8721), reinforcing the need to report independent validation rather than a search maximum.
''')
page(r'''\subsection{Probability quality and score distributions}
\fig{validation_plots.png}{.88}{Frozen final-validation discrimination, quantile-bin calibration, and score distributions. The decision threshold was fixed before these records were evaluated.}
The probability distributions show useful separation but also overlap. A high ROC AUC does not remove the need to choose an operating threshold, and it does not imply that each individual probability is accurate. Reliability bins summarize empirical agreement but can conceal local errors and depend on how probabilities are grouped.
'''+table(['Frozen-score diagnostic','Raw classifier','Sigmoid-calibrated'],[('Brier score','0.07714','0.06691'),('ECE, ten equal-width bins','0.09183','0.02940')],'p{.49\\linewidth}rr')+r'''Brier score is $N^{-1}\sum_i(p_i-y_i)^2$. Expected calibration error weights each nonempty bin's absolute observed-minus-predicted frequency gap by its sample fraction. The figure uses quantile bins, whereas ECE here uses ten fixed-width bins. These different binning choices are intentional and must not be conflated.

The before/after diagnostics were computed from the same frozen validation predictions for documentation. No calibrator or threshold was fitted using these records. Their improvement is descriptive evidence in this sample, not a significance claim or permission to retune the model. Calibration learned in ADNI may change under different prevalence, recruitment, missingness, or assay procedures.
''')
page(r'''\subsection{Subgroups, coefficients, and errors}
All 65 final-validation converters are in the MCI subgroup ($N=125$). Its precision is 0.8438, recall 0.8308, $F_1=0.8372$, and ROC AUC 0.9277. The CN/SMC subgroup has 87 participants and no converter, so sensitivity and positive-case discrimination cannot be estimated there. ADNI3 validation also has no events; ADNI-GO has only two. Pooled accuracy cannot resolve these subgroup evidence gaps.
\fig{coefficients.png}{.82}{Largest absolute coefficients in the selected logistic model. Numeric-value terms are on the training-standardized scale; the plot describes model associations, not causal effects.}
Functional impairment (FAQ), CDR sum of boxes, and ADAS13 have positive coefficients, while FDG, hippocampal volume, and immediate verbal recall have negative coefficients. These signs are compatible with expected impairment patterns, conditional on the other transformed predictors. Correlation can redistribute coefficients, so their ordering is not a stable causal ranking. Calibration acts on the combined score; coefficients are not additive percentage-point effects.
'''+table(['Error group','n','Baseline composition','Age median','MMSE median'],[('False positive',11,'8 LMCI, 2 EMCI, 1 CN','73.5','27'),('False negative',11,'8 LMCI, 3 EMCI','69.7','28')],'lrlrr')+r'''Both groups have median baseline CDRSB 1.5. These small aggregate summaries do not establish a clinical mechanism for each error. A missed five-year converter is a false negative; conversion only after year five is not, by itself, a false negative for this endpoint. The report publishes no individual participant explanations or records.
''')
page(r'''\section{Software implementation and prediction interface}
\subsection{Package organization}
The implementation is a Python package named \code{btp}, with configuration outside code and source inputs outside version control. Its modules separate responsibilities rather than combining preparation and evaluation into an opaque notebook.
'''+table(['Subsystem','Responsibility'],[(r'\code{schema, cohort}','Predictor contract, incident labels and eligibility'),(r'\code{prepare}','Dated corrections, strict merge and provenance'),(r'\code{preprocessing, splits}','Training-only transforms and persistent partitions'),(r'\code{training, search}','Common folds, experiments, selection and final fitting'),(r'\code{artifact}','Serialized predictor and baseline input validation'),(r'\code{evaluation, reproduce}','Frozen metrics, aggregate export and independent rebuild')],'p{.37\\linewidth}p{.54\\linewidth}')+r'''The experiment runner checkpoints completed configuration records, reports fold metrics, and records the split hash. Saved model, matrix, and selection hashes are checked before final validation. A validation-opening marker prevents preparation or training from overwriting that run. These are safeguards against accidental reuse, not a security boundary against deliberate manual modification.
\subsection{Command-line workflow}
\begin{verbatim}
uv sync --locked
uv run --locked python -m btp prepare
uv run --locked python -m btp train
uv run --locked python -m btp evaluate
uv run --locked python -m btp predict --input examples/baseline.json
uv run --locked python -m btp reproduce
\end{verbatim}
The default configuration uses \code{../data} for authorized local inputs and \code{local} for generated outputs. Paths resolve relative to the configuration file. A new configuration/local directory is necessary for a distinct exploratory experiment. Repeated evaluation of the original frozen run returns its saved result rather than silently selecting another model.

Prediction accepts a baseline JSON object or nonempty list of objects. Source names and source units must be retained. Optional missing measurements use the saved preprocessing. Unknown fields, nonfinite numeric values, selected impossible ranges, invalid APOE4 counts, and ineligible baseline diagnoses are rejected. At least three measured source fields are required, but that minimal input check is not evidence that every three-field pattern yields a reliable risk estimate.
''')
page(r'''\subsection{Prediction semantics, verification, and data stewardship}
Outputs include a probability (\path{risk_probability}) and percentage (\code{risk_percent}). It also includes the threshold, binary decision (\path{predicted_positive}), and model version. A synthetic 72-year-old LMCI example with selected cognitive observations yields probability about 0.3656, below the frozen threshold 0.5131. It illustrates execution, not clinical validation of that person's risk. Baseline diagnosis is an eligibility declaration and is not passed to the classifier.

The artifact bundles transformations with the classifier so that input records are not standardized or imputed afresh at inference. Serialization tests verify equality before and after reloading. Only trusted locally generated joblib artifacts should be loaded. No web interface, clinical deployment, or authentication service is implemented.
\subsection{Tests and independent reproduction}
Thirteen synthetic tests cover calendar boundaries, baseline dementia exclusion, short follow-up, agreeing and conflicting visits, undated dementia, laboratory strings, unavailable history, date-safe NEUROBAT selection, handoff contracts, fold-local preprocessing, unseen categories, serialization, and the complete preparation-to-evaluation workflow. The integration fixture contains synthetic records rather than redistributed ADNI data. GitHub CI executes the synthetic suite.

The reproduction command launches the selected preparation and fitting workflow into a separate local directory. It rebuilds the 2,155-row baseline matrix from original inputs, recreates fixed subject partitions, refits the chosen development model, and recalibrates using the same reserved subjects. The resulting matrix and split files match their original hashes; the threshold is identical and probabilities agree within $10^{-12}$. This is stronger evidence than merely reloading a saved model, although it does not measure performance on a new cohort.
\subsection{Privacy and experiment evidence}
Input filenames and SHA-256 hashes identify the source snapshot. Local provenance maps selected features and labels back to subjects and dates. Patient rows, split IDs, individual predictions, model binaries, and original source documents remain untracked. Aggregate JSON results, plots, software, configuration, and synthetic examples are published through an explicit allowlist.

The public repository was created before implementation and updated in small milestone commits. A locked Python 3.12 environment records tested dependencies. This combination supports review, rerunning, and collaboration without treating the downloadable vendor packages or planning PDFs as necessary application code. Original input-file hashes were checked after the run and were unchanged.
''')
page(r'''\section{Discussion, limitations, and conclusions}
\subsection{What explains the outcome}
The project shows that carefully prepared baseline summaries support strong discrimination in the eligible ADNI cohort, and that a strongly regularized linear classifier can outperform the tested tree ensembles under the chosen criterion. This does not imply universal superiority of logistic regression. The result depends on cohort construction, correlated summaries, search budget, panel definitions, preprocessing, and the operating objective.

Correcting labels and enforcing timing were fundamental. Existing baseline disease would otherwise make the prediction problem easier for the wrong reason. Preserving absent history as missing and fitting preprocessing only on training data protect validity even when they do not increase a score. These decisions make the performance interpretable as the stated research experiment.
\subsection{Limits of the evidence}
Only 1,056 of 2,155 baseline participants have accepted binary outcomes. Excluding insufficient follow-up can select for health, attendance, enrollment era, and survival. Missing biomarkers may also be informative. KNN does not guarantee unbiased recovery when missingness depends on unobserved disease severity. A complete-case analysis would introduce a different selection mechanism rather than automatically solve the problem.

ADNIMERGE baseline suffixes establish merged-table provenance, but original acquisition timestamps are not supplied for every retained assay and imaging composite. Diagnosis visits do not reveal exact onset time. Conservative ambiguity exclusions reduce sample size and may alter the population. ADNI recruitment and phase composition also differ from routine care.

The final-validation sample is modest, and its event-free CN/SMC subgroup prevents a sensitivity claim for initially unimpaired converters. Class-count-conditional bootstrap intervals do not include model-development variability. The dedicated calibration set is small and serves two fitted purposes, calibration and threshold selection. Searching many configurations introduces optimism into development maxima even with fold-local preprocessing.
\subsection{Next steps and conclusion}
The priority is independent temporal or external validation with sufficient CN/SMC events, followed by subgroup and missingness-shift analyses. A prespecified survival study could retain censored participants; matched-subject modality and imputation studies could clarify incremental value. These are extensions, not hidden components of the completed classifier.

The implemented artifact achieves 89.62\% untouched-validation accuracy and 83.08\% precision and recall for observed five-year any-dementia conversion. Its substantive contribution is the complete auditable chain from source records to calibrated output, supported by tests, manifests, and a fresh-process rebuild. The result is a credible research baseline with explicit boundaries, not a clinical deployment claim.
''')
page(r'''\appendix\section{Implemented feature and reproduction reference}
\subsection{Source-feature contract}
The table identifies all 60 assigned source predictors by group. ECog domain names expand exactly as shown. The final chosen core panel comprises the first four demographic/genetic fields, all ten core cognitive fields, three CSF fields, seven MRI fields, FDG, and the first three history flags.
'''+table(['Group','Exact source fields'],[('Demographics / genetics',r'\code{AGE, PTGENDER, PTEDUCAT, APOE4}; \code{PTRACCAT, PTETHCAT, PTMARRY}'),('CSF',r'\code{ABETA_bl, TAU_bl, PTAU_bl}'),('Core cognition',r'\code{CDRSB_bl, ADAS11_bl, ADAS13_bl, MMSE_bl}; \code{RAVLT_immediate_bl, RAVLT_learning_bl}; \code{RAVLT_forgetting_bl, RAVLT_perc_forgetting_bl}; \code{FAQ_bl, MOCA_bl}'),('MRI',r'\code{Hippocampus_bl, Ventricles_bl, WholeBrain_bl}; \code{Entorhinal_bl, Fusiform_bl, MidTemp_bl, ICV_bl}'),('PET',r'\code{FDG_bl, AV45_bl}'),('ECog (14)',r'\code{EcogPt} and \code{EcogSP} each followed by \code{Mem, Lang, Visspat, Plan, Organ, Divatt, Total}, then \code{_bl}'),('NEUROBAT (10)',r'\code{TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC}; \code{CATANIMSC, CLOCKSCOR, LIMMTOTAL, LDELTOTAL}; \code{BNTTOTAL, DIGITSCOR}'),('History (7)',r'\code{MH4CARD, MH9ENDO, MH16SMOK}; \code{MHPSYCH, MH3HEAD, MH14ALCH, MH2NEURL}')],'p{.22\\linewidth}p{.69\\linewidth}')+r'''\subsection{Selected artifact and software}
Core logistic regression: $C=0.01$, balanced weights, 3,000 maximum iterations; KNN $k=5$, distance weighting after standardization; no MRI/ICV normalization; sigmoid calibration $C=10^6$; threshold 0.5130783821; seed 42. Final transformed dimension: 60. This count includes missing indicators and encoded terms, not 60 selected original measurements.
'''+table(['Component','Version'],[('Python','3.12.10')]+list(versions.items())+[('matplotlib / pytest','3.10.3 / 8.3.5')],'p{.45\\linewidth}l')+r'''Use the repository README for complete commands and authorized source-file locations. The input and model hashes in \path{results/run_summary.json} identify the recorded snapshot; \path{results/development_comparison.json} stores the comparisons, and \path{results/validation_results.json} stores the frozen metrics and intervals.
''')
page(r'''\begin{thebibliography}{9}\addcontentsline{toc}{section}{References and evidence record}
\bibitem{adni} Alzheimer's Disease Neuroimaging Initiative. \emph{Diagnostic Information: harmonized DXSUM diagnosis}. Official documentation. \url{https://adni.loni.usc.edu/quick-start-guide-asset101625/diagnostic.html}. Accessed 2 November 2026.
\bibitem{tadpole} EuroPOND / TADPOLE consortium. \emph{The Alzheimer's Disease Prediction Of Longitudinal Evolution Challenge}. Official data and challenge resources. \url{https://tadpole.grand-challenge.org/}.
\bibitem{xgb} T. Chen and C. Guestrin. \emph{XGBoost: A Scalable Tree Boosting System}. 2016. \url{https://arxiv.org/abs/1603.02754}.
\bibitem{calibration} scikit-learn developers. \emph{Probability calibration}. Official user guide. \url{https://scikit-learn.org/stable/modules/calibration.html}. Software versions used in the experiment are recorded in the appendix.
\bibitem{pipeline} scikit-learn developers. \emph{Common pitfalls and recommended practices}. Official guidance on preprocessing and leakage. \url{https://scikit-learn.org/stable/common_pitfalls.html}.
\bibitem{project} Project source and experiment records. \emph{ADNI dementia prediction}. November 2026. \url{https://github.com/soumil-kumar/adni-dementia-prediction}. Primary records: data audit, feature registry, development comparisons, selection, final validation, and run summary under \code{results/}.
\end{thebibliography}
\subsection*{Evidence statement}
The thesis is grounded in the implemented Python source, frozen model, saved metrics, and reproduction records. No classifier was retrained or threshold changed while preparing this report. Cohort profile tables and coefficient/error summaries are descriptive documentation. Calibration diagnostics computed after final validation are identified as supplemental analyses and do not constitute independent confirmatory studies.

The local corrected feature audit and two-person assignment informed candidate selection and handoff ownership. Their historical feature counts are distinguished from the actual fitted panels. Report content is organized around the completed implementation, not an aspirational list of analyses.

ADNI and TADPOLE data remain subject to their applicable access and reuse conditions. Author identity, institutional declarations, and supervisor certification have not been invented. Add required institutional front matter before formal submission.
''')
parts.append(r'\end{document}')
tex=''.join(parts).replace('95\\% bootstrap interval',r'95\% bootstrap interval')
(root/'BTP_Thesis.tex').write_text(tex,encoding='utf-8')
print('Generated implementation-centred thesis')
