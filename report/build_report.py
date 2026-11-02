"""Build editable LaTeX from recorded aggregate results; no training occurs here."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
repo=root.parent
read=lambda p:json.loads(p.read_text())
results=read(repo/'results/validation_results.json')
records=read(repo/'results/development_comparison.json')['records']
diag=read(root/'diagnostics.json')
inventory=read(repo.parent/'full_inventory.json')
def esc(x):
    return str(x).replace('\\', '/').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(headers,rows,widths=None):
    spec=widths or ('l'+'r'*(len(headers)-1))
    return '\\begin{center}\\small\n\\begin{tabular}{'+spec+'}\\hline\n'+' & '.join(headers)+r'\\\hline'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\hline\\end{tabular}\\end{center}\n'
parts=[r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,longtable}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,citecolor=blue!45!black]{hyperref}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small ADNI five-year dementia prediction}\fancyhead[R]{\small BTP technical report}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{3em}
\newcommand{\field}[1]{\texttt{\detokenize{#1}}}
\newcommand{\fig}[3]{\begin{figure}[!ht]\centering\includegraphics[width=#2\linewidth]{figures/#1}\caption{#3}\end{figure}}
\begin{document}
\begin{titlepage}\centering
\vspace*{22mm}
{\Large Bachelor of Technology Project\par}\vspace{14mm}
{\Huge\bfseries Predictive Modeling of\par Neurocognitive Disease\par Progression\par}\vspace{10mm}
{\LARGE Five-year dementia prediction from\par baseline ADNI tabular biomarkers\par}\vspace{18mm}
{\large Technical thesis report\par 2 November 2026\par}\vfill
\begin{minipage}{.88\textwidth}
\textbf{Implemented endpoint:} any incident dementia within five calendar years, among participants without dementia at their original baseline. Alzheimer-attributed dementia is a narrower endpoint and is not separately established by this classifier.

\textbf{Evidence base:} frozen model, immutable input snapshots, recorded development comparisons, untouched validation, and independent fresh-process reproduction.

\textbf{Repository:} \url{https://github.com/soumil-kumar/adni-dementia-prediction}
\end{minipage}\vspace{12mm}
\end{titlepage}
''']
def page(body):parts.append(body+'\n\\clearpage\n')
page(r'''\section*{Abstract}\addcontentsline{toc}{section}{Abstract}
This project develops a reproducible, baseline-only machine-learning pipeline for estimating the probability of dementia within five years using tabular data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The original project motivation emphasizes Alzheimer's disease (AD), but the implemented outcome is \emph{any dementia}, defined by the harmonized DXSUM diagnosis, rather than AD-specific etiology. Raw MRI and PET images are excluded; numeric imaging composites remain eligible predictors.

Seventeen local source tables and an earlier 3,066-field audit informed a 67-candidate feature specification. Seven plasma candidates were deferred, leaving 60 predictors divided between two feature-extraction handoffs. These handoffs were validated, corrected for assessment timing and unavailable medical history, and merged without duplicate subjects. Of 2,155 baseline participants, 1,056 had defensible binary outcomes: 324 incident dementia cases and 732 observed five-year nonconverters. Baseline dementia, insufficient observation, ambiguous diagnoses, and other eligibility failures were excluded.

The labeled cohort was divided into development (633), calibration (211), and final validation (212) subjects. Five development folds compared logistic regression, random forest, XGBoost, and CatBoost across clinical, core, and expanded panels. Learned preprocessing remained inside training folds, and inner training holdouts selected thresholds. Across 109 recorded comparisons, the selected model was balanced, strongly regularized logistic regression with scaled K-nearest-neighbor imputation on the core panel. A reserved calibration partition fitted sigmoid calibration and selected an $F_1$ threshold of 0.513078.

The frozen model achieved validation accuracy 89.62\%, precision 83.08\%, recall 83.08\%, specificity 92.52\%, and $F_1=0.8308$. ROC AUC was 0.9667; average precision, used here as the precision--recall summary, was 0.9250; Brier score was 0.06691. The confusion matrix contained 136 true negatives, 11 false positives, 11 false negatives, and 54 true positives. C-index is not reported because a survival estimator was not fitted. Supplemental calibration diagnostics were computed from frozen predictions without retuning.

Thirteen synthetic tests and repository CI passed. Independent reproduction recovered identical prepared data and subject partitions and reproduced probabilities within $10^{-12}$. Important limitations are right-censoring selection, limited external validity, uncertain acquisition timing for some merged baseline fields, and no converters in the validation CN/SMC subgroup. These results demonstrate a reproducible research workflow; they do not establish a clinically deployable probability for the general population.

\textbf{Keywords:} ADNI; dementia prognosis; baseline biomarkers; temporal leakage; missing data; logistic regression; calibration; subject-level validation.
''')
page(r'''\setcounter{tocdepth}{1}\tableofcontents
\vfill\textbf{Reading guide.} Chapters 1--5 define the implemented task and methods. Chapter 6 separates development comparisons from final validation. Chapter 7 identifies limitations and unimplemented extensions. The appendix specifies all 67 audited candidates and the reproducibility contract. All page counts include this front matter, appendices, and references.
''')
abbr=[('AD / NCD','Alzheimer\textquotesingle s disease / neurocognitive disease'),('CN / SMC','Cognitively normal / significant memory concern'),('MCI; EMCI / LMCI','Mild cognitive impairment; early / late MCI'),('RID / VISCODE','Participant roster identifier / visit code'),('\field{bl}; \field{_bl}','Original baseline visit; baseline-version field suffix'),('APOE4','Number of apolipoprotein E $\\varepsilon4$ alleles (0--2)'),('CDRSB','Clinical Dementia Rating sum of boxes'),('ADAS11 / ADAS13','11-/13-item Alzheimer\textquotesingle s Disease Assessment Scale'),('MMSE / MoCA','Mini-Mental State Examination / Montreal Cognitive Assessment'),('RAVLT','Rey Auditory Verbal Learning Test'),('FAQ / ECog','Functional Activities Questionnaire / Everyday Cognition'),('CSF / ICV','Cerebrospinal fluid / intracranial volume'),('FDG / AV45','Fluorodeoxyglucose PET / florbetapir amyloid PET'),('ABETA / TAU / PTAU','Amyloid-$\\beta$ / total tau / phosphorylated tau'),('\field{pT217_F}','Fujirebio plasma phosphorylated tau-217'),('\field{NfL_Q} / \field{GFAP_Q}','Quanterix neurofilament light / glial fibrillary acidic protein'),('ROC / AP / ECE','Receiver operating characteristic / average precision / expected calibration error'),('TP / FP / TN / FN','True positive / false positive / true negative / false negative')]
page(r'\section*{Nomenclature and abbreviations}\addcontentsline{toc}{section}{Nomenclature and abbreviations}'+table(['Term','Meaning'],abbr,'p{.26\\linewidth}p{.65\\linewidth}')+r'''\textbf{Notation.} $i$ identifies a subject, $t_i$ the index date, $h_i=t_i+5$ calendar years, $x_i$ baseline predictors, $y_i$ observed five-year conversion status, $q_i$ raw risk, $p_i$ calibrated risk, and $\tau$ the fixed decision threshold. ``Validation'' denotes the final held-out partition unless explicitly described as a development fold. ``PR AUC'' is reported as AP, not a trapezoidal integral.
''')
page(r'''\section{Introduction and clinical problem formulation}
\subsection{Motivation and research question}
Clinical prognosis asks what will happen after an assessment, rather than simply recognizing current disease. Earlier identification of participants likely to develop dementia could help prioritize follow-up, investigate biomarkers, or support research recruitment. The intended beneficiaries include CN/SMC and MCI populations, although performance for these groups must be established separately. Existing impairment is not a future conversion event: including baseline AD patients would turn prognosis into partial diagnosis recognition.

The research question is: \emph{using only information available at a participant's original baseline, can tabular demographic, cognitive, genetic, medical-history, and biomarker fields discriminate observed dementia conversion within five years?} The thesis evaluates this question in the local ADNI snapshot. ADNI spans multiple enrollment phases and is a research cohort, not a representative primary-care sample.
\subsection{Scope and modality constraints}
No raw three-dimensional MRI, PET volumes, image slices, or image-learning network is used. Their processing would introduce a separate acquisition, segmentation, quality-control, and computational pipeline. Already derived numeric measures---hippocampal volume, ventricular volume, whole-brain volume, entorhinal measures, and PET composites---are tabular predictors and remain in scope. Their availability and timing still require scrutiny; the word ``tabular'' does not eliminate temporal leakage.
\subsection{Output contract}
For an eligible baseline record, the pipeline estimates
\begin{equation}p_i\approx P(y_i=1\mid x_i,\text{eligible ADNI population}),\qquad
\widehat y_i=\mathbf{1}[p_i\geq\tau].\end{equation}
The interface returns a probability in $[0,1]$, percentage, threshold, binary decision, and model version. A below-threshold decision does not imply zero possibility. The continuous score allows threshold choices to be distinguished from ranking ability.

The selected operational objective is $F_1$, balancing precision and recall. Accuracy alone is insufficient: predicting nonconversion for everyone achieves 69.34\% accuracy on final validation but detects none of its 65 converters. No minimum accuracy or diagnostic guarantee was assumed. The completed contribution is a validated research artifact with transparent uncertainty, rather than an unsupported clinical claim.
''')
roles=['Reference PACC input','Reference PACC input','Reference PACC input','Outcome trajectories','History flags','Redundant item scores','Supplemental neuropsychology','Demographic reference','Deferred free text','Deferred plasma','Primary baseline features','Granular reference','Dictionary','Challenge subset','Challenge targets','Redundant MRI extraction','Redundant MRI extraction']
names=['PACC ADAS Q4','PACC MMSE','PACC NEUROBAT','DXSUM','MEDHIST','MMSE','NEUROBAT','PTDEMOG','RECMHIST','UPENN plasma','ADNIMERGE','TADPOLE D1/D2','TADPOLE dictionary','TADPOLE D3','TADPOLE D4 corrected','dfMri','dfMri D12']
page(r'''\section{Data architecture and multi-phase inventory}
\subsection{Provenance and source roles}
The downloaded material spans ADNI1, ADNI-GO, ADNI2, ADNI3, and ADNI4 resources. The primary baseline backbone is the local ADNIMERGE CSV, distributed with TADPOLE material; newer diagnosis and supplemental CSVs extend its information. Presence of newer source tables does not imply that all phases contribute equally to the final labeled cohort. No new participant downloads were required.

The 17-table inventory below counts table fields, not independent clinical concepts. Dictionaries and duplicated measures contribute to the earlier 3,066-field inventory. The extracted A/B handoffs, shared list, and old-label files are derived inputs and are not additional raw tables in this inventory.
'''+table(['Source','Rows','Columns','Functional role'],[(names[i],f"{x['rows']:,}",x['cols'],roles[i]) for i,x in enumerate(inventory)],'p{.25\\linewidth}rrp{.36\\linewidth}')+r'''The ADNIMERGE2 directory additionally contains a packaged R resource with bundled datasets, documentation, and upstream helper code. Its existence does not imply that those R routines trained the Python classifier. TADPOLE challenge subsets are not independent external validation for this implementation \cite{tadpole}.
''')
page(r'''\subsection{Relational structure and excluded tables}
'''+table(['Table','Rows/subject','Required handling'],[('TADPOLE D1/D2','7.34','Original baseline selection'),('ADNIMERGE','6.46','One original baseline record'),('DXSUM','4.37','Ordered diagnosis trajectory'),('NEUROBAT','3.77','Dated baseline/screening selection'),('MEDHIST','1.24','Pre-index condition aggregation')],'p{.27\\linewidth}rp{.46\\linewidth}')+r'''These values are whole-source averages, not labeled-cohort distributions. If one subject has $a_i$ eligible rows in one table and $b_i$ in another, a naive RID-only join can produce $a_ib_i$ rows. Counting those rows as independent patients inflates sample size and can put the same person's information into training and evaluation. Temporal filtering, explicit aggregation, and one-to-one merge validation address different parts of this problem; none substitutes for the others.

The two-person architecture assigns Person A 29 demographic/genetic, CSF, cognitive, MRI, and PET features; Person B supplies 14 ECog measures, ten neuropsychological scores, and seven history flags. Both outputs retain every shared RID, even if some measurements are missing. Their intersection is therefore not used to select only complete participants. The handoffs contain 2,155 rows each, with no duplicate RIDs or overlapping predictor names.

\textbf{Exclusion rationale.} TADPOLE D1/D2's numerous granular imaging fields are retained as reference rather than reintroduced alongside already selected composites. Its dictionary supplies metadata. D3 and D4 serve a different challenge design and can overlap the underlying ADNI population; treating them as an independent test set would require an explicit identity and timing audit. The two dfMri extracts add redundant measurements. Item-level MMSE and PACC helper files are redundant with selected summary scores or require a separately specified composite reconstruction.

RECMHIST contains 30,810 free-text history records. Interpreting its narratives would require an additional natural-language processing protocol and temporal validation. PTDEMOG remains a reference for dated demographic corrections; no undocumented demographic backfilling is claimed in this run. The plasma table has 2,422 rows but only 251 subjects overlapping the baseline pool with a draw at or before the index in the earlier feasibility audit. Table completeness is not cohort coverage, and an ``earliest available'' draw may still occur years after original baseline.

Source files and original handoffs remain immutable. A local manifest stores filenames and SHA-256 hashes, allowing the September 2026 source snapshot to be distinguished from a later refresh.
''')
page(r'''\section{Feature auditing and temporal leakage prevention}
\subsection{Tiered audit and reconciliation}
The corrected audit defines 34 core and 33 secondary candidates. Selection emphasizes clinically interpretable summaries, plausible prognostic relevance, manageable dimensionality, and defensible baseline timing. The audit is a candidate specification, not evidence that every feature improves prediction. The final primary candidate matrix has 60 fields because seven plasma fields were deferred.
'''+table(['Group','Tier 1','Tier 2','Implemented handling'],[('Demographics/genetics',4,3,'Original baseline'),('CSF / plasma',9,1,'3 CSF retained; 7 plasma deferred'),('Core cognition',10,0,'Baseline suffix fields'),('ECog',0,14,'Baseline suffix fields'),('NEUROBAT',0,10,'Dated screening/baseline'),('MRI / PET',8,1,'Baseline composites'),('History',3,4,'Pre-index binary aggregation'),('Total',34,33,'60 source predictors')],'p{.27\\linewidth}rrp{.40\\linewidth}')+r'''A full field-level specification appears in Appendix A. A demographic or genotype field may be biologically stable, but this implementation still reads the original baseline record. Repeated cognition, imaging, and CSF measures use baseline versions. Medical history is not replaced by a random row; it is aggregated over eligible dated records.

The Tier 3 accounting requires qualification. Arithmetic gives $3066-34-33=2999$ fields outside the candidate set, whereas the corrected PDF prints 2,943 Tier 3 fields. Approximate narrative descriptions such as ``700 MRI ROIs'' and ``500 PET sub-ROIs'' are not a reconciled partition of unique clinical variables. Reported exclusion buckets also include metadata and cross-table duplicates. The implementation preserves this discrepancy in the data audit rather than fabricating a clean breakdown.

The main deferred families are redundant granular MRI/PET variables, DTI measures, challenge metadata, identifiers, diagnostic target fields, redundant neuropsychological subscores, free text, and future-visit measurements. Direct diagnosis fields such as DX, DXCHANGE, and DXAD must not become predictors. Their clinical importance does not make their inclusion valid for the intended prognosis.
''')
page(r'''\subsection{Baseline timing, aggregation, and sample-size interpretation}
Let $s_{ij}$ denote the measurement date for feature $j$. An eligible predictor must satisfy $s_{ij}\leq t_i$ under the available provenance. A follow-up cognitive score at $t_i+12$ months contains disease evolution that has not occurred at index. It can improve apparent accuracy while making the baseline prediction impossible to reproduce prospectively.

ADNIMERGE's \field{_bl} convention addresses baseline versus follow-up fields. However, individual acquisition timestamps are not supplied for every merged value. Therefore, the implementation establishes original-baseline table provenance, not a complete raw acquisition-date audit for every assay and image-derived composite.

NEUROBAT uses actual VISDATE and translated VISCODE2: accept screening or baseline records from index day or the previous 90 days, prefer screening for Logical Memory and baseline for other tests, and resolve eligible alternatives deterministically. Same-day conflicting values become missing. This corrected 30--120 cells per neuropsychological column in the supplied B handoff.

For history, the condition flag is one if any eligible pre-index record explicitly reports it, zero when eligible records document absence, and missing when no usable history exists. The B handoff had encoded unavailable history as zero; 417 subjects were restored to missing for every assigned history field. \field{MH3HEAD} covers head, eyes, ears, nose, and throat history and must not be described as a specific traumatic-brain-injury measure.

Laboratory strings such as \field{>1700} and \field{<80} are parsed into their boundary and a signed bound flag, not silently treated as exact observations. Three bound flags are generated; only two are nonconstant in final development fitting. Legitimate negative RAVLT change scores are preserved.
'''+table(['Population / panel','Subjects','Source features','Ratio'],[('Historical complete demographic/cognitive pool','2,022','34','59.5:1'),('Historical same pool, all candidates','2,022','67','30.2:1'),('Historical multimodal complete cases','1,003','34','29.5:1'),('Final labeled cohort / core','1,056','28','37.7:1'),('Development cohort / core','633','28','22.6:1')],'p{.48\\linewidth}rrr')+r'''The historical ratios use different denominators from training. They neither guarantee freedom from overfitting nor establish the size of an eligible complete-case experiment. Development has only 194 converters, about 6.9 events per core source feature before encoding and indicators. Regularization and held-out evaluation remain essential.
''')
page(r'''\section{Target construction and leakage-free preprocessing}
\subsection{Five-year outcome algorithm}
For each unique original baseline RID, parse $t_i$ and form $h_i$ using a five-calendar-year offset. This handles leap-day baselines without approximating five years by a fixed number of days. Sort valid DXSUM dates and use harmonized diagnoses 1 (unimpaired), 2 (MCI), and 3 (dementia) \cite{adni}.
\begin{equation}
y_i=\begin{cases}
1,&\exists\,d\in(t_i,h_i]\text{ with diagnosis }3,\\
0,&\text{no such event and valid observation at/after }h_i,\\
\text{unknown},&\text{otherwise.}
\end{cases}
\end{equation}
Before assigning outcomes, exclude baseline AD, missing/unrecognized baseline diagnoses, and dementia at or before index. Agreeing repeated diagnoses do not change the outcome. Conflicting valid diagnoses on the same date quarantine the subject, as does an undated dementia record. Exclusion reason precedence is explicit, so cohort counts sum to the baseline pool without double counting.

An event exactly on the anniversary is positive; an event the next day is outside the five-year outcome. A later diagnosis reversion does not undo an observed event. A valid dementia observation after the horizon may support a zero five-year label if no within-window event was observed, but it cannot prove that biological onset occurred only after the horizon. Visit spacing therefore limits event-time precision.
'''+table(['Disposition','Count'],[(esc(k.replace('_',' ')),v) for k,v in read(repo/'results/data_audit.json')['cohort_flow'].items()],'p{.70\\linewidth}r')+r'''The old labels were not trusted as ground truth: 363 originally positive participants had baseline AD, and the preliminary reconstruction identified 22 old negatives with inadequate follow-up. Final stricter ambiguity checks produced 1,056 known outcomes. Unknown labels are excluded from binary training, never imputed as negatives. No censored participant was silently routed to a survival model in this completed experiment.
''')
page(r'''\subsection{Integrity contract, partitions, and preprocessing}
The merge checks exact feature names, matching RID sets, one unique row per handoff, and no predictor-name collision. It then joins with one-to-one validation and attaches corrected label metadata separately. The master matrix contains 60 source fields and three bound indicators. Source dates, diagnoses, phase, and IDs remain audit metadata and are excluded by an explicit predictor allowlist.
'''+table(['Partition','Total','Positive','Negative'],[('Development',633,194,439),('Calibration / threshold',211,65,146),('Final validation',212,65,147)])+r'''A seeded, outcome-stratified 60/20/20 subject split precedes fitting any learned transformation. IDs are stored locally. With exactly one record per subject, stratified K-fold splitting is also subject-disjoint; GroupKFold would become necessary if multiple visits or repeated anchors were introduced. The current implementation uses five-fold StratifiedKFold with explicit RID-isolation checks, not a falsely claimed GroupKFold execution.

Within each training fold, drop fields with more than 80\% missingness and constant fields. Categorical missing values receive a distinct category and one-hot encoding supports unseen categories. The primary simple comparator uses numeric median imputation, missingness indicators, and standardization. The alternative first standardizes numeric values using observed training values, then applies distance-weighted KNN imputation with $k=5$, and appends numeric missingness indicators. Scaling before KNN prevents volume units from dominating distances.

For standardized partially observed vectors, KNN uses a distance based on jointly observed coordinates. Missing features are estimated from nearby training subjects; validation and calibration records supply no neighbors to the fitted training imputer. Categorical values are encoded separately rather than averaged into fractional categories.

No blanket subject pruning for missing APOE4 or demographics was performed after labels were established. Eligible missing predictors are handled by fitted preprocessing. This differs from the earlier proposal's pruning language and preserves useful subjects. No imputer is fitted separately to the two handoffs, no full-cohort normalization precedes splitting, and no outcome column is passed to imputation.

MissForest and iterative random-forest imputation were not executed. They are scientifically plausible alternatives but cannot be included as measured sensitivity results. Their addition would require development-only comparisons under the same split discipline.
''')
page(r'''\section{Predictive modeling, calibration, and survival framework}
\subsection{Classifier architectures}
The majority-class classifier establishes a minimal reference. Logistic regression models a raw probability
\begin{equation}q_i=\sigma(\beta_0+z_i^\top\beta),\quad \sigma(a)=\frac{1}{1+e^{-a}},\end{equation}
where $z_i$ is the fold-transformed predictor vector. The objective combines class-weighted log loss with an $L_2$ penalty; smaller $C$ implies stronger regularization. Balanced weights are computed from the fitted training sample, not final validation frequencies. The selected $C=0.01$ favors coefficient shrinkage in a moderate-size, correlated biomarker panel.

Random forest averages predictions over randomized trees, controlling depth, minimum leaf size, and feature subsampling. XGBoost builds an additive tree score $f(x)=\sum_m f_m(x)$ by sequentially minimizing regularized loss \cite{xgb}. It is a candidate, not the final selected model. CatBoost supplies an additional gradient-boosting comparator. In this pipeline its inputs use the common numeric/one-hot preprocessing, so no claim of a specialized native-categorical experiment is made.

Three source-feature panels are fixed before comparison: clinical (age, sex, education, APOE4, ten cognitive fields), core (28 non-plasma Tier 1 fields), and expanded (all 60 assigned fields). Thus the clinical panel includes genotype; it is not literally a questionnaire-only screen. The implementation does not include a separate clinical-plus-history-only or blood-panel experiment.
\subsection{Search and fold-local threshold selection}
Eight sampled configurations per model/panel are evaluated, below the 30-configuration cap. These include regularization and class weights for logistic regression; tree depth and leaf size for forest; depth, learning rate, sampling, regularization, and positive weights for XGBoost; and depth, iterations, learning rate, and regularization for CatBoost. The appendix records the actual spaces.

Each outer development fold reserves 25\% of its training portion as an inner threshold holdout. Fit on the remaining inner subjects, maximize threshold $F_1$ on that holdout, refit on all outer-training subjects, and score the untouched outer fold. All models share the outer folds. Rank configurations by mean outer-fold $F_1$, then recall, then simplicity. These scores support selection but can be optimistic after searching many configurations; final validation remains the independent assessment.
''')
page(r'''\subsection{Probability calibration and decision policy}
After selection, refit the winning pipeline on all development subjects. For calibration probabilities $q_i$, use clipped log odds $u_i=\log(q_i/(1-q_i))$ and fit
\begin{equation}p_i=\sigma(a u_i+b)\end{equation}
on the reserved 211-subject calibration partition. The fitted calibrator uses logistic regression with $C=10^6$; the implementation rejects a negative calibration slope. Sigmoid calibration maps scores to empirically better aligned probabilities, but calibration does not establish population-wide clinical validity \cite{calibration}.

The final threshold maximizes calibration-set $F_1$, resolving ties by higher recall. Calibration and threshold share that partition, so their fitted performance is not an independent result. The frozen threshold is 0.5130783821. The general family
\begin{equation}F_\beta=(1+\beta^2)\frac{\mathrm{precision}\,\mathrm{recall}}{\beta^2\mathrm{precision}+\mathrm{recall}}\end{equation}
clarifies the alternatives: $F_1$ balances precision and recall, $F_2$ increases recall's weight, and $F_{0.5}$ increases precision's weight. Precision is not the same as specificity. Separate $F_2$-optimized and $F_{0.5}$-optimized policies were not selected or evaluated; their values in Chapter 6 describe the existing frozen decision rule.
\subsection{Survival analysis: an unimplemented extension}
Right-censored participants contribute partial observation rather than known nonconversion. A survival model would use $(T_i,\delta_i)$ instead of dropping them, where $T_i$ is observed event/censoring time and $\delta_i$ indicates an event. A Cox model assumes
\begin{equation}\lambda(t\mid x)=\lambda_0(t)\exp(\gamma^\top x),\qquad P(T\leq5\mid x)=1-S(5\mid x).\end{equation}
This would require checking proportional hazards, defining censoring and competing death, and considering interval-censored diagnosis times. The existing binary model cannot recover a survival curve from a single five-year probability.

Accordingly, lifelines, Cox fitting, C-index, and low/medium/high-risk Kaplan--Meier curves are \emph{not completed results}. Risk-tertile curves would require predeclared cut points and survival times with adequate follow-up; naive curves from the known-outcome binary subset could conceal the original selection bias. A future survival comparison should retain censored subjects, use development-only fitting, and report time-dependent discrimination and calibration on a defensible evaluation cohort.
''')
metricrows=[]
for key,name in [('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('specificity','Specificity'),('f1','$F_1$'),('balanced_accuracy','Balanced accuracy'),('roc_auc','ROC AUC'),('average_precision','Average precision'),('brier','Brier score')]:
    value=results['primary'][key];lo,hi=results['confidence_intervals_95'][key]
    metricrows.append((name,f'{value:.4f}',f'{lo:.4f}--{hi:.4f}'))
for b,k in [('2','f2'),('0.5','f05')]:
    lo,hi=diag['f'+b+'_ci'];metricrows.append(('$F_{'+b+'}$',f'{diag[k]:.4f}',f'{lo:.4f}--{hi:.4f}'))
page(r'''\section{Experimental results, ablations, and explainability}
\subsection{Final validation discrimination}
The model, selected features, calibration, and threshold were frozen before final validation. No result in this chapter was used to change them. Evaluation contains 212 independent participants, including 65 positives and 147 negatives.
'''+table(['Metric','Estimate',r'95\% bootstrap interval'],metricrows)+r'''Intervals use 2,000 stratified subject-bootstrap samples (seed 42), keeping observed class counts, model, and threshold fixed. They describe conditional held-out sampling uncertainty, not the variability of an entire retraining procedure. Supplemental $F_2$ and $F_{0.5}$ intervals use the same fixed-policy bootstrap design.

For positive conversion, precision is $TP/(TP+FP)$ and recall is $TP/(TP+FN)$. Specificity is $TN/(TN+FP)$, accuracy is $(TP+TN)/N$, and balanced accuracy averages sensitivity and specificity. Here $TP=54$, $FP=11$, $FN=11$, and $TN=136$. Equal FP and FN counts make precision, recall, and all three reported $F_\beta$ point estimates equal; the bootstrap intervals need not be equal.

ROC AUC summarizes ranking across thresholds. Average precision summarizes the precision--recall staircase and is not the same estimator as trapezoidal PR AUC. The validation positive fraction is 0.3066; it supplies an important reference for interpreting AP. The majority-class model has accuracy 0.6934 and zero recall. Threshold 0.5 happens to yield the same decisions as the selected threshold on this sample; this observation did not motivate a threshold revision.
''')
page(r'''\subsection{Ranking curves and calibration quality}
\fig{validation_plots.png}{.73}{Frozen validation ROC, precision--recall curve, calibration, and probability distributions. The displayed threshold was fixed on calibration subjects.}
The combined validation figure uses quantile-based reliability bins. The additional before/after plot below uses ten quantile bins. Quantile bins are not equal-width bins; both their construction and the small number of events matter when interpreting apparent agreement.
\fig{calibration.png}{.55}{Descriptive before/after sigmoid calibration on the same frozen validation records; no calibration was fitted on these records.}
'''+table(['Diagnostic','Before calibration','After calibration'],[('Brier score',f"{diag['before']['brier']:.5f}",f"{diag['after']['brier']:.5f}"),('ECE (10 fixed-width bins)',f"{diag['before']['ece']:.5f}",f"{diag['after']['ece']:.5f}")])+r'''Brier score is $N^{-1}\sum_i(p_i-y_i)^2$. ECE weights each nonempty bin's absolute observed-minus-predicted probability gap by its frequency. ECE is bin-sensitive and is reported descriptively, without a significance claim. These additional diagnostics do not constitute another model-selection round.
''')
best=[]
for panel in ['clinical','core','expanded']:
    for model in ['logistic','forest','xgboost','catboost']:
        pool=[r for r in records if r['spec']['panel']==panel and r['spec']['model']==model]
        winner=max(pool,key=lambda r:r['mean']['f1'])
        best.append((panel,model,f"{winner['mean']['f1']:.3f}",f"{winner['mean']['precision']:.3f}",f"{winner['mean']['recall']:.3f}"))
page(r'''\subsection{Development panel and preprocessing ablations}
'''+table(['Panel','Model','Mean $F_1$','Precision','Recall'],best,'llrrr')+r'''Each row shows the best recorded development configuration for that model and panel, including eligible preprocessing alternatives. They are selected maxima, not independent final-validation experiments. The expanded panel does not automatically dominate: adding correlated ECog and neuropsychological measurements can add missingness and estimation variance as well as signal.

The strongest selected comparison was core logistic regression with KNN imputation: mean development $F_1=0.8721$ (fold SD 0.0442), accuracy 0.9179, precision 0.8411, and recall 0.9074. Its matching median-imputation specification had mean $F_1$ about 0.867. MRI/ICV normalization and KNN alternatives were tested for the three strongest development configurations; they were not additional held-out model comparisons. Final-validation $F_1$ of 0.8308 is lower than the selected development mean, illustrating why development maxima must not be reported as test results.

The measured panels are 14/28/60 source features, not the audit's original 34/67 including plasma. No inference about the benefit of blood biomarkers is supported by these ablations. Nor is a broad ``non-invasive versus full multimodal'' cost comparison established: the clinical panel includes APOE4, and no separate history-augmented clinical panel was fitted. A direct KNN--MissForest experiment and complete-case-versus-imputed comparison were not run. The historical 1,003 complete multimodal subjects are not automatically eligible labeled complete cases.

\textbf{Subgroup validity.} All 65 validation converters are in the MCI subgroup ($N=125$), with precision 0.8438, recall 0.8308, $F_1=0.8372$, and ROC AUC 0.9277. CN/SMC has 87 subjects and zero events, so its sensitivity and positive-class discrimination are not estimable. ADNI3 validation also has zero events; ADNIGO has only two. Zero-division conventions in JSON output must not be interpreted as evidence of zero or perfect sensitivity.
''')
page(r'''\subsection{Coefficient interpretation and aggregate error audit}
\fig{coefficients.png}{.87}{Largest absolute coefficients of the selected standardized logistic model. These are conditional model associations, not SHAP values or causal effects.}
Positive coefficients increase the raw model log odds; negative coefficients reduce them, conditional on other transformed inputs. FAQ, CDRSB, and ADAS13 have positive coefficients, while FDG, hippocampal volume, and immediate verbal recall have negative coefficients. This is compatible with functional/cognitive impairment and lower structural/metabolic measures being informative within this model. It does not establish biological causation or stable rankings across correlated features.

Because the KNN pipeline standardizes numeric values before imputation, numeric-value coefficients are on that standardized scale. Missing indicators are separate terms, and categorical terms have a different interpretation. A raw coefficient cannot be compared uncritically across all term types. Sigmoid calibration subsequently transforms the combined raw score, so a coefficient is not an additive percentage-point effect.
'''+table(['Error group','Count','Baseline composition','Median age','Median MMSE'],[('False positive',11,'8 LMCI, 2 EMCI, 1 CN','73.5','27'),('False negative',11,'8 LMCI, 3 EMCI','69.7','28')],'lrlrr')+r'''Both error groups have median baseline CDRSB 1.5. These descriptive summaries do not explain every mistake or establish an imaging-defined failure mechanism. A false negative here is a converter inside the five-year window missed by the threshold; conversion after year five alone is not a false negative for this endpoint.

Global SHAP summaries and patient-specific waterfall plots were not produced. The coefficient plot is a transparent alternative for the actual selected estimator and is explicitly labeled as such. Individual ADNI risk explanations and patient records are not included in the public report. A future explainability study should use a fixed background distribution, record transformations, examine correlated features, and separate model explanations from causal clinical explanations.
''')
page(r'''\section{Limitations, future clinical scope, and conclusion}
\subsection{What the present evidence supports}
The study establishes a working baseline-only preparation, training, calibration, and evaluation pipeline with reproducible predictions. Its performance supports discrimination among eligible observed ADNI participants. It does not demonstrate specifically AD-attributed dementia, transportability to general practice, benefit from an intervention, or sensitivity among CN/SMC converters, because final validation contains none.

Excluding 698 insufficiently followed subjects can select for attendance, health status, enrollment phase, and survival. Biomarker absence may be informative rather than missing completely at random. KNN preserves sample size but cannot guarantee unbiased recovery under informative missingness. Complete-case deletion would introduce a different selection mechanism rather than automatically solve this problem.

The baseline pool and newer source files originate from different snapshots. RID matching and original-index timing address basic alignment, but not every measurement has an independently supplied acquisition timestamp. Diagnoses are observed at visits, so onset is interval-observed; a later diagnosis is not proof of exact onset. Conflicting and undated records are handled conservatively at the cost of sample size.

Calibration is specific to cohort prevalence and measurement procedures. Its apparent improvement on this validation sample is encouraging but cannot certify a patient's ``true probability.'' The 212-subject sample and correlated feature panel limit precision, while selected development maxima are affected by search. Bootstrap intervals condition on the frozen model and class counts.
\subsection{Prioritized extensions}
First, obtain independent temporal or external validation with adequate CN/SMC events and assess phase, demographic, and site heterogeneity. Second, add a prespecified censored survival analysis, accounting for diagnosis intervals and competing death. Third, reassess a truly pre-index plasma subgroup and compare modality panels on matched subjects. Fourth, test MissForest and complete-case sensitivity analyses within development data. Finally, evaluate explanation stability, decision-curve utility, and clinician-defined operating constraints before any interface or deployment.

\subsection{Conclusion}
Correct outcome construction and temporal provenance were at least as consequential as model choice. A regularized logistic model outperformed the considered alternatives under the selected development criterion and achieved 89.62\% held-out accuracy with 83.08\% precision and recall. The principal contribution is an auditable prognosis workflow that preserves source data, isolates evaluation, expresses uncertainty, and makes unsupported claims visible rather than concealing them behind a high accuracy number.
''')
# Appendix: exact 67-feature specification split by tier into compact lists.
import sys
sys.path.insert(0,str(repo/'src'))
from btp.schema import PANELS,FEATURES,CSF,COGNITION,MRI,ECOG,NEURO,HISTORY
plasma=['pT217_F','AB42_F','AB40_F','AB42_AB40_F','NfL_Q','GFAP_Q']
def featuretable(fields):
    rows=[]
    for f in fields:
        source='Plasma' if f in plasma+['pT217_AB42_F'] else 'MEDHIST' if f in HISTORY else 'NEUROBAT' if f in NEURO else 'ADNIMERGE'
        timing='Deferred; pre-index draw required' if source=='Plasma' else 'Pre-index aggregation' if source=='MEDHIST' else 'Dated sc/bl, 0--90 days' if source=='NEUROBAT' else 'Original baseline / static' if not f.endswith('_bl') else 'Baseline suffix'
        rows.append((r'\field{'+f+'}',source,timing))
    return table(['Exact field','Source','Temporal rule / inclusion'],rows,'p{.43\\linewidth}p{.16\\linewidth}p{.32\\linewidth}')
page(r'''\appendix\section{Complete 67-candidate feature specification}
\subsection{Tier 1: demographics, biomarkers, and cognition (23 fields)}
The audit's 34 Tier 1 fields comprise this page's 23 fields and the next page's 11 fields. Plasma belongs to the historical candidate specification but is excluded from the 60-feature primary matrix. The signed laboratory flags are derived inputs and do not inflate the count of audited source fields.
'''+featuretable(PANELS['core'][:4]+CSF+plasma+COGNITION)+r'''\textbf{Modality interpretation.} APOE4 is an allele count rather than a disease diagnosis. CSF ABETA, total tau, and phosphorylated tau are distinct assays. Plasma suffixes identify the assay conventions in the local source; they must not be substituted across platforms without validation. The temporal rule is the availability of a measurement by index, not simply a row's nominal baseline visit label.
''')
page(r'''\subsection{Tier 1: imaging and history (11 fields)}
'''+featuretable(MRI+['FDG_bl']+HISTORY[:3])+r'''\subsection{Tier 2: neuropsychology and other supplementary fields (19 fields)}
'''+featuretable(NEURO+['PTRACCAT','PTETHCAT','PTMARRY']+HISTORY[3:]+['AV45_bl','pT217_AB42_F'])+r'''Tier 2 totals 33: the 19 fields above plus 14 ECog fields on the following page. Race and ethnicity were evaluated only in the expanded candidate panel; they were not retained by the chosen core-panel model. Their presence in an audit is not a causal claim about ancestry or disease.
''')
page(r'''\subsection{Tier 2: self-report and informant ECog (14 fields)}
'''+featuretable(ECOG)+r'''\subsection{Operational panel membership}
Clinical: age, sex, education, APOE4, and ten core cognitive measures (14). Core: clinical plus three CSF, seven MRI, FDG, and three Tier 1 history flags (28). Expanded: all 60 non-plasma assigned predictors, including 14 ECog, ten NEUROBAT, three supplementary demographics, four supplementary history fields, and AV45.

Bound indicators accompany CSF where retained. Training-fold constant and missingness filters can alter the transformed feature count. The final fitted pipeline retained 28 source fields and two bound indicators before missing-indicator expansion and categorical encoding. This should not be confused with 30 original clinical measurements or with a fitted 67-feature model.

\subsection{Missingness inspection}
The repository supplies a 60-field eligible-cohort missingness table and a plot of the least complete features. These describe the data, whereas all feature-removal decisions used training folds only. In the final known-outcome cohort, sparse Digit Span and Digit Symbol variables remain potential expanded-panel inputs but have substantial missingness; full-table audit percentages are not substituted for these cohort-specific values.
''')
page(r'''\section{Hyperparameters, software, and reproducibility}
\subsection{Actual search spaces and selected configuration}
'''+table(['Estimator','Recorded search dimensions'],[('Logistic regression',r'$C\in\{.01,.05,.1,.5,1,5,10\}$; weights none/balanced'),('Random forest','160 trees; depth 3/5/8/unlimited; leaf 2/5/10; max features sqrt/.7; weights none/balanced'),('XGBoost','120/240 trees; depth 2/3/4; learning rate .03/.07; subsampling .8/1; column sampling .8/1; lambda 1/10; child weight 3/8; positive weight 1/2'),('CatBoost','150/300 iterations; depth 3/4/5; learning rate .03/.07; leaf L2 3/10; class weights none/balanced')],'p{.23\\linewidth}p{.68\\linewidth}')+r'''The selected estimator uses $C=.01$, balanced weights, maximum 3,000 logistic iterations, seed 42, core panel, unnormalized MRI values, distance-weighted KNN with $k=5$, and a separate sigmoid calibrator with $C=10^6$. The threshold is 0.5130783821. Additional candidates normalize MRI by ICV; this transformation was not selected.
'''+table(['Component','Recorded version'],[('Python','3.12.10')]+[(esc(k),v) for k,v in read(repo/'results/run_summary.json')['versions'].items()]+[('matplotlib / pytest','3.10.3 / 8.3.5'),('lifelines / SHAP / fpdf2','Not used')],'p{.50\\linewidth}l')+r'''\subsection{Reproduction and integrity}
The package provides \field{prepare}, \field{train}, \field{evaluate}, \field{predict}, and \field{reproduce} commands. Use Python 3.12 and \field{uv sync --locked}. Original data are external inputs; no patient data or model binaries are published. Local manifests contain input, matrix, model, and split hashes. The final validation marker blocks retraining in the same experiment directory; repeated evaluation returns the saved result.

Thirteen synthetic tests and GitHub CI passed. Fresh-process reproduction regenerated all 2,155 baseline rows, identical subject splits and threshold, and probabilities within $10^{-12}$. Public aggregate results identify the data snapshot without distributing subject records. LaTeX source, plots, and this report are additional documentation; they do not modify the fitted artifact.
''')
page(r'''\begin{thebibliography}{9}\addcontentsline{toc}{section}{References and evidence sources}
\bibitem{adni} Alzheimer's Disease Neuroimaging Initiative. \emph{Diagnostic Information: harmonized DXSUM diagnosis}. Official documentation, accessed 2 November 2026. \url{https://adni.loni.usc.edu/quick-start-guide-asset101625/diagnostic.html}.
\bibitem{tadpole} EuroPOND / TADPOLE consortium. \emph{The Alzheimer's Disease Prediction Of Longitudinal Evolution Challenge}. Official challenge resources. \url{https://tadpole.grand-challenge.org/}.
\bibitem{xgb} T. Chen and C. Guestrin. \emph{XGBoost: A Scalable Tree Boosting System}. 2016. \url{https://arxiv.org/abs/1603.02754}.
\bibitem{calibration} scikit-learn developers. \emph{Probability calibration}. Official user guide; implementation versions are recorded separately in Appendix B. \url{https://scikit-learn.org/stable/modules/calibration.html}.
\bibitem{local} Project implementation and aggregate experiment record. \emph{ADNI dementia prediction}, frozen baseline study, November 2026. \url{https://github.com/soumil-kumar/adni-dementia-prediction}. Evidence files: \field{results/validation_results.json}, \field{results/development_comparison.json}, \field{results/selection.json}, \field{results/data_audit.json}, and \field{results/run_summary.json}.
\bibitem{audit} Local project documents. \emph{ADNI Feature Audit v3 (Corrected)}, \emph{Two-Person Table Assignment}, and \emph{Data Preparation Pipeline}, September 2026. Candidate specification and division of work; numerical discrepancies are explicitly reconciled in Chapter 3.
\end{thebibliography}
\subsection*{Evidence and reporting statement}
All reported primary metrics come from the saved frozen validation experiment. Additional before/after ECE, $F_\beta$ intervals, coefficient plots, and aggregate error summaries were computed for this thesis from the existing predictor and validation records. They are descriptive follow-up analyses, not independent confirmatory experiments. No model, label policy, feature panel, or threshold was changed using these diagnostics.

The report intentionally distinguishes requested analyses from available evidence. C-index, Kaplan--Meier curves, a fitted Cox model, MissForest comparisons, plasma-panel validation, complete-case experiments, and SHAP waterfalls are not reported as completed work. Their absence is documented rather than filled with illustrative numbers.

The source data originate from ADNI and local TADPOLE-distributed material. Dataset access and reuse must follow the applicable ADNI agreement and TADPOLE acknowledgement requirements. This technical report does not substitute for institutional submission declarations, author signatures, or supervisor approval.
''')
parts.append(r'\end{document}')
tex=''.join(parts).replace(chr(12)+'ield', r'\field').replace(chr(9)+'extquotesingle', r'\textquotesingle')
tex=tex.replace(r'\end{titlepage}', r'\end{titlepage}\setcounter{page}{2}')
(root/'BTP_Report.tex').write_text(tex,encoding='utf-8')
print('Authored',len(parts)-2,'page blocks')
