"""Reproduce the two LaTeX reports from the joint Word manuscript and saved results.

Run: python build_reports.py <path-to-ADNI_BTP_report.docx>
Only document text and embedded illustrations are read from the manuscript.
"""
import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def plain(element):
    return ''.join(t.text or '' for t in element.findall('.//w:t', NS))


def tex(s):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '’': "'", '‘': '`', '“': '``', '”': "''",
        '–': '--', '—': '---', '−': '$-$', '×': r'\ensuremath{\times}',
        'β': r'\ensuremath{\beta}', '⁻': r'\textsuperscript{-}', '⁴': r'\textsuperscript{4}',
        '\u00a0': ' '}
    s = s.replace('1 × 10⁻⁴', 'RATEPLACEHOLDER')
    return ''.join(replacements.get(c, c) for c in s).replace('RATEPLACEHOLDER', r'$1\times 10^{-4}$')


def p(s):
    s = tex(s)
    for old, new in [('[2, 3]', r'\cite{collection,docs}'), ('[1]', r'\cite{pathak}'),
                     ('[2]', r'\cite{collection}'), ('[3]', r'\cite{docs}')]:
        s = s.replace(old, new)
    return s + '\n\n'


def table(caption, rows, widths):
    spec = '@{}' + ''.join(r'>{\raggedright\arraybackslash}p{' + str(w) + r'\linewidth}' for w in widths) + '@{}'
    header = ' & '.join(r'\textbf{' + tex(v) + '}' for v in rows[0]) + r'\\'
    return (r'\begingroup\small\renewcommand{\arraystretch}{1.16}'+'\n'
        +r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+p(caption).strip()+r'}\\'+'\n'
        +r'\toprule '+header+r'\midrule\endfirsthead'+'\n'
        +r'\toprule '+header+r'\midrule\endhead'+'\n'
        +r'\midrule\endfoot\bottomrule\endlastfoot'+'\n'
        +'\n'.join(' & '.join(tex(v) for v in row)+r'\\' for row in rows[1:])+'\n'
        +r'\end{longtable}\endgroup'+'\n')


def heading(text, level=1):
    return ('\\section' if level==1 else '\\subsection')+'{'+tex(text)+'}\n'


def figure(name, caption, width):
    return (r'\par\medskip\noindent\begin{minipage}{\linewidth}\centering'+'\n'+r'\includegraphics[width='+width
        +r'\linewidth,height=.63\textheight,keepaspectratio]{figures/'+name+'}\n'
        +r'\par\smallskip\refstepcounter{figure}\small Figure \thefigure: '+tex(caption)+r'\end{minipage}\par\medskip'+'\n')


def read_docx(source):
    with ZipFile(source) as z:
        blocks=list(ET.fromstring(z.read('word/document.xml')).find('w:body', NS))
        assets={new:z.read('word/media/'+old) for old,new in [('image1.png','iit_kharagpur.png'),
            ('image2.png','workflow.png'),('image3.png','validation.png'),('image4.png','coefficients.png')]}
    assert plain(blocks[37])=='Abstract' and plain(blocks[106])=='6. Methodology and Model'
    assert plain(blocks[184])=='References', 'The supplied Word manuscript has changed; review its section mapping.'
    return blocks,assets


CAPTIONS={75:('Main components of the attention-based rs-fMRI model reported by Pathak et al. [1]',[.28,.68]),
 82:('Major ADNI data categories and their typical collection routes [2, 3]',[.20,.30,.45]),
 91:('Important local source tables and their role in the project',[.29,.66]),
 97:('Feature families in the candidate specification',[.33,.62]),
 102:('Core feature panel used by the final project predictor',[.30,.15,.48]),
 116:('Participant counts after outcome construction and eligibility checks',[.75,.20]),
 134:('Performance of the frozen project predictor on final validation',[.25,.20,.49]),
 142:('Confusion matrix for the frozen project predictor on 212 validation participants',[.29,.32,.33])}


def source_range(blocks,start,end,overrides=None):
    overrides=overrides or {}
    out=[]
    for i in range(start,end+1):
        b=blocks[i];v=overrides.get(i,plain(b)).strip()
        if b.tag.endswith('}tbl'):
            rows=[[plain(c) for c in row.findall('w:tc',NS)] for row in b.findall('w:tr',NS)]
            out.append(table(CAPTIONS[i][0],rows,CAPTIONS[i][1]));continue
        if not v or re.match(r'^Table \d+[:.]',v):continue
        if re.match(r'^\d+\.\s+',v):out.append(heading(re.sub(r'^\d+\.\s+','',v)))
        elif re.match(r'^\d+\.\d+\s+',v):out.append(heading(re.sub(r'^\d+\.\d+\s+','',v),2))
        else:out.append(p(v))
    return ''.join(out)


PREAMBLE=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,longtable}
\usepackage{xcolor,fancyhdr}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue!40!black]{hyperref}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\tabcolsep}{4.5pt}\setlength{\emergencystretch}{3em}
\setlength{\LTcapwidth}{.9\linewidth}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Five-year dementia prediction}\fancyhead[R]{\small BTP report}\fancyfoot[C]{\thepage}
\newcommand{\field}[1]{\texttt{\detokenize{#1}}}
\setcounter{tocdepth}{1}
\begin{document}
'''


def title(name,roll):
    return r'''\begin{titlepage}\centering\vspace*{8mm}
{\huge\bfseries Predictive Modeling of\par Neurocognitive Disease Progression\par}\vspace{6mm}
{\Large Five-Year Dementia Prediction Using\par Baseline ADNI Tabular Data\par}\vspace{8mm}
\includegraphics[height=38mm]{figures/iit_kharagpur.png}\par\vspace{7mm}
{\large Bachelor's Thesis Project\par}\vspace{3mm}
Thesis submitted to\par
{\Large\bfseries Indian Institute of Technology Kharagpur\par}\vspace{7mm}
In partial fulfillment of the requirements for the award of the degree of\par
{\large B.Tech. in Biotechnology and Biochemical Engineering\par}\vspace{8mm}
Submitted by\par
{\Large\bfseries '''+name+r'''\par}
Roll No.: '''+roll+r'''\par\vfill
Under the guidance of\par
{\large\bfseries Prof. Manjunatha Mahadevappa\par}\vspace{3mm}
Department of Medical Science and Technology\par
IIT Kharagpur\par\vspace{8mm}
\end{titlepage}\setcounter{page}{2}
'''


def common_data(blocks):
    return source_range(blocks,45,49)+heading('Objectives')+r'\begin{itemize}'+'\n'+''.join(r'\item '+p(plain(blocks[i])) for i in range(58,62))+r'\end{itemize}'+'\n'+source_range(blocks,63,77,{
        64:'Pathak et al. (2026) study Alzheimer’s disease classification from resting-state fMRI [1]. Their work illustrates a separate ADNI use case: learning from functional connectivity to identify current disease status. This project instead forecasts observed dementia conversion from baseline tabular measurements.',
        77:'The paper motivates the use of brain measurements and subject-level evaluation. Its attention architecture, rs-fMRI processing and present-diagnosis target are distinct from the baseline tabular prognosis implemented here.'})+source_range(blocks,79,104,{
        81:'4.1 Major data categories',
        100:'Three panels were compared: clinical (14 source variables), core (28 source variables), and expanded (60 assigned non-plasma source variables). The project’s frozen predictor uses the 28-variable core panel. It adds CSF biomarkers, imaging-derived measures and three medical-history flags to the clinical panel.'})


HANDOFF_A=r'''\subsubsection*{Individual report focus: ADNIMERGE handoff}
The two handoffs were prepared as part of a joint project. This report documents the 29-field ADNIMERGE block: seven demographic/genetic fields, three CSF biomarkers, ten cognitive measures, seven MRI summaries and two PET composites. The block retains all 2,155 baseline identifiers and is checked against the shared subject list. Repeated measurements use baseline-version fields such as \field{MMSE_bl} and \field{Hippocampus_bl}; no later visit is used to replace a missing baseline value.

Baseline dates in the legacy ADNIMERGE extract were parsed using their day--month--year format and checked against the ISO-formatted shared baseline dates. The team retained the original numeric measurement units before fitting. Supported numeric missing codes were set to missing, without treating all negative values as invalid: negative RAVLT change scores can be legitimate. For CSF strings such as \field{>1700} or \field{<80}, the parser stores the numeric bound and an indicator of the direction of the bound. An assay reported at a bound is therefore distinguishable from an exact measurement at the same numeric value.

The normalization focus of this handoff is the continuous ADNIMERGE block. Age, cognitive scores, CSF and imaging variables have different numerical ranges. In the selected pipeline they are standardized using the observed training values before distance-weighted imputation, so volume units do not dominate the distance calculation. Intracranial-volume normalization was also evaluated on selected development configurations. The frozen predictor uses the original MRI quantities followed by training-fitted scaling, rather than MRI-to-ICV ratios. These transformations are fitted only after joining the handoffs and partitioning the cohort.
'''

HANDOFF_B=r'''\subsubsection*{Individual report focus: ECog, NEUROBAT and MEDHIST handoff}
The two handoffs were prepared as part of a joint project. This report documents the 31-field supplementary block: 14 baseline Everyday Cognition (ECog) scores from ADNIMERGE, ten selected NEUROBAT measures and seven MEDHIST flags. Its keys are checked against the same 2,155 baseline participants before joining the handoffs.

NEUROBAT rows were matched using actual assessment dates and screening or baseline visit codes. Eligible visits had to fall at the index date or within the previous 90 days. Screening was preferred for Logical Memory; baseline was preferred for other measures. Within the preferred visit group, the most recent eligible assessment was chosen. Conflicting results at the same priority and date were treated as missing rather than resolved arbitrarily. The audit recorded between 30 and 120 changed values per test column relative to the supplied handoff.

MEDHIST rows were filtered to dates at or before index. A condition became 1 if any eligible record documented it and 0 if eligible records documented absence without a positive record. Unavailable history remained missing. This corrected the earlier zero coding for 417 participants without usable pre-baseline history. This distinction matters because lack of a history record does not imply lack of a condition.

The normalization focus here is harmonizing visit timing, numeric test-score fields and binary history coding. Values were left in the source test units and history retained its 0/1/missing representation in the prepared matrix. No independent full-cohort scaler was applied to this handoff. After joining, numerical transformations were learned on each training fold; this preserves the common validation boundary and retains unknown history for imputation.
'''

SHARED_TRANSFORMS=r'''\subsubsection*{Shared preprocessing after the merge}
Constant variables and variables with more than 80\% missingness were filtered within each fitted training portion. The median comparator imputes numeric values, adds missingness indicators and standardizes them. The KNN alternative first standardizes observed numeric values using training statistics, then applies distance-weighted KNN imputation with $k=5$ and retains missingness indicators. Categorical inputs use missing-category handling and one-hot encoding. Calibration and validation rows receive the stored training transformations; they do not update them.
'''

MODEL_A=r'''\subsection{Logistic regression and XGBoost}
Regularized logistic regression estimates a raw conversion probability $q=\sigma(\beta_0+\boldsymbol{z}^{\mathsf{T}}\boldsymbol{\beta})$, with $\sigma(s)=(1+\exp(-s))^{-1}$ and $\boldsymbol{z}$ the fitted preprocessing output. Training minimizes class-weighted log loss with an $L_2$ penalty. Smaller $C$ means stronger regularization. The search varied $C$ and class weighting, with balanced weights estimated from the fitted training portion.

XGBoost builds an additive score from sequential decision trees. Each tree improves the regularized classification objective given the current ensemble \cite{xgb}. The experiment used histogram-based tree construction and varied tree count, depth, learning rate, row and feature sampling, leaf regularization, minimum child weight and positive-class weight. The strongest XGBoost configuration used the expanded panel, 120 trees, depth 4, learning rate 0.07, $\lambda=1$, minimum child weight 3 and positive weight 2; row and column sampling fractions were both 1.0. Its median-based preprocessing was fitted inside the folds.

The selected project estimator is the core-panel logistic classifier with $C=0.01$, balanced class weights and scaled, distance-weighted KNN imputation. The expanded logistic variant with MRI-to-ICV ratios reached a nearby development score, but the saved selection follows the unrounded mean $F_1$ ranking. The chosen classifier was refitted on all 633 development subjects before calibration.
'''

MODEL_B=r'''\subsection{CatBoost and Random Forest}
Random Forest averages probability estimates from randomized trees. Bootstrap resampling and feature subsampling reduce correlation between trees; depth and minimum leaf size control complexity \cite{forest}. The experiment used 160 trees and varied depth, minimum leaf size, feature sampling and class weights. The strongest recorded forest used the core panel, no maximum depth, a minimum leaf size of five, square-root feature sampling and no class weights.

CatBoost fits trees sequentially to improve a classification objective \cite{cat}. The search varied boosting iterations, depth, learning rate, leaf regularization and automatic class weighting. In this experiment it received numeric and one-hot encoded features from the same fold-fitted preprocessing as the other candidates. This is not a test of a separate native categorical-input configuration. The strongest CatBoost run used the core panel, 300 iterations, depth 4, learning rate 0.07, leaf $L_2$ regularization 3 and balanced automatic class weights.

Both of these best configurations use median-based numeric imputation with missingness indicators and training-fitted scaling. Their reported scores come from development folds. CatBoost led this two-family comparison by mean $F_1$. Neither tree candidate was the frozen project predictor, so no independently calibrated held-out performance is assigned to these two models.
'''

CALIBRATION=r'''\subsection{Calibration and decision rule for the shared predictor}
The team refitted the selected predictor on development subjects and used the reserved calibration partition to fit a sigmoid mapping of raw log odds. For raw probability $q$ (clipped away from zero and one), the calibrated score is $p=\sigma(a\log(q/(1-q))+b)$. The model reports $100p$ as the estimated percentage risk of dementia within five years from baseline. The operating threshold, chosen to maximize $F_1$ on calibration data, was 0.513078. At or above the threshold the model flags a positive prediction; below it the probability is still reported.

Calibration and threshold selection use the same reserved partition. Those fitted results are not an independent assessment. The pipeline, calibrator and cutoff were frozen before the final 212 participants were evaluated. The held-out figures and metrics below refer to that shared frozen artifact only.
'''


def comparisons(records,keys):
    names={'logistic':'Logistic regression','xgboost':'XGBoost','catboost':'CatBoost','forest':'Random Forest'}
    panels=[('clinical','Clinical (14 features)'),('core','Core (28 features)'),('expanded','Expanded (60 features)')]
    best=lambda key,panel:max((r for r in records if r['spec']['model']==key and r['spec']['panel']==panel),key=lambda r:(r['mean']['f1'],r['mean']['recall']))
    rows=[['Feature panel',*[names[k] for k in keys]]]
    for panel,name in panels: rows.append([name,*[f"{best(k,panel)['mean']['f1']:.4f}" for k in keys]])
    details=[['Model and strongest panel','Mean F1','Fold SD','Precision','Recall','ROC AUC']]
    for key in keys:
        winner=max((best(key,panel) for panel,_ in panels),key=lambda r:r['mean']['f1']);m=winner['mean']
        details.append([names[key]+' / '+winner['spec']['panel'],f"{m['f1']:.4f}",f"{winner['std']['f1']:.4f}",
            f"{m['precision']:.4f}",f"{m['recall']:.4f}",f"{m['roc_auc']:.4f}"])
    return heading('Development comparisons for the assigned models',2)+p('The tables report the best recorded mean development scores for the model families covered in this report. These are selected cross-validation results, not held-out test results.')+table('Best recorded mean development F1 by model and panel',rows,[.45,.24,.24])+table('Development scores for the strongest configuration of each assigned model',details,[.33,.105,.105,.12,.10,.12])


def bibliography(is_a):
    out=r'''\begin{thebibliography}{9}\addcontentsline{toc}{section}{References}
\bibitem{pathak} H. Pathak, G. Reddy N, M. Acharya, and M. Mahadevappa, \emph{An Attention-Based Framework for Alzheimer's Disease Classification Using Resting-State fMRI}, 2026. \url{https://arxiv.org/abs/2607.26746}.
\bibitem{collection} Alzheimer's Disease Neuroimaging Initiative, \emph{Data Collection}. \url{https://adni.loni.usc.edu/quick-start-guide-asset/datacollection.html}.
\bibitem{docs} Alzheimer's Disease Neuroimaging Initiative, \emph{ADNI Documentation}. \url{https://adni.loni.usc.edu/help-faqs/adni-documentation/}.
'''
    if is_a:out+=r'''\bibitem{xgb} T. Chen and C. Guestrin, \emph{XGBoost: A Scalable Tree Boosting System}, KDD, 2016. \url{https://arxiv.org/abs/1603.02754}.
'''
    else:out+=r'''\bibitem{forest} L. Breiman, \emph{Random Forests}, Machine Learning 45, pp. 5--32, 2001. \url{https://doi.org/10.1023/A:1010933404324}.
\bibitem{cat} L. Prokhorenkova et al., \emph{CatBoost: unbiased boosting with categorical features}, NeurIPS, 2018. \url{https://arxiv.org/abs/1706.09516}.
'''
    return out+r'''\bibitem{project} BTP project repository and frozen aggregate experiment records. \url{https://github.com/soumil-kumar/adni-dementia-prediction}.
\end{thebibliography}
\end{document}
'''


def make(blocks,records,is_a):
    name,roll=('Soumil Kumar','23BT10034') if is_a else ('Ritam Kundu','23BT10031')
    output=PREAMBLE+title(name,roll)+r'\section*{Abstract}\addcontentsline{toc}{section}{Abstract}'+'\n'
    output+=source_range(blocks,38,40,{39:plain(blocks[39]).replace('Raw MRI and PET images are used here, although numerical measurements derived from imaging are included as tabular features.','Imaging is represented by numeric measurements derived from scans. Raw MRI and PET files are not model inputs.')})
    output+=p('The labeled cohort was divided into development (633 participants), calibration (211) and final validation (212) sets. '+('This report documents the ADNIMERGE handoff and the logistic regression and XGBoost comparisons. The selected core-panel logistic model with scaled KNN imputation was retained for final calibration and validation.' if is_a else 'This report documents the ECog, NEUROBAT and MEDHIST handoff and the CatBoost and Random Forest comparisons. The strongest of these tree candidates was core-panel CatBoost, with mean development F1 of 0.8505. The tree comparisons were development experiments.')+' The shared frozen project predictor achieved 89.62% accuracy, 83.08% precision, 83.08% recall, 92.52% specificity, F1 of 0.8308 and ROC AUC of 0.9667 on final validation.')
    output+=p(plain(blocks[42]))+r'\clearpage\tableofcontents\clearpage'+'\n'
    output+=common_data(blocks)
    output+=r'\clearpage'+'\n'+heading('Methodology and Model')+heading('Overall workflow',2)+figure('workflow.png','Overview of the baseline tabular prediction workflow.', '.98')
    output+=source_range(blocks,109,116,{114:plain(blocks[114]).replace('an undated dementia records','undated dementia records')})
    output+=source_range(blocks,118,119)+(HANDOFF_A if is_a else HANDOFF_B)+SHARED_TRANSFORMS
    output+=source_range(blocks,122,123)+p('Within development, five-fold stratified cross-validation was applied to each model and feature panel. For each outer fold, one quarter of its training records was used as an inner threshold holdout. A candidate was fitted on the remaining inner records, its threshold was selected by F1 on that holdout, and it was refitted on all outer training records before scoring the untouched outer fold. Hyperparameter search sampled eight configurations per model and panel. Candidates were ranked by mean fold F1, then recall and simplicity. All learned preprocessing was fitted within the corresponding training partition.')
    output+=(MODEL_A if is_a else MODEL_B)+CALIBRATION
    output+=r'\clearpage'+'\n'+heading('Results')+heading('Final validation of the shared predictor',2)+p(plain(blocks[132]))
    output+=p('This section presents the result of the frozen project predictor. It does not assign those metrics to every candidate in the development comparisons.')
    output+=source_range(blocks,133,134)+figure('validation.png','Held-out performance of the shared frozen project predictor: ROC, precision--recall, calibration and risk distributions.', '.73')+source_range(blocks,138,144)
    output+=comparisons(records,['logistic','xgboost'] if is_a else ['catboost','forest'])+p(plain(blocks[150]))
    if is_a:
        output+=heading('Interpretation of important model features',2)+figure('coefficients.png','Largest absolute coefficients in the selected regularized logistic regression model.', '.89')+source_range(blocks,153,155)
    else:
        output+=heading('Interpreting the tree model comparison',2)+p('Moving from the clinical to the core panel raised mean development F1 for both models. Adding the expanded candidates did not raise the best recorded F1 for either tree model. Core CatBoost had greater mean recall than core Random Forest (0.9022 versus 0.8457), while the forest had greater mean precision (0.8373 versus 0.8077). This comparison describes an operating tradeoff on development folds; it does not identify causal features or establish a significant difference.')+p('The saved experiments do not provide a tree-specific feature importance or final validation confusion matrix. A tree feature-attribution analysis would need its own fitted estimator and evaluation protocol. No coefficient plot from the shared frozen predictor is attributed to the tree models.')
    output+=source_range(blocks,157,162,{160:('The development comparison suggests that additional features do not consistently improve performance. The selected core logistic configuration balances missingness, dimensionality and interpretability.' if is_a else 'The core panel gave the highest recorded F1 for both tree families covered here. Their development scores help compare settings, while the shared final validation describes a separately frozen predictor. Extra features may bring additional missingness and correlation.')})
    output+=source_range(blocks,164,166,{166:('The selected core logistic regression model achieved 89.62% accuracy, 83.08% precision and recall, and ROC AUC of 0.9667 on the 212-person final validation set. XGBoost was compared on development folds; its strongest recorded configuration had mean F1 of 0.8486. Further validation is needed before clinical use.' if is_a else 'The strongest tree candidate was CatBoost on the core panel, with mean development F1 of 0.8505 versus 0.8397 for the strongest Random Forest. These scores are separate from the shared frozen predictor’s held-out accuracy of 89.62%, precision and recall of 83.08%, and ROC AUC of 0.9667. Further validation is needed before clinical use.')})
    output+=source_range(blocks,168,181,{175:('Interpretability is limited to conditional model coefficients. They do not establish causal effects; individual-level explanation plots were not produced.' if is_a else 'Tree-model evidence is limited to the recorded development comparisons. Independent tree-model calibration and individual-level feature-attribution plots were not produced.'),180:'Study explanation stability across folds and cohorts with carefully designed feature-attribution methods. Explanations should not be interpreted as causal evidence.'})
    output+=p('The experiment records and implementation are available in the project repository [4]. The archived split and fitted artifact support reproduction; patient data, individual predictions and model binaries remain local. The supplemental comparisons in these reports were recovered from saved development records without retraining or retuning the frozen predictor.').replace('[4]',r'\cite{project}')
    return output+r'\clearpage'+'\n'+bibliography(is_a)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('joint_report',type=Path)
    args=parser.parse_args();blocks,assets=read_docx(args.joint_report)
    records=json.loads((ROOT/'results/development_comparison.json').read_text())['records']
    for folder,is_a in [('soumil',True),('ritam',False)]:
        destination=BASE/folder;figures=destination/'figures';figures.mkdir(parents=True,exist_ok=True)
        for name,data in assets.items():
            if is_a or name!='coefficients.png':(figures/name).write_bytes(data)
        content=make(blocks,records,is_a)
        forbidden=('catboost','random forest','ritam kundu') if is_a else ('logistic regression','xgboost','soumil kumar')
        assert all(s not in content.lower() for s in forbidden),folder
        (destination/'main.tex').write_text(content,encoding='utf-8')
        print('Built',destination/'main.tex')


if __name__=='__main__':main()
