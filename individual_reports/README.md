# Individual BTP reports

Compile `main.tex` twice with `pdflatex` from either `soumil/` or `ritam/`.
Keep that directory's `figures/` alongside the source. The two projects are
independent and do not need the Word manuscript to compile.

Both reports document the same jointly completed study. Soumil's individual
sections cover the ADNIMERGE handoff, numeric normalization and the logistic
regression and XGBoost comparisons. Ritam's sections cover the ECog, NEUROBAT
and MEDHIST handoff and the CatBoost and Random Forest comparisons. The shared
held-out evaluation is labeled separately from each report's development scores.
Each report includes an unsigned candidate declaration and supervisor certificate.

In the project repository, to regenerate both sources, run `python build_reports.py <path-to-joint-report.docx>`
from this directory. The builder uses the final joint manuscript and saved aggregate
development records. It does not refit the model or read participant data.
