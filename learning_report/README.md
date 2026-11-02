# Understanding the model: learning companion

This separate 13-page explanatory report covers the post-extraction workflow,
data splits, preprocessing, KNN imputation, logistic-regression training, model
selection, calibration, threshold choice, validation, and probability semantics.
It includes a verified numerical walkthrough using the existing synthetic input:
raw classifier probability 54.05%, calibrated probability 36.56%.

Compile `main.tex` twice using pdflatex. Keep `guide.tex` alongside it; both files
are required. Compilation does not need ADNI records or the saved model.

The guide explains the actual frozen artifact without retraining it. It predicts
observed any-dementia conversion within five years of original baseline, rather
than onset only after five years or every condition under the term NCD.
