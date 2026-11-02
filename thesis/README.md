# Implementation-centred B.Tech. thesis

This independently reorganized report documents the project actually implemented.
It covers the research question, source provenance, cohort and label construction,
feature corrections, fold-local preprocessing, model selection, calibration,
held-out evaluation, software, tests, reproduction, and limitations.

The PDF has 21 A4 pages including front matter, appendix, and references.
No institutional or author identity has been invented.

## Compile the editable source

From this directory, run twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error BTP_Thesis.tex
pdflatex -interaction=nonstopmode -halt-on-error BTP_Thesis.tex
```

Compilation requires the `.tex` file and `figures/` only, without patient data.
`build_thesis.py` is an optional authoring helper that uses repository aggregates
and existing documentation figures; it does not fit or change the model.
The source bundle contains the ready-to-edit LaTeX and referenced figures.
