# BTP LaTeX thesis report

`BTP_Report.tex` is a self-contained editable report with five figure files in
`figures/`. It compiles to 23 A4 pages, including front matter, references, and
the complete 67-candidate feature appendix. The title page identifies BTP as Bachelor's Thesis Project, IIT Kharagpur,
students Soumil Kumar (23BT10034) and Ritam Kundu (23BT10031), and guidance by
Manjunath Mahadevappa, using the details supplied by the project authors.

From this directory, compile twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error BTP_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error BTP_Report.tex
```

The report documents the frozen any-dementia model. It explicitly identifies
survival models, C-index, Kaplan-Meier curves, MissForest, plasma experiments,
and SHAP as unimplemented extensions rather than fabricated results.

`diagnostics.py` computes supplemental aggregate diagnostics from the existing
trusted local model and validation records; it does not train or retune anything.
`diagnostics.json` contains those aggregates only. `build_report.py` regenerates
the LaTeX using local results and the original inventory; it is optional for
editing or compiling the supplied source. Compilation does not require ADNI data.
