# ADNI five-year dementia prediction

Research pipeline predicting any dementia within five calendar years from a
non-demented participant's original baseline tabular measurements.

The primary selection metric is F1. Accuracy, precision, recall, calibration,
and uncertainty are reported on a frozen subject-level validation partition.

## Data policy

This repository contains code, configuration, synthetic tests, and aggregate
results only. Obtain ADNI data through authorized access. Raw data, prepared
tables, subject IDs, individual predictions, and trained model binaries stay local.
Original source tables and A/B handoffs are never overwritten.

## Environment

Use Python 3.12. Dependency versions and reproducible commands will be recorded
as the implementation milestones are completed. `config.json` defaults to the
existing handoffs and source tables in `../data`; generated artifacts go to
ignored `local/`.

## Method

- One baseline row per RID; exclude dementia at or before baseline.
- DXSUM diagnosis 3 is any dementia; incomplete follow-up is unknown, not negative.
- Split subjects 60/20/20 into development, calibration, and final validation.
- Fit preprocessing inside development folds and freeze before final validation.
- No raw imaging, post-baseline predictors, UI, or clinical deployment.

Clinical coding reference: https://adni.loni.usc.edu/quick-start-guide-asset/diagnostic.html
