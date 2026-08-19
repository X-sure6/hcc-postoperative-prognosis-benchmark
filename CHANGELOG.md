# Changelog

## Temporal split correction — 2026-08-19

- Corrected the prespecified temporal validation split to match the study analysis:
  development 2015-10-05 to 2019-06-30, excluded 3-month interval gap
  2019-07-01 to 2019-09-30, and held-out validation 2019-10-01 to 2020-12-25.
- Internal fixed five-fold cross-validation and OOF SHAP definitions are unchanged.
- Temporal validation outputs must be regenerated under the corrected split.


## Audited analysis release — 2026-08

- Standardized recurrence-only fixed-time endpoints as TTR throughout active code.
- Disabled silent binary outcome recoding.
- Enforced complete five-fold success and exact OOF coverage before official aggregation.
- Replaced TabPFN constructor fallback chain with strict explicit local-checkpoint CUDA policy.
- Added checkpoint SHA256 verification and network-offline guard.
- Updated TabPFN reproduction target to 6.4.1.
- Replaced first-fold/subsampled SHAP with five-fold held-out OOF patient-level SHAP aggregation.
- Added endpoint/feature-set subset execution (`--targets`, `--feature-sets`, `--partial-cv-only`) for reproducible multi-process SHAP scheduling without changing analysis logic.
- Fixed `--skip-shap` so summary/export steps do not attempt SHAP aggregation when SHAP is intentionally disabled.
- Set temporal validation dates to development 2015-10-05 to 2019-06-30, excluded 3-month interval gap 2019-07-01 to 2019-09-30, and held-out validation 2019-10-01 to 2020-12-25.
- Added CGH restricted-cohort, L2-logistic, paired incremental-value and temporal-shift helpers.
- Added R9.3 reference-bundle validator, tests and GitHub Actions CI.
