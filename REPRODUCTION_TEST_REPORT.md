# Reproduction Test Report

## Release status

**FINAL RELEASE GATE: PASS (25/25)**

This repository was validated before public release using the locked clinical input, fixed five-fold assignments, the prespecified local TabPFN checkpoint, and the final TTR-only endpoint definitions.

## Automated tests

- Unit tests: **11/11 PASS**
- Python syntax compilation: **PASS**
- Endpoint contract: **PASS**
- Strict binary-outcome validation: **PASS**
- Partial feature-set routing regression test: **PASS**
- Strict local-checkpoint TabPFN no-fallback policy: **PASS**

## Endpoint contract

The released analysis uses ten fixed-time endpoints:

- OS12m
- OS24m
- OS36m
- OS48m
- OS60m
- TTR12m
- TTR24m
- TTR36m
- TTR48m
- TTR60m

Temporal validation is restricted to:

- OS12m
- OS24m
- TTR12m
- TTR24m

## Prediction reproducibility

A final-release TabPFN equivalence test was performed for:

- Endpoint: TTR24m
- Feature set: ICPI (`full_data`)
- Analyzable patients: 251
- Outer folds: 5

The final release reproduced the previously validated run exactly:

- Patient coverage: identical
- Fold assignments: identical
- Outcomes: identical
- Predicted probabilities: identical
- Mean absolute probability difference: 0
- Maximum absolute probability difference: 0
- AUROC difference: 0
- AUPRC difference: 0
- Brier-score difference: 0

All tested model wrappers also reproduced their corresponding validated predictions exactly.

## Main-pipeline validation

The complete RandomForest release smoke test successfully executed:

- 10 endpoints
- 3 clinical-pathway feature sets
- 30 internal-CV configurations
- Calibration
- Decision-curve analysis
- Bootstrap summaries
- Temporal validation
- Source-data export
- Final release audit

All 30 RandomForest internal-CV configurations and all 12 temporal configurations passed the release checks.

## Deterministic five-fold OOF SHAP

Five-fold held-out OOF SHAP was tested with explicit deterministic per-configuration/per-fold seeds.

Two independent runs of the same TTR24m/ICPI/TabPFN configuration produced:

- 5/5 fold completion
- Identical held-out patient metadata
- Identical feature names
- Identical SHAP arrays
- Identical base values
- Identical model-input arrays
- Identical fold-level feature importance
- Maximum SHAP difference: 0

Five-fold OOF SHAP aggregation also passed with complete patient coverage and no fallback.

The deterministic SHAP implementation was additionally compared with the earlier validated SHAP run. The leading clinical interpretation was stable, including complete overlap of the top 10 features for the tested configuration.

## CGH supplementary-analysis validation

The following supplementary modules were executed successfully:

- Restricted surgical-cohort analysis
- L2-regularized logistic-regression comparator
- Paired incremental-value analysis
- Continuous-time safety gate
- Temporal case-mix analysis
- Five-fold SHAP-stability analysis

All corresponding release-gate checks passed.

## Continuous-time safety

Continuous survival analyses are not fabricated from fixed-time labels.

If validated continuous OS/TTR times and competing-event information are not supplied, continuous-time survival modules explicitly skip execution.

## Reproducibility policy

The released workflow uses:

- Fixed five-fold assignments
- Explicit binary outcomes
- Strict fail-closed execution
- Explicit local TabPFN checkpoint
- Checkpoint SHA256 validation
- CUDA requirement for TabPFN
- No default TabPFN constructor
- No remote model download
- No CPU fallback
- Deterministic five-fold held-out OOF SHAP
- Complete OOF-coverage auditing

Patient-level clinical data, prediction outputs, SHAP arrays, and model checkpoints are not included in the public repository.
