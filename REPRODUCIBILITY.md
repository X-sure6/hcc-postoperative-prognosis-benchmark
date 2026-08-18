# Reproducibility contract

## Official analysis semantics

- OS and TTR fixed-time binary tasks at 12/24/36/48/60 months.
- TTR is the recurrence-only fixed-time endpoint family used throughout the final program.
- Shared, prespecified five-fold train/validation/test assignments are reused exactly.
- Any fold failure aborts the configuration and prevents pooled/ranked official output.
- Outcome recoding is forbidden: non-missing outcomes must be explicitly 0/1.

## TabPFN

Policy: `STRICT_LOCAL_CHECKPOINT_NO_FALLBACK_V1`.

- local checkpoint required;
- checkpoint SHA256 is checked before fitting;
- CUDA required;
- remote downloads blocked during constructor/fit/predict;
- default constructor forbidden;
- CPU fallback forbidden;
- audited R9.3 package used TabPFN 6.4.1 and random seed 42.

Checkpoint SHA256:

`5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`

## OOF SHAP

For every one of the 10 endpoints × 3 feature sets:

- SHAP is computed separately in each of the 5 outer held-out folds;
- background <=60 training instances;
- all held-out test patients are explained by default;
- chunk size 8;
- patient-level SHAP, model-input values, cleaned feature values and metadata are saved;
- aggregation requires exact OOF patient coverage and y_true identity;
- 30 complete configurations correspond to 150 fold-level SHAP tasks.

## Temporal validation

S3 only:

- train: 2015-10-05 through 2019-09-30;
- excluded gap: 2019-10-01 through 2019-12-31;
- held-out validation: 2020-01-01 through 2020-12-25;
- OS12m, OS24m, TTR12m, TTR24m.

## Verification levels

1. **CI/static level**: compilation, unit contracts and no-fallback static checks.
2. **Private preflight level**: data schema + endpoint + exact fixed-fold coverage.
3. **Reference-bundle level**: verifies the completed R9.3 30-config/150-fold strict-no-fallback artifact.
4. **Full GPU reproduction**: requires the private TTR-labelled dataset, private fixed folds,
   audited checkpoint and a CUDA environment. This is the only level that refits all models and SHAP.
