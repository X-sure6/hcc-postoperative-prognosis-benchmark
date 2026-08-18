# HCC Postoperative Prognosis Benchmark — CGH/TTR audited release

This repository contains the public analysis code for fixed-time postoperative
risk prediction after hepatocellular carcinoma surgery. The final release uses
**overall survival (OS)** and **time to recurrence (TTR)** at 12, 24, 36, 48 and
60 months. Recurrence-only fixed-time outcomes are represented exclusively as
`TTR12m`–`TTR60m` throughout the public code and input contract.

Private patient-level data, fixed-fold assignments and the TabPFN checkpoint are
**not** included.

## What changed from the earlier V8 repository

- Active recurrence endpoints are `TTR12m`–`TTR60m` only; unsupported endpoint names
  fail validation rather than being silently aliased.
- Binary outcomes must already be encoded as `0/1`; values such as `{1,2}` are
  never silently remapped.
- Fixed folds are strict: every analyzable patient must appear exactly once as a
  held-out test patient across the five outer folds, and any failed fold aborts
  the configuration. Partial OOF results are never ranked as official results.
- TabPFN uses one explicit local checkpoint on CUDA. Default-constructor,
  remote-download and CPU fallback are forbidden.
- The audited checkpoint SHA256 is
  `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`.
- R9.3 reproduction used `tabpfn==6.4.1`, random seed 42 and the local v2.5
  classifier checkpoint.
- SHAP is computed in **all five held-out outer folds**, then pooled exactly once
  per analyzable patient. The default is background `<=60`, no held-out test cap
  (`--shap-test 0`), chunk size 8 and permutation-SHAP
  `max_evals=max(2*n_features+1, 101)`.
- Historical/reference prediction comparison is post-run diagnostic only and
  never blocks SHAP generation.
- The sole temporal experiment is S3:
  - development: 2015-10-05 to 2019-09-30
  - gap: 2019-10-01 to 2019-12-31
  - held-out validation: 2020-01-01 to 2020-12-25
  - endpoints: OS12m, OS24m, TTR12m, TTR24m.
- A separate `cgh_supplementary_analyses.py` implements the fixed-endpoint CGH
  robustness analyses and reads five-fold SHAP stability from the primary run.

## Repository layout

```text
hcc_postoperative_prognosis_benchmark.py   primary CV + S3 temporal + OOF SHAP
cgh_supplementary_analyses.py              CGH supplementary analyses
requirements.txt
requirements-reproduction.txt
REPRODUCIBILITY.md
CHANGELOG.md
tools/
  validate_r93_reference_bundle.py
tests/
.github/workflows/ci.yml
```

## Input contract

### Analysis workbook

The private workbook must contain 330 patient rows in the study dataset and the
10 fixed-time endpoints:

```text
OS12m OS24m OS36m OS48m OS60m
TTR12m TTR24m TTR36m TTR48m TTR60m
```

The program expects the prespecified PCI/PPEI/ICPI feature counts 22/41/56,
plus an initial-treatment date (`初始治疗时间` by default). `Tumor Size >5 cm`
is ICPI-only; continuous tumour size is included in all three feature sets.

### Fixed-fold file

The fold-long file must contain:

```text
target, sample_index, sample_id, fold, split
```

with targets already named `TTR*`. Folds must be 1–5 and roles must be
`train/val/test`. Every analyzable patient must be test exactly once.

### Endpoint naming

Private analysis and fixed-fold files must already use the exact endpoint names
listed above. The public release does not perform endpoint-name migration or aliasing.


## Installation

Use Python 3.12.3 for the closest R9.3 environment. Install a CUDA-compatible
PyTorch build separately (R9.3 used PyTorch 2.10.0), then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-reproduction.txt
# For optional KM/CIF/Cox CGH analyses:
# pip install -r requirements-cgh.txt
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```


## Optional subset execution for reproducibility testing

The primary program can run a prespecified subset of internal CV/SHAP tasks without
changing preprocessing, fixed folds, TabPFN construction or SHAP computation:

```bash
python hcc_postoperative_prognosis_benchmark.py \
  --excel V8_TTR.xlsx \
  --fold-file CV_Fold_Assignments_Long_TTR.xlsx \
  --models TabPFN \
  --targets OS12m \
  --feature-sets classic_preop \
  --partial-cv-only \
  --feature-set-workers 1 \
  --tabpfn-checkpoint /private/path/tabpfn-v2.5-classifier-v2.5_default.ckpt \
  --output outputs/os12_pci
```

This mode skips temporal validation and global final aggregation and is intended for
parallel reproduction/smoke testing. A full official run should omit these subset flags.

## Strict preflight

A CPU-only preflight can be run without TabPFN by selecting a non-TabPFN model:

```bash
python hcc_postoperative_prognosis_benchmark.py \
  --excel /private/V8_TTR.xlsx \
  --fold-file /private/CV_Fold_Assignments_Long_TTR.xlsx \
  --models RandomForest \
  --output outputs/preflight \
  --preflight-only
```

If the source-value audit contains clinically verified exceptional values, use
`--allow-data-warnings` only after source-record review.

## Full audited run

```bash
python hcc_postoperative_prognosis_benchmark.py \
  --excel /private/V8_TTR.xlsx \
  --fold-file /private/CV_Fold_Assignments_Long_TTR.xlsx \
  --tabpfn-checkpoint /private/tabpfn-v2.5-classifier-v2.5_default.ckpt \
  --tabpfn-checkpoint-sha256 5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12 \
  --output outputs/final
```

Do **not** use `--skip-shap` for the final five-fold OOF-SHAP release.

Important defaults:

```text
random state                  42
outer folds                    5
SHAP background               60
SHAP held-out test cap         0 (all patients)
SHAP chunk size                8
internal bootstrap           500
S3 temporal threshold        0.5
```

## Mean AUROC vs pooled OOF AUROC

These two estimates are intentionally kept separate:

- **five-fold Mean AUROC**: arithmetic mean of the five held-out-fold AUROCs;
  used for the primary internal model-ranking summary/Fig. 2.
- **pooled OOF AUROC**: all held-out predictions concatenated and AUROC computed
  once; used for pooled ROC/PR, calibration, DCA and bootstrap analyses.

Do not mix the two definitions in the same ranking statement.

## CGH supplementary analyses

The final CGH helper consumes the same TTR-labelled input/folds and a patient-level
OOF prediction table:

```bash
python cgh_supplementary_analyses.py \
  --excel /private/V8_TTR.xlsx \
  --fold-file /private/CV_Fold_Assignments_Long_TTR.xlsx \
  --oof-predictions /private/oof_predictions_all_V8_R93_merged.csv \
  --primary-output outputs/final \
  --output outputs/cgh_supplementary \
  --modules all
```

It provides:

1. BCLC 0/A, tumour <=5 cm and solitary-HCC restricted-cohort analyses
   (OS36m/TTR24m);
2. an L2-logistic comparator over 30 endpoint × feature-set configurations;
3. paired PCI→PPEI→ICPI OOF incremental-value bootstrap analyses;
4. OS36 risk-tertile KM, TTR24 competing-risk CIF and penalized Cox/cause-specific Cox
   when an independently audited `--survival-data` file is supplied; otherwise a transparent
   **SKIPPED** record is written (fixed-time labels are never used to fabricate continuous times);
5. S3 case-mix SMD/CV-vs-temporal comparison when `--temporal-predictions` is supplied;
6. five-fold SHAP stability copied from the primary audited OOF-SHAP output.

The final five-fold SHAP computation itself belongs to the primary benchmark,
not to the supplementary helper.

## R9.3 reference-bundle audit

A private R9.3 result bundle can be checked structurally without rerunning the GPU:

```bash
python tools/validate_r93_reference_bundle.py /private/R93_reference.zip
```

Hard gates include 30/30 configurations, 150/150 fold tasks, TTR terminology,
complete OOF coverage and 150 strict no-fallback checkpoint audits. Probability
changes relative to a historical run are diagnostic, not a SHAP gate.

## Privacy-safe GitHub upload

Do not commit:

- patient-level Excel/CSV files;
- fixed-fold files containing patient identifiers;
- model checkpoints;
- patient-level OOF predictions or SHAP arrays;
- local output directories.

Run `git status` before every public push. See `GITHUB_UPLOAD_CHECKLIST.md`.
