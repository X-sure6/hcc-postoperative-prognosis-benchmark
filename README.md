# HCC Postoperative Prognosis Benchmark

This repository contains the reproducible analysis pipeline used to benchmark
postoperative prognostic models for hepatocellular carcinoma (HCC) across
multiple fixed postoperative time horizons.

The workflow includes:

- strict reuse of prespecified five-fold internal cross-validation assignments;
- three clinicopathological feature configurations: PCI, PPEI, and ICPI;
- comparisons among TabPFN, TabNet, XGBoost, LightGBM, Random Forest, CNLC, and BCLC;
- the sole prespecified S3 3-month interval-gap temporal validation;
- bootstrap confidence intervals, calibration assessment, decision-curve analysis,
  paired AUROC comparisons, and optional SHAP analysis.

Patient-level data, fixed-fold assignments, model checkpoints, and generated
results are not included because they may contain sensitive or
institution-specific information.

## Repository structure

```text
.
├── hcc_postoperative_prognosis_benchmark.py
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
└── GITHUB_UPLOAD_CHECKLIST.md
```

## Analysis design

### Internal cross-validation

The pipeline strictly reuses an existing fold-long file and never silently
generates replacement folds.

| Feature configuration | Internal name | Number of features |
|---|---|---:|
| Preoperative clinical indicators (PCI) | `classic_preop` | 22 |
| Postoperative and pathology-enhanced indicators (PPEI) | `postop_total` | 41 |
| Integrated clinical-pathway indicators (ICPI) | `full_data` | 56 |

Continuous tumour size is represented by the largest recorded diameter.
The binary indicator `Tumor Size >5 cm` is included only in ICPI.

### Prespecified S3 interval-gap temporal validation

The sole temporal experiment uses the prespecified S3 protocol:

- development period: 2015-10-05 to 2019-06-30;
- excluded 3-month interval: 2019-07-01 to 2019-09-30;
- temporal validation period: 2019-10-01 to 2020-12-25;
- endpoints: OS12m, OS24m, RFS12m, and RFS24m;
- preprocessing fitted on the complete development set and applied unchanged
  to the temporal validation set;
- no internal temporal split and no temporal cross-validation;
- a fixed classification threshold of 0.5 for every model;
- temporal calibration estimated with unpenalized scikit-learn logistic regression;
- a 20-seed TabNet probability-mean ensemble using seeds 42–61 and 100 fixed
  epochs per member, with GPU cache cleanup after each member.

The program does not search across alternative temporal cut-offs and does not
select a split according to model performance. S3 is the only temporal
experiment defined and executed by the public main program.

## Input requirements

### Analysis dataset

Provide a private Excel workbook containing:

- one unique patient identifier column;
- an initial-treatment-date column, default name `初始治疗时间`;
- ten binary endpoint columns:
  `OS12m`, `OS24m`, `OS36m`, `OS48m`, `OS60m`,
  `RFS12m`, `RFS24m`, `RFS36m`, `RFS48m`, and `RFS60m`;
- variables required for the 22 PCI, 41 PPEI, and 56 ICPI features;
- CNLC and BCLC columns when these staging baselines are requested.

The program performs strict checks before model fitting. Unexpected feature
counts, duplicated columns, missing endpoints, incompatible fixed folds,
invalid dates, and residual decimal commas cause an explicit failure.

### Fixed-fold assignment file

The fixed fold-long CSV or Excel file must contain:

```text
target, sample_index, sample_id, fold, split
```

For every endpoint, folds must be numbered 1–5 and each fold must contain
`train`, `val`, and `test` assignments. Patient IDs and row indices are checked
against the analysis dataset.

## Installation

Python 3.10 or 3.11 is recommended. For GPU execution, install a PyTorch build
compatible with the CUDA version on the target machine before installing the
remaining dependencies.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## TabPFN checkpoint

A local checkpoint can be supplied as a command-line argument:

```bash
--tabpfn-checkpoint /path/to/tabpfn_checkpoint.ckpt
```

or through the `TABPFN_CHECKPOINT` environment variable.

## Usage

### Preflight checks only

```bash
python hcc_postoperative_prognosis_benchmark.py \
  --excel /path/to/private_analysis_data.xlsx \
  --fold-file /path/to/fixed_fold_long.csv \
  --output outputs/hcc_postoperative_prognosis_benchmark \
  --preflight-only
```

### Complete analysis

```bash
python hcc_postoperative_prognosis_benchmark.py \
  --excel /path/to/private_analysis_data.xlsx \
  --fold-file /path/to/fixed_fold_long.csv \
  --tabpfn-checkpoint /path/to/tabpfn_checkpoint.ckpt \
  --output outputs/hcc_postoperative_prognosis_benchmark
```

The default thread settings preserve the original component protocols:

```text
--model-n-jobs 2
--temporal-model-n-jobs 1
```

Display all command-line options:

```bash
python hcc_postoperative_prognosis_benchmark.py --help
```

## Privacy-safe defaults

By default:

- CV and temporal prediction files do not contain direct patient identifiers;
- temporal assignment files do not contain exact treatment dates;
- reproducibility metadata does not expose local absolute paths.

The following options are intended only for controlled local analyses and their
outputs should not be committed to a public repository:

```text
--save-direct-identifiers
--record-absolute-paths
--save-fold-data
```

Always inspect `git status` before committing files.

## Main outputs

The default output directory is:

```text
outputs/hcc_postoperative_prognosis_benchmark/
```

Major subdirectories include:

- `cv/`: fold-level and pooled internal cross-validation results;
- `temporal/`: the sole prespecified S3 full-development interval-gap temporal validation results;
- `summary/`: consolidated result tables and source-data workbooks;
- `reproducibility/`: parameters, seeds, software metadata, file hashes, and GPU logs.

## Reproducibility notes

- Random seed for non-TabNet models: 42.
- Internal CV tree-model threads default to 2; S3 temporal tree-model threads default to 1.
- Temporal bootstrap seeds include the S3 split name for exact deterministic regeneration.
- Temporal TabNet seeds: 42–61; all 20 probabilities are averaged.
- The pipeline never silently generates new internal folds.
- Preprocessing is fitted separately within each CV training fold.
- S3 temporal preprocessing is fitted once on the complete 2015-10-05 to 2019-06-30 development set.
- Direct identifiers are used internally only to validate alignment with the
  prespecified fold file unless explicitly exported.

## Data availability

The repository intentionally excludes patient-level clinical data. Data access
and sharing must comply with the applicable ethics approval, consent framework,
institutional policy, and journal requirements.
