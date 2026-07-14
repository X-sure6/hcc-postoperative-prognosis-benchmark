#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCC postoperative prognosis benchmark — fixed-fold cross-validation and prespecified S3 temporal validation.

This single script retains the original strict five-fold internal CV benchmark
and implements the sole prespecified S3 full-development interval-gap temporal validation protocol.

Final analysis definition
-------------------------
- One private analysis dataset is supplied at runtime for both phases
- Strict reuse of an existing fold-long file for internal CV; no automatic re-splitting
- PCI / PPEI / ICPI feature counts: 22 / 41 / 56
- Continuous tumour size is parsed as the largest recorded diameter
- Tumor Size >5 cm is included only in ICPI
- "无" and unresolvable abnormal values become missing before training-set-only KNN
- Initial treatment date is used only for temporal splitting
- Internal CV keeps three feature-set workers and the original CV model protocol
- S3 temporal validation uses all development-period patients for training, with no internal split or cross-validation
- Temporal validation is restricted to OS12m, OS24m, RFS12m and RFS24m
- Temporal classification threshold is prespecified at 0.5 for every model
- Temporal TabNet is a 20-seed (42–61), 100-epoch probability-mean ensemble
- Temporal bootstrap CIs, calibration, DCA and paired bootstrap comparisons are exported
- Random seed for non-TabNet models: 42

The script fails before model fitting when input, features or fixed folds do not
match the prespecified analysis. It never silently generates replacement folds. Patient-level data and fold files are not included in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Stable analysis constants
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_OUTER_FOLDS = 5
KNN_IMPUTER_K = 5
RIGHT_SKEW_THRESHOLD = 1.0
CATEGORICAL_UNIQUE_THRESHOLD = 8
DEFAULT_BOOTSTRAP_ROUNDS = 500
DEFAULT_TEMPORAL_BOOTSTRAP_ROUNDS = 2000
DEFAULT_PAIRED_BOOTSTRAP_ROUNDS = 5000
TEMPORAL_FIXED_THRESHOLD = 0.5
TEMPORAL_TABNET_SEEDS = list(range(42, 62))
TEMPORAL_TABNET_MAX_EPOCHS = 100
TEMPORAL_TABNET_PATIENCE = 0
TEMPORAL_TABNET_BATCH_SIZE = 64
TEMPORAL_TABNET_VIRTUAL_BATCH_SIZE = 32
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.99, 0.01), 2)

ALL_TARGETS = [
    "OS12m", "OS24m", "OS36m", "OS48m", "OS60m",
    "RFS12m", "RFS24m", "RFS36m", "RFS48m", "RFS60m",
]
TEMPORAL_TARGETS = ["OS12m", "OS24m", "RFS12m", "RFS24m"]
FEATURE_SET_ORDER = ["classic_preop", "postop_total", "full_data"]
FEATURE_SET_DISPLAY = {
    "classic_preop": "PCI",
    "postop_total": "PPEI",
    "full_data": "ICPI",
}
EXPECTED_FEATURE_COUNTS = {
    "classic_preop": 22,
    "postop_total": 41,
    "full_data": 56,
}
CORE_MODELS = ["TabPFN", "TabNet", "XGBoost", "LightGBM", "RandomForest"]
STAGE_MODELS = ["CNLC", "BCLC"]
ALL_MODELS = CORE_MODELS + STAGE_MODELS

TIME_COL_DEFAULT = "初始治疗时间"
ID_CANDIDATES = ["住院号", "病案号", "住院ID", "ID", "id", "patient_id", "PatientID", "编号"]
CNLC_CANDIDATES = [
    "CNLC", "CNLC分期", "CNLC stage", "CNLC_stage", "cNLC",
    "CNLC分期（1=Ia，2=Ib，3=IIa,4=IIb，5=IIIa，6=IIIb）",
]
BCLC_CANDIDATES = [
    "BCLC", "BCLC分期", "BCLC stage", "BCLC_stage",
    "BCLC分期（0=0期，1=A，2=B,3=C,4=D）",
]

TEMPORAL_DEV_START = pd.Timestamp("2015-10-05")
TEMPORAL_DEV_END = pd.Timestamp("2019-06-30")
TEMPORAL_GAP_START = pd.Timestamp("2019-07-01")
TEMPORAL_GAP_END = pd.Timestamp("2019-09-30")
TEMPORAL_VAL_START = pd.Timestamp("2019-10-01")
TEMPORAL_VAL_END = pd.Timestamp("2020-12-25")
TEMPORAL_SPLIT_NAME = "S3_gap3m_2019Q4_to_2020"
TEMPORAL_SPLIT_IS_SOLE_PRESPECIFIED_ANALYSIS = True

# Canonical names are resolved against aliases, then actual Excel column names are used.
FEATURE_ALIASES: Dict[str, List[str]] = {
    "Sex": ["Sex", "性别（男 1,女 0）", "性别", "Gender"],
    "Age": ["Age", "年龄（岁）", "年龄"],
    "Cirrhosis": ["Cirrhosis", "肝硬化（是 1,否 0）", "肝硬化"],
    "Viral hepatitis": ["Viral hepatitis", "病毒性肝炎（是1，否0）", "病毒性肝炎"],
    "Multiple tumors": ["Multiple tumors", "肿瘤多发", "多发肿瘤"],
    "Tumor Size >5 cm": ["Tumor Size >5 cm", "肿瘤长径大于5cm", "Tumor size >5 cm"],
    "Pre-op AFP": ["Pre-op AFP", "术前AFP (ng/ml)", "术前AFP"],
    "Pre-op CEA": ["Pre-op CEA", "术前CEA (ng/ml)", "术前CEA"],
    "Pre-op CA125": ["Pre-op CA125", "术前CA125（U/ml）", "术前CA125"],
    "Pre-op CA19-9": ["Pre-op CA19-9", "术前CA199（U/ml）", "术前CA19-9", "术前CA199"],
    "HBV DNA": ["HBV DNA", "HBV-DNA（IU/ml）", "HBV-DNA"],
    "Pre-op AST": ["Pre-op AST", "术前AST（U/L）", "术前 AST（U/L）"],
    "Pre-op ALT": ["Pre-op ALT", "术前 ALT（U/L）", "术前ALT（U/L）"],
    "Pre-op total bilirubin": ["Pre-op total bilirubin", "术前总胆红素（umol/L）"],
    "Pre-op direct bilirubin": ["Pre-op direct bilirubin", "术前直接胆红素（umol/L）"],
    "Pre-op total protein": ["Pre-op total protein", "术前总蛋白（g/L）"],
    "Pre-op albumin": ["Pre-op albumin", "术前白蛋白（g/L）"],
    "Pre-op A/G ratio": ["Pre-op A/G ratio", "术前白球比"],
    "Neutrophils": ["Neutrophils", "中性粒（X109/L）", "中性粒（*109/L）"],
    "Lymphocytes": ["Lymphocytes", "淋巴"],
    "Platelets": ["Platelets", "血小板"],
    "PT": ["PT", "凝血酶原时间"],
    "Tumor size": ["Tumor size", "肿瘤大小(CM)", "Tumour size"],
    "Surgery": ["Surgery", "术式"],
    "Differentiation": ["Differentiation", "分化程度（高分化 1，中分化 2，低分化3）", "分化程度"],
    "Post-op targeted tx": ["Post-op targeted tx", "术后靶向（是1，否0）", "术后靶向"],
    "Post-op chemo": ["Post-op chemo", "术后化疗（是1，否0）", "术后化疗"],
    "Post-op RT": ["Post-op RT", "术后放疗（是1，否0）", "术后放疗"],
    "Post-op immunotx": ["Post-op immunotx", "术后免疫治疗（是1，否0）", "术后免疫治疗"],
    "Post-op TCM": ["Post-op TCM", "术后中药（是1，否0）", "术后中药"],
    "Prophylactic TACE": ["Prophylactic TACE", "术后预防性TACE", "预防性TACE"],
    "Margin <=1 cm": ["Margin <=1 cm", "切缘小于等于1cm"],
    "MVI grade": ["MVI grade", "MVI分级"],
    "Post-op AST": ["Post-op AST", "术后 AST（U/L）", "术后AST（U/L）"],
    "Post-op ALT": ["Post-op ALT", "术后 ALT（U/L）", "术后ALT（U/L）"],
    "Post-op total bilirubin": ["Post-op total bilirubin", "术后总胆红素postoperative total bilirubin（umol/L）", "术后总胆红素（umol/L）"],
    "Post-op direct bilirubin": ["Post-op direct bilirubin", "术后直接胆红素（umol/L）"],
    "Post-op total protein": ["Post-op total protein", "术后总蛋白（g/L）"],
    "Post-op albumin": ["Post-op albumin", "术后白蛋白（g/L）"],
    "Post-op A/G ratio": ["Post-op A/G ratio", "术后白球比"],
    "Margin status": ["Margin status", "切缘"],
    "TB count": ["TB count", "TB数量"],
}

PCI_CANONICAL = [
    "Sex", "Age", "Cirrhosis", "Viral hepatitis", "Multiple tumors",
    "Pre-op AFP", "Pre-op CEA", "Pre-op CA125", "Pre-op CA19-9", "HBV DNA",
    "Pre-op AST", "Pre-op ALT", "Pre-op total bilirubin", "Pre-op direct bilirubin",
    "Pre-op total protein", "Pre-op albumin", "Pre-op A/G ratio", "Neutrophils",
    "Lymphocytes", "Platelets", "PT", "Tumor size",
]
PPEI_CANONICAL = PCI_CANONICAL + [
    "Surgery", "Differentiation", "Post-op targeted tx", "Post-op chemo",
    "Post-op RT", "Post-op immunotx", "Post-op TCM", "Prophylactic TACE",
    "Margin <=1 cm", "MVI grade", "Post-op AST", "Post-op ALT",
    "Post-op total bilirubin", "Post-op direct bilirubin", "Post-op total protein",
    "Post-op albumin", "Post-op A/G ratio", "Margin status", "TB count",
]

MISSING_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "missing", "未知", "不详", "未查",
    "未检测", "无", "-", "--", "/", "未记录",
}
NO_TOKENS = {"否", "无治疗", "未治疗", "未行", "no", "n", "false", "0"}
YES_TOKENS = {"是", "有", "已治疗", "行", "yes", "y", "true", "1"}
TREATMENT_PATTERNS = [
    r"treatment", r"therapy", r"\btx\b", r"chemo", r"immuno", r"target",
    r"radiotherapy", r"\brt\b", r"tace", r"化疗", r"靶向", r"免疫", r"放疗",
    r"中药", r"治疗",
]

# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except Exception:
    TabPFNClassifier = None
    TABPFN_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TabNetClassifier = None
    TABNET_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    sm = None
    STATSMODELS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class ModelResult:
    phase: str
    target: str
    feature_set: str
    model: str
    status: str
    error_message: str
    n_samples: int
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    fold: int
    threshold: float
    AUROC: float
    AUPRC: float
    Accuracy: float
    Sensitivity: float
    Specificity: float
    Precision: float
    F1: float
    BrierScore: float
    TN: int
    FP: int
    FN: int
    TP: int


@dataclass
class WorkerConfig:
    excel_path: str
    sheet_name: Any
    fold_file: str
    output_root: str
    time_col: str
    id_col: str
    cnlc_col: Optional[str]
    bclc_col: Optional[str]
    feature_sets: Dict[str, List[str]]
    models: List[str]
    tabpfn_checkpoint: str
    model_n_jobs: int
    bootstrap_rounds: int
    run_shap: bool
    shap_background: int
    shap_test: int
    save_fold_data: bool
    save_direct_identifiers: bool


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def log_line(message: str, log_file: Optional[str | Path] = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def normalize_name(value: Any) -> str:
    s = str(value).strip().replace("\n", " ")
    s = re.sub(r"\s+", "", s)
    return (
        s.replace("（", "(").replace("）", ")")
        .replace("，", ",").replace("：", ":").replace("／", "/")
        .lower()
    )


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", " ") for c in out.columns]
    return out


def find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    exact = set(columns)
    normalized = {normalize_name(c): c for c in columns}
    for c in candidates:
        if c in exact:
            return c
        if normalize_name(c) in normalized:
            return normalized[normalize_name(c)]
    return None


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0", s):
        s = s[:-2]
    return s


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_slug(text: str) -> str:
    x = re.sub(r"[^0-9A-Za-z_]+", "_", str(text))
    return re.sub(r"_+", "_", x).strip("_") or "item"


def detect_duplicate_columns(columns: Sequence[str]) -> None:
    mangled = [c for c in columns if re.search(r"\.\d+$", str(c))]
    if mangled:
        raise ValueError(
            "检测到 pandas 自动生成的重复表头列（例如 Sex.1/Age.1），已停止运行: "
            + ", ".join(mangled)
        )
    normalized_map: Dict[str, List[str]] = {}
    for c in columns:
        normalized_map.setdefault(normalize_name(c), []).append(str(c))
    duplicates = [v for v in normalized_map.values() if len(v) > 1]
    if duplicates:
        raise ValueError(f"检测到标准化后重复的表头，已停止运行: {duplicates}")


def scan_residual_decimal_commas(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    pat = re.compile(r"(?<!\d)\d+,\d+(?!\d)")
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        for idx, value in s.items():
            if isinstance(value, str) and pat.search(value):
                hits.append({"row_index": idx, "column": col, "value": value})
                if len(hits) >= 50:
                    return hits
    return hits


def resolve_canonical_features(columns: Sequence[str], canonical_features: Sequence[str]) -> Tuple[List[str], List[str]]:
    normalized_to_actual = {normalize_name(c): c for c in columns}
    resolved: List[str] = []
    missing: List[str] = []
    used: set[str] = set()
    for canonical in canonical_features:
        aliases = FEATURE_ALIASES.get(canonical, [canonical])
        found = None
        for alias in aliases:
            if alias in columns:
                found = alias
                break
            found = normalized_to_actual.get(normalize_name(alias))
            if found is not None:
                break
        if found is None:
            missing.append(canonical)
        elif found in used:
            raise ValueError(f"特征解析发生重复映射: {canonical} -> {found}")
        else:
            resolved.append(found)
            used.add(found)
    return resolved, missing


# -----------------------------------------------------------------------------
# Deterministic input cleaning before split-local preprocessing
# -----------------------------------------------------------------------------
def is_treatment_column(column: str) -> bool:
    return any(re.search(p, str(column), flags=re.IGNORECASE) for p in TREATMENT_PATTERNS)


def parse_boundary_number(text: str) -> Optional[float]:
    s = text.strip().replace("＜", "<").replace("〉", ">").replace("〈", "<").replace("＞", ">")
    s = s.replace("≤", "<").replace("≥", ">")
    s = s.replace(" ", "")
    m = re.fullmatch(r"[<>]?([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_tumor_size_max(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if normalize_name(s) in MISSING_TOKENS:
        return np.nan
    if re.search(r"\d,\d", s):
        raise ValueError(f"Tumor size 中仍存在小数逗号: {s}")
    s = s.replace("×", "*").replace("X", "*").replace("x", "*")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", s)
    if not nums:
        return np.nan
    try:
        vals = [float(x) for x in nums]
        return float(max(vals))
    except Exception:
        return np.nan


def clean_generic_value(value: Any, treatment: bool = False) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        return value
    s = str(value).strip()
    key = normalize_name(s)
    if key in MISSING_TOKENS:
        return np.nan
    if treatment:
        if key in {normalize_name(x) for x in NO_TOKENS}:
            return 0
        if key in {normalize_name(x) for x in YES_TOKENS}:
            return 1
        # A named drug or explicit regimen denotes treatment received.
        return 1
    boundary = parse_boundary_number(s)
    if boundary is not None:
        return boundary
    return s


def clean_input_dataframe(df: pd.DataFrame, tumor_size_col: str, protected_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    audit_rows: List[Dict[str, Any]] = []
    for col in out.columns:
        if col == tumor_size_col:
            before_missing = int(out[col].isna().sum())
            out[col] = out[col].map(parse_tumor_size_max)
            after_missing = int(out[col].isna().sum())
            audit_rows.append({
                "column": col,
                "cleaning": "tumor_size_largest_diameter",
                "missing_before": before_missing,
                "missing_after": after_missing,
            })
            continue
        if col in protected_cols:
            continue
        treatment = is_treatment_column(col)
        before_missing = int(out[col].isna().sum())
        out[col] = out[col].map(lambda v, t=treatment: clean_generic_value(v, treatment=t))
        after_missing = int(out[col].isna().sum())
        audit_rows.append({
            "column": col,
            "cleaning": "treatment_binary" if treatment else "generic_deterministic",
            "missing_before": before_missing,
            "missing_after": after_missing,
        })
    return out, pd.DataFrame(audit_rows)


# -----------------------------------------------------------------------------
# Binary outcomes, metrics and curves
# -----------------------------------------------------------------------------
def sanitize_binary_y(series: pd.Series) -> pd.Series:
    mapping = {
        "0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1,
        "negative": 0, "positive": 1, "neg": 0, "pos": 1,
        "alive": 0, "dead": 1, "event_free": 0, "event": 1,
    }
    y = series.copy()
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().replace(mapping)
    y = pd.to_numeric(y, errors="coerce").dropna()
    unique = sorted(pd.Series(y).unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"标签不是二分类，唯一值: {unique}")
    if unique != [0, 1]:
        y = y.map({unique[0]: 0, unique[1]: 1})
    return y.astype(int)


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan
    except Exception:
        return np.nan


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan
    except Exception:
        return np.nan


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


def compute_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    if len(np.unique(y_true)) < 2:
        return 0.5, pd.DataFrame()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    finite = np.isfinite(thresholds)
    if finite.any():
        idxs = np.where(finite)[0]
        best = idxs[int(np.nanargmax(youden[finite]))]
    else:
        best = int(np.nanargmax(youden))
    return float(thresholds[best]), pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "youden_index": youden,
    })


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "AUROC": safe_auroc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "BrierScore": float(brier_score_loss(y_true, y_prob)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        if STATSMODELS_AVAILABLE:
            fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
            return float(fit.params[0]), float(fit.params[1])
        try:
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        except Exception:
            clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        clf.fit(lp, y)
        return float(clf.intercept_[0]), float(clf.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def decision_curve_df(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    prevalence = float(y.mean())
    rows = []
    for pt in DCA_THRESHOLDS:
        pred = (p >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n = len(y)
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(tp / n - fp / n * pt / (1 - pt)),
            "net_benefit_treat_all": float(prevalence - (1 - prevalence) * pt / (1 - pt)),
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, rounds: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    metric_names = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "BrierScore"]
    store: Dict[str, List[float]] = {m: [] for m in metric_names}
    point = {
        "AUROC": safe_auroc(y, p),
        "AUPRC": safe_auprc(y, p),
        "Accuracy": float(accuracy_score(y, pred)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(specificity_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "BrierScore": float(brier_score_loss(y, p)),
    }
    attempts = 0
    while min(len(v) for v in store.values()) < rounds and attempts < rounds * 30:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals = compute_metrics(y[idx], p[idx], 0.5)
        # threshold-independent AUROC/AUPRC/Brier; threshold-dependent values use saved predictions.
        vals["Accuracy"] = float(accuracy_score(y[idx], pred[idx]))
        vals["Sensitivity"] = float(recall_score(y[idx], pred[idx], zero_division=0))
        vals["Specificity"] = float(specificity_score(y[idx], pred[idx]))
        vals["Precision"] = float(precision_score(y[idx], pred[idx], zero_division=0))
        vals["F1"] = float(f1_score(y[idx], pred[idx], zero_division=0))
        for m in metric_names:
            if pd.notna(vals[m]):
                store[m].append(float(vals[m]))
    rows = []
    for m in metric_names:
        arr = np.asarray(store[m], dtype=float)
        rows.append({
            "metric": m,
            "point_estimate": point[m] if m in point else np.nan,
            "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "n_boot": int(len(arr)),
        })
    return pd.DataFrame(rows)


def curve_points(y_true: np.ndarray, y_prob: np.ndarray, metadata: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    for x, y, t in zip(fpr, tpr, thresholds):
        rows.append({**metadata, "curve": "ROC", "x": float(x), "y": float(y), "threshold": float(t)})
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    for i, (x, y) in enumerate(zip(recall, precision)):
        t = float(thresholds_pr[i]) if i < len(thresholds_pr) else np.nan
        rows.append({**metadata, "curve": "PRC", "x": float(x), "y": float(y), "threshold": t})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Fold-local preprocessor
# -----------------------------------------------------------------------------
class FoldPreprocessor:
    def __init__(self, categorical_unique_threshold: int = CATEGORICAL_UNIQUE_THRESHOLD):
        self.categorical_unique_threshold = categorical_unique_threshold
        self.feature_names_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.continuous_cols_: List[str] = []
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.category_bounds_: Dict[str, Tuple[int, int]] = {}
        self.all_nan_cols_: List[str] = []
        self.log1p_cols_: List[str] = []
        self.imputer_: Optional[KNNImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.fitted_ = False

    @staticmethod
    def _integer_like(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return bool(len(s) and np.all(np.isclose(s, np.round(s))))

    def _identify_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical, continuous = [], []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
                categorical.append(col)
            else:
                numeric = pd.to_numeric(s, errors="coerce")
                if numeric.dropna().nunique() <= self.categorical_unique_threshold and self._integer_like(numeric):
                    categorical.append(col)
                else:
                    continuous.append(col)
        return categorical, continuous

    def _fit_category_maps(self, df: pd.DataFrame) -> None:
        for col in self.categorical_cols_:
            values = sorted(df[col].dropna().astype(str).unique().tolist())
            mapping = {v: i for i, v in enumerate(values)}
            self.category_maps_[col] = mapping
            self.category_bounds_[col] = (0, max(mapping.values())) if mapping else (0, 0)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_names_:
            s = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            if col in self.categorical_cols_:
                ss = s.astype(str).where(~s.isna(), np.nan)
                out[col] = ss.map(self.category_maps_[col])
            else:
                out[col] = pd.to_numeric(s, errors="coerce")
        return out

    def fit(self, raw_train: pd.DataFrame) -> "FoldPreprocessor":
        self.feature_names_ = list(raw_train.columns)
        self.categorical_cols_, self.continuous_cols_ = self._identify_types(raw_train)
        self._fit_category_maps(raw_train)
        encoded = self._encode(raw_train)
        self.all_nan_cols_ = [c for c in encoded.columns if encoded[c].notna().sum() == 0]
        self.feature_names_ = [c for c in self.feature_names_ if c not in self.all_nan_cols_]
        self.categorical_cols_ = [c for c in self.categorical_cols_ if c in self.feature_names_]
        self.continuous_cols_ = [c for c in self.continuous_cols_ if c in self.feature_names_]
        encoded = encoded[self.feature_names_]
        if encoded.shape[1] == 0:
            raise ValueError("预处理后没有可用特征")
        k = min(KNN_IMPUTER_K, max(1, len(encoded)))
        self.imputer_ = KNNImputer(n_neighbors=k, weights="uniform")
        imputed = pd.DataFrame(self.imputer_.fit_transform(encoded), columns=self.feature_names_, index=encoded.index)
        for col in self.categorical_cols_:
            lo, hi = self.category_bounds_.get(col, (0, 0))
            imputed[col] = np.round(imputed[col]).clip(lo, hi).astype(int)
        self.log1p_cols_ = []
        for col in self.continuous_cols_:
            s = pd.to_numeric(imputed[col], errors="coerce")
            if len(s.dropna()) and s.min() >= 0 and s.skew() >= RIGHT_SKEW_THRESHOLD:
                self.log1p_cols_.append(col)
                imputed[col] = np.log1p(s)
        if self.continuous_cols_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(imputed[self.continuous_cols_])
        self.fitted_ = True
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.imputer_ is None:
            raise RuntimeError("FoldPreprocessor 尚未拟合")
        encoded = self._encode(raw)[self.feature_names_]
        out = pd.DataFrame(self.imputer_.transform(encoded), columns=self.feature_names_, index=raw.index)
        for col in self.categorical_cols_:
            lo, hi = self.category_bounds_.get(col, (0, 0))
            out[col] = np.round(out[col]).clip(lo, hi).astype(int)
        for col in self.log1p_cols_:
            out[col] = np.log1p(np.clip(pd.to_numeric(out[col], errors="coerce"), 0, None))
        if self.scaler_ is not None and self.continuous_cols_:
            out.loc[:, self.continuous_cols_] = self.scaler_.transform(out[self.continuous_cols_])
        return out

    def fit_transform(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_train).transform(raw_train)

    def metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names_,
            "categorical_cols": self.categorical_cols_,
            "continuous_cols": self.continuous_cols_,
            "log1p_cols": self.log1p_cols_,
            "dropped_all_nan_cols": self.all_nan_cols_,
            "knn_neighbors": KNN_IMPUTER_K,
        }


# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------
class BaseModel:
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "BaseModel":
        raise NotImplementedError

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class TabPFNModel(BaseModel):
    def __init__(self, checkpoint: str):
        if not TABPFN_AVAILABLE:
            raise ImportError("tabpfn 未安装")
        self.checkpoint = checkpoint
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "TabPFNModel":
        kwargs: Dict[str, Any] = {}
        if self.checkpoint:
            kwargs["model_path"] = self.checkpoint
        for candidate in (
            {**kwargs, "device": "cuda", "random_state": RANDOM_STATE},
            {**kwargs, "device": "cuda"},
            kwargs,
            {},
        ):
            try:
                self.model = TabPFNClassifier(**candidate)
                break
            except TypeError:
                continue
        if self.model is None:
            raise RuntimeError("无法构建 TabPFNClassifier")
        self.model.fit(X_train.to_numpy(dtype=np.float32), y_train.astype(int))
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class TabNetModel(BaseModel):
    def __init__(self):
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet 未安装")
        self.model = TabNetClassifier(verbose=0, seed=RANDOM_STATE)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "TabNetModel":
        kwargs: Dict[str, Any] = {
            "max_epochs": 200,
            "patience": 30,
            "batch_size": min(256, max(16, len(X_train))),
            "virtual_batch_size": min(128, max(8, len(X_train))),
        }
        if X_val is not None and y_val is not None and len(X_val) and len(np.unique(y_val)) >= 2:
            kwargs.update({
                "eval_set": [(X_val.to_numpy(dtype=np.float32), y_val.astype(int))],
                "eval_name": ["val"],
                "eval_metric": ["auc"],
            })
        self.model.fit(X_train.to_numpy(dtype=np.float32), y_train.astype(int), **kwargs)
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class XGBoostModel(BaseModel):
    def __init__(self, phase: str, n_jobs: int):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost 未安装")
        self.phase = phase
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "XGBoostModel":
        pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
        spw = float(neg / pos) if pos else 1.0
        if self.phase != "cv":
            raise ValueError("XGBoostModel is reserved for the internal CV phase")
        params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        )
        params.update(dict(
            objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=self.n_jobs, scale_pos_weight=spw, verbosity=0,
        ))
        if self.phase == "cv" and X_val is not None and y_val is not None and len(X_val):
            try:
                self.model = XGBClassifier(**params, early_stopping_rounds=30)
                self.model.fit(X_train.to_numpy(np.float32), y_train.astype(np.int32), eval_set=[(X_val.to_numpy(np.float32), y_val.astype(np.int32))], verbose=False)
                return self
            except Exception:
                pass
        self.model = XGBClassifier(**params)
        self.model.fit(X_train.to_numpy(np.float32), y_train.astype(np.int32), verbose=False)
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


def lightgbm_safe_df(X: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if mapping is None:
        mapping = {c: f"f_{i:04d}_{safe_slug(c)}" for i, c in enumerate(X.columns)}
    out = X.copy()
    out.columns = [mapping[c] for c in X.columns]
    return out, mapping


class LightGBMModel(BaseModel):
    def __init__(self, phase: str, n_jobs: int):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm 未安装")
        self.phase = phase
        self.n_jobs = n_jobs
        self.model = None
        self.mapping: Optional[Dict[str, str]] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "LightGBMModel":
        pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
        spw = float(neg / pos) if pos else 1.0
        if self.phase != "cv":
            raise ValueError("LightGBMModel is reserved for the internal CV phase")
        params = dict(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
        )
        params.update(dict(random_state=RANDOM_STATE, n_jobs=self.n_jobs, scale_pos_weight=spw, verbosity=-1))
        self.model = LGBMClassifier(**params)
        xtr, self.mapping = lightgbm_safe_df(X_train)
        fit_kwargs: Dict[str, Any] = {}
        if self.phase == "cv" and X_val is not None and y_val is not None and len(X_val):
            xval, _ = lightgbm_safe_df(X_val, self.mapping)
            fit_kwargs["eval_set"] = [(xval, y_val)]
        self.model.fit(xtr, y_train.astype(int), **fit_kwargs)
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        xx, _ = lightgbm_safe_df(X, self.mapping)
        p = self.model.predict_proba(xx)
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class RandomForestModel(BaseModel):
    def __init__(self, phase: str, n_jobs: int):
        if phase != "cv":
            raise ValueError("RandomForestModel is reserved for the internal CV phase")
        kwargs = dict(n_estimators=300, class_weight="balanced")
        self.model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "RandomForestModel":
        self.model.fit(X_train.to_numpy(dtype=float), y_train.astype(int))
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=float))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class StageRateModel(BaseModel):
    def __init__(self, stage_col: str):
        self.stage_col = stage_col
        self.mapping: Dict[str, float] = {}
        self.global_mean = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "StageRateModel":
        stages = X_train[self.stage_col].astype(str).where(~X_train[self.stage_col].isna(), "__MISSING__")
        tmp = pd.DataFrame({"stage": stages, "y": y_train.astype(int)})
        self.mapping = tmp.groupby("stage")["y"].mean().to_dict()
        self.global_mean = float(tmp["y"].mean())
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        stages = X[self.stage_col].astype(str).where(~X[self.stage_col].isna(), "__MISSING__")
        return np.asarray([self.mapping.get(x, self.global_mean) for x in stages], dtype=float)


def make_model(model_name: str, phase: str, cfg: WorkerConfig) -> BaseModel:
    if model_name == "TabPFN":
        return TabPFNModel(cfg.tabpfn_checkpoint)
    if model_name == "TabNet":
        return TabNetModel()
    if model_name == "XGBoost":
        return XGBoostModel(phase, cfg.model_n_jobs)
    if model_name == "LightGBM":
        return LightGBMModel(phase, cfg.model_n_jobs)
    if model_name == "RandomForest":
        return RandomForestModel(phase, cfg.model_n_jobs)
    if model_name == "CNLC":
        if not cfg.cnlc_col:
            raise ValueError("CNLC 列不存在")
        return StageRateModel(cfg.cnlc_col)
    if model_name == "BCLC":
        if not cfg.bclc_col:
            raise ValueError("BCLC 列不存在")
        return StageRateModel(cfg.bclc_col)
    raise ValueError(f"未知模型: {model_name}")


def dependency_status(model_name: str, cfg: WorkerConfig) -> Tuple[bool, str]:
    if model_name == "TabPFN":
        return TABPFN_AVAILABLE, "tabpfn missing"
    if model_name == "TabNet":
        return TABNET_AVAILABLE, "pytorch-tabnet missing"
    if model_name == "XGBoost":
        return XGB_AVAILABLE, "xgboost missing"
    if model_name == "LightGBM":
        return LGBM_AVAILABLE, "lightgbm missing"
    if model_name == "RandomForest":
        return True, ""
    if model_name == "CNLC":
        return cfg.cnlc_col is not None, "CNLC column missing"
    if model_name == "BCLC":
        return cfg.bclc_col is not None, "BCLC column missing"
    return False, "unknown model"


# -----------------------------------------------------------------------------
# Strict fold reuse
# -----------------------------------------------------------------------------
def read_fold_long(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        book = pd.ExcelFile(p)
        preferred = [s for s in ["with_index", "folds", "Sheet1"] if s in book.sheet_names]
        sheet = preferred[0] if preferred else book.sheet_names[0]
        df = pd.read_excel(p, sheet_name=sheet)
    else:
        df = pd.read_csv(p)
    df = canonicalize_columns(df)
    if "target" not in df.columns and "endpoint" in df.columns:
        df["target"] = df["endpoint"]
    required = {"target", "sample_index", "sample_id", "fold", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"固定折文件缺少列: {sorted(missing)}")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="raise").astype(int)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["sample_id"] = df["sample_id"].map(normalize_id)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip()
    return df


def validate_target_folds(fold_long: pd.DataFrame, target: str, df_task: pd.DataFrame, sample_ids: pd.Series) -> pd.DataFrame:
    f = fold_long[fold_long["target"] == target].copy()
    if f.empty:
        raise ValueError(f"固定折文件中不存在终点 {target}")
    if not set(f["fold"].unique()) == set(range(1, N_OUTER_FOLDS + 1)):
        raise ValueError(f"{target} 的 fold 不是完整 1-5: {sorted(f['fold'].unique())}")
    if not set(f["split"].unique()).issubset({"train", "val", "test"}) or set(f["split"].unique()) != {"train", "val", "test"}:
        raise ValueError(f"{target} 的 split 必须严格包含 train/val/test")
    expected = set(map(int, df_task.index.tolist()))
    observed = set(map(int, f["sample_index"].tolist()))
    if expected != observed:
        raise ValueError(
            f"{target} 固定折样本覆盖不一致: missing={sorted(expected-observed)[:10]}, extra={sorted(observed-expected)[:10]}"
        )
    id_map = {int(idx): normalize_id(v) for idx, v in sample_ids.items()}
    mismatches = []
    for _, row in f[["sample_index", "sample_id"]].drop_duplicates().iterrows():
        idx = int(row["sample_index"])
        if id_map.get(idx, "") != normalize_id(row["sample_id"]):
            mismatches.append((idx, id_map.get(idx, ""), normalize_id(row["sample_id"])))
    if mismatches:
        raise ValueError(f"{target} 固定折 sample_id 不一致，示例: {mismatches[:10]}")
    test_counts = f[f["split"] == "test"].groupby("sample_index").size()
    if len(test_counts) != len(expected) or not (test_counts == 1).all():
        raise ValueError(f"{target} 每位患者必须恰好作为一次 test")
    for fold in range(1, N_OUTER_FOLDS + 1):
        sub = f[f["fold"] == fold]
        for split in ["train", "val", "test"]:
            if sub[sub["split"] == split].empty:
                raise ValueError(f"{target} fold={fold} 的 {split} 为空")
        # Within one fold each sample must have exactly one role.
        if sub["sample_index"].duplicated().any():
            raise ValueError(f"{target} fold={fold} 中同一样本出现多个角色")
    return f


def split_indices_from_fold(fold_df: pd.DataFrame, fold: int) -> Tuple[pd.Index, pd.Index, pd.Index]:
    sub = fold_df[fold_df["fold"] == fold]
    return (
        pd.Index(sub.loc[sub["split"] == "train", "sample_index"].astype(int)),
        pd.Index(sub.loc[sub["split"] == "val", "sample_index"].astype(int)),
        pd.Index(sub.loc[sub["split"] == "test", "sample_index"].astype(int)),
    )


# -----------------------------------------------------------------------------
# SHAP generation and aggregation
# -----------------------------------------------------------------------------
def run_generic_shap(model: BaseModel, X_train: pd.DataFrame, X_test: pd.DataFrame, outdir: Path, background_n: int, test_n: int) -> None:
    ensure_dir(outdir)
    if not SHAP_AVAILABLE:
        (outdir / "shap_unavailable.txt").write_text("shap is not installed", encoding="utf-8")
        return
    if any(re.search(r"\.\d+$", c) for c in X_train.columns):
        raise ValueError("SHAP 检测到 .1/.2 重复特征")
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = np.arange(len(X_train)) if len(X_train) <= background_n else np.sort(rng.choice(len(X_train), background_n, replace=False))
    ts_idx = np.arange(len(X_test)) if len(X_test) <= test_n else np.sort(rng.choice(len(X_test), test_n, replace=False))
    background = X_train.iloc[bg_idx]
    test = X_test.iloc[ts_idx]

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        xx = pd.DataFrame(arr, columns=X_train.columns)
        p1 = model.predict_proba_1(xx)
        return np.column_stack([1 - p1, p1])

    try:
        explainer = shap.PermutationExplainer(predict_fn, background.to_numpy(dtype=float), feature_names=list(X_train.columns))
        explanation = explainer(test.to_numpy(dtype=float), max_evals=max(2 * X_train.shape[1] + 1, 101))
        values = explanation.values
        np.save(outdir / "generic_shap_values.npy", values)
        if values.ndim == 3:
            class1 = values[:, :, 1]
            base = explanation.base_values[:, 1] if np.ndim(explanation.base_values) > 1 else explanation.base_values
            plot_exp = shap.Explanation(values=class1, base_values=base, data=test.to_numpy(float), feature_names=list(X_train.columns))
        else:
            class1 = values
            plot_exp = explanation
        imp = pd.DataFrame({"feature": X_train.columns, "mean_abs_shap": np.abs(class1).mean(axis=0)})
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(outdir / "generic_shap_importance.csv", index=False, encoding="utf-8-sig")
        try:
            shap.plots.beeswarm(plot_exp, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(outdir / "generic_shap_beeswarm.png", dpi=180, bbox_inches="tight")
            plt.close()
            shap.plots.bar(plot_exp, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(outdir / "generic_shap_bar.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception:
            plt.close("all")
    except Exception:
        (outdir / "generic_shap_error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def aggregate_shap(cv_root: Path, summary_root: Path) -> pd.DataFrame:
    rows = []
    for path in cv_root.glob("*/**/generic_shap_importance.csv"):
        # Expected: cv/target/feature_set/TabPFN/folds/fold_1/shap/generic_shap_importance.csv
        parts = path.relative_to(cv_root).parts
        if len(parts) < 3:
            continue
        target, feature_set = parts[0], parts[1]
        df = pd.read_csv(path)
        if any(re.search(r"\.\d+$", str(x)) for x in df["feature"]):
            raise ValueError(f"SHAP 汇总检测到重复特征: {path}")
        df["target"] = target
        df["feature_set"] = feature_set
        df["model"] = "TabPFN"
        df["fold"] = 1
        df["rank"] = df["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    ensure_dir(summary_root)
    long_df.to_csv(summary_root / "shap_long_table.csv", index=False, encoding="utf-8-sig")
    long_df[long_df["rank"] <= 20].to_csv(summary_root / "shap_top20.csv", index=False, encoding="utf-8-sig")
    mean_df = long_df.groupby(["feature_set", "feature"], as_index=False)["mean_abs_shap"].mean().sort_values(["feature_set", "mean_abs_shap"], ascending=[True, False])
    mean_df.to_csv(summary_root / "shap_mean_importance.csv", index=False, encoding="utf-8-sig")
    freq = long_df.assign(top10=long_df["rank"] <= 10).groupby(["feature_set", "feature"], as_index=False)["top10"].sum().rename(columns={"top10": "top10_frequency"})
    freq.to_csv(summary_root / "shap_top10_frequency.csv", index=False, encoding="utf-8-sig")
    return long_df


# -----------------------------------------------------------------------------
# Worker data loading
# -----------------------------------------------------------------------------
def configure_worker_threads(n_jobs: int) -> None:
    value = str(max(1, int(n_jobs)))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = value
    np.random.seed(RANDOM_STATE)


def load_clean_data_for_worker(cfg: WorkerConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_name)
    df = canonicalize_columns(df)
    tumor_col = cfg.feature_sets["classic_preop"][-1]
    protected = [cfg.id_col, cfg.time_col] + [t for t in ALL_TARGETS if t in df.columns]
    cleaned, _ = clean_input_dataframe(df, tumor_size_col=tumor_col, protected_cols=protected)
    cleaned[cfg.id_col] = cleaned[cfg.id_col].map(normalize_id)
    cleaned[cfg.time_col] = pd.to_datetime(cleaned[cfg.time_col], errors="coerce")
    return cleaned


def prepare_target(df: pd.DataFrame, target: str, id_col: str, require_date: bool = False, time_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    task = df[df[target].notna()].copy()
    if require_date:
        assert time_col is not None
        task = task[task[time_col].notna()].copy()
    y = sanitize_binary_y(task[target])
    task = task.loc[y.index].copy()
    y = y.loc[task.index]
    sample_ids = task[id_col].map(normalize_id)
    return task, y, sample_ids


# -----------------------------------------------------------------------------
# CV feature-set worker
# -----------------------------------------------------------------------------
def empty_model_result(phase: str, target: str, fs: str, model: str, status: str, error: str, fold: int = 0, n_samples: int = 0, n_train: int = 0, n_val: int = 0, n_test: int = 0, n_features: int = 0) -> ModelResult:
    return ModelResult(
        phase=phase, target=target, feature_set=fs, model=model, status=status,
        error_message=error, n_samples=n_samples, n_train=n_train, n_val=n_val,
        n_test=n_test, n_features=n_features, fold=fold, threshold=np.nan,
        AUROC=np.nan, AUPRC=np.nan, Accuracy=np.nan, Sensitivity=np.nan,
        Specificity=np.nan, Precision=np.nan, F1=np.nan, BrierScore=np.nan,
        TN=0, FP=0, FN=0, TP=0,
    )


def run_cv_feature_set_worker(feature_set: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = WorkerConfig(**cfg_dict)
    configure_worker_threads(cfg.model_n_jobs)
    df = load_clean_data_for_worker(cfg)
    fold_long = read_fold_long(cfg.fold_file)
    cv_root = ensure_dir(Path(cfg.output_root) / "cv")
    worker_log = cv_root / f"worker_{feature_set}.log"
    log_line(f"CV worker started: {feature_set} ({FEATURE_SET_DISPLAY[feature_set]})", worker_log)
    fs_cols = cfg.feature_sets[feature_set]
    run_models = [m for m in cfg.models if m in CORE_MODELS]
    # Stage baselines are run only in the ICPI worker, once per target.
    if feature_set == "full_data":
        run_models += [m for m in cfg.models if m in STAGE_MODELS]

    result_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    cal_rows: List[Dict[str, Any]] = []
    dca_rows: List[pd.DataFrame] = []
    boot_rows: List[pd.DataFrame] = []

    for target in ALL_TARGETS:
        task, y, sample_ids = prepare_target(df, target, cfg.id_col)
        folds = validate_target_folds(fold_long, target, task, sample_ids)
        for model_name in run_models:
            available, reason = dependency_status(model_name, cfg)
            model_fold_results: List[ModelResult] = []
            model_preds: List[pd.DataFrame] = []
            model_dir = ensure_dir(cv_root / target / (feature_set if model_name in CORE_MODELS else f"{model_name}_only") / model_name)
            if not available:
                result_rows.append(asdict(empty_model_result("cv", target, feature_set, model_name, "dependency_missing", reason, n_samples=len(task), n_features=len(fs_cols))))
                continue
            for fold in range(1, N_OUTER_FOLDS + 1):
                train_idx, val_idx, test_idx = split_indices_from_fold(folds, fold)
                log_line(f"CV | {FEATURE_SET_DISPLAY[feature_set]} | {target} | {model_name} | fold {fold}/5", worker_log)
                try:
                    if model_name in STAGE_MODELS:
                        stage_col = cfg.cnlc_col if model_name == "CNLC" else cfg.bclc_col
                        assert stage_col is not None
                        X_raw = task[[stage_col]].copy()
                        X_train, X_val, X_test = X_raw.loc[train_idx], X_raw.loc[val_idx], X_raw.loc[test_idx]
                        n_features = 1
                    else:
                        X_raw = task[fs_cols].copy()
                        pre = FoldPreprocessor()
                        X_train = pre.fit_transform(X_raw.loc[train_idx])
                        X_val = pre.transform(X_raw.loc[val_idx])
                        X_test = pre.transform(X_raw.loc[test_idx])
                        n_features = X_train.shape[1]
                        fold_pre_dir = ensure_dir(cv_root / target / feature_set / "preprocessing" / f"fold_{fold}")
                        save_json(pre.metadata(), fold_pre_dir / "preprocessor_meta.json")
                        if cfg.save_fold_data:
                            for name, raw_idx, proc in [
                                ("train", train_idx, X_train), ("val", val_idx, X_val), ("test", test_idx, X_test)
                            ]:
                                raw_out = X_raw.loc[raw_idx].copy()
                                raw_out["y"] = y.loc[raw_idx].to_numpy()
                                if cfg.save_direct_identifiers:
                                    raw_out["sample_id"] = sample_ids.loc[raw_idx].to_numpy()
                                raw_out.to_csv(fold_pre_dir / f"{name}_raw.csv", index=False, encoding="utf-8-sig")
                                proc_out = proc.copy()
                                proc_out["y"] = y.loc[raw_idx].to_numpy()
                                if cfg.save_direct_identifiers:
                                    proc_out["sample_id"] = sample_ids.loc[raw_idx].to_numpy()
                                proc_out.to_csv(fold_pre_dir / f"{name}_processed.csv", index=False, encoding="utf-8-sig")

                    y_train = y.loc[train_idx].to_numpy(dtype=int)
                    y_val = y.loc[val_idx].to_numpy(dtype=int)
                    y_test = y.loc[test_idx].to_numpy(dtype=int)
                    model = make_model(model_name, "cv", cfg)
                    model.fit(X_train, y_train, X_val, y_val)
                    val_prob = model.predict_proba_1(X_val)
                    threshold, roc_df = compute_youden_threshold(y_val, val_prob)
                    test_prob = model.predict_proba_1(X_test)
                    metrics = compute_metrics(y_test, test_prob, threshold)
                    fold_dir = ensure_dir(model_dir / "folds" / f"fold_{fold}")
                    if not roc_df.empty:
                        roc_df.to_csv(fold_dir / "val_roc_curve.csv", index=False, encoding="utf-8-sig")
                    pred = pd.DataFrame({
                        "phase": "cv", "target": target,
                        "feature_set": feature_set if model_name in CORE_MODELS else f"{model_name}_only",
                        "model": model_name, "fold": fold, "split": "test",
                        "sample_index": test_idx.astype(int),
                        "y_true": y_test, "y_prob": test_prob,
                        "threshold": threshold, "y_pred": (test_prob >= threshold).astype(int),
                    })
                    if cfg.save_direct_identifiers:
                        pred["sample_id"] = sample_ids.loc[test_idx].map(normalize_id).to_numpy()
                    pred.to_csv(fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
                    model_preds.append(pred)
                    row = ModelResult(
                        phase="cv", target=target,
                        feature_set=feature_set if model_name in CORE_MODELS else f"{model_name}_only",
                        model=model_name, status="ok", error_message="", n_samples=len(task),
                        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
                        n_features=n_features, fold=fold, threshold=metrics["threshold"],
                        AUROC=metrics["AUROC"], AUPRC=metrics["AUPRC"], Accuracy=metrics["Accuracy"],
                        Sensitivity=metrics["Sensitivity"], Specificity=metrics["Specificity"],
                        Precision=metrics["Precision"], F1=metrics["F1"], BrierScore=metrics["BrierScore"],
                        TN=metrics["TN"], FP=metrics["FP"], FN=metrics["FN"], TP=metrics["TP"],
                    )
                    model_fold_results.append(row)
                    if model_name == "TabPFN" and fold == 1 and cfg.run_shap:
                        run_generic_shap(model, X_train, X_test, fold_dir / "shap", cfg.shap_background, cfg.shap_test)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    error_dir = ensure_dir(model_dir / "folds" / f"fold_{fold}")
                    (error_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    log_line(f"ERROR | CV | {feature_set} | {target} | {model_name} | fold {fold}: {err}", worker_log)
                    model_fold_results.append(empty_model_result("cv", target, feature_set, model_name, "error", err, fold=fold, n_samples=len(task), n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx), n_features=len(fs_cols)))

            for r in model_fold_results:
                result_rows.append(asdict(r))
            if model_preds:
                oof = pd.concat(model_preds, ignore_index=True)
                oof.to_csv(model_dir / "all_test_predictions.csv", index=False, encoding="utf-8-sig")
                fold_metrics = pd.DataFrame([asdict(r) for r in model_fold_results])
                fold_metrics.to_csv(model_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
                pred_rows.append(oof)
                y_oof = oof["y_true"].to_numpy(int)
                p_oof = oof["y_prob"].to_numpy(float)
                pred_oof = oof["y_pred"].to_numpy(int)
                intercept, slope = calibration_intercept_slope(y_oof, p_oof)
                cal_row = {
                    "target": target,
                    "feature_set": oof["feature_set"].iloc[0],
                    "model": model_name,
                    "CalibrationIntercept": intercept,
                    "CalibrationSlope": slope,
                    "BrierScore": float(brier_score_loss(y_oof, p_oof)),
                }
                cal_rows.append(cal_row)
                save_json(cal_row, model_dir / "calibration_metrics.json")
                frac, mean_pred = calibration_curve(y_oof, p_oof, n_bins=min(10, max(3, len(y_oof) // 20)), strategy="quantile")
                pd.DataFrame({"mean_predicted_probability": mean_pred, "observed_event_rate": frac}).to_csv(model_dir / "calibration_curve_points.csv", index=False, encoding="utf-8-sig")
                dca = decision_curve_df(y_oof, p_oof)
                dca.insert(0, "model", model_name)
                dca.insert(0, "feature_set", oof["feature_set"].iloc[0])
                dca.insert(0, "target", target)
                dca.to_csv(model_dir / "decision_curve_analysis.csv", index=False, encoding="utf-8-sig")
                dca_rows.append(dca)
                boot = bootstrap_ci(y_oof, p_oof, pred_oof, cfg.bootstrap_rounds)
                boot.insert(0, "model", model_name)
                boot.insert(0, "feature_set", oof["feature_set"].iloc[0])
                boot.insert(0, "target", target)
                boot.to_csv(model_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
                boot_rows.append(boot)

    worker_dir = ensure_dir(cv_root / "_workers")
    pd.DataFrame(result_rows).to_csv(worker_dir / f"{feature_set}_fold_metrics.csv", index=False, encoding="utf-8-sig")
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(worker_dir / f"{feature_set}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(worker_dir / f"{feature_set}_calibration.csv", index=False, encoding="utf-8-sig")
    if dca_rows:
        pd.concat(dca_rows, ignore_index=True).to_csv(worker_dir / f"{feature_set}_dca.csv", index=False, encoding="utf-8-sig")
    if boot_rows:
        pd.concat(boot_rows, ignore_index=True).to_csv(worker_dir / f"{feature_set}_bootstrap.csv", index=False, encoding="utf-8-sig")
    log_line(f"CV worker completed: {feature_set}", worker_log)
    return {"feature_set": feature_set, "status": "completed", "rows": len(result_rows)}


# -----------------------------------------------------------------------------
# Temporal validation — complete development-set training
# -----------------------------------------------------------------------------
@dataclass
class TemporalPrimaryResult:
    split_scheme: str
    target: str
    feature_set: str
    model: str
    status: str
    error_message: str
    seed_specification: str
    n_total_analyzable: int
    n_training: int
    n_gap: int
    n_test: int
    n_features: int
    training_event_n: int
    training_non_event_n: int
    test_event_n: int
    test_non_event_n: int
    threshold_source: str
    threshold: float
    AUROC: float
    AUPRC: float
    Accuracy: float
    Sensitivity: float
    Specificity: float
    Precision: float
    F1: float
    BrierScore: float
    CalibrationIntercept: float
    CalibrationSlope: float
    TN: int
    FP: int
    FN: int
    TP: int
    fixed_epochs_or_estimators: float
    ensemble_members: int


def stable_seed(*parts: Any, base: int = RANDOM_STATE) -> int:
    text_value = "|".join(map(str, parts)).encode("utf-8")
    value = int(hashlib.sha256(text_value).hexdigest()[:8], 16)
    return int((value + base) % (2**31 - 1))


def temporal_indices(task: pd.DataFrame, time_col: str) -> Tuple[pd.Index, pd.Index, pd.Index]:
    dates = pd.to_datetime(task[time_col], errors="coerce")
    training = task.index[(dates >= TEMPORAL_DEV_START) & (dates <= TEMPORAL_DEV_END)]
    gap = task.index[(dates >= TEMPORAL_GAP_START) & (dates <= TEMPORAL_GAP_END)]
    test = task.index[(dates >= TEMPORAL_VAL_START) & (dates <= TEMPORAL_VAL_END)]
    return pd.Index(training), pd.Index(gap), pd.Index(test)


def build_study_id_map(df: pd.DataFrame) -> Dict[Any, str]:
    return {idx: f"HCC{position:04d}" for position, idx in enumerate(df.index, start=1)}


def serializable_index(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return str(value)


def temporal_model_available(
    model_name: str,
    cnlc_col: Optional[str],
    bclc_col: Optional[str],
) -> Tuple[bool, str]:
    if model_name == "TabPFN":
        return TABPFN_AVAILABLE, "tabpfn missing"
    if model_name == "TabNet":
        return TABNET_AVAILABLE, "pytorch-tabnet missing"
    if model_name == "XGBoost":
        return XGB_AVAILABLE, "xgboost missing"
    if model_name == "LightGBM":
        return LGBM_AVAILABLE, "lightgbm missing"
    if model_name == "RandomForest":
        return True, ""
    if model_name == "CNLC":
        return cnlc_col is not None, "CNLC column missing"
    if model_name == "BCLC":
        return bclc_col is not None, "BCLC column missing"
    return False, f"unknown model: {model_name}"


class TemporalTabPFNModel(BaseModel):
    def __init__(self, checkpoint: str, seed: int):
        if not TABPFN_AVAILABLE:
            raise ImportError("tabpfn 未安装")
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val=None, y_val=None) -> "TemporalTabPFNModel":
        kwargs: Dict[str, Any] = {}
        if self.checkpoint:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(f"TabPFN checkpoint 不存在: {self.checkpoint}")
            kwargs["model_path"] = self.checkpoint
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        candidates = (
            {**kwargs, "device": device, "random_state": self.seed},
            {**kwargs, "device": device},
            {**kwargs, "random_state": self.seed},
            kwargs,
            {},
        )
        for candidate in candidates:
            try:
                self.model = TabPFNClassifier(**candidate)
                break
            except TypeError:
                continue
        if self.model is None:
            raise RuntimeError("无法构建 TabPFNClassifier")
        self.model.fit(X_train.to_numpy(dtype=np.float32), y_train.astype(int))
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class TemporalTabNetModel(BaseModel):
    """One fixed-epoch TabNet member; no validation set and no early stopping."""

    def __init__(self, seed: int):
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet 未安装")
        self.seed = int(seed)
        device_name = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        try:
            self.model = TabNetClassifier(verbose=0, seed=self.seed, device_name=device_name)
        except TypeError:
            self.model = TabNetClassifier(verbose=0, seed=self.seed)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val=None, y_val=None) -> "TemporalTabNetModel":
        batch_size = min(TEMPORAL_TABNET_BATCH_SIZE, max(16, len(X_train)))
        virtual_batch_size = min(
            TEMPORAL_TABNET_VIRTUAL_BATCH_SIZE,
            max(8, batch_size // 2),
        )
        if virtual_batch_size >= batch_size:
            virtual_batch_size = max(8, batch_size // 2)
        self.model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train.astype(int),
            max_epochs=TEMPORAL_TABNET_MAX_EPOCHS,
            patience=TEMPORAL_TABNET_PATIENCE,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False,
        )
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class TemporalXGBoostModel(BaseModel):
    def __init__(self, seed: int, n_jobs: int):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost 未安装")
        self.seed = int(seed)
        self.n_jobs = max(1, int(n_jobs))
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val=None, y_val=None) -> "TemporalXGBoostModel":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / pos) if pos else 1.0
        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=self.n_jobs,
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
        )
        self.model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train.astype(np.int32),
            verbose=False,
        )
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class TemporalLightGBMModel(BaseModel):
    def __init__(self, seed: int, n_jobs: int):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm 未安装")
        self.seed = int(seed)
        self.n_jobs = max(1, int(n_jobs))
        self.model = None
        self.mapping: Optional[Dict[str, str]] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val=None, y_val=None) -> "TemporalLightGBMModel":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / pos) if pos else 1.0
        self.model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
        )
        x_train, self.mapping = lightgbm_safe_df(X_train)
        self.model.fit(x_train, y_train.astype(int))
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        x, _ = lightgbm_safe_df(X, self.mapping)
        p = self.model.predict_proba(x)
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


class TemporalRandomForestModel(BaseModel):
    def __init__(self, seed: int, n_jobs: int):
        self.model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=max(1, int(n_jobs)),
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val=None, y_val=None) -> "TemporalRandomForestModel":
        self.model.fit(X_train.to_numpy(dtype=float), y_train.astype(int))
        return self

    def predict_proba_1(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X.to_numpy(dtype=float))
        return p[:, 1] if p.shape[1] >= 2 else np.zeros(len(X))


def make_temporal_model(model_name: str, seed: int, cfg: WorkerConfig) -> BaseModel:
    if model_name == "TabPFN":
        return TemporalTabPFNModel(cfg.tabpfn_checkpoint, seed)
    if model_name == "TabNet":
        return TemporalTabNetModel(seed)
    if model_name == "XGBoost":
        return TemporalXGBoostModel(seed, cfg.model_n_jobs)
    if model_name == "LightGBM":
        return TemporalLightGBMModel(seed, cfg.model_n_jobs)
    if model_name == "RandomForest":
        return TemporalRandomForestModel(seed, cfg.model_n_jobs)
    if model_name == "CNLC":
        if cfg.cnlc_col is None:
            raise ValueError("CNLC 列不存在")
        return StageRateModel(cfg.cnlc_col)
    if model_name == "BCLC":
        if cfg.bclc_col is None:
            raise ValueError("BCLC 列不存在")
        return StageRateModel(cfg.bclc_col)
    raise ValueError(f"未知模型: {model_name}")


def temporal_decision_curve_df(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    dca = decision_curve_df(y_true, y_prob)
    if dca.empty:
        return dca
    for key, value in reversed(list(metadata.items())):
        dca.insert(0, key, value)
    return dca


def temporal_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    rounds: int,
    rng_seed: int,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    point = compute_metrics(y, p, threshold)
    metric_names = [
        "AUROC", "AUPRC", "Accuracy", "Sensitivity",
        "Specificity", "Precision", "F1", "BrierScore",
    ]
    values: Dict[str, List[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(rng_seed)
    attempts = 0
    while min(len(v) for v in values.values()) < rounds and attempts < rounds * 50:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        metrics = compute_metrics(y[idx], p[idx], threshold)
        metrics["Accuracy"] = float(accuracy_score(y[idx], pred[idx]))
        metrics["Sensitivity"] = float(recall_score(y[idx], pred[idx], zero_division=0))
        metrics["Specificity"] = float(specificity_score(y[idx], pred[idx]))
        metrics["Precision"] = float(precision_score(y[idx], pred[idx], zero_division=0))
        metrics["F1"] = float(f1_score(y[idx], pred[idx], zero_division=0))
        for name in metric_names:
            if pd.notna(metrics[name]):
                values[name].append(float(metrics[name]))
    rows: List[Dict[str, Any]] = []
    for name in metric_names:
        arr = np.asarray(values[name], dtype=float)
        rows.append({
            **metadata,
            "metric": name,
            "point_estimate": point[name],
            "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "n_boot": int(len(arr)),
        })
    return pd.DataFrame(rows)


def paired_bootstrap_auroc(
    y_true: np.ndarray,
    reference_prob: np.ndarray,
    comparison_prob: np.ndarray,
    rounds: int,
    rng_seed: int,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    reference = np.asarray(reference_prob, dtype=float)
    comparison = np.asarray(comparison_prob, dtype=float)
    observed = safe_auroc(y, comparison) - safe_auroc(y, reference)
    rng = np.random.default_rng(rng_seed)
    differences: List[float] = []
    attempts = 0
    while len(differences) < rounds and attempts < rounds * 50:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        differences.append(
            safe_auroc(y[idx], comparison[idx]) - safe_auroc(y[idx], reference[idx])
        )
    arr = np.asarray(differences, dtype=float)
    return {
        "AUROC_difference_comparison_minus_TabPFN": observed,
        "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "probability_difference_gt_0": float(np.mean(arr > 0)) if len(arr) else np.nan,
        "two_sided_bootstrap_p": (
            float(min(1.0, 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))))
            if len(arr) else np.nan
        ),
        "n_boot": int(len(arr)),
    }


def run_temporal_primary_model(
    target: str,
    task: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    study_id_map: Dict[Any, str],
    feature_set: str,
    feature_cols: List[str],
    model_name: str,
    training_idx: pd.Index,
    gap_idx: pd.Index,
    test_idx: pd.Index,
    cfg: WorkerConfig,
    temporal_root: Path,
    log_file: Path,
    bootstrap_rounds: int,
    save_direct_identifiers: bool,
) -> Tuple[TemporalPrimaryResult, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    output_feature_set = feature_set if model_name in CORE_MODELS else f"{model_name}_only"
    model_dir = ensure_dir(temporal_root / target / output_feature_set / model_name)
    available, reason = temporal_model_available(model_name, cfg.cnlc_col, cfg.bclc_col)

    n_train = len(training_idx)
    n_test = len(test_idx)
    train_event_n = int(y.loc[training_idx].sum()) if n_train else 0
    test_event_n = int(y.loc[test_idx].sum()) if n_test else 0

    def empty_result(status: str, message: str) -> TemporalPrimaryResult:
        return TemporalPrimaryResult(
            split_scheme=TEMPORAL_SPLIT_NAME,
            target=target,
            feature_set=output_feature_set,
            model=model_name,
            status=status,
            error_message=message,
            seed_specification=(
                "42-61 probability mean ensemble" if model_name == "TabNet" else str(RANDOM_STATE)
            ),
            n_total_analyzable=len(task),
            n_training=n_train,
            n_gap=len(gap_idx),
            n_test=n_test,
            n_features=(1 if model_name in STAGE_MODELS else len(feature_cols)),
            training_event_n=train_event_n,
            training_non_event_n=n_train - train_event_n,
            test_event_n=test_event_n,
            test_non_event_n=n_test - test_event_n,
            threshold_source="prespecified_fixed_0.5",
            threshold=TEMPORAL_FIXED_THRESHOLD,
            AUROC=np.nan,
            AUPRC=np.nan,
            Accuracy=np.nan,
            Sensitivity=np.nan,
            Specificity=np.nan,
            Precision=np.nan,
            F1=np.nan,
            BrierScore=np.nan,
            CalibrationIntercept=np.nan,
            CalibrationSlope=np.nan,
            TN=0,
            FP=0,
            FN=0,
            TP=0,
            fixed_epochs_or_estimators=(
                float(TEMPORAL_TABNET_MAX_EPOCHS) if model_name == "TabNet" else np.nan
            ),
            ensemble_members=(len(TEMPORAL_TABNET_SEEDS) if model_name == "TabNet" else 1),
        )

    if not available:
        return empty_result("dependency_missing", reason), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    if n_train < 5 or n_test < 1 or len(np.unique(y.loc[training_idx])) < 2:
        return (
            empty_result("invalid_split", "training/test sample size or class distribution is invalid"),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
        )

    try:
        if model_name == "CNLC":
            assert cfg.cnlc_col is not None
            cols = [cfg.cnlc_col]
            X_raw = task[cols].copy()
            X_train = X_raw.loc[training_idx]
            X_test = X_raw.loc[test_idx]
            n_features = 1
            preprocessing_meta = {
                "fit_scope": "not_applicable_stage_rate_baseline",
                "feature_names": cols,
            }
        elif model_name == "BCLC":
            assert cfg.bclc_col is not None
            cols = [cfg.bclc_col]
            X_raw = task[cols].copy()
            X_train = X_raw.loc[training_idx]
            X_test = X_raw.loc[test_idx]
            n_features = 1
            preprocessing_meta = {
                "fit_scope": "not_applicable_stage_rate_baseline",
                "feature_names": cols,
            }
        else:
            X_raw = task[feature_cols].copy()
            preprocessor = FoldPreprocessor()
            X_train = preprocessor.fit_transform(X_raw.loc[training_idx])
            X_test = preprocessor.transform(X_raw.loc[test_idx])
            n_features = X_train.shape[1]
            preprocessing_meta = preprocessor.metadata()
            preprocessing_meta["fit_scope"] = "complete_temporal_training_set_only"
            save_json(preprocessing_meta, model_dir / "preprocessor_meta.json")

        y_train = y.loc[training_idx].to_numpy(dtype=int)
        y_test = y.loc[test_idx].to_numpy(dtype=int)
        extras: Dict[str, pd.DataFrame] = {}

        if model_name == "TabNet":
            member_probabilities: List[np.ndarray] = []
            member_metric_rows: List[Dict[str, Any]] = []
            member_prediction_frames: List[pd.DataFrame] = []
            for position, seed in enumerate(TEMPORAL_TABNET_SEEDS, start=1):
                log_line(
                    f"TABNET ENSEMBLE {position}/{len(TEMPORAL_TABNET_SEEDS)} | "
                    f"{target} | {FEATURE_SET_DISPLAY[feature_set]} | seed={seed} | "
                    f"epochs={TEMPORAL_TABNET_MAX_EPOCHS}",
                    log_file,
                )
                member = TemporalTabNetModel(seed)
                member.fit(X_train, y_train)
                member_prob = np.asarray(member.predict_proba_1(X_test), dtype=float)
                member_probabilities.append(member_prob)
                member_metrics = compute_metrics(y_test, member_prob, TEMPORAL_FIXED_THRESHOLD)
                member_metric_rows.append({
                    "target": target,
                    "feature_set": feature_set,
                    "display": FEATURE_SET_DISPLAY[feature_set],
                    "model": "TabNet",
                    "seed": seed,
                    "fixed_epochs": TEMPORAL_TABNET_MAX_EPOCHS,
                    "threshold": TEMPORAL_FIXED_THRESHOLD,
                    **member_metrics,
                })
                member_prediction_frames.append(pd.DataFrame({
                    "target": target,
                    "feature_set": feature_set,
                    "model": "TabNet",
                    "seed": seed,
                    "sample_index": [serializable_index(i) for i in test_idx],
                    "study_id": [study_id_map[i] for i in test_idx],
                    "y_true": y_test,
                    "y_prob": member_prob,
                }))
            test_prob = np.mean(np.vstack(member_probabilities), axis=0)
            member_metrics_df = pd.DataFrame(member_metric_rows)
            member_predictions_df = pd.concat(member_prediction_frames, ignore_index=True)
            member_metrics_df.to_csv(
                model_dir / "tabnet_ensemble_member_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            member_predictions_df.to_csv(
                model_dir / "tabnet_ensemble_member_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            extras["tabnet_seed_results"] = member_metrics_df
            extras["tabnet_seed_predictions"] = member_predictions_df
            fixed_epochs_or_estimators = float(TEMPORAL_TABNET_MAX_EPOCHS)
            seed_specification = "42-61 probability mean ensemble"
            ensemble_members = len(TEMPORAL_TABNET_SEEDS)
        else:
            model = make_temporal_model(model_name, RANDOM_STATE, cfg)
            model.fit(X_train, y_train)
            test_prob = np.asarray(model.predict_proba_1(X_test), dtype=float)
            fixed_epochs_or_estimators = {
                "XGBoost": 400.0,
                "LightGBM": 400.0,
                "RandomForest": 300.0,
            }.get(model_name, np.nan)
            seed_specification = str(RANDOM_STATE)
            ensemble_members = 1

        metrics = compute_metrics(y_test, test_prob, TEMPORAL_FIXED_THRESHOLD)
        calibration_intercept, calibration_slope = calibration_intercept_slope(y_test, test_prob)

        predictions = pd.DataFrame({
            "split_scheme": TEMPORAL_SPLIT_NAME,
            "target": target,
            "feature_set": output_feature_set,
            "model": model_name,
            "seed_specification": seed_specification,
            "sample_index": [serializable_index(i) for i in test_idx],
            "study_id": [study_id_map[i] for i in test_idx],
            "y_true": y_test,
            "y_prob": test_prob,
            "threshold": TEMPORAL_FIXED_THRESHOLD,
            "y_pred": (test_prob >= TEMPORAL_FIXED_THRESHOLD).astype(int),
        })
        if save_direct_identifiers:
            predictions["sample_id"] = patient_ids.loc[test_idx].to_numpy()
        predictions.to_csv(model_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        metadata = {
            "split_scheme": TEMPORAL_SPLIT_NAME,
            "target": target,
            "feature_set": output_feature_set,
            "model": model_name,
        }
        curves = curve_points(y_test, test_prob, metadata)
        curves.to_csv(model_dir / "roc_pr_curve_points.csv", index=False, encoding="utf-8-sig")
        dca = temporal_decision_curve_df(y_test, test_prob, metadata)
        dca.to_csv(model_dir / "decision_curve_analysis.csv", index=False, encoding="utf-8-sig")
        bootstrap = temporal_bootstrap_confidence_intervals(
            y_test,
            test_prob,
            TEMPORAL_FIXED_THRESHOLD,
            bootstrap_rounds,
            stable_seed(target, output_feature_set, model_name, "bootstrap"),
            metadata,
        )
        bootstrap.to_csv(
            model_dir / "bootstrap_confidence_intervals.csv",
            index=False,
            encoding="utf-8-sig",
        )

        result = TemporalPrimaryResult(
            split_scheme=TEMPORAL_SPLIT_NAME,
            target=target,
            feature_set=output_feature_set,
            model=model_name,
            status="ok",
            error_message="",
            seed_specification=seed_specification,
            n_total_analyzable=len(task),
            n_training=n_train,
            n_gap=len(gap_idx),
            n_test=n_test,
            n_features=n_features,
            training_event_n=train_event_n,
            training_non_event_n=n_train - train_event_n,
            test_event_n=test_event_n,
            test_non_event_n=n_test - test_event_n,
            threshold_source="prespecified_fixed_0.5",
            threshold=TEMPORAL_FIXED_THRESHOLD,
            AUROC=metrics["AUROC"],
            AUPRC=metrics["AUPRC"],
            Accuracy=metrics["Accuracy"],
            Sensitivity=metrics["Sensitivity"],
            Specificity=metrics["Specificity"],
            Precision=metrics["Precision"],
            F1=metrics["F1"],
            BrierScore=metrics["BrierScore"],
            CalibrationIntercept=calibration_intercept,
            CalibrationSlope=calibration_slope,
            TN=metrics["TN"],
            FP=metrics["FP"],
            FN=metrics["FN"],
            TP=metrics["TP"],
            fixed_epochs_or_estimators=fixed_epochs_or_estimators,
            ensemble_members=ensemble_members,
        )
        save_json(
            {"result": asdict(result), "preprocessing": preprocessing_meta},
            model_dir / "run_meta.json",
        )
        extras["dca"] = dca
        return result, predictions, curves, bootstrap, extras

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        (model_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log_line(
            f"ERROR | TEMPORAL | {target} | {output_feature_set} | "
            f"{model_name}: {error_message}",
            log_file,
        )
        return empty_result("error", error_message), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def summarize_tabnet_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    metric_names = [
        "AUROC", "AUPRC", "Accuracy", "Sensitivity",
        "Specificity", "Precision", "F1", "BrierScore",
    ]
    rows: List[Dict[str, Any]] = []
    for (target, feature_set), group in seed_df.groupby(["target", "feature_set"]):
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append({
                "target": target,
                "feature_set": feature_set,
                "display": FEATURE_SET_DISPLAY.get(feature_set, feature_set),
                "model": "TabNet",
                "metric": metric,
                "n_seeds": len(values),
                "median": float(np.median(values)) if len(values) else np.nan,
                "minimum": float(np.min(values)) if len(values) else np.nan,
                "maximum": float(np.max(values)) if len(values) else np.nan,
                "q025": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "q975": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def write_temporal_metric_wide_tables(results: pd.DataFrame, temporal_root: Path) -> None:
    if results.empty:
        return
    ok = results[results["status"] == "ok"].copy()
    for metric in [
        "AUROC", "AUPRC", "Accuracy", "Sensitivity",
        "Specificity", "Precision", "F1", "BrierScore",
    ]:
        wide = ok.pivot_table(
            index=["target"],
            columns=["feature_set", "model"],
            values=metric,
            aggfunc="first",
        )
        wide.to_csv(temporal_root / f"metric_{metric}_wide.csv", encoding="utf-8-sig")


def export_temporal_excel_summary(
    temporal_root: Path,
    tables: Dict[str, pd.DataFrame],
) -> None:
    path = temporal_root / "temporal_S3_summary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wrote_sheet = False
        for sheet, frame in tables.items():
            if frame is None:
                continue
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)
            wrote_sheet = True
        if not wrote_sheet:
            pd.DataFrame({"status": ["no output"]}).to_excel(
                writer, sheet_name="status", index=False
            )


def run_temporal_validation(
    df: pd.DataFrame,
    cfg: WorkerConfig,
    temporal_bootstrap_rounds: int,
    paired_bootstrap_rounds: int,
    save_direct_identifiers: bool,
) -> Dict[str, pd.DataFrame]:
    configure_worker_threads(cfg.model_n_jobs)
    temporal_root = ensure_dir(Path(cfg.output_root) / "temporal")
    log_file = temporal_root / "run.log"
    log_line("Prespecified S3 full-development interval-gap temporal validation started", log_file)

    study_id_map = build_study_id_map(df)
    all_results: List[Dict[str, Any]] = []
    all_predictions: List[pd.DataFrame] = []
    all_curves: List[pd.DataFrame] = []
    all_bootstrap: List[pd.DataFrame] = []
    all_dca: List[pd.DataFrame] = []
    all_assignments: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    tabnet_seed_rows: List[pd.DataFrame] = []
    tabnet_seed_prediction_rows: List[pd.DataFrame] = []

    core_models = [model for model in cfg.models if model in CORE_MODELS]
    stage_models = [model for model in cfg.models if model in STAGE_MODELS]
    total_primary = len(TEMPORAL_TARGETS) * (
        len(FEATURE_SET_ORDER) * len(core_models) + len(stage_models)
    )
    primary_counter = 0

    for target in TEMPORAL_TARGETS:
        log_line("=" * 100, log_file)
        log_line(f"TARGET | {target}", log_file)
        task, y, patient_ids = prepare_target(
            df,
            target,
            cfg.id_col,
            require_date=True,
            time_col=cfg.time_col,
        )
        training_idx, gap_idx, test_idx = temporal_indices(task, cfg.time_col)

        role_sets = [set(training_idx), set(gap_idx), set(test_idx)]
        labels = ["training", "gap", "held_out_validation"]
        for i in range(len(role_sets)):
            for j in range(i + 1, len(role_sets)):
                overlap = role_sets[i] & role_sets[j]
                if overlap:
                    raise ValueError(
                        f"{target}: {labels[i]} 与 {labels[j]} 存在重叠: "
                        f"{list(overlap)[:10]}"
                    )
        if len(training_idx) < 5 or len(test_idx) < 1:
            raise ValueError(f"{target}: training/test 样本量不足")
        if len(np.unique(y.loc[training_idx])) < 2 or len(np.unique(y.loc[test_idx])) < 2:
            raise ValueError(f"{target}: training 或 temporal validation 只有一个类别")

        training_events = int(y.loc[training_idx].sum())
        test_events = int(y.loc[test_idx].sum())
        event_rows.append({
            "split_scheme": TEMPORAL_SPLIT_NAME,
            "target": target,
            "training_n": len(training_idx),
            "training_event_n": training_events,
            "training_non_event_n": len(training_idx) - training_events,
            "training_event_rate": training_events / len(training_idx),
            "gap_n": len(gap_idx),
            "gap_event_n": int(y.loc[gap_idx].sum()) if len(gap_idx) else 0,
            "validation_n": len(test_idx),
            "validation_event_n": test_events,
            "validation_non_event_n": len(test_idx) - test_events,
            "validation_event_rate": test_events / len(test_idx),
            "training_start": str(TEMPORAL_DEV_START.date()),
            "training_end": str(TEMPORAL_DEV_END.date()),
            "gap_start": str(TEMPORAL_GAP_START.date()),
            "gap_end": str(TEMPORAL_GAP_END.date()),
            "validation_start": str(TEMPORAL_VAL_START.date()),
            "validation_end": str(TEMPORAL_VAL_END.date()),
        })

        for role, indices in [
            ("temporal_training", training_idx),
            ("gap_excluded", gap_idx),
            ("temporal_validation", test_idx),
        ]:
            for idx in indices:
                row: Dict[str, Any] = {
                    "split_scheme": TEMPORAL_SPLIT_NAME,
                    "target": target,
                    "sample_index": serializable_index(idx),
                    "study_id": study_id_map[idx],
                    "role": role,
                    "y": int(y.loc[idx]),
                }
                if save_direct_identifiers:
                    row["sample_id"] = patient_ids.loc[idx]
                    row["initial_treatment_date"] = str(
                        pd.Timestamp(task.loc[idx, cfg.time_col]).date()
                    )
                all_assignments.append(row)

        for feature_set in FEATURE_SET_ORDER:
            feature_cols = cfg.feature_sets[feature_set]
            for model_name in core_models:
                primary_counter += 1
                log_line(
                    f"PRIMARY {primary_counter}/{total_primary} | {target} | "
                    f"{FEATURE_SET_DISPLAY[feature_set]} | {model_name}",
                    log_file,
                )
                result, predictions, curves, bootstrap, extras = run_temporal_primary_model(
                    target=target,
                    task=task,
                    y=y,
                    patient_ids=patient_ids,
                    study_id_map=study_id_map,
                    feature_set=feature_set,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    training_idx=training_idx,
                    gap_idx=gap_idx,
                    test_idx=test_idx,
                    cfg=cfg,
                    temporal_root=temporal_root,
                    log_file=log_file,
                    bootstrap_rounds=temporal_bootstrap_rounds,
                    save_direct_identifiers=save_direct_identifiers,
                )
                all_results.append(asdict(result))
                if not predictions.empty:
                    all_predictions.append(predictions)
                if not curves.empty:
                    all_curves.append(curves)
                if not bootstrap.empty:
                    all_bootstrap.append(bootstrap)
                if "dca" in extras and not extras["dca"].empty:
                    all_dca.append(extras["dca"])
                if "tabnet_seed_results" in extras and not extras["tabnet_seed_results"].empty:
                    tabnet_seed_rows.append(extras["tabnet_seed_results"])
                if "tabnet_seed_predictions" in extras and not extras["tabnet_seed_predictions"].empty:
                    tabnet_seed_prediction_rows.append(extras["tabnet_seed_predictions"])

        for model_name in stage_models:
            primary_counter += 1
            log_line(
                f"PRIMARY {primary_counter}/{total_primary} | {target} | {model_name}_only",
                log_file,
            )
            result, predictions, curves, bootstrap, extras = run_temporal_primary_model(
                target=target,
                task=task,
                y=y,
                patient_ids=patient_ids,
                study_id_map=study_id_map,
                feature_set="full_data",
                feature_cols=[],
                model_name=model_name,
                training_idx=training_idx,
                gap_idx=gap_idx,
                test_idx=test_idx,
                cfg=cfg,
                temporal_root=temporal_root,
                log_file=log_file,
                bootstrap_rounds=temporal_bootstrap_rounds,
                save_direct_identifiers=save_direct_identifiers,
            )
            all_results.append(asdict(result))
            if not predictions.empty:
                all_predictions.append(predictions)
            if not curves.empty:
                all_curves.append(curves)
            if not bootstrap.empty:
                all_bootstrap.append(bootstrap)
            if "dca" in extras and not extras["dca"].empty:
                all_dca.append(extras["dca"])

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    bootstrap_df = pd.concat(all_bootstrap, ignore_index=True) if all_bootstrap else pd.DataFrame()
    dca_df = pd.concat(all_dca, ignore_index=True) if all_dca else pd.DataFrame()
    assignments_df = pd.DataFrame(all_assignments)
    events_df = pd.DataFrame(event_rows)
    tabnet_seed_df = pd.concat(tabnet_seed_rows, ignore_index=True) if tabnet_seed_rows else pd.DataFrame()
    tabnet_seed_predictions_df = (
        pd.concat(tabnet_seed_prediction_rows, ignore_index=True)
        if tabnet_seed_prediction_rows else pd.DataFrame()
    )
    tabnet_seed_summary_df = summarize_tabnet_seed_results(tabnet_seed_df)

    results_df.to_csv(temporal_root / "temporal_long_results.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(temporal_root / "predictions_all.csv", index=False, encoding="utf-8-sig")
    curves_df.to_csv(temporal_root / "roc_pr_curve_points.csv", index=False, encoding="utf-8-sig")
    bootstrap_df.to_csv(temporal_root / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    dca_df.to_csv(temporal_root / "dca_all.csv", index=False, encoding="utf-8-sig")
    assignments_df.to_csv(temporal_root / "train_test_assignments_all.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(temporal_root / "split_event_distribution.csv", index=False, encoding="utf-8-sig")
    tabnet_seed_df.to_csv(temporal_root / "tabnet_ensemble_member_metrics.csv", index=False, encoding="utf-8-sig")
    tabnet_seed_predictions_df.to_csv(
        temporal_root / "tabnet_ensemble_member_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tabnet_seed_summary_df.to_csv(
        temporal_root / "tabnet_ensemble_member_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not results_df.empty:
        ok = results_df[results_df["status"] == "ok"].copy()
        best = (
            ok.sort_values(
                ["target", "AUROC", "AUPRC", "BrierScore"],
                ascending=[True, False, False, True],
            )
            .groupby("target", as_index=False)
            .head(1)
        )
    else:
        best = pd.DataFrame()
    best.to_csv(temporal_root / "best_by_target.csv", index=False, encoding="utf-8-sig")
    write_temporal_metric_wide_tables(results_df, temporal_root)

    paired_rows: List[Dict[str, Any]] = []
    if not predictions_df.empty:
        core_prediction_df = predictions_df[
            predictions_df["feature_set"].isin(FEATURE_SET_ORDER)
        ]
        for (target, feature_set), group in core_prediction_df.groupby(["target", "feature_set"]):
            reference = group[group["model"] == "TabPFN"].sort_values("sample_index")
            if reference.empty:
                continue
            for model_name in [m for m in core_models if m != "TabPFN"]:
                comparison = group[group["model"] == model_name].sort_values("sample_index")
                if comparison.empty:
                    continue
                merged = reference[["sample_index", "y_true", "y_prob"]].merge(
                    comparison[["sample_index", "y_true", "y_prob"]],
                    on="sample_index",
                    suffixes=("_TabPFN", "_comparison"),
                    validate="one_to_one",
                )
                if not np.array_equal(
                    merged["y_true_TabPFN"].to_numpy(),
                    merged["y_true_comparison"].to_numpy(),
                ):
                    raise ValueError(
                        f"paired bootstrap 标签不一致: {target} {feature_set} {model_name}"
                    )
                comparison_result = paired_bootstrap_auroc(
                    merged["y_true_TabPFN"].to_numpy(dtype=int),
                    merged["y_prob_TabPFN"].to_numpy(dtype=float),
                    merged["y_prob_comparison"].to_numpy(dtype=float),
                    paired_bootstrap_rounds,
                    stable_seed(target, feature_set, model_name, "paired_bootstrap"),
                )
                paired_rows.append({
                    "target": target,
                    "feature_set": feature_set,
                    "reference_model": "TabPFN",
                    "comparison_model": model_name,
                    **comparison_result,
                })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(
        temporal_root / "paired_bootstrap_vs_TabPFN.csv",
        index=False,
        encoding="utf-8-sig",
    )

    export_temporal_excel_summary(temporal_root, {
        "primary_results": results_df,
        "best_by_target": best,
        "event_distribution": events_df,
        "bootstrap_CI": bootstrap_df,
        "paired_vs_TabPFN": paired_df,
        "TabNet_members": tabnet_seed_df,
        "TabNet_member_summary": tabnet_seed_summary_df,
    })

    error_count = int((results_df["status"] == "error").sum()) if not results_df.empty else 0
    dependency_missing_count = (
        int((results_df["status"] == "dependency_missing").sum())
        if not results_df.empty else 0
    )
    expected_rows = len(TEMPORAL_TARGETS) * (
        len(FEATURE_SET_ORDER) * len(core_models) + len(stage_models)
    )
    completion = {
        "completed": True,
        "protocol": "temporal_S3_full_development_gap3m",
        "primary_rows": len(results_df),
        "expected_primary_rows_for_requested_models": expected_rows,
        "primary_ok": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "primary_errors": error_count,
        "dependency_missing": dependency_missing_count,
        "tabnet_ensemble_member_rows": len(tabnet_seed_df),
        "expected_tabnet_member_rows_if_TabNet_requested": (
            len(TEMPORAL_TARGETS) * len(FEATURE_SET_ORDER) * len(TEMPORAL_TABNET_SEEDS)
            if "TabNet" in core_models else 0
        ),
        "threshold": TEMPORAL_FIXED_THRESHOLD,
        "internal_split": False,
        "internal_cross_validation": False,
        "temporal_bootstrap_rounds": temporal_bootstrap_rounds,
        "paired_bootstrap_rounds": paired_bootstrap_rounds,
        "direct_identifiers_exported": save_direct_identifiers,
        "output_root": str(temporal_root.resolve()),
    }
    save_json(completion, temporal_root / "completion_summary.json")
    log_line(f"Temporal validation completed | {completion}", log_file)
    if error_count:
        log_line("注意：存在模型错误，请查看各模型目录中的 error.txt", log_file)

    return {
        "results": results_df,
        "predictions": predictions_df,
        "curves": curves_df,
        "bootstrap": bootstrap_df,
        "dca": dca_df,
        "events": events_df,
        "assignments": assignments_df,
        "paired": paired_df,
        "tabnet_members": tabnet_seed_df,
        "tabnet_member_predictions": tabnet_seed_predictions_df,
        "tabnet_summary": tabnet_seed_summary_df,
        "best": best,
    }


# -----------------------------------------------------------------------------
# Phase orchestration and summaries
# -----------------------------------------------------------------------------
def run_phase_in_executor(executor: ProcessPoolExecutor, worker_fn, phase_name: str, cfg: WorkerConfig) -> None:
    log_line(f"Starting {phase_name} with 3 feature-set workers", Path(cfg.output_root) / "master.log")
    futures = {executor.submit(worker_fn, fs, asdict(cfg)): fs for fs in FEATURE_SET_ORDER}
    for future in as_completed(futures):
        fs = futures[future]
        try:
            result = future.result()
            log_line(f"{phase_name} worker result: {result}", Path(cfg.output_root) / "master.log")
        except Exception as exc:
            log_line(f"FATAL worker failure | {phase_name} | {fs}: {type(exc).__name__}: {exc}", Path(cfg.output_root) / "master.log")
            raise


def concat_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_cv(output_root: Path) -> Dict[str, pd.DataFrame]:
    cv_root = output_root / "cv"
    workers = cv_root / "_workers"
    fold_metrics = concat_csvs(workers.glob("*_fold_metrics.csv"))
    oof = concat_csvs(workers.glob("*_oof_predictions.csv"))
    calibration = concat_csvs(workers.glob("*_calibration.csv"))
    dca = concat_csvs(workers.glob("*_dca.csv"))
    bootstrap = concat_csvs(workers.glob("*_bootstrap.csv"))

    ok = fold_metrics[fold_metrics["status"] == "ok"].copy() if not fold_metrics.empty else pd.DataFrame()
    metric_cols = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "BrierScore"]
    if not ok.empty:
        grouped = ok.groupby(["target", "feature_set", "model"], as_index=False)
        summary = grouped.agg(
            n_samples=("n_samples", "max"), n_features=("n_features", "max"),
            **{f"{m}_mean": (m, "mean") for m in metric_cols},
            **{f"{m}_std": (m, "std") for m in metric_cols},
        )
        if not calibration.empty:
            summary = summary.merge(calibration, on=["target", "feature_set", "model"], how="left")
    else:
        summary = pd.DataFrame()
    fold_metrics.to_csv(cv_root / "fold_metrics_all.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(cv_root / "oof_predictions_all.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(cv_root / "calibration_all.csv", index=False, encoding="utf-8-sig")
    dca.to_csv(cv_root / "dca_all.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(cv_root / "bootstrap_all.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(cv_root / "overall_summary_all_models.csv", index=False, encoding="utf-8-sig")

    leading = pd.DataFrame()
    if not summary.empty:
        leading = summary.sort_values(["target", "AUROC_mean", "AUPRC_mean", "BrierScore_mean"], ascending=[True, False, False, True]).groupby("target", as_index=False).head(1)
        leading.to_csv(cv_root / "leading_configurations.csv", index=False, encoding="utf-8-sig")
    return {
        "fold_metrics": fold_metrics, "oof": oof, "calibration": calibration,
        "dca": dca, "bootstrap": bootstrap, "summary": summary, "leading": leading,
    }


def summarize_temporal(output_root: Path) -> Dict[str, pd.DataFrame]:
    root = output_root / "temporal"

    def read_csv(name: str) -> pd.DataFrame:
        path = root / name
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "results": read_csv("temporal_long_results.csv"),
        "predictions": read_csv("predictions_all.csv"),
        "curves": read_csv("roc_pr_curve_points.csv"),
        "bootstrap": read_csv("bootstrap_confidence_intervals.csv"),
        "dca": read_csv("dca_all.csv"),
        "events": read_csv("split_event_distribution.csv"),
        "assignments": read_csv("train_test_assignments_all.csv"),
        "paired": read_csv("paired_bootstrap_vs_TabPFN.csv"),
        "tabnet_members": read_csv("tabnet_ensemble_member_metrics.csv"),
        "tabnet_member_predictions": read_csv("tabnet_ensemble_member_predictions.csv"),
        "tabnet_summary": read_csv("tabnet_ensemble_member_summary.csv"),
        "best": read_csv("best_by_target.csv"),
    }


def export_source_data(
    output_root: Path,
    cv: Dict[str, pd.DataFrame],
    temporal: Dict[str, pd.DataFrame],
    shap_long: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
) -> None:
    summary = ensure_dir(output_root / "summary")
    source1 = summary / "Source_Data_1_full_cross_validation_metrics_current.xlsx"
    with pd.ExcelWriter(source1, engine="openpyxl") as writer:
        cv["summary"].to_excel(writer, sheet_name="overall_summary", index=False)
        cv["fold_metrics"].to_excel(writer, sheet_name="fold_metrics", index=False)
        cv["oof"].head(1_000_000).to_excel(writer, sheet_name="oof_predictions", index=False)
        cv["calibration"].to_excel(writer, sheet_name="calibration", index=False)
        cv["dca"].head(1_000_000).to_excel(writer, sheet_name="dca", index=False)
        cv["leading"].to_excel(writer, sheet_name="leading", index=False)

    source2 = summary / "Source_Data_2_temporal_validation_metric_matrix.xlsx"
    with pd.ExcelWriter(source2, engine="openpyxl") as writer:
        temporal["results"].to_excel(writer, sheet_name="temporal_results", index=False)
        temporal["best"].to_excel(writer, sheet_name="best_by_target", index=False)
        temporal["events"].to_excel(writer, sheet_name="sample_distribution", index=False)
        temporal["assignments"].to_excel(writer, sheet_name="assignments", index=False)
        temporal["predictions"].head(1_000_000).to_excel(writer, sheet_name="predictions", index=False)
        temporal["curves"].head(1_000_000).to_excel(writer, sheet_name="curve_points", index=False)
        temporal["bootstrap"].to_excel(writer, sheet_name="bootstrap_CI", index=False)
        temporal["paired"].to_excel(writer, sheet_name="paired_vs_TabPFN", index=False)
        temporal["dca"].head(1_000_000).to_excel(writer, sheet_name="DCA", index=False)
        temporal["tabnet_members"].to_excel(writer, sheet_name="TabNet_members", index=False)
        temporal["tabnet_summary"].to_excel(writer, sheet_name="TabNet_summary", index=False)

    source3 = summary / "Source_Data_3_bootstrap_confidence_intervals.xlsx"
    with pd.ExcelWriter(source3, engine="openpyxl") as writer:
        cv["bootstrap"].to_excel(writer, sheet_name="bootstrap_ci", index=False)

    source4 = summary / "Source_Data_4_complete_SHAP_feature_rankings.xlsx"
    with pd.ExcelWriter(source4, engine="openpyxl") as writer:
        shap_long.to_excel(writer, sheet_name="SHAP_long", index=False)
        if not shap_long.empty:
            shap_long[shap_long["rank"] <= 20].to_excel(writer, sheet_name="top20", index=False)

    all_features: List[Dict[str, Any]] = []
    for feature_set, columns in feature_sets.items():
        for order, column in enumerate(columns, start=1):
            all_features.append({
                "feature": column,
                "PCI": int(column in feature_sets["classic_preop"]),
                "PPEI": int(column in feature_sets["postop_total"]),
                "ICPI": int(column in feature_sets["full_data"]),
                "order_within_current_set": order,
                "feature_set_source": feature_set,
            })
    feature_dict = (
        pd.DataFrame(all_features)
        .drop_duplicates(subset=["feature"])
        .sort_values("feature")
    )
    source5 = summary / "Source_Data_5_clinical_pathway_feature_dictionary.xlsx"
    feature_dict.to_excel(source5, index=False)


def build_model_parameter_manifest(model_n_jobs: int) -> Dict[str, Any]:
    return {
        "CV": {
            "XGBoost": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": model_n_jobs},
            "LightGBM": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": model_n_jobs},
            "RandomForest": {"n_estimators": 300, "class_weight": "balanced", "n_jobs": model_n_jobs},
            "TabNet": {"max_epochs": 200, "patience": 30},
        },
        "Temporal_Validation": {
            "training_scope": "complete development period; no internal split or CV",
            "classification_threshold": TEMPORAL_FIXED_THRESHOLD,
            "TabPFN": {"seed": RANDOM_STATE, "training": "complete temporal development set"},
            "TabNet": {
                "seeds": TEMPORAL_TABNET_SEEDS,
                "ensemble": "mean of 20 validation probabilities; no seed selection",
                "max_epochs": TEMPORAL_TABNET_MAX_EPOCHS,
                "patience": TEMPORAL_TABNET_PATIENCE,
                "early_stopping": False,
            },
            "XGBoost": {
                "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
                "n_jobs": model_n_jobs, "early_stopping": False,
            },
            "LightGBM": {
                "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                "subsample": 0.9, "colsample_bytree": 0.9,
                "n_jobs": model_n_jobs, "early_stopping": False,
            },
            "RandomForest": {
                "n_estimators": 300, "class_weight": "balanced", "n_jobs": model_n_jobs,
            },
        },
    }


# -----------------------------------------------------------------------------
# GPU monitor and reproducibility metadata
# -----------------------------------------------------------------------------
class GPUMonitor:
    def __init__(self, output_csv: Path, interval_seconds: int = 30):
        self.output_csv = output_csv
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        header_written = False
        while not self.stop_event.is_set():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                query = "index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
                output = subprocess.check_output([
                    "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"
                ], stderr=subprocess.STDOUT, text=True, timeout=15)
                proc_output = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"
                ], stderr=subprocess.STDOUT, text=True, timeout=15)
                process_text = " | ".join(x.strip() for x in proc_output.splitlines())
                rows = []
                for line in output.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 7:
                        rows.append([timestamp] + parts[:7] + [process_text])
                df = pd.DataFrame(rows, columns=[
                    "timestamp", "gpu_index", "gpu_name", "utilization_gpu_percent",
                    "memory_used_mib", "memory_total_mib", "power_draw_w", "temperature_c", "gpu_processes",
                ])
                df.to_csv(self.output_csv, mode="a", header=not header_written, index=False, encoding="utf-8-sig")
                header_written = True
            except Exception as exc:
                if not header_written:
                    pd.DataFrame([{"timestamp": timestamp, "error": str(exc)}]).to_csv(self.output_csv, index=False, encoding="utf-8-sig")
                    header_written = True
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _metadata_path(value: Any, record_absolute_paths: bool) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value))
    return str(path.resolve()) if record_absolute_paths else path.name


def _sanitized_command_line(args: argparse.Namespace) -> str:
    parts = [Path(sys.argv[0]).name]
    path_options = {
        "--excel": args.excel,
        "--fold-file": args.fold_file,
        "--output": args.output,
        "--tabpfn-checkpoint": args.tabpfn_checkpoint,
        "--icpi-feature-file": args.icpi_feature_file,
    }
    for option, value in path_options.items():
        if value:
            parts.extend([option, _metadata_path(value, args.record_absolute_paths)])
    parts.extend(["--sheet", str(args.sheet), "--time-col", str(args.time_col)])
    if args.id_col:
        parts.extend(["--id-col", str(args.id_col)])
    parts.extend(["--models", *map(str, args.models)])
    return " ".join(parts)


def save_reproducibility(output_root: Path, args: argparse.Namespace, feature_sets: Dict[str, List[str]]) -> None:
    rep = ensure_dir(output_root / "reproducibility")
    params = vars(args).copy()
    for key in ("excel", "fold_file", "output", "tabpfn_checkpoint", "icpi_feature_file"):
        params[key] = _metadata_path(params.get(key, ""), args.record_absolute_paths)
    params["feature_sets"] = feature_sets
    params["expected_feature_counts"] = EXPECTED_FEATURE_COUNTS
    params["temporal_design"] = {
        "protocol": "full-development interval-gap fixed protocol",
        "split_name": TEMPORAL_SPLIT_NAME,
        "sole_prespecified_temporal_experiment": True,
        "training": f"{TEMPORAL_DEV_START.date()} to {TEMPORAL_DEV_END.date()}",
        "gap": f"{TEMPORAL_GAP_START.date()} to {TEMPORAL_GAP_END.date()}",
        "validation": f"{TEMPORAL_VAL_START.date()} to {TEMPORAL_VAL_END.date()}",
        "internal_split": False,
        "internal_cross_validation": False,
        "threshold": TEMPORAL_FIXED_THRESHOLD,
        "TabNet_ensemble_seeds": TEMPORAL_TABNET_SEEDS,
        "TabNet_fixed_epochs": TEMPORAL_TABNET_MAX_EPOCHS,
        "TabNet_seed_selection": "none; all members averaged",
    }
    save_json(params, rep / "run_parameters.json")
    save_json({
        "global_seed_for_non_TabNet_models": RANDOM_STATE,
        "numpy": RANDOM_STATE,
        "TabNet_temporal_ensemble_seeds": TEMPORAL_TABNET_SEEDS,
        "TabNet_seed_selection": "none; all 20 members averaged",
    }, rep / "random_seeds.json")
    save_json(build_model_parameter_manifest(args.model_n_jobs), rep / "model_and_analysis_parameters.json")
    (rep / "command_line.txt").write_text(_sanitized_command_line(args), encoding="utf-8")
    fingerprints = {}
    for label, path in [("input_excel", args.excel), ("fixed_fold_file", args.fold_file), ("tabpfn_checkpoint", args.tabpfn_checkpoint)]:
        if path and Path(path).exists():
            fingerprints[label] = {
                "file": _metadata_path(path, args.record_absolute_paths),
                "sha256": sha256_file(path),
                "size_bytes": Path(path).stat().st_size,
            }
    save_json(fingerprints, rep / "file_fingerprints.json")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable if args.record_absolute_paths else Path(sys.executable).name,
        "cwd": os.getcwd() if args.record_absolute_paths else "<redacted>",
        "dependencies": {
            "tabpfn": TABPFN_AVAILABLE, "pytorch_tabnet": TABNET_AVAILABLE,
            "xgboost": XGB_AVAILABLE, "lightgbm": LGBM_AVAILABLE,
            "shap": SHAP_AVAILABLE, "statsmodels": STATSMODELS_AVAILABLE,
            "torch": TORCH_AVAILABLE, "torch_cuda_available": bool(TORCH_AVAILABLE and torch.cuda.is_available()),
        },
    }
    save_json(environment, rep / "software_environment.json")
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=120)
    except Exception as exc:
        freeze = f"pip freeze failed: {exc}"
    (rep / "pip_freeze.txt").write_text(freeze, encoding="utf-8")


# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
def check_disk_space(path: Path, required_gb: float, allow_low_disk: bool) -> None:
    probe = path if path.exists() else path.parent
    probe.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    if free_gb < required_gb and not allow_low_disk:
        raise RuntimeError(f"输出文件系统剩余空间仅 {free_gb:.1f} GB，低于要求 {required_gb:.1f} GB。可清理空间或显式使用 --allow-low-disk。")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空: {output_root}。为避免混入旧结果，请改目录或使用 --overwrite。")
        try:
            shutil.rmtree(output_root)
        except Exception as exc:
            raise RuntimeError(f"无法删除旧输出目录，可能仍有进程占用 .nfs 文件: {exc}") from exc
    output_root.mkdir(parents=True, exist_ok=True)


def resolve_feature_sets_strict(df: pd.DataFrame, id_col: str, time_col: str, icpi_feature_file: str = "") -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    pci, missing_pci = resolve_canonical_features(df.columns.tolist(), PCI_CANONICAL)
    ppei, missing_ppei = resolve_canonical_features(df.columns.tolist(), PPEI_CANONICAL)
    if missing_pci or missing_ppei:
        raise ValueError(f"PCI/PPEI 特征缺失。PCI missing={missing_pci}; PPEI missing={missing_ppei}")
    excluded = set([id_col, time_col] + [t for t in ALL_TARGETS if t in df.columns])
    eligible = [c for c in df.columns if c not in excluded]
    if icpi_feature_file:
        p = Path(icpi_feature_file)
        if p.suffix.lower() == ".json":
            requested = json.loads(p.read_text(encoding="utf-8"))
        else:
            tmp = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
            requested = tmp.iloc[:, 0].dropna().astype(str).tolist()
        icpi, missing = resolve_canonical_features(df.columns.tolist(), requested)
        if missing:
            raise ValueError(f"ICPI 显式清单存在缺失特征: {missing}")
    else:
        # Default: freeze the complete eligible structured set only when it is exactly 56.
        # Any unexpected extra or missing column causes a hard failure, preventing silent feature capture.
        icpi = eligible
    feature_sets = {"classic_preop": pci, "postop_total": ppei, "full_data": icpi}
    for fs, expected in EXPECTED_FEATURE_COUNTS.items():
        actual = len(feature_sets[fs])
        if actual != expected:
            raise ValueError(f"{FEATURE_SET_DISPLAY[fs]} 特征数错误: expected={expected}, actual={actual}. 实际列={feature_sets[fs]}")
    tumor_size = pci[-1]
    threshold_col = resolve_canonical_features(df.columns.tolist(), ["Tumor Size >5 cm"])[0]
    threshold_col = threshold_col[0] if threshold_col else None
    if tumor_size not in pci or tumor_size not in ppei or tumor_size not in icpi:
        raise ValueError("Tumor size 必须同时进入 PCI/PPEI/ICPI")
    if threshold_col is None or threshold_col in pci or threshold_col in ppei or threshold_col not in icpi:
        raise ValueError("Tumor Size >5 cm 必须仅进入 ICPI")
    if time_col in set(sum(feature_sets.values(), [])):
        raise ValueError("初始治疗时间误入特征集")
    audit = []
    for fs, cols in feature_sets.items():
        for order, col in enumerate(cols, start=1):
            audit.append({"feature_set": fs, "display": FEATURE_SET_DISPLAY[fs], "order": order, "feature": col})
    return feature_sets, pd.DataFrame(audit)


def validate_temporal_s3_definition() -> None:
    """Fail fast if the public main program no longer matches the prespecified S3 design."""
    expected = {
        "development_start": pd.Timestamp("2015-10-05"),
        "development_end": pd.Timestamp("2019-06-30"),
        "gap_start": pd.Timestamp("2019-07-01"),
        "gap_end": pd.Timestamp("2019-09-30"),
        "validation_start": pd.Timestamp("2019-10-01"),
        "validation_end": pd.Timestamp("2020-12-25"),
        "split_name": "S3_gap3m_2019Q4_to_2020",
    }
    actual = {
        "development_start": TEMPORAL_DEV_START,
        "development_end": TEMPORAL_DEV_END,
        "gap_start": TEMPORAL_GAP_START,
        "gap_end": TEMPORAL_GAP_END,
        "validation_start": TEMPORAL_VAL_START,
        "validation_end": TEMPORAL_VAL_END,
        "split_name": TEMPORAL_SPLIT_NAME,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"Temporal S3 definition mismatch: {mismatches}")
    if not (
        TEMPORAL_DEV_START <= TEMPORAL_DEV_END
        < TEMPORAL_GAP_START <= TEMPORAL_GAP_END
        < TEMPORAL_VAL_START <= TEMPORAL_VAL_END
    ):
        raise ValueError("Invalid temporal S3 date ordering")


def preflight(args: argparse.Namespace, output_root: Path) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], Dict[str, List[str]]]:
    validate_temporal_s3_definition()
    excel = Path(args.excel)
    fold = Path(args.fold_file)
    if not excel.exists():
        raise FileNotFoundError(f"输入文件不存在: {excel}")
    if not fold.exists():
        raise FileNotFoundError(f"固定折文件不存在: {fold}")
    if "TabPFN" in args.models and args.tabpfn_checkpoint and not Path(args.tabpfn_checkpoint).exists():
        raise FileNotFoundError(f"TabPFN checkpoint 不存在: {args.tabpfn_checkpoint}")
    df = canonicalize_columns(pd.read_excel(excel, sheet_name=args.sheet))
    detect_duplicate_columns(df.columns)
    id_col = args.id_col or find_first_existing(df.columns.tolist(), ID_CANDIDATES)
    if not id_col:
        raise ValueError("未找到患者 ID 列，请使用 --id-col 指定")
    if args.time_col not in df.columns:
        raise ValueError(f"未找到时间列: {args.time_col}")
    if df[id_col].isna().any() or (df[id_col].map(normalize_id) == "").any():
        raise ValueError("患者 ID 存在缺失")
    normalized_ids = df[id_col].map(normalize_id)
    if normalized_ids.duplicated().any():
        dup = normalized_ids[normalized_ids.duplicated(keep=False)].head(20).tolist()
        raise ValueError(f"患者 ID 重复: {dup}")
    missing_targets = [t for t in ALL_TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"缺少 10 个预设终点中的列: {missing_targets}")
    dates = pd.to_datetime(df[args.time_col], errors="coerce")
    if dates.isna().any():
        bad = df.index[dates.isna()].tolist()[:20]
        raise ValueError(f"初始治疗时间存在缺失或无法解析，行号示例: {bad}")
    cnlc_col = find_first_existing(df.columns.tolist(), CNLC_CANDIDATES)
    bclc_col = find_first_existing(df.columns.tolist(), BCLC_CANDIDATES)
    feature_sets, feature_audit = resolve_feature_sets_strict(df, id_col, args.time_col, args.icpi_feature_file)
    scan_cols = [c for c in df.columns if c not in {id_col, args.time_col, *ALL_TARGETS}]
    comma_hits = scan_residual_decimal_commas(df, scan_cols)
    if comma_hits:
        pd.DataFrame(comma_hits).to_csv(output_root / "preflight_decimal_comma_errors.csv", index=False, encoding="utf-8-sig")
        raise ValueError(f"检测到残余小数逗号，已停止。示例: {comma_hits[:5]}")
    tumor_col = feature_sets["classic_preop"][-1]
    cleaned, cleaning_audit = clean_input_dataframe(df, tumor_col, [id_col, args.time_col] + ALL_TARGETS)
    cleaned[id_col] = cleaned[id_col].map(normalize_id)
    cleaned[args.time_col] = dates
    feature_audit.to_csv(output_root / "preflight_feature_set_audit.csv", index=False, encoding="utf-8-sig")
    cleaning_audit.to_csv(output_root / "preflight_cleaning_audit.csv", index=False, encoding="utf-8-sig")
    # Strict fold compatibility for every endpoint before workers are started.
    fold_long = read_fold_long(fold)
    fold_audit = []
    for target in ALL_TARGETS:
        task, y, sample_ids = prepare_target(cleaned, target, id_col)
        validated = validate_target_folds(fold_long, target, task, sample_ids)
        fold_audit.append({
            "target": target, "n_task": len(task), "fold_rows": len(validated),
            "unique_samples": validated["sample_index"].nunique(), "status": "PASS",
        })
    pd.DataFrame(fold_audit).to_csv(output_root / "preflight_fixed_fold_audit.csv", index=False, encoding="utf-8-sig")
    return cleaned, id_col, cnlc_col, bclc_col, feature_sets


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def coerce_sheet_arg(value: Any) -> Any:
    """Convert command-line numeric sheet strings such as '0' back to an index."""
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HCC fixed-fold cross-validation and prespecified S3 interval-gap temporal validation")
    parser.add_argument("--excel", required=True, help="Private Excel input used for both CV and temporal validation")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    parser.add_argument("--fold-file", required=True, help="Existing fixed fold-long CSV/XLSX; strict reuse, no fallback")
    parser.add_argument("--output", default="outputs/hcc_postoperative_prognosis_benchmark", help="Output root")
    parser.add_argument("--time-col", default=TIME_COL_DEFAULT)
    parser.add_argument("--id-col", default="", help="Patient ID column; auto-detected when omitted")
    parser.add_argument("--icpi-feature-file", default="", help="Optional explicit 56-feature ICPI list in JSON/CSV/XLSX; default requires exactly 56 eligible columns")
    parser.add_argument("--tabpfn-checkpoint", default=os.environ.get("TABPFN_CHECKPOINT", ""), help="Optional local TabPFN checkpoint; may also be supplied through TABPFN_CHECKPOINT")
    parser.add_argument("--feature-set-workers", type=int, default=3)
    parser.add_argument("--model-n-jobs", type=int, default=2)
    parser.add_argument("--bootstrap-rounds", type=int, default=DEFAULT_BOOTSTRAP_ROUNDS, help="Internal CV bootstrap rounds")
    parser.add_argument("--temporal-bootstrap-rounds", type=int, default=DEFAULT_TEMPORAL_BOOTSTRAP_ROUNDS, help="Temporal validation bootstrap confidence-interval rounds")
    parser.add_argument("--paired-bootstrap-rounds", type=int, default=DEFAULT_PAIRED_BOOTSTRAP_ROUNDS, help="Temporal paired AUROC bootstrap rounds versus TabPFN")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--skip-shap", action="store_true", help="Skip fold-1 TabPFN generic SHAP")
    parser.add_argument("--shap-background", type=int, default=60)
    parser.add_argument("--shap-test", type=int, default=40)
    parser.add_argument("--save-fold-data", action="store_true", help="Save raw/processed matrices once per target-feature set-fold; uses substantial disk")
    parser.add_argument("--gpu-monitor-interval", type=int, default=30)
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    parser.add_argument("--allow-low-disk", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--save-direct-identifiers", action="store_true", help="Export direct patient IDs in CV/temporal outputs and exact dates in temporal outputs; disabled by default")
    parser.add_argument("--record-absolute-paths", action="store_true", help="Record local absolute paths in reproducibility metadata; disabled by default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sheet = coerce_sheet_arg(args.sheet)
    if args.feature_set_workers != 3:
        raise ValueError("The prespecified analysis uses exactly three feature-set workers; set --feature-set-workers 3")
    if args.model_n_jobs < 1:
        raise ValueError("--model-n-jobs 必须 >=1")
    output_root = Path(args.output).resolve()
    check_disk_space(output_root, args.minimum_free_gb, args.allow_low_disk)
    prepare_output_root(output_root, args.overwrite)
    log_line("HCC postoperative prognosis benchmark started", output_root / "master.log")

    cleaned_df, id_col, cnlc_col, bclc_col, feature_sets = preflight(args, output_root)
    args.id_col = id_col
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds 必须 >=1")
    if args.temporal_bootstrap_rounds < 1:
        raise ValueError("--temporal-bootstrap-rounds 必须 >=1")
    if args.paired_bootstrap_rounds < 1:
        raise ValueError("--paired-bootstrap-rounds 必须 >=1")
    save_reproducibility(output_root, args, feature_sets)
    if args.preflight_only:
        log_line("Preflight completed successfully; no models were run", output_root / "master.log")
        return

    cfg = WorkerConfig(
        excel_path=str(Path(args.excel).resolve()), sheet_name=args.sheet,
        fold_file=str(Path(args.fold_file).resolve()), output_root=str(output_root),
        time_col=args.time_col, id_col=id_col, cnlc_col=cnlc_col, bclc_col=bclc_col,
        feature_sets=feature_sets, models=list(args.models),
        tabpfn_checkpoint=str(Path(args.tabpfn_checkpoint).resolve()) if args.tabpfn_checkpoint else "",
        model_n_jobs=args.model_n_jobs, bootstrap_rounds=args.bootstrap_rounds,
        run_shap=not args.skip_shap, shap_background=args.shap_background,
        shap_test=args.shap_test, save_fold_data=args.save_fold_data,
        save_direct_identifiers=args.save_direct_identifiers,
    )

    monitor = GPUMonitor(output_root / "reproducibility" / "gpu_usage_nvidia_smi.csv", args.gpu_monitor_interval)
    monitor.start()
    try:
        context = mp.get_context("spawn")
        # Internal CV retains the original three feature-set workers.
        with ProcessPoolExecutor(max_workers=args.feature_set_workers, mp_context=context) as executor:
            run_phase_in_executor(executor, run_cv_feature_set_worker, "internal CV", cfg)
        cv_summary = summarize_cv(output_root)

        # Temporal validation is intentionally sequential: every model is trained once on
        # the complete development period, and TabNet members are averaged without
        # validation-based seed, epoch or threshold selection.
        run_temporal_validation(
            cleaned_df,
            cfg,
            temporal_bootstrap_rounds=args.temporal_bootstrap_rounds,
            paired_bootstrap_rounds=args.paired_bootstrap_rounds,
            save_direct_identifiers=args.save_direct_identifiers,
        )
        temporal_summary = summarize_temporal(output_root)
        shap_long = aggregate_shap(output_root / "cv", output_root / "summary")
        export_source_data(output_root, cv_summary, temporal_summary, shap_long, feature_sets)
    finally:
        monitor.stop()

    log_line("All analyses completed", output_root / "master.log")
    log_line(f"Output: {output_root}", output_root / "master.log")


if __name__ == "__main__":
    main()
