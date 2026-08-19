#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CGH supplementary analyses for the final TTR-labelled HCC V8 release.

This script is deliberately separate from the primary benchmark. It consumes the
same private TTR-labelled dataset/fixed folds plus patient-level OOF predictions
from the primary analysis (or the audited OLD-V8/R9.3 merged OOF table).

Implemented CGH modules
-----------------------
1. Restricted surgical-cohort sensitivity: BCLC 0/A, tumour <=5 cm, solitary HCC
   for OS36m and TTR24m.
2. L2-regularized logistic-regression comparator using the same endpoint-specific
   five-fold train/validation/test assignments.
3. Paired PCI -> PPEI -> ICPI incremental value using patient-level OOF bootstrap.
4. Clinical risk stratification when continuous survival/recurrence data are supplied.
5. Temporal case-mix SMD and CV-vs-temporal-validation performance comparison when temporal
   prediction files are supplied.
6. Penalized Cox/cause-specific Cox sensitivity when continuous data are supplied.
7. Five-fold SHAP stability is read from the primary benchmark's OOF-SHAP outputs;
   SHAP is not re-fit here, preventing a second implementation from drifting away
   from the audited strict-no-fallback TabPFN path.

The fixed-time recurrence endpoint is TTR throughout. Unsupported endpoint names
are rejected rather than silently remapped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import hcc_postoperative_prognosis_benchmark as core

TARGETS = core.ALL_TARGETS
FEATURES = core.FEATURE_SET_ORDER
DISPLAY = core.FEATURE_SET_DISPLAY
RANDOM_STATE = core.RANDOM_STATE
FINAL_TEMPORAL_SPLIT = core.TEMPORAL_SPLIT_NAME


def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def save_json(obj: Any, p: str | Path) -> None:
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else np.nan


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else np.nan


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def youden_threshold(y, p) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    score = tpr - fpr
    finite = np.isfinite(thresholds)
    if finite.any():
        idx = np.where(finite)[0][int(np.argmax(score[finite]))]
        return float(thresholds[idx])
    return 0.5


def bootstrap_metric_ci(y, p, rounds: int, seed: int) -> Dict[str, Tuple[float, float]]:
    y=np.asarray(y,int); p=np.asarray(p,float); rng=np.random.default_rng(seed)
    store={"AUROC":[],"AUPRC":[],"BrierScore":[]}; attempts=0
    while min(map(len,store.values())) < rounds and attempts < rounds*40:
        attempts += 1
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]; pp=p[idx]
        if len(np.unique(yy))<2: continue
        store["AUROC"].append(safe_auc(yy,pp)); store["AUPRC"].append(safe_ap(yy,pp)); store["BrierScore"].append(float(brier_score_loss(yy,pp)))
    out={}
    for k,v in store.items():
        a=np.asarray(v,float); out[k]=(float(np.quantile(a,.025)),float(np.quantile(a,.975))) if len(a) else (np.nan,np.nan)
    return out


@dataclass
class DataBundle:
    raw: pd.DataFrame
    clean: pd.DataFrame
    id_col: str
    time_col: str
    feature_sets: Dict[str,List[str]]
    folds: pd.DataFrame


def load_data(excel: str | Path, fold_file: str | Path, sheet: Any=0, time_col: str=core.TIME_COL_DEFAULT, id_col: str="") -> DataBundle:
    core.validate_temporal_definition()
    raw=core.canonicalize_columns(pd.read_excel(excel,sheet_name=sheet))
    core.detect_duplicate_columns(raw.columns)
    id_col=id_col or core.find_first_existing(raw.columns.tolist(),core.ID_CANDIDATES)
    if not id_col: raise ValueError("Patient ID column not found")
    if time_col not in raw.columns: raise ValueError(f"Missing temporal date column: {time_col}")
    missing=[t for t in TARGETS if t not in raw.columns]
    if missing: raise ValueError(f"Missing fixed-time TTR/OS endpoints: {missing}")
    for t in TARGETS:
        vals=pd.to_numeric(raw[t],errors="coerce").dropna().unique()
        if not set(map(float,vals)).issubset({0.0,1.0}): raise ValueError(f"{t} must contain only 0/1/NA")
    feature_sets,_=core.resolve_feature_sets_strict(raw,id_col,time_col,"")
    tumor_col=feature_sets["classic_preop"][-1]
    clean,_=core.clean_input_dataframe(raw,tumor_col,[id_col,time_col]+TARGETS)
    clean,_=core.recompute_v8_derived_indicators(clean)
    clean[id_col]=clean[id_col].map(core.normalize_id)
    clean[time_col]=pd.to_datetime(raw[time_col],errors="raise")
    folds=core.read_fold_long(fold_file)
    for t in TARGETS:
        task,y,ids=core.prepare_target(clean,t,id_col)
        core.validate_target_folds(folds,t,task,ids)
    return DataBundle(raw,clean,id_col,time_col,feature_sets,folds)


def load_oof(path: str | Path, data: DataBundle) -> pd.DataFrame:
    p=pd.read_csv(path)
    req={"target","feature_set","model","sample_index","y_true","y_prob"}
    if not req.issubset(p.columns): raise ValueError(f"OOF missing columns: {sorted(req-set(p.columns))}")
    core.validate_endpoint_names(p.target.astype(str).unique().tolist(), "OOF predictions", allow_subset=True)
    p=p[p.target.isin(TARGETS)].copy(); p.sample_index=pd.to_numeric(p.sample_index,errors="raise").astype(int)
    for (t,fs,m),g in p.groupby(["target","feature_set","model"]):
        expected=pd.to_numeric(data.raw.loc[g.sample_index,t],errors="coerce").to_numpy()
        if np.isnan(expected).any() or not np.array_equal(expected.astype(int),g.y_true.to_numpy(int)):
            raise ValueError(f"OOF y_true mismatch: {t}/{fs}/{m}")
    return p


class LogisticPreprocessor:
    """Conventional logistic preprocessing used for the CGH comparator."""
    def __init__(self, columns: Sequence[str]):
        self.columns=list(columns)
        self.cont=[c for c in self.columns if core.normalize_name(c) in core.V8_CONTINUOUS_NORMALIZED]
        self.cat=[c for c in self.columns if core.normalize_name(c) in core.V8_CATEGORICAL_NORMALIZED]
        if len(self.cont)+len(self.cat)!=len(self.columns):
            raise ValueError(f"Unknown feature roles: {[c for c in self.columns if c not in self.cont+self.cat]}")
        self.cimp=SimpleImputer(strategy="median"); self.kimp=SimpleImputer(strategy="most_frequent")
        self.scaler=StandardScaler(); self.logcols=[]
        try: self.encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except TypeError: self.encoder=OneHotEncoder(handle_unknown="ignore",sparse=False)
    def fit(self,X):
        if self.cont:
            a=self.cimp.fit_transform(X[self.cont])
            for j in range(a.shape[1]):
                s=pd.Series(a[:,j])
                if len(s) and s.min()>=0 and s.skew()>=core.RIGHT_SKEW_THRESHOLD:
                    self.logcols.append(j); a[:,j]=np.log1p(a[:,j])
            self.scaler.fit(a)
        if self.cat:
            c=self.kimp.fit_transform(X[self.cat].astype(object)); self.encoder.fit(c.astype(str))
        return self
    def transform(self,X):
        blocks=[]
        if self.cont:
            a=self.cimp.transform(X[self.cont])
            for j in self.logcols: a[:,j]=np.log1p(np.clip(a[:,j],0,None))
            blocks.append(self.scaler.transform(a))
        if self.cat:
            c=self.kimp.transform(X[self.cat].astype(object)); blocks.append(self.encoder.transform(c.astype(str)))
        return np.hstack(blocks).astype(float)


def module_subgroup(data: DataBundle,oof:pd.DataFrame,out:Path,rounds:int):
    root=ensure_dir(out/"01_restricted_cohorts")
    bclc=core.find_first_existing(data.clean.columns,core.BCLC_CANDIDATES)
    multiple=core.resolve_canonical_features(data.clean.columns.tolist(),["Multiple tumors"])[0][0]
    tumor=core.resolve_canonical_features(data.clean.columns.tolist(),["Tumor size"])[0][0]
    masks={"BCLC_0A":pd.to_numeric(data.clean[bclc],errors="coerce").isin([0,1]),"Tumor_le_5cm":pd.to_numeric(data.clean[tumor],errors="coerce").le(5),"Solitary":pd.to_numeric(data.clean[multiple],errors="coerce").eq(0)}
    rows=[]
    configs=[("classic_preop","TabPFN"),("postop_total","TabPFN"),("full_data","TabPFN"),("CNLC_only","CNLC"),("BCLC_only","BCLC")]
    for subgroup,mask in masks.items():
        ids=set(data.clean.index[mask].astype(int))
        for t in ["OS36m","TTR24m"]:
            for fs,m in configs:
                g=oof[(oof.target==t)&(oof.feature_set==fs)&(oof.model==m)&oof.sample_index.isin(ids)]
                if g.empty or g.y_true.nunique()<2: continue
                y=g.y_true.to_numpy(int); p=g.y_prob.to_numpy(float); ci=bootstrap_metric_ci(y,p,rounds,stable_seed(subgroup,t,fs,m))
                rows.append({"subgroup":subgroup,"target":t,"feature_set":fs,"model":m,"n":len(g),"events":int(y.sum()),"AUROC":safe_auc(y,p),"AUROC_lo":ci["AUROC"][0],"AUROC_hi":ci["AUROC"][1],"AUPRC":safe_ap(y,p),"BrierScore":float(brier_score_loss(y,p))})
    pd.DataFrame(rows).to_csv(root/"restricted_cohort_performance.csv",index=False,encoding="utf-8-sig")


def module_logistic(data:DataBundle,oof:pd.DataFrame,out:Path,rounds:int):
    root=ensure_dir(out/"02_L2_logistic_comparator"); fold_rows=[]; preds=[]
    for t in TARGETS:
        task,y,ids=core.prepare_target(data.clean,t,data.id_col); vf=core.validate_target_folds(data.folds,t,task,ids)
        for fs in FEATURES:
            X=task[data.feature_sets[fs]]
            for k in range(1,6):
                tr,va,te=core.split_indices_from_fold(vf,k); pre=LogisticPreprocessor(data.feature_sets[fs]); pre.fit(X.loc[tr])
                Xtr=pre.transform(X.loc[tr]); Xva=pre.transform(X.loc[va]); Xte=pre.transform(X.loc[te])
                model=LogisticRegression(penalty="l2",C=1.0,solver="liblinear",max_iter=5000,random_state=RANDOM_STATE)
                model.fit(Xtr,y.loc[tr].to_numpy(int)); pva=model.predict_proba(Xva)[:,1]; thr=youden_threshold(y.loc[va].to_numpy(int),pva); pte=model.predict_proba(Xte)[:,1]
                fold_rows.append({"target":t,"feature_set":fs,"fold":k,"AUROC":safe_auc(y.loc[te],pte),"AUPRC":safe_ap(y.loc[te],pte),"BrierScore":float(brier_score_loss(y.loc[te],pte)),"threshold":thr,"encoded_features":Xtr.shape[1]})
                preds.append(pd.DataFrame({"target":t,"feature_set":fs,"model":"L2_Logistic","fold":k,"sample_index":te,"y_true":y.loc[te].to_numpy(int),"y_prob":pte,"threshold":thr,"y_pred":(pte>=thr).astype(int)}))
    fp=pd.DataFrame(fold_rows); pp=pd.concat(preds,ignore_index=True); fp.to_csv(root/"logistic_fold_metrics.csv",index=False,encoding="utf-8-sig"); pp.to_csv(root/"logistic_oof_predictions.csv",index=False,encoding="utf-8-sig")
    rows=[]
    for (t,fs),g in pp.groupby(["target","feature_set"]):
        f=fp[(fp.target==t)&(fp.feature_set==fs)]; y=g.y_true.to_numpy(int); p=g.y_prob.to_numpy(float); tab=oof[(oof.target==t)&(oof.feature_set==fs)&(oof.model=="TabPFN")].sort_values("sample_index")
        gg=g.sort_values("sample_index")
        if not np.array_equal(tab.sample_index.to_numpy(),gg.sample_index.to_numpy()): raise ValueError(f"TabPFN/logistic OOF alignment mismatch {t}/{fs}")
        y_aligned=gg.y_true.to_numpy(int); p_log=gg.y_prob.to_numpy(float); p_tab=tab.y_prob.to_numpy(float)
        if not np.array_equal(y_aligned,tab.y_true.to_numpy(int)): raise ValueError(f"TabPFN/logistic y_true mismatch {t}/{fs}")
        rows.append({"target":t,"feature_set":fs,"logistic_AUROC_fold_mean":f.AUROC.mean(),"logistic_AUROC_pooled":safe_auc(y_aligned,p_log),"TabPFN_AUROC_pooled":safe_auc(y_aligned,p_tab),"delta_TabPFN_minus_logistic_pooled":safe_auc(y_aligned,p_tab)-safe_auc(y_aligned,p_log)})
    pd.DataFrame(rows).to_csv(root/"logistic_vs_TabPFN_summary.csv",index=False,encoding="utf-8-sig")


def paired_delta(y,a,b,rounds,seed):
    y=np.asarray(y,int);a=np.asarray(a,float);b=np.asarray(b,float); point=[safe_auc(y,b)-safe_auc(y,a),safe_ap(y,b)-safe_ap(y,a),float(brier_score_loss(y,b)-brier_score_loss(y,a))]
    rng=np.random.default_rng(seed); vals=[[],[],[]]; attempts=0
    while len(vals[0])<rounds and attempts<rounds*40:
        attempts+=1;idx=rng.integers(0,len(y),len(y));yy=y[idx]
        if len(np.unique(yy))<2:continue
        vals[0].append(safe_auc(yy,b[idx])-safe_auc(yy,a[idx]));vals[1].append(safe_ap(yy,b[idx])-safe_ap(yy,a[idx]));vals[2].append(float(brier_score_loss(yy,b[idx])-brier_score_loss(yy,a[idx])))
    ci=[(float(np.quantile(v,.025)),float(np.quantile(v,.975))) for v in vals]
    return point,ci


def module_incremental(oof:pd.DataFrame,out:Path,rounds:int):
    root=ensure_dir(out/"03_incremental_value"); rows=[]; pairs=[("classic_preop","postop_total"),("postop_total","full_data"),("classic_preop","full_data")]
    tab=oof[oof.model=="TabPFN"].copy()
    for t in TARGETS:
        for a,b in pairs:
            ga=tab[(tab.target==t)&(tab.feature_set==a)][["sample_index","y_true","y_prob"]].rename(columns={"y_prob":"pa"}); gb=tab[(tab.target==t)&(tab.feature_set==b)][["sample_index","y_true","y_prob"]].rename(columns={"y_true":"yb","y_prob":"pb"}); m=ga.merge(gb,on="sample_index",validate="one_to_one")
            if not np.array_equal(m.y_true.to_numpy(int),m.yb.to_numpy(int)): raise ValueError(f"Outcome mismatch in paired comparison {t}")
            point,ci=paired_delta(m.y_true,m.pa,m.pb,rounds,stable_seed("delta",t,a,b)); rows.append({"target":t,"from":a,"to":b,"n":len(m),"delta_AUROC":point[0],"delta_AUROC_lo":ci[0][0],"delta_AUROC_hi":ci[0][1],"delta_AUPRC":point[1],"delta_AUPRC_lo":ci[1][0],"delta_AUPRC_hi":ci[1][1],"delta_Brier_to_minus_from":point[2],"delta_Brier_lo":ci[2][0],"delta_Brier_hi":ci[2][1]})
    pd.DataFrame(rows).to_csv(root/"paired_incremental_performance.csv",index=False,encoding="utf-8-sig")


def smd_cont(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna();b=pd.to_numeric(b,errors="coerce").dropna();den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if len(a)>1 and len(b)>1 else np.nan;return float((b.mean()-a.mean())/den) if pd.notna(den) and den>0 else np.nan


def module_temporal(data:DataBundle,oof:pd.DataFrame,temporal_path:str|Path,out:Path):
    root=ensure_dir(out/"05_temporal_case_mix"); tp=pd.read_csv(temporal_path); core.validate_endpoint_names(tp.target.astype(str).unique().tolist(), "temporal predictions", allow_subset=True)
    dates=pd.to_datetime(data.clean[data.time_col]); dev=(dates>=core.TEMPORAL_DEV_START)&(dates<=core.TEMPORAL_DEV_END); val=(dates>=core.TEMPORAL_VAL_START)&(dates<=core.TEMPORAL_VAL_END)
    rows=[]
    for c in data.feature_sets["full_data"]:
        if core.normalize_name(c) in core.V8_CONTINUOUS_NORMALIZED: rows.append({"feature":c,"type":"continuous","SMD":smd_cont(data.clean.loc[dev,c],data.clean.loc[val,c])})
        else:
            aa=data.clean.loc[dev,c].dropna().astype(str);bb=data.clean.loc[val,c].dropna().astype(str)
            for lev in sorted(set(aa)|set(bb)):
                pa=(aa==lev).mean() if len(aa) else np.nan;pb=(bb==lev).mean() if len(bb) else np.nan;den=np.sqrt((pa*(1-pa)+pb*(1-pb))/2) if pd.notna(pa) and pd.notna(pb) else np.nan;rows.append({"feature":c,"level":lev,"type":"categorical_level","SMD":float((pb-pa)/den) if den and den>0 else np.nan})
    s=pd.DataFrame(rows);s["abs_SMD"]=s.SMD.abs();s.sort_values("abs_SMD",ascending=False).to_csv(root/"temporal_case_mix_SMD.csv",index=False,encoding="utf-8-sig")
    perf=[]
    for (t,fs,m),g in tp.groupby(["target","feature_set","model"]):
        if t not in core.TEMPORAL_TARGETS: continue
        cv=oof[(oof.target==t)&(oof.feature_set==fs)&(oof.model==m)]; y=g.y_true.to_numpy(int);p=g.y_prob.to_numpy(float)
        perf.append({"target":t,"feature_set":fs,"model":m,"temporal_split":FINAL_TEMPORAL_SPLIT,"temporal_AUROC":safe_auc(y,p),"temporal_AUPRC":safe_ap(y,p),"CV_AUROC":safe_auc(cv.y_true.to_numpy(int),cv.y_prob.to_numpy(float)) if len(cv) else np.nan,"delta_temporal_minus_CV_AUROC":safe_auc(y,p)-safe_auc(cv.y_true.to_numpy(int),cv.y_prob.to_numpy(float)) if len(cv) else np.nan})
    pd.DataFrame(perf).to_csv(root/"CV_vs_temporal_performance.csv",index=False,encoding="utf-8-sig")


def module_shap_stability(primary_output:str|Path,out:Path):
    root=ensure_dir(out/"07_fivefold_SHAP_stability"); src=Path(primary_output)/"summary"/"shap_fivefold_stability_all_30config.csv"
    if not src.exists():
        pd.DataFrame({"status":["SKIPPED"],"reason":[f"Primary five-fold SHAP stability file not found: {src}"]}).to_csv(root/"SHAP_NOT_AVAILABLE.csv",index=False);return
    x=pd.read_csv(src); core.validate_endpoint_names(x.target.astype(str).unique().tolist(), "SHAP stability", allow_subset=True)
    x.to_csv(root/"shap_fivefold_stability_all_30config.csv",index=False,encoding="utf-8-sig")
    x[(x.target.isin(["OS36m","TTR24m"])) & (x.feature_set.isin(["postop_total","full_data"]))].to_csv(root/"shap_stability_OS36_TTR24_focus.csv",index=False,encoding="utf-8-sig")


def load_survival_data(path: str | Path, data: DataBundle) -> pd.DataFrame:
    """Load independently audited continuous-time data; never derive it from fixed labels."""
    p=Path(path)
    surv=pd.read_excel(p) if p.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(p)
    surv=core.canonicalize_columns(surv)
    id_col=data.id_col if data.id_col in surv.columns else core.find_first_existing(surv.columns.tolist(),core.ID_CANDIDATES)
    if not id_col: raise ValueError("Continuous survival file has no patient ID column")
    aliases={"OS.EVENT":["OS.EVENT","OS"],"OS.TIME":["OS.TIME"],"TTR.EVENT":["TTR.EVENT"],"TTR.TIME":["TTR.TIME"],"TTR.STATUS":["TTR.STATUS"]}
    resolved={}
    for key,cands in aliases.items():
        col=core.find_first_existing(surv.columns.tolist(),cands)
        if not col: raise ValueError(f"Continuous survival file missing {key}")
        resolved[key]=col
    x=pd.DataFrame({"patient_id":surv[id_col].map(core.normalize_id)})
    for key,col in resolved.items(): x[key]=pd.to_numeric(surv[col],errors="raise")
    if x.patient_id.duplicated().any() or (x.patient_id=="").any(): raise ValueError("Continuous survival patient IDs must be unique/non-missing")
    if (x[["OS.TIME","TTR.TIME"]]<0).any().any(): raise ValueError("Negative continuous times are not allowed")
    if not set(x["OS.EVENT"].dropna().astype(int).unique()).issubset({0,1}): raise ValueError("OS.EVENT must be 0/1")
    if not set(x["TTR.EVENT"].dropna().astype(int).unique()).issubset({0,1}): raise ValueError("TTR.EVENT must be 0/1")
    if not set(x["TTR.STATUS"].dropna().astype(int).unique()).issubset({0,1,2}): raise ValueError("TTR.STATUS must be 0=censor, 1=recurrence, 2=death-before-recurrence")
    index_by_id={core.normalize_id(v):idx for idx,v in data.raw[data.id_col].items()}
    x["sample_index"]=x.patient_id.map(index_by_id)
    if x.sample_index.isna().any(): raise ValueError("Continuous survival file contains IDs absent from analysis workbook")
    x.sample_index=x.sample_index.astype(int)
    return x


def assign_tertiles(prob: pd.Series) -> pd.Series:
    ranks=prob.rank(method="first",pct=True)
    return pd.cut(ranks,bins=[0,1/3,2/3,1],labels=["Low","Intermediate","High"],include_lowest=True).astype(str)


def km_survival_at(times, events, horizon):
    times=np.asarray(times,float); events=np.asarray(events,int); s=1.0
    for t in sorted(np.unique(times[(times<=horizon)&(events==1)])):
        at_risk=np.sum(times>=t); d=np.sum((times==t)&(events==1))
        if at_risk: s*=1-d/at_risk
    return float(s)


def cif_at(times,status,horizon):
    times=np.asarray(times,float); status=np.asarray(status,int); S=1.0; cif=0.0
    for t in sorted(np.unique(times[times<=horizon])):
        n=np.sum(times>=t); d1=np.sum((times==t)&(status==1)); dall=np.sum((times==t)&np.isin(status,[1,2]))
        if n:
            cif += S*(d1/n); S *= 1-(dall/n)
    return float(cif)


def module_risk_and_survival(data:DataBundle,oof:pd.DataFrame,survival_data:str,out:Path):
    root=ensure_dir(out/"04_clinical_risk_stratification")
    surv=load_survival_data(survival_data,data)
    summaries=[]
    for target,horizon,kind in [("OS36m",36,"OS"),("TTR24m",24,"TTR")]:
        g=oof[(oof.target==target)&(oof.feature_set=="full_data")&(oof.model=="TabPFN")][["sample_index","y_prob"]].copy()
        g["risk_group"]=assign_tertiles(g.y_prob)
        m=g.merge(surv,on="sample_index",how="left",validate="one_to_one")
        if m[["OS.TIME","OS.EVENT","TTR.TIME","TTR.EVENT","TTR.STATUS"]].isna().any().any(): raise ValueError(f"Missing continuous-time data for {target} OOF patients")
        for risk in ["Low","Intermediate","High"]:
            q=m[m.risk_group==risk]
            if kind=="OS":
                estimate=km_survival_at(q["OS.TIME"],q["OS.EVENT"],horizon); events=int(q["OS.EVENT"].sum())
                metric="survival_probability"
            else:
                estimate=cif_at(q["TTR.TIME"],q["TTR.STATUS"],horizon); events=int((q["TTR.STATUS"]==1).sum())
                metric="recurrence_CIF"
            summaries.append({"target":target,"risk_group":risk,"n":len(q),"events":events,"horizon_months":horizon,"metric":metric,"estimate":estimate})
        m.to_csv(root/f"{target}_OOF_risk_groups.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(root/"risk_group_horizon_estimates.csv",index=False,encoding="utf-8-sig")

    # Conventional penalized Cox sensitivity.
    cox_root=ensure_dir(out/"06_survival_sensitivity"); rows=[]
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        pd.DataFrame({"status":["SKIPPED"],"reason":[f"lifelines unavailable: {exc}"]}).to_csv(cox_root/"COX_NOT_RUN.csv",index=False); return
    merged=data.clean.copy(); merged["_patient_id"]=data.raw[data.id_col].map(core.normalize_id); merged=merged.merge(surv,left_on="_patient_id",right_on="patient_id",how="inner",validate="one_to_one")
    for outcome,event_col,time_col in [("OS","OS.EVENT","OS.TIME"),("TTR_cause_specific","TTR.EVENT","TTR.TIME")]:
        for fs in FEATURES:
            pre=LogisticPreprocessor(data.feature_sets[fs]); pre.fit(merged[data.feature_sets[fs]]); X=pre.transform(merged[data.feature_sets[fs]])
            dd=pd.DataFrame(X,columns=[f"x{i}" for i in range(X.shape[1])]); dd["duration"]=merged[time_col].to_numpy(float); dd["event"]=merged[event_col].to_numpy(int)
            cph=CoxPHFitter(penalizer=0.1); cph.fit(dd,duration_col="duration",event_col="event",show_progress=False)
            rows.append({"outcome":outcome,"feature_set":fs,"display":DISPLAY[fs],"n":len(dd),"events":int(dd.event.sum()),"penalizer":0.1,"concordance_index":float(cph.concordance_index_),"status":"ok"})
    pd.DataFrame(rows).to_csv(cox_root/"penalized_cox_summary.csv",index=False,encoding="utf-8-sig")


def module_continuous(survival_data:str,data:DataBundle,oof:pd.DataFrame,out:Path):
    if not survival_data:
        root=ensure_dir(out/"04_06_continuous_time_sensitivity")
        pd.DataFrame({"status":["SKIPPED"],"reason":["Continuous OS/TTR times and competing-event status were not supplied. Fixed-time labels are not used to fabricate KM/CIF/Cox analyses."]}).to_csv(root/"CONTINUOUS_TIME_NOT_RUN.csv",index=False,encoding="utf-8-sig")
        return
    module_risk_and_survival(data,oof,survival_data,out)


def parse_args():
    p=argparse.ArgumentParser(description="CGH supplementary analyses for the final TTR-labelled V8/R9.3 release")
    p.add_argument("--excel",required=True);p.add_argument("--sheet",default=0);p.add_argument("--fold-file",required=True);p.add_argument("--oof-predictions",required=True);p.add_argument("--output",required=True);p.add_argument("--id-col",default="");p.add_argument("--time-col",default=core.TIME_COL_DEFAULT)
    p.add_argument("--temporal-predictions",default="");p.add_argument("--primary-output",default="");p.add_argument("--survival-data",default="");p.add_argument("--bootstrap-rounds",type=int,default=2000)
    p.add_argument("--modules",nargs="+",default=["subgroup","logistic","incremental","continuous","temporal","shap"],choices=["subgroup","logistic","incremental","continuous","temporal","shap","all"])
    return p.parse_args()


def main():
    args=parse_args();args.sheet=core.coerce_sheet_arg(args.sheet);out=ensure_dir(args.output);mods=args.modules
    if "all" in mods:mods=["subgroup","logistic","incremental","continuous","temporal","shap"]
    data=load_data(args.excel,args.fold_file,args.sheet,args.time_col,args.id_col);oof=load_oof(args.oof_predictions,data)
    if "subgroup" in mods:module_subgroup(data,oof,out,args.bootstrap_rounds)
    if "logistic" in mods:module_logistic(data,oof,out,args.bootstrap_rounds)
    if "incremental" in mods:module_incremental(oof,out,args.bootstrap_rounds)
    if "continuous" in mods:module_continuous(args.survival_data,data,oof,out)
    if "temporal" in mods:
        if args.temporal_predictions:module_temporal(data,oof,args.temporal_predictions,out)
        else:pd.DataFrame({"status":["SKIPPED"],"reason":["--temporal-predictions not supplied"]}).to_csv(ensure_dir(out/"05_temporal_case_mix")/"TEMPORAL_NOT_RUN.csv",index=False)
    if "shap" in mods:
        if args.primary_output:module_shap_stability(args.primary_output,out)
        else:pd.DataFrame({"status":["SKIPPED"],"reason":["--primary-output not supplied"]}).to_csv(ensure_dir(out/"07_fivefold_SHAP_stability")/"SHAP_NOT_RUN.csv",index=False)
    save_json({"status":"PASS","endpoint_semantics":"TTR recurrence-only fixed-time classification","temporal_split":FINAL_TEMPORAL_SPLIT,"modules":mods},out/"CGH_SUPPLEMENTARY_COMPLETION.json")

if __name__=="__main__":main()
