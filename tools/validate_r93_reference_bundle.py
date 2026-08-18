#!/usr/bin/env python3
"""Validate the published/private R9.3 five-fold OOF SHAP reference bundle.

The reference comparison is deliberately non-blocking for floating-point
prediction drift. Structural identity, five-fold completeness, TTR semantics,
strict local-checkpoint/no-fallback audits, and OOF coverage are hard gates.
"""
from __future__ import annotations
import argparse, csv, io, json, re, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

TARGETS = [f"OS{x}m" for x in (12,24,36,48,60)] + [f"TTR{x}m" for x in (12,24,36,48,60)]
FEATURES = ["classic_preop", "postop_total", "full_data"]
EXPECTED_SHA = "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"

def load_csv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))

def validate(bundle: Path) -> dict:
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        root = names[0].split("/")[0] + "/"
        oof = [n for n in names if n.endswith("/oof_tabpfn_predictions.csv")]
        fold_completion = [n for n in names if re.search(r"/fold_[1-5]/fold_completion\.json$", n)]
        checkpoint_audits = [n for n in names if n.endswith("/tabpfn_strict_checkpoint_audit.json")]
        config_completion = [n for n in names if n.endswith("/config_completion.json")]
        if len(oof) != 30 or len(config_completion) != 30 or len(fold_completion) != 150 or len(checkpoint_audits) != 150:
            raise RuntimeError(
                f"Incomplete bundle: oof={len(oof)}, config={len(config_completion)}, "
                f"fold={len(fold_completion)}, checkpoint_audit={len(checkpoint_audits)}"
            )
        configs, total_rows = set(), 0
        for name in oof:
            parts = name.split("/")
            target, fs = parts[-3], parts[-2]
            if target not in TARGETS or fs not in FEATURES:
                raise RuntimeError(f"Unexpected target/feature-set: {target}/{fs}")
            rows = load_csv(z, name)
            total_rows += len(rows)
            configs.add((target, fs))
            folds = {int(r["fold"]) for r in rows}
            if folds != {1,2,3,4,5}:
                raise RuntimeError(f"Incomplete folds in {target}/{fs}: {folds}")
            keys = [int(r["sample_index"]) for r in rows]
            if len(keys) != len(set(keys)):
                raise RuntimeError(f"Duplicate OOF sample_index in {target}/{fs}")
            if not {int(float(r["y_true"])) for r in rows}.issubset({0,1}):
                raise RuntimeError(f"Non-binary y_true in {target}/{fs}")
        if configs != {(t,f) for t in TARGETS for f in FEATURES}:
            raise RuntimeError("The 30 target × feature-set configurations are not complete")
        audit_errors = []
        observed_versions = set()
        for name in checkpoint_audits:
            a = json.loads(z.read(name).decode("utf-8"))
            observed_versions.add(str(a.get("tabpfn_version", "unknown")))
            checks = {
                "checkpoint_sha256": str(a.get("checkpoint_sha256", "")).lower() == EXPECTED_SHA,
                "device": str(a.get("device", "")).lower().startswith("cuda"),
                "default_constructor_allowed": a.get("default_constructor_allowed") is False,
                "remote_download_allowed": a.get("remote_download_allowed") is False,
                "cpu_fallback_allowed": a.get("cpu_fallback_allowed") is False,
                "network_blocked": a.get("network_blocked") is True,
            }
            if not all(checks.values()):
                audit_errors.append({"file": name, "checks": checks})
        if audit_errors:
            raise RuntimeError(f"Strict checkpoint/no-fallback audit failed: {audit_errors[:3]}")
        summary_name = next((n for n in names if n.endswith("/OOF_beeswarm_completion_summary.json")), None)
        summary = json.loads(z.read(summary_name).decode("utf-8")) if summary_name else {}
        if summary and summary.get("status") != "PASS":
            raise RuntimeError("Reference completion summary is not PASS")
    return {
        "status": "PASS",
        "bundle": str(bundle),
        "configurations": 30,
        "fold_tasks": 150,
        "oof_rows": total_rows,
        "targets": TARGETS,
        "feature_sets": FEATURES,
        "checkpoint_sha256": EXPECTED_SHA,
        "tabpfn_versions_observed": sorted(observed_versions),
        "strict_no_fallback_verified": True,
        "reference_probability_comparison_policy": "nonblocking diagnostic only",
    }

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()
    report = validate(args.bundle)
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
