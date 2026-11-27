"""Validate: enforce data checks; produce a ValidatedDataset or a FAILED execution."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..recorder import PipelineError, StepRun, new_artifact


def combined_text(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    return df[fields].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def run(store, contexts, raw, cfg: dict, out_dir: Path, code_ver: str):
    vcfg, dcfg = cfg["validation"], cfg["data"]
    label, fields = dcfg["label"], dcfg["text_fields"]
    step = StepRun(store, "Validate", contexts, code_ver=code_ver)
    step.add_input(raw, "raw_dataset")

    passed, failed = [], []

    def check(name: str, ok: bool, detail: str = ""):
        (passed if ok else failed).append(f"{name}{': ' + detail if detail else ''}")

    try:
        df = pd.read_csv(raw.uri)
    except Exception as e:  # unreadable CSV is itself a validation failure
        step.fail(f"could not read CSV: {e}", {"checks_failed": json.dumps([f"readable_csv: {e}"])})
        raise PipelineError(f"Validation failed: could not read {raw.uri}: {e}")

    missing_cols = [c for c in vcfg["required_columns"] if c not in df.columns]
    check("required_columns", not missing_cols, f"missing {missing_cols}" if missing_cols else "")
    check("min_rows", len(df) >= vcfg["min_rows"], f"{len(df)} rows (min {vcfg['min_rows']})")

    stats, empty_rate, fraud_rate = {}, None, None
    if label in df.columns:
        labels = pd.to_numeric(df[label], errors="coerce")
        bad = df[label][~labels.isin(vcfg["allowed_labels"])]
        check("label_values", bad.empty,
              f"{len(bad)} rows with labels outside {vcfg['allowed_labels']} "
              f"(e.g. {bad.astype(str).unique()[:3].tolist()})" if not bad.empty else "")
    present_fields = [f for f in fields if f in df.columns]
    if present_fields:
        text = combined_text(df, present_fields)
        empty_rate = float((text == "").mean())
        check("empty_text_rate", empty_rate <= vcfg["max_empty_text_rate"],
              f"{empty_rate:.2%} empty (max {vcfg['max_empty_text_rate']:.2%})")
        stats = {c: round(float(df[c].isna().mean()), 4) for c in present_fields + ([label] if label in df.columns else [])}

    if not failed:
        df = df.assign(text=text, **{label: pd.to_numeric(df[label])})
        clean = df[df["text"] != ""][["text", label]].reset_index(drop=True)
        fraud_rate = float(clean[label].mean())
        check("min_fraud_rate", fraud_rate >= vcfg["min_fraud_rate"],
              f"{fraud_rate:.2%} fraud (min {vcfg['min_fraud_rate']:.2%})")

    checks = {"checks_passed": json.dumps(passed), "checks_failed": json.dumps(failed)}
    if failed:
        step.fail("validation checks failed: " + "; ".join(failed), checks)
        for f in failed:
            print(f"  [validate] FAILED {f}")
        raise PipelineError(f"Validation failed ({len(failed)} check(s)); execution id={step.id} recorded as FAILED")

    out_dir.mkdir(parents=True, exist_ok=True)
    uri = out_dir / "validated.csv"
    clean.to_csv(uri, index=False)
    art = new_artifact(store, "ValidatedDataset", uri=str(uri), props={
        "rows_in": int(len(df)), "rows_kept": int(len(clean)), "rows_dropped": int(len(df) - len(clean)),
        "missing_value_stats": json.dumps(stats), "fraud_rate": round(fraud_rate, 5),
        "empty_text_rate": round(empty_rate, 5), "text_fields": json.dumps(fields),
    })
    step.complete(outputs=[(art, "validated_dataset")], props=checks)
    print(f"  [validate] {len(passed)} checks passed; kept {len(clean)}/{len(df)} rows, "
          f"fraud rate {fraud_rate:.2%} -> ValidatedDataset id={art.id}")
    return art, clean
