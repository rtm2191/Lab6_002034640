"""Split: deterministic stratified train/test split recorded as a TrainTestSplit artifact."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..recorder import StepRun, new_artifact


def run(store, contexts, validated, clean, cfg: dict, out_dir: Path, code_ver: str):
    scfg, label = cfg["split"], cfg["data"]["label"]
    step = StepRun(store, "Split", contexts, code_ver=code_ver)
    step.add_input(validated, "validated_dataset")

    idx = list(range(len(clean)))
    train_idx, test_idx = train_test_split(
        idx, test_size=scfg["test_size"], random_state=scfg["random_state"],
        stratify=clean[label] if scfg.get("stratify", True) else None,
    )
    uri = out_dir / "split_indices.json"
    uri.write_text(json.dumps({"train": train_idx, "test": test_idx}))
    y = clean[label]
    art = new_artifact(store, "TrainTestSplit", uri=str(uri), props={
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "test_size": float(scfg["test_size"]), "random_state": int(scfg["random_state"]),
        "stratified": int(bool(scfg.get("stratify", True))),
        "train_fraud_rate": round(float(y.iloc[train_idx].mean()), 5),
        "test_fraud_rate": round(float(y.iloc[test_idx].mean()), 5),
    })
    step.complete(outputs=[(art, "split")])
    print(f"  [split] train={len(train_idx)} test={len(test_idx)} -> TrainTestSplit id={art.id}")
    return art, train_idx, test_idx
