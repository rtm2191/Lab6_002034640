"""Evaluate: score a Model on the held-out split and register a Metrics artifact."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)

from ..recorder import StepRun, new_artifact


def run(store, contexts, model_art, split, pipe, clean, test_idx, label: str,
        out_dir: Path, code_ver: str):
    step = StepRun(store, "Evaluate", contexts, code_ver=code_ver)
    step.add_input(model_art, "model")
    step.add_input(split, "split")

    test = clean.iloc[test_idx]
    X, y = test["text"], test[label]
    pred = pipe.predict(X)
    scores = pipe.decision_function(X) if hasattr(pipe, "decision_function") else pipe.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    uri = out_dir / f"metrics_model{model_art.id}.json"
    uri.write_text(json.dumps({**metrics, "report": classification_report(y, pred, output_dict=True)}, indent=2))
    art = new_artifact(store, "Metrics", uri=str(uri), props={
        **metrics, "positive_class": "fraudulent=1", "test_rows": int(len(test)),
    })
    step.complete(outputs=[(art, "metrics")])
    print("  [evaluate] model id={:<3} acc={accuracy:.4f} P={precision:.4f} R={recall:.4f} "
          "F1={f1:.4f} AUC={roc_auc:.4f} -> Metrics id={mid}".format(model_art.id, mid=art.id, **metrics))
    return art
