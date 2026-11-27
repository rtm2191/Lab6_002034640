"""SelectBest + Deploy.

SelectBest queries MLMD (not pipeline memory) for every Model + Metrics pair in
the experiment Context and picks the winner by the configured metric. Deploy
copies only that model to outputs/serving/ and registers a ServingModel.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..lineage import Lineage
from ..mlmd_types import get_prop
from ..recorder import PipelineError, StepRun, new_artifact, next_version, now_iso

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def select_best(store, contexts, experiment_ctx, metric: str, code_ver: str):
    if metric not in METRICS:
        raise PipelineError(f"selection_metric must be one of {METRICS}, got '{metric}'")
    step = StepRun(store, "SelectBest", contexts, code_ver=code_ver, props={"selection_metric": metric})
    rows = Lineage(store).experiment_runs(experiment_ctx.id)
    for r in rows:
        step.add_input(r["model"], "candidate_model")
        step.add_input(r["metrics"], "candidate_metrics")
    if not rows:
        step.fail("no Model/Metrics pairs found in experiment context")
        raise PipelineError("SelectBest found no candidate models")
    best = max(rows, key=lambda r: (get_prop(r["metrics"], metric), -r["model"].id))
    value = get_prop(best["metrics"], metric)
    step.complete(props={"selected_model_id": best["model"].id, "selected_value": float(value),
                         "n_candidates": len(rows)})
    print(f"  [select] queried {len(rows)} candidates from MLMD; best by {metric}: "
          f"model id={best['model'].id} ({get_prop(best['model'], 'run_name')}) {metric}={value:.4f}")
    return best["model"], best["metrics"], float(value)


def deploy(store, contexts, model, metrics, metric: str, value: float, serving_dir: Path, code_ver: str):
    step = StepRun(store, "Deploy", contexts, code_ver=code_ver)
    step.add_input(model, "model")
    step.add_input(metrics, "metrics")

    serving_dir.mkdir(parents=True, exist_ok=True)
    target = serving_dir / "model.joblib"
    shutil.copy2(model.uri, target)

    # Retire the previously active serving model (kept for history, flagged inactive).
    previous = [a for a in store.get_artifacts_by_type("ServingModel") if get_prop(a, "is_active") == 1]
    for p in previous:
        p.properties["is_active"].int_value = 0
    if previous:
        store.put_artifacts(previous)

    version = next_version(store, "ServingModel", key="serving_version")
    art = new_artifact(store, "ServingModel", uri=str(target), name=f"serving_v{version}", props={
        "source_model_id": model.id, "model_version": get_prop(model, "version"),
        "serving_version": version, "selection_metric": metric, "selection_value": value,
        "deployed_at": now_iso(), "is_active": 1,
    })
    step.complete(outputs=[(art, "serving_model")], props={"serving_uri": str(target)})
    print(f"  [deploy] model v{get_prop(model, 'version')} (id={model.id}) -> {target} "
          f"as ServingModel id={art.id} serving_v{version}"
          + (f"; retired serving id(s) {[p.id for p in previous]}" if previous else ""))
    return art
