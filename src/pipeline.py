"""End-to-end pipeline: ingest -> validate -> split -> train/evaluate (xN) -> select best -> deploy."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .mlmd_types import PIPELINE_NAME, PIPELINE_STEPS, get_prop, register_types
from .recorder import PipelineError, code_version, get_or_create_context, now_iso, update_context
from .steps import deploy, evaluate, ingest, split, train, validate

OUTPUTS = Path("outputs")


def run_pipeline(store, cfg: dict, config_path: str, data_path: str | None = None,
                 name: str | None = None):
    register_types(store)
    code_ver = code_version()
    data_path = data_path or cfg["data"]["path"]
    exp_cfg = cfg["experiment"]
    metric = exp_cfg.get("selection_metric", "f1")
    label = cfg["data"]["label"]

    pipeline_ctx = get_or_create_context(store, "Pipeline", PIPELINE_NAME, {
        "description": "Fake job posting detection: TF-IDF text classifiers on EMSCAD",
        "steps": " -> ".join(PIPELINE_STEPS),
    })
    exp_name = name or f"{exp_cfg['name']}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if store.get_context_by_type_and_name("Experiment", exp_name) is not None:
        raise PipelineError(f"Experiment '{exp_name}' already exists; choose another --name")
    exp_ctx = get_or_create_context(store, "Experiment", exp_name, {
        "description": exp_cfg.get("description", ""), "created_at": now_iso(),
        "config_path": config_path, "selection_metric": metric, "status": "RUNNING",
        "code_version": code_ver,
    })
    contexts = [pipeline_ctx, exp_ctx]
    run_dir = OUTPUTS / "runs" / exp_name
    print(f"Experiment '{exp_name}' (context id={exp_ctx.id}) | pipeline context id={pipeline_ctx.id} "
          f"| code {code_ver} | data {data_path}")

    try:
        raw = ingest.run(store, contexts, data_path, code_ver)
        update_context(store, exp_ctx, {"data_sha256": get_prop(raw, "sha256")})
        validated, clean = validate.run(store, contexts, raw, cfg, run_dir, code_ver)
        split_art, train_idx, test_idx = split.run(store, contexts, validated, clean, cfg, run_dir, code_ver)
        for run_cfg in cfg["runs"]:
            model_art, pipe = train.run(store, contexts, split_art, clean, train_idx, run_cfg, label,
                                        OUTPUTS / "models", code_ver)
            evaluate.run(store, contexts, model_art, split_art, pipe, clean, test_idx, label,
                         run_dir, code_ver)
        best_model, best_metrics, value = deploy.select_best(store, contexts, exp_ctx, metric, code_ver)
        serving = deploy.deploy(store, contexts, best_model, best_metrics, metric, value,
                                OUTPUTS / "serving", code_ver)
    except PipelineError:
        update_context(store, exp_ctx, {"status": "FAILED"})
        raise
    except Exception:
        update_context(store, exp_ctx, {"status": "FAILED"})
        raise

    update_context(store, exp_ctx, {"status": "COMPLETE", "best_model_id": best_model.id})
    return exp_name, best_model, serving
