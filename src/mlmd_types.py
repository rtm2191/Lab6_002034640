"""Custom MLMD ArtifactTypes, ExecutionTypes and ContextTypes for this project.

Every type is registered with typed properties. `put_*_type(..., can_add_fields=True)`
makes registration idempotent and lets the schema grow without breaking an
existing store.
"""
from __future__ import annotations

from ml_metadata.proto import metadata_store_pb2 as pb

STRING, INT, DOUBLE = pb.STRING, pb.INT, pb.DOUBLE

# ---------------------------------------------------------------- artifacts
ARTIFACT_TYPES: dict[str, dict[str, int]] = {
    "RawDataset": {
        "filename": STRING,
        "row_count": INT,
        "column_count": INT,
        "size_bytes": INT,
        "sha256": STRING,
        "version": INT,
    },
    "ValidatedDataset": {
        "rows_in": INT,
        "rows_kept": INT,
        "rows_dropped": INT,
        "missing_value_stats": STRING,  # JSON {column: missing fraction}
        "fraud_rate": DOUBLE,
        "empty_text_rate": DOUBLE,
        "text_fields": STRING,  # JSON list
    },
    "TrainTestSplit": {
        "train_rows": INT,
        "test_rows": INT,
        "test_size": DOUBLE,
        "random_state": INT,
        "stratified": INT,
        "train_fraud_rate": DOUBLE,
        "test_fraud_rate": DOUBLE,
    },
    "Model": {
        "run_name": STRING,
        "framework": STRING,
        "framework_version": STRING,
        "algorithm": STRING,
        "hyperparameters": STRING,  # JSON (classifier + TF-IDF params)
        "version": INT,
        "train_seconds": DOUBLE,
        "vocabulary_size": INT,
    },
    "Metrics": {
        "accuracy": DOUBLE,
        "precision": DOUBLE,
        "recall": DOUBLE,
        "f1": DOUBLE,
        "roc_auc": DOUBLE,
        "positive_class": STRING,
        "test_rows": INT,
    },
    "ServingModel": {
        "source_model_id": INT,
        "model_version": INT,
        "serving_version": INT,
        "selection_metric": STRING,
        "selection_value": DOUBLE,
        "deployed_at": STRING,
        "is_active": INT,  # 1 = currently serving, 0 = replaced by a newer deploy
    },
}

# --------------------------------------------------------------- executions
_COMMON_EXEC = {
    "start_time": STRING,
    "end_time": STRING,
    "duration_seconds": DOUBLE,
    "status": STRING,  # RUNNING | COMPLETE | FAILED
    "code_version": STRING,  # git commit hash
    "error_message": STRING,
}

EXECUTION_TYPES: dict[str, dict[str, int]] = {
    "Ingest": {**_COMMON_EXEC, "source_path": STRING, "reused_existing": INT},
    "Validate": {**_COMMON_EXEC, "checks_passed": STRING, "checks_failed": STRING},
    "Split": {**_COMMON_EXEC},
    "Train": {**_COMMON_EXEC, "run_name": STRING, "algorithm": STRING},
    "Evaluate": {**_COMMON_EXEC},
    "SelectBest": {
        **_COMMON_EXEC,
        "selection_metric": STRING,
        "selected_model_id": INT,
        "selected_value": DOUBLE,
        "n_candidates": INT,
    },
    "Deploy": {**_COMMON_EXEC, "serving_uri": STRING},
}

# ----------------------------------------------------------------- contexts
CONTEXT_TYPES: dict[str, dict[str, int]] = {
    "Experiment": {
        "description": STRING,
        "created_at": STRING,
        "config_path": STRING,
        "selection_metric": STRING,
        "status": STRING,
        "best_model_id": INT,
        "data_sha256": STRING,
        "code_version": STRING,
    },
    "Pipeline": {
        "description": STRING,
        "steps": STRING,
    },
}

PIPELINE_NAME = "fake-job-detection"
PIPELINE_STEPS = ["Ingest", "Validate", "Split", "Train", "Evaluate", "SelectBest", "Deploy"]


def register_types(store) -> dict[str, int]:
    """Register (or fetch) every custom type. Returns {type_name: type_id}."""
    ids: dict[str, int] = {}
    for name, props in ARTIFACT_TYPES.items():
        t = pb.ArtifactType(name=name)
        for k, v in props.items():
            t.properties[k] = v
        ids[name] = store.put_artifact_type(t, can_add_fields=True)
    for name, props in EXECUTION_TYPES.items():
        t = pb.ExecutionType(name=name)
        for k, v in props.items():
            t.properties[k] = v
        ids[name] = store.put_execution_type(t, can_add_fields=True)
    for name, props in CONTEXT_TYPES.items():
        t = pb.ContextType(name=name)
        for k, v in props.items():
            t.properties[k] = v
        ids[name] = store.put_context_type(t, can_add_fields=True)
    return ids


# ----------------------------------------------------------------- helpers
def set_props(msg, props: dict) -> None:
    """Write python values into a proto's typed `properties` map."""
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, bool):
            msg.properties[k].int_value = int(v)
        elif isinstance(v, int):
            msg.properties[k].int_value = v
        elif isinstance(v, float):
            msg.properties[k].double_value = v
        else:
            msg.properties[k].string_value = str(v)


def get_prop(msg, key: str, default=None):
    """Read a property (typed first, then custom) as a python value."""
    for container in (msg.properties, msg.custom_properties):
        if key in container:
            val = container[key]
            which = val.WhichOneof("value")
            return getattr(val, which) if which else default
    return default


def props_dict(msg) -> dict:
    return {k: get_prop(msg, k) for k in list(msg.properties) + list(msg.custom_properties)}
