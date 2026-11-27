"""Ingest: hash the CSV and register it as a versioned RawDataset.

An unchanged file (same sha256) re-uses the existing RawDataset artifact instead
of creating a duplicate; a changed file becomes version N+1.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from ml_metadata.metadata_store import ListOptions

from ..mlmd_types import get_prop
from ..recorder import PipelineError, StepRun, new_artifact, next_version


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_by_hash(store, sha: str):
    # MLMD filter query (SQL-like predicate over typed properties).
    matches = store.get_artifacts(list_options=ListOptions(
        filter_query=f"type = 'RawDataset' AND properties.sha256.string_value = '{sha}'"))
    return matches[0] if matches else None


def run(store, contexts, data_path: str, code_ver: str):
    path = Path(data_path)
    step = StepRun(store, "Ingest", contexts, props={"source_path": str(path)}, code_ver=code_ver)
    if not path.exists():
        step.fail(f"dataset not found: {path}")
        raise PipelineError(f"Dataset not found at {path}. See README 'Get the dataset'.")

    sha = sha256_of(path)
    existing = find_by_hash(store, sha)
    if existing is not None:
        # Same bytes as an already-registered dataset: link the existing artifact
        # as this execution's output (new Event + Attribution, no new Artifact).
        step.complete(outputs=[(existing, "raw_dataset")], props={"reused_existing": 1})
        print(f"  [ingest] unchanged file -> reusing RawDataset id={existing.id} "
              f"v{get_prop(existing, 'version')} sha256={sha[:12]}")
        return existing

    df = pd.read_csv(path)
    version = next_version(store, "RawDataset")
    art = new_artifact(
        store, "RawDataset", uri=str(path), name=f"raw_dataset_v{version}",
        props={"filename": path.name, "row_count": int(len(df)), "column_count": int(df.shape[1]),
               "size_bytes": path.stat().st_size, "sha256": sha, "version": version},
    )
    step.complete(outputs=[(art, "raw_dataset")], props={"reused_existing": 0})
    print(f"  [ingest] new RawDataset id={art.id} v{version} rows={len(df)} sha256={sha[:12]}")
    return art
