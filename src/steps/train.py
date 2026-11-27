"""Train: fit one TF-IDF + classifier configuration and register a versioned Model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..recorder import StepRun, new_artifact, next_version

ALGORITHMS = {
    "LogisticRegression": LogisticRegression,
    "LinearSVC": LinearSVC,
    "RandomForest": RandomForestClassifier,
}


def build(run_cfg: dict) -> Pipeline:
    tfidf = dict(run_cfg.get("tfidf", {}))
    if "ngram_range" in tfidf:
        tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    clf = ALGORITHMS[run_cfg["algorithm"]](**run_cfg.get("params", {}))
    return Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, **tfidf)), ("clf", clf)])


def run(store, contexts, split, clean, train_idx, run_cfg: dict, label: str,
        models_dir: Path, code_ver: str):
    step = StepRun(store, "Train", contexts, name=None, code_ver=code_ver,
                   props={"run_name": run_cfg["name"], "algorithm": run_cfg["algorithm"]})
    step.add_input(split, "split")

    pipe = build(run_cfg)
    train = clean.iloc[train_idx]
    t0 = time.time()
    pipe.fit(train["text"], train[label])
    secs = round(time.time() - t0, 2)

    version = next_version(store, "Model")
    models_dir.mkdir(parents=True, exist_ok=True)
    uri = models_dir / f"model_v{version}.joblib"
    joblib.dump(pipe, uri)
    hyper = {"classifier": run_cfg.get("params", {}), "tfidf": run_cfg.get("tfidf", {})}
    art = new_artifact(store, "Model", uri=str(uri), name=f"model_v{version}", props={
        "run_name": run_cfg["name"], "framework": "scikit-learn",
        "framework_version": sklearn.__version__, "algorithm": run_cfg["algorithm"],
        "hyperparameters": json.dumps(hyper, sort_keys=True), "version": version,
        "train_seconds": secs, "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_),
    })
    step.complete(outputs=[(art, "model")])
    print(f"  [train] {run_cfg['name']:<20} {run_cfg['algorithm']:<18} {secs:>6.1f}s -> Model id={art.id} v{version}")
    return art, pipe
