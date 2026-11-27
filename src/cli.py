"""Command-line interface: run the pipeline and query MLMD lineage.

    python -m src.cli run --config config/experiments.yaml [--data PATH] [--name NAME]
    python -m src.cli list-experiments
    python -m src.cli list-artifacts [--type Model]
    python -m src.cli executions [--state FAILED] [--experiment NAME]
    python -m src.cli compare --experiment NAME|latest [--metric f1]
    python -m src.cli trace-model --model-id ID
    python -m src.cli trace-data --artifact-id ID
    python -m src.cli graph --experiment NAME|latest  |  graph --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from tabulate import tabulate

from .lineage import INPUT, OUTPUT, STATE_NAMES, Lineage, describe, render_tree
from .mlmd_types import get_prop, register_types
from .recorder import PipelineError
from .store import db_path, get_store

OUT = Path("outputs")
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _lineage() -> Lineage:
    store = get_store()
    register_types(store)
    return Lineage(store)


def _resolve_experiment(lin: Lineage, name: str):
    if name == "latest":
        exps = lin.experiments()
        if not exps:
            raise SystemExit("No experiments recorded yet. Run: python -m src.cli run")
        return exps[-1]
    return lin.experiment(name)


# ------------------------------------------------------------------ commands
def cmd_run(args):
    from .pipeline import run_pipeline
    cfg = yaml.safe_load(Path(args.config).read_text())
    store = get_store()
    print(f"MLMD store: {db_path()}")
    try:
        exp_name, best, serving = run_pipeline(store, cfg, args.config, args.data, args.name)
    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}", file=sys.stderr)
        return 1
    print(f"\nDone. Experiment '{exp_name}': deployed Model id={best.id} "
          f"(v{get_prop(best, 'version')}, {get_prop(best, 'run_name')}) as ServingModel id={serving.id}")
    print(f"Next: python -m src.cli compare --experiment {exp_name}")
    return 0


def cmd_list_experiments(args):
    lin = _lineage()
    rows = []
    for c in lin.experiments():
        arts = lin.store.get_artifacts_by_context(c.id)
        n_models = sum(lin.atype[a.type_id] == "Model" for a in arts)
        raw = [a for a in arts if lin.atype[a.type_id] == "RawDataset"]
        rows.append([c.id, c.name, get_prop(c, "status"), get_prop(c, "created_at"),
                     f"v{get_prop(raw[0], 'version')} (#{raw[0].id})" if raw else "-",
                     n_models, get_prop(c, "best_model_id") or "-", get_prop(c, "selection_metric"),
                     get_prop(c, "code_version")])
    print(tabulate(rows, headers=["ctx id", "experiment", "status", "created (UTC)", "raw data",
                                  "models", "best model", "metric", "code"], tablefmt="github"))
    if not rows:
        print("(no experiments yet)")


def cmd_list_artifacts(args):
    lin = _lineage()
    arts = lin.store.get_artifacts_by_type(args.type) if args.type else lin.store.get_artifacts()
    deployed = lin.deployed_model_ids()
    rows = []
    for a in arts:
        t = lin.atype[a.type_id]
        v = get_prop(a, "version") or get_prop(a, "serving_version")
        extra = {"RawDataset": lambda: f"sha256={get_prop(a, 'sha256')[:12]} rows={get_prop(a, 'row_count')}",
                 "Model": lambda: f"{get_prop(a, 'run_name')}{'  ★ deployed' if a.id in deployed else ''}",
                 "Metrics": lambda: f"f1={get_prop(a, 'f1'):.4f}",
                 "ServingModel": lambda: f"model #{get_prop(a, 'source_model_id')} active={get_prop(a, 'is_active')}",
                 }.get(t, lambda: "")()
        rows.append([a.id, t, a.name or "-", f"v{v}" if v else "-", extra, a.uri])
    print(tabulate(rows, headers=["id", "type", "name", "ver", "details", "uri"], tablefmt="github"))


def cmd_executions(args):
    lin = _lineage()
    execs = (lin.store.get_executions_by_context(_resolve_experiment(lin, args.experiment).id)
             if args.experiment else lin.store.get_executions())
    rows = []
    for e in execs:
        state = STATE_NAMES.get(e.last_known_state, "?")
        if args.state and state != args.state.upper():
            continue
        ins = [ev.artifact_id for ev in lin.store.get_events_by_execution_ids([e.id]) if ev.type == INPUT]
        outs = [ev.artifact_id for ev in lin.store.get_events_by_execution_ids([e.id]) if ev.type == OUTPUT]
        rows.append([e.id, lin.etype[e.type_id], state, get_prop(e, "start_time"),
                     get_prop(e, "duration_seconds"), ins, outs,
                     (get_prop(e, "error_message") or "")[:90]])
    print(tabulate(rows, headers=["id", "type", "state", "start", "secs", "inputs", "outputs", "error"],
                   tablefmt="github"))


def cmd_compare(args):
    lin = _lineage()
    ctx = _resolve_experiment(lin, args.experiment)
    metric = args.metric or get_prop(ctx, "selection_metric") or "f1"
    runs = lin.experiment_runs(ctx.id)
    if not runs:
        raise SystemExit(f"Experiment '{ctx.name}' has no evaluated models (status={get_prop(ctx, 'status')}).")
    best_id = max(runs, key=lambda r: get_prop(r["metrics"], metric))["model"].id
    deployed = lin.deployed_model_ids()
    header = ["model id", "ver", "run", "algorithm", "classifier params", "tfidf params", *METRIC_COLS, "best", "serving now"]
    rows = []
    for r in sorted(runs, key=lambda r: -get_prop(r["metrics"], metric)):
        m, met = r["model"], r["metrics"]
        hp = json.loads(get_prop(m, "hyperparameters"))
        rows.append([m.id, f"v{get_prop(m, 'version')}", get_prop(m, "run_name"), get_prop(m, "algorithm"),
                     json.dumps(hp["classifier"]), json.dumps(hp["tfidf"]),
                     *[f"{get_prop(met, k):.4f}" for k in METRIC_COLS],
                     "★" if m.id == best_id else "", "yes" if m.id in deployed else ""])
    print(f"Experiment '{ctx.name}' (ctx id={ctx.id}, status={get_prop(ctx, 'status')}), ranked by {metric} "
          f"on the fraud class:\n")
    print(tabulate(rows, headers=header, tablefmt="github"))
    OUT.mkdir(exist_ok=True)
    base = OUT / f"compare_{ctx.name}"
    base.with_suffix(".md").write_text(
        f"# Experiment `{ctx.name}`: ranked by `{metric}` (fraud class)\n\n"
        + tabulate(rows, headers=header, tablefmt="github") + "\n")
    with open(base.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    print(f"\nSaved {base}.md and {base}.csv")


def _save_trace(name: str, lines: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved outputs/{name}")


def cmd_trace_model(args):
    lin = _lineage()
    art = lin.artifact(args.model_id)
    if lin.type_of(art) not in ("Model", "ServingModel", "Metrics"):
        print(f"note: artifact {art.id} is a {lin.type_of(art)}, tracing it anyway")
    tree = lin.backward(art.id)
    lines = [f"Backward lineage of artifact #{art.id} (contexts: {', '.join(lin.contexts_of_artifact(art.id))})", ""]
    lines += render_tree(tree, arrow="<-", highlight=lin.deployed_model_ids())
    raws = [n.obj for n in tree.walk() if n.type_name == "RawDataset"]
    for r in raws:
        lines.append(f"\n=> trained on RawDataset #{r.id} v{get_prop(r, 'version')} "
                     f"({get_prop(r, 'filename')}, sha256 {get_prop(r, 'sha256')})")
    print("\n".join(lines))
    _save_trace(f"trace_model_{art.id}.txt", lines)


def cmd_trace_data(args):
    lin = _lineage()
    art = lin.artifact(args.artifact_id)
    tree = lin.forward(art.id)
    deployed, ever = lin.deployed_model_ids(), lin.ever_deployed_model_ids()
    header = [f"Forward lineage of artifact #{art.id}: {describe(tree)}", ""]
    tree_lines = render_tree(tree, arrow="->", highlight=deployed)
    models = {n.obj.id: n.obj for n in tree.walk() if n.type_name == "Model"}
    metrics_by_model = {}
    for mid in models:
        for ex in lin.consumers(mid):
            for out in lin.io(ex.id, OUTPUT):
                if lin.type_of(out) == "Metrics":
                    metrics_by_model[mid] = out
    rows = []
    for mid, m in sorted(models.items()):
        met = metrics_by_model.get(mid)
        exp = [c for c in lin.store.get_contexts_by_artifact(mid) if lin.ctype[c.type_id] == "Experiment"]
        rows.append([mid, f"v{get_prop(m, 'version')}", get_prop(m, "run_name"), exp[0].name if exp else "-",
                     f"{get_prop(met, 'f1'):.4f}" if met else "-",
                     "★ ACTIVE" if mid in deployed else ("previously" if mid in ever else "")])
    summary = [f"{len(models)} model(s) were trained downstream of artifact #{art.id}:", "",
               tabulate(rows, headers=["model id", "ver", "run", "experiment", "f1", "deployed"],
                        tablefmt="github")]
    lines = header + tree_lines + [""] + summary
    if args.verbose:
        print("\n".join(lines))
    else:
        print("\n".join(header + summary))
        print("\n(full lineage tree saved to the file below; pass --verbose to print it)")
    _save_trace(f"trace_data_{art.id}.txt", lines)


def cmd_graph(args):
    from .graph import build, render
    lin = _lineage()
    if args.all:
        g, name, title = build(lin, None), "lineage_all", "MLMD lineage: entire store"
    else:
        ctx = _resolve_experiment(lin, args.experiment)
        g, name, title = build(lin, [ctx.id]), f"lineage_{ctx.name}", f"MLMD lineage: experiment {ctx.name}"
    paths = render(g, OUT / "graphs" / name, title)
    n_a = sum(d["kind"] == "artifact" for _, d in g.nodes(data=True))
    print(f"Graph: {n_a} artifacts, {g.number_of_nodes() - n_a} executions, {g.number_of_edges()} events")
    for p in paths:
        print(f"  wrote {p}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.cli", description="MLMD lineage pipeline for fake job postings")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline for an experiment config")
    r.add_argument("--config", default="config/experiments.yaml")
    r.add_argument("--data", help="override data.path from the config")
    r.add_argument("--name", help="experiment name (default: <config name>-<UTC timestamp>)")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("list-experiments", help="list Experiment contexts").set_defaults(fn=cmd_list_experiments)
    la = sub.add_parser("list-artifacts", help="list artifacts (ids for trace commands)")
    la.add_argument("--type", choices=["RawDataset", "ValidatedDataset", "TrainTestSplit", "Model", "Metrics", "ServingModel"])
    la.set_defaults(fn=cmd_list_artifacts)
    ex = sub.add_parser("executions", help="list executions, e.g. --state FAILED")
    ex.add_argument("--state", choices=["COMPLETE", "FAILED", "RUNNING"], type=str.upper)
    ex.add_argument("--experiment")
    ex.set_defaults(fn=cmd_executions)
    c = sub.add_parser("compare", help="table of runs/params/metrics in an experiment")
    c.add_argument("--experiment", required=True, help="experiment name, or 'latest'")
    c.add_argument("--metric", choices=METRIC_COLS)
    c.set_defaults(fn=cmd_compare)
    tm = sub.add_parser("trace-model", help="backward lineage: model -> ... -> raw dataset")
    tm.add_argument("--model-id", type=int, required=True)
    tm.set_defaults(fn=cmd_trace_model)
    td = sub.add_parser("trace-data", help="forward lineage: dataset -> models -> deployment")
    td.add_argument("--artifact-id", type=int, required=True)
    td.add_argument("--verbose", action="store_true", help="print the full lineage tree")
    td.set_defaults(fn=cmd_trace_data)
    g = sub.add_parser("graph", help="render lineage graph (PNG, SVG, interactive HTML)")
    grp = g.add_mutually_exclusive_group(required=True)
    grp.add_argument("--experiment", help="experiment name, or 'latest'")
    grp.add_argument("--all", action="store_true", help="whole store (shows every dataset version)")
    g.set_defaults(fn=cmd_graph)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
