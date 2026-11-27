"""Lineage graph: static PNG/SVG (graphviz) + interactive HTML (pyvis)."""
from __future__ import annotations

import json
from pathlib import Path

import graphviz
import networkx as nx
from ml_metadata.proto import metadata_store_pb2 as pb
from pyvis.network import Network

from .lineage import STATE_NAMES, Lineage
from .mlmd_types import get_prop, props_dict

ARTIFACT_STYLE = {  # type -> (graphviz shape, fill colour)
    "RawDataset": ("cylinder", "#cfe2ff"),
    "ValidatedDataset": ("cylinder", "#d1e7dd"),
    "TrainTestSplit": ("folder", "#e2d9f3"),
    "Model": ("box3d", "#fff3cd"),
    "Metrics": ("note", "#f8f9fa"),
    "ServingModel": ("doubleoctagon", "#ffe69c"),
}
EXEC_FILL = {"COMPLETE": "#adb5bd", "FAILED": "#f1aeb5", "RUNNING": "#9ec5fe"}
DEPLOYED = "#ffc107"


def _label(lin: Lineage, obj) -> str:
    if isinstance(obj, pb.Artifact):
        t = lin.atype[obj.type_id]
        p = props_dict(obj)
        line = {
            "RawDataset": lambda: f"v{p['version']} rows={p['row_count']}\\nsha={p['sha256'][:10]}",
            "ValidatedDataset": lambda: f"kept={p['rows_kept']}",
            "TrainTestSplit": lambda: f"{p['train_rows']}/{p['test_rows']}",
            "Model": lambda: f"v{p['version']} {p['run_name']}",
            "Metrics": lambda: f"F1={p['f1']:.3f} AUC={p['roc_auc']:.3f}",
            "ServingModel": lambda: f"serving v{p['serving_version']}{' (active)' if p['is_active'] else ''}",
        }.get(t, lambda: "")()
        return f"{t} #{obj.id}\\n{line}"
    t = lin.etype[obj.type_id]
    extra = f"\\n{get_prop(obj, 'run_name')}" if t == "Train" else ""
    return f"{t} #{obj.id}{extra}\\n{STATE_NAMES.get(obj.last_known_state, '?')}"


def build(lin: Lineage, context_ids: list[int] | None) -> nx.DiGraph:
    store = lin.store
    if context_ids is None:
        arts = store.get_artifacts()
        execs = store.get_executions()
    else:
        arts, execs, seen_a, seen_e = [], [], set(), set()
        for cid in context_ids:
            arts += [a for a in store.get_artifacts_by_context(cid) if a.id not in seen_a and not seen_a.add(a.id)]
            execs += [e for e in store.get_executions_by_context(cid) if e.id not in seen_e and not seen_e.add(e.id)]
    active = lin.deployed_model_ids()
    active_serving = {a.id for a in lin.active_serving()}

    g = nx.DiGraph()
    for a in arts:
        t = lin.atype[a.type_id]
        deployed = (t == "Model" and a.id in active) or a.id in active_serving
        g.add_node(f"a{a.id}", kind="artifact", type=t, label=_label(lin, a), deployed=deployed,
                   title=json.dumps({"id": a.id, "type": t, "uri": a.uri, **props_dict(a)}, indent=1, default=str))
    for e in execs:
        t = lin.etype[e.type_id]
        g.add_node(f"e{e.id}", kind="execution", type=t, label=_label(lin, e), deployed=False,
                   state=STATE_NAMES.get(e.last_known_state, "UNKNOWN"),
                   title=json.dumps({"id": e.id, "type": t, **props_dict(e)}, indent=1, default=str))
    exec_ids = [e.id for e in execs]
    for ev in store.get_events_by_execution_ids(exec_ids) if exec_ids else []:
        a, e = f"a{ev.artifact_id}", f"e{ev.execution_id}"
        if a not in g:
            continue
        key = ev.path.steps[0].key if ev.path.steps else ""
        if ev.type == pb.Event.INPUT:
            g.add_edge(a, e, label=key, kind="INPUT")
        elif ev.type == pb.Event.OUTPUT:
            g.add_edge(e, a, label=key, kind="OUTPUT")
    return g


def render(g: nx.DiGraph, out_base: Path, title: str) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(title, graph_attr={"rankdir": "LR", "label": title, "labelloc": "t",
                                              "fontsize": "18", "fontname": "Helvetica", "nodesep": "0.25"},
                           node_attr={"fontname": "Helvetica", "fontsize": "10", "style": "filled"},
                           edge_attr={"fontname": "Helvetica", "fontsize": "8", "color": "#6c757d"})
    for n, d in g.nodes(data=True):
        if d["kind"] == "artifact":
            shape, fill = ARTIFACT_STYLE.get(d["type"], ("ellipse", "white"))
            attrs = {"shape": shape, "fillcolor": DEPLOYED if d["deployed"] else fill}
            if d["deployed"]:
                attrs.update(penwidth="3", color="#b8860b", label=d["label"] + "\\n★ DEPLOYED")
        else:
            attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": EXEC_FILL.get(d["state"], "white")}
            if d["state"] == "FAILED":
                attrs.update(color="#dc3545", penwidth="2")
        dot.node(n, attrs.pop("label", d["label"]), **attrs)
    for u, v, d in g.edges(data=True):
        dot.edge(u, v, label=d["label"], style="solid" if d["kind"] == "OUTPUT" else "dashed")

    written = []
    for fmt in ("png", "svg"):
        dot.format = fmt
        written.append(Path(dot.render(filename=out_base.name, directory=out_base.parent, cleanup=True)))

    net = Network(height="850px", width="100%", directed=True, cdn_resources="remote", heading=title)
    for n, d in g.nodes(data=True):
        if d["kind"] == "artifact":
            color = DEPLOYED if d["deployed"] else ARTIFACT_STYLE.get(d["type"], ("", "#ffffff"))[1]
            net.add_node(n, label=d["label"].replace("\\n", "\n"), title=d["title"], shape="ellipse",
                         color={"background": color, "border": "#b8860b" if d["deployed"] else "#495057"},
                         borderWidth=4 if d["deployed"] else 1)
        else:
            net.add_node(n, label=d["label"].replace("\\n", "\n"), title=d["title"], shape="box",
                         color={"background": EXEC_FILL.get(d["state"], "#ffffff"), "border": "#212529"})
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, title=f"{d['kind']}: {d['label']}", dashes=d["kind"] == "INPUT")
    net.set_options(json.dumps({
        "layout": {"hierarchical": {"enabled": True, "direction": "LR", "sortMethod": "directed",
                                    "levelSeparation": 220, "nodeSpacing": 90}},
        "physics": {"enabled": False},
        "edges": {"arrows": {"to": {"enabled": True}}, "smooth": {"type": "cubicBezier"}},
        "interaction": {"hover": True, "navigationButtons": True},
    }))
    html = out_base.with_suffix(".html")
    net.write_html(str(html), notebook=False, open_browser=False)
    written.append(html)
    return written
