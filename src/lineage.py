"""Lineage queries over the MLMD store (artifacts <-> events <-> executions <-> contexts).

Every function here reads ONLY from MLMD, never from pipeline memory, so
lineage can be reconstructed later from the store alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ml_metadata.proto import metadata_store_pb2 as pb

from .mlmd_types import get_prop, props_dict

INPUT, OUTPUT = pb.Event.INPUT, pb.Event.OUTPUT
STATE_NAMES = {v: k for k, v in pb.Execution.State.items()}


class Lineage:
    def __init__(self, store):
        self.store = store
        self.atype = {t.id: t.name for t in store.get_artifact_types()}
        self.etype = {t.id: t.name for t in store.get_execution_types()}
        self.ctype = {t.id: t.name for t in store.get_context_types()}

    # ------------------------------------------------------------- basics
    def artifact(self, aid: int) -> pb.Artifact:
        found = self.store.get_artifacts_by_id([aid])
        if not found:
            raise SystemExit(f"No artifact with id {aid}")
        return found[0]

    def type_of(self, node) -> str:
        if isinstance(node, pb.Artifact):
            return self.atype[node.type_id]
        return self.etype[node.type_id]

    def producers(self, aid: int) -> list[pb.Execution]:
        ev = [e for e in self.store.get_events_by_artifact_ids([aid]) if e.type == OUTPUT]
        return self.store.get_executions_by_id([e.execution_id for e in ev]) if ev else []

    def consumers(self, aid: int) -> list[pb.Execution]:
        ev = [e for e in self.store.get_events_by_artifact_ids([aid]) if e.type == INPUT]
        return self.store.get_executions_by_id(sorted({e.execution_id for e in ev})) if ev else []

    def io(self, eid: int, etype) -> list[pb.Artifact]:
        ids = [e.artifact_id for e in self.store.get_events_by_execution_ids([eid]) if e.type == etype]
        return self.store.get_artifacts_by_id(ids) if ids else []

    def experiments(self) -> list[pb.Context]:
        return sorted(self.store.get_contexts_by_type("Experiment"), key=lambda c: c.id)

    def experiment(self, name: str) -> pb.Context:
        ctx = self.store.get_context_by_type_and_name("Experiment", name)
        if ctx is None:
            names = ", ".join(c.name for c in self.experiments()) or "(none)"
            raise SystemExit(f"No experiment named '{name}'. Known: {names}")
        return ctx

    def active_serving(self) -> list[pb.Artifact]:
        return [a for a in self.store.get_artifacts_by_type("ServingModel") if get_prop(a, "is_active") == 1]

    def deployed_model_ids(self) -> set[int]:
        return {get_prop(a, "source_model_id") for a in self.active_serving()}

    def ever_deployed_model_ids(self) -> set[int]:
        return {get_prop(a, "source_model_id") for a in self.store.get_artifacts_by_type("ServingModel")}

    # ------------------------------------------------- experiment queries
    def experiment_runs(self, ctx_id: int) -> list[dict]:
        """Every Model in the experiment joined with its Metrics, via lineage.

        Metrics -> (OUTPUT event) -> Evaluate execution -> (INPUT event) -> Model.
        """
        arts = self.store.get_artifacts_by_context(ctx_id)
        rows = []
        for m in (a for a in arts if self.atype[a.type_id] == "Metrics"):
            for ex in self.producers(m.id):
                for model in (a for a in self.io(ex.id, INPUT) if self.atype[a.type_id] == "Model"):
                    rows.append({"model": model, "metrics": m})
        return sorted(rows, key=lambda r: r["model"].id)

    # --------------------------------------------------- lineage traversal
    def backward(self, aid: int) -> "Node":
        """Upstream tree: artifact <- producing execution <- its inputs <- ..."""
        art = self.artifact(aid)
        node = Node(art, self.type_of(art))
        prods = self.producers(aid)
        if self.type_of(art) == "RawDataset":
            node.note = f"registered by {len(prods)} Ingest execution(s): {[p.id for p in prods]}"
            return node
        for ex in prods:
            exn = Node(ex, self.type_of(ex))
            for inp in self.io(ex.id, INPUT):
                exn.children.append(self.backward(inp.id))
            node.children.append(exn)
        return node

    def forward(self, aid: int, seen: set | None = None) -> "Node":
        """Downstream tree: artifact -> consuming executions -> their outputs -> ..."""
        seen = seen if seen is not None else set()
        art = self.artifact(aid)
        node = Node(art, self.type_of(art))
        for ex in self.consumers(aid):
            if ex.id in seen:
                continue
            seen.add(ex.id)
            exn = Node(ex, self.type_of(ex))
            for out in self.io(ex.id, OUTPUT):
                exn.children.append(self.forward(out.id, seen))
            node.children.append(exn)
        return node

    def contexts_of_artifact(self, aid: int) -> list[str]:
        return [c.name for c in self.store.get_contexts_by_artifact(aid)]


@dataclass
class Node:
    obj: object
    type_name: str
    children: list = field(default_factory=list)
    note: str = ""

    @property
    def is_artifact(self) -> bool:
        return isinstance(self.obj, pb.Artifact)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def describe(node: Node) -> str:
    """One-line human summary of an artifact/execution for trace output."""
    o, t = node.obj, node.type_name
    if not node.is_artifact:
        state = STATE_NAMES.get(o.last_known_state, "?")
        extra = ""
        if t == "Train":
            extra = f" run={get_prop(o, 'run_name')}"
        return (f"[{t} execution #{o.id}] state={state} start={get_prop(o, 'start_time')} "
                f"code={get_prop(o, 'code_version')}{extra}")
    p = props_dict(o)
    if t == "RawDataset":
        d = f"v{p['version']} rows={p['row_count']} sha256={p['sha256'][:16]}… file={p['filename']}"
    elif t == "ValidatedDataset":
        d = f"rows_kept={p['rows_kept']}/{p['rows_in']} fraud_rate={p['fraud_rate']:.4f}"
    elif t == "TrainTestSplit":
        d = f"train={p['train_rows']} test={p['test_rows']} seed={p['random_state']}"
    elif t == "Model":
        d = f"v{p['version']} {p['algorithm']} run={p['run_name']} params={json.loads(p['hyperparameters'])['classifier']}"
    elif t == "Metrics":
        d = f"f1={p['f1']:.4f} precision={p['precision']:.4f} recall={p['recall']:.4f} roc_auc={p['roc_auc']:.4f}"
    elif t == "ServingModel":
        d = f"serving v{p['serving_version']} from model v{p['model_version']} active={bool(p['is_active'])}"
    else:
        d = ""
    return f"<{t} #{o.id}> {d}  uri={o.uri}"


def render_tree(node: Node, prefix: str = "", last: bool = True, root: bool = True,
                arrow: str = "<-", highlight: set[int] | None = None) -> list[str]:
    highlight = highlight or set()
    mark = ""
    if node.is_artifact and node.type_name == "Model" and node.obj.id in highlight:
        mark = "  ★ DEPLOYED"
    line = describe(node) + mark + (f"  ({node.note})" if node.note else "")
    lines = [line if root else f"{prefix}{'└' if last else '├'}{arrow} {line}"]
    child_prefix = prefix + ("" if root else ("    " if last else "│   "))
    for i, c in enumerate(node.children):
        lines += render_tree(c, child_prefix, i == len(node.children) - 1, False, arrow, highlight)
    return lines
