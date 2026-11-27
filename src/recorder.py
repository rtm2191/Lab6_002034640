"""Helpers that turn every pipeline step into a fully-recorded MLMD Execution.

A StepRun:
  1. inserts the Execution as RUNNING as soon as the step starts (so a crash
     still leaves a trace) and associates it with the run's Contexts;
  2. on success, writes INPUT/OUTPUT Events + output Artifacts and attributes the
     artifacts to the Contexts in ONE atomic `put_execution` call;
  3. on failure, marks the Execution FAILED, keeps its INPUT events, and stores
     the error message.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from ml_metadata.proto import metadata_store_pb2 as pb

from .mlmd_types import get_prop, set_props


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_version(repo_root: str | Path = ".") -> str:
    """Current git commit hash, read straight from .git (no git binary needed)."""
    git = Path(repo_root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1]
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text().strip()[:12]
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ")[0][:12]
    except OSError:
        pass
    return "unknown"


class PipelineError(RuntimeError):
    """Raised when a step fails; the pipeline stops after recording it."""


def new_artifact(store, type_name: str, uri: str, props: dict, name: str | None = None,
                 custom: dict | None = None) -> pb.Artifact:
    a = pb.Artifact()
    a.type_id = store.get_artifact_type(type_name).id
    a.uri = uri
    a.state = pb.Artifact.LIVE
    if name:
        a.name = name
    set_props(a, props)
    for k, v in (custom or {}).items():
        if isinstance(v, float):
            a.custom_properties[k].double_value = v
        elif isinstance(v, int):
            a.custom_properties[k].int_value = v
        else:
            a.custom_properties[k].string_value = str(v)
    return a


class StepRun:
    def __init__(self, store, exec_type: str, contexts: list[pb.Context], name: str | None = None,
                 props: dict | None = None, code_ver: str = "unknown"):
        self.store = store
        self.contexts = contexts
        self.t0 = time.time()
        self.inputs: list[tuple[pb.Artifact, str]] = []
        self.execution = pb.Execution()
        self.execution.type_id = store.get_execution_type(exec_type).id
        if name:
            self.execution.name = name
        self.execution.last_known_state = pb.Execution.RUNNING
        set_props(self.execution, {"start_time": now_iso(), "status": "RUNNING",
                                   "code_version": code_ver, **(props or {})})
        # Insert immediately, associated with all contexts, with no events yet.
        eid, _, _ = store.put_execution(self.execution, [], contexts)
        self.execution.id = eid

    @property
    def id(self) -> int:
        return self.execution.id

    def add_input(self, artifact: pb.Artifact, key: str) -> None:
        self.inputs.append((artifact, key))

    def _event(self, etype, key: str, index: int = 0) -> pb.Event:
        ev = pb.Event(type=etype)
        ev.path.steps.add().key = key
        ev.path.steps.add().index = index
        return ev

    def _close(self, state, status: str, extra: dict) -> None:
        self.execution.last_known_state = state
        set_props(self.execution, {"end_time": now_iso(), "status": status,
                                   "duration_seconds": round(time.time() - self.t0, 3), **extra})

    def complete(self, outputs: list[tuple[pb.Artifact, str]] | None = None,
                 props: dict | None = None) -> list[int]:
        """Record success. Returns the ids of the output artifacts (in order)."""
        self._close(pb.Execution.COMPLETE, "COMPLETE", props or {})
        pairs = [(a, self._event(pb.Event.INPUT, k, i)) for i, (a, k) in enumerate(self.inputs)]
        outputs = outputs or []
        pairs += [(a, self._event(pb.Event.OUTPUT, k, i)) for i, (a, k) in enumerate(outputs)]
        _, art_ids, _ = self.store.put_execution(self.execution, pairs, self.contexts)
        out_ids = art_ids[len(self.inputs):]
        for (a, _), aid in zip(outputs, out_ids):
            a.id = aid
        return out_ids

    def fail(self, error: str, props: dict | None = None) -> None:
        self._close(pb.Execution.FAILED, "FAILED", {"error_message": error[:4000], **(props or {})})
        pairs = [(a, self._event(pb.Event.INPUT, k, i)) for i, (a, k) in enumerate(self.inputs)]
        self.store.put_execution(self.execution, pairs, self.contexts)


# --------------------------------------------------------------- contexts
def get_or_create_context(store, type_name: str, name: str, props: dict) -> pb.Context:
    existing = store.get_context_by_type_and_name(type_name, name)
    if existing is not None:
        return existing
    ctx = pb.Context(type_id=store.get_context_type(type_name).id, name=name)
    set_props(ctx, props)
    [ctx.id] = store.put_contexts([ctx])
    return ctx


def update_context(store, ctx: pb.Context, props: dict) -> None:
    set_props(ctx, props)
    store.put_contexts([ctx])


def next_version(store, type_name: str, key: str = "version") -> int:
    versions = [get_prop(a, key) or 0 for a in store.get_artifacts_by_type(type_name)]
    return max(versions, default=0) + 1
