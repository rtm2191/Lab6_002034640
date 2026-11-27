"""Connection to the MLMD metadata store (SQLite file on a mounted volume)."""
from __future__ import annotations

import os
from pathlib import Path

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

DEFAULT_DB = "mlmd_store/metadata.db"


def db_path() -> Path:
    return Path(os.environ.get("MLMD_DB", DEFAULT_DB))


def get_store(path: str | os.PathLike | None = None) -> mlmd.MetadataStore:
    """Open (or create) the SQLite-backed metadata store.

    The original lab used TFX's InteractiveContext, which hides the store in a
    temporary pipeline directory. Here we configure the store explicitly so it
    lives on a Docker volume and persists across container runs.
    """
    path = Path(path) if path else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = metadata_store_pb2.ConnectionConfig()
    config.sqlite.filename_uri = str(path)
    config.sqlite.connection_mode = (
        metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
    )
    return mlmd.MetadataStore(config)
