# ml-metadata has no Windows wheels; its manylinux2014 wheels need CPython 3.9-3.11.
# Pin the Debian release so the base image does not drift underneath the wheels.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLMD_DB=/app/mlmd_store/metadata.db

# graphviz system package provides the `dot` binary used to render PNG/SVG lineage graphs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import ml_metadata; print('ml-metadata', ml_metadata.__version__)"

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
