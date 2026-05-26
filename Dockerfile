# Pin the Debian codename explicitly so a future ``slim`` re-alias to a
# new Debian release doesn't change the base out from under us. Full
# digest pinning (``python@sha256:...``) belongs in the prod-build CI
# step where the digest is captured at release time.
FROM python:3.12-slim-bookworm AS base

# Don't write .pyc files (keeps the image lean) and flush stdout/stderr
# so container logs surface immediately. PIP_* vars keep the layer cache
# clean and silence pip's self-update nag in build output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# --- Dev target: includes test/lint tools ---
FROM base AS dev
RUN pip install --no-cache-dir ".[dev]"
COPY app/ app/
RUN addgroup --system --gid 1000 appuser \
 && adduser --system --no-create-home --uid 1000 --gid 1000 appuser \
 # /cache is the HuggingFace + fastembed model cache mount point. Docker's
 # named-volume first-mount semantics copy this directory's ownership into
 # the volume, so creating it as appuser here is what lets the non-root
 # uvicorn process write the BM42 sparse model + (optional) reranker model
 # caches inside the volume. Mirrors the ingestion-service Dockerfile.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser /cache \
 && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Prod target: runtime deps only ---
FROM base AS prod
RUN pip install --no-cache-dir .
COPY app/ app/
RUN addgroup --system --gid 1000 appuser \
 && adduser --system --no-create-home --uid 1000 --gid 1000 appuser \
 # See dev-target comment — same ownership setup is required in prod for
 # the named model-cache volume to be writable by the non-root user.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser /cache \
 && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
