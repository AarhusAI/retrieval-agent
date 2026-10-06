# Pin the Debian codename explicitly so a future ``slim`` re-alias to a
# new Debian release doesn't change the base out from under us. Full
# digest pinning (``python@sha256:...``) belongs in the prod-build CI
# step where the digest is captured at release time.
FROM python:3.12-slim-bookworm AS base

# Don't write .pyc files (keeps the image lean) and flush stdout/stderr
# so container logs surface immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Pin the in-container appuser to a uid/gid that match the host developer's
# user so ruff/pytest can write caches in the /app bind mount. Override at
# build time for CI / other-uid hosts (compose passes APP_UID/APP_GID).
ARG APP_UID=1000
ARG APP_GID=1000

# The user is created up front, together with every directory it must own.
# All of these directories are empty at this point, so a non-recursive chown
# is free. Creating files as appuser later (uv sync, COPY --chown) avoids the
# recursive `chown -R` that used to duplicate the whole venv into a new layer
# and made the "exporting layers" step very slow.
#
# /opt/venv: lives outside /app because the dev bind mount (./:/app) would
#   hide it. Owned by appuser so `task install` can re-sync it in-container.
# /cache: HuggingFace + fastembed model cache mount point. Docker's
#   named-volume first-mount semantics copy this directory's ownership into
#   the volume, so creating it as appuser here is what lets the non-root
#   uvicorn process write the BM42 sparse model + (optional) reranker model
#   caches inside the volume. Mirrors the ingestion-service Dockerfile.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && addgroup --system --gid ${APP_GID} appuser \
 && adduser --system --no-create-home --uid ${APP_UID} --ingroup appuser appuser \
 && mkdir -p /opt/venv /cache/hf /cache/fastembed \
 && chown appuser:appuser /app /opt/venv /cache /cache/hf /cache/fastembed

# Dependencies come from uv.lock, so dev and prod install exactly what CI tested.
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/opt/venv/bin:$PATH

# Everything from here on runs as appuser, so files are born with the right owner.
USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# --- Dev target: includes test/lint tools ---
FROM base AS dev
RUN uv sync --frozen --no-cache --no-install-project --extra dev
COPY --chown=appuser:appuser app/ app/
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Prod target: runtime deps only ---
FROM base AS prod
RUN uv sync --frozen --no-cache --no-install-project
COPY --chown=appuser:appuser app/ app/
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]