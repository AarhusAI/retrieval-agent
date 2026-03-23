FROM python:3.12-slim

ARG ENV=prod

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN if [ "$ENV" = "dev" ]; then \
      pip install --no-cache-dir ".[dev]"; \
    else \
      pip install --no-cache-dir .; \
    fi

COPY app/ app/

EXPOSE 8000

HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
