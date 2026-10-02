# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_DIR=/models \
    PAPERPILOT_ALEMBIC_ROOT=/app

RUN pip install --no-cache-dir uv==0.12.22
WORKDIR /app

# 1) dependencies only (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project

# 2) application code
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini docker-entrypoint.sh ./
RUN uv sync --frozen --no-dev --extra ml

# 3) optionally bake the ~130 MB embedding model into the image so containers start offline
ARG PRELOAD_MODEL=true
RUN mkdir -p /models && if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')"; \
    fi

RUN useradd --system --uid 10001 --home /app app && chown -R app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
