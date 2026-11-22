FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project \
    && mkdir -p /app/figures \
    && chown appuser:appuser /app/figures \
    && chmod -R a+rX /app

ENV UV_NO_SYNC=1
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY --chown=appuser:appuser collect.py analyze.py visualize.py ./
COPY --chown=appuser:appuser data ./data
RUN chmod a+rwx /app/figures /app/data && chmod a+rw /app/data/*.csv

USER appuser

CMD ["sh", "-c", "uv run --frozen --no-dev python analyze.py && uv run --frozen --no-dev python visualize.py"]
