FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface"

WORKDIR /app

# Install dependency metadata first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install third-party dependencies without installing project source yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy files required to build/install the project package
COPY README.md ./
COPY src ./src

# Install the project itself
RUN uv sync --frozen --no-dev

# Runtime directories used by the app / mounts
RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/.cache/huggingface

EXPOSE 8011
EXPOSE 8501