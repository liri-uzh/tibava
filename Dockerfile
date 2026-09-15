FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagickwand-dev imagemagick git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

ARG WORKSPACE_MEMBER

# Copy only workspace metadata first. This layer changes when locked third-party
# dependencies change, but not for ordinary application source edits.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY packages/tibava_data/pyproject.toml packages/tibava_data/pyproject.toml
COPY packages/tibava_interface/pyproject.toml packages/tibava_interface/pyproject.toml
COPY packages/tibava_utils/pyproject.toml packages/tibava_utils/pyproject.toml
COPY analyser/pyproject.toml analyser/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
COPY inference_ray/pyproject.toml inference_ray/pyproject.toml

# Install the selected member's locked third-party dependency closure directly
# into the runtime image. Workspace packages are installed after their sources
# are copied below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package "${WORKSPACE_MEMBER}" \
      --no-install-workspace

# backend depends on analyser, which depends on inference_ray; all three shared
# workspace sources are therefore required for at least one supported target.
COPY packages/ /app/packages/
COPY analyser/ /app/analyser/
COPY backend/ /app/backend/
COPY inference_ray/ /app/inference_ray/
COPY scripts/ /app/scripts/

# Build and install the selected member and all transitive local workspace
# dependencies as regular wheels. --no-editable avoids runtime source links.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package "${WORKSPACE_MEMBER}" --no-editable

ENTRYPOINT []
