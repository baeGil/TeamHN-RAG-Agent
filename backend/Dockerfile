FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml /app/pyproject.toml

# Install deps via uv
RUN uv venv /opt/venv --python 3.11 \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && uv pip install --python /opt/venv/bin/python -e "."


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY backend /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
