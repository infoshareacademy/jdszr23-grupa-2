FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* /app/
RUN pip install --upgrade pip && pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --only main

COPY backend /app/backend
COPY backend/data /app/backend/data

ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
