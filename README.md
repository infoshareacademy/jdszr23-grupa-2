# E-commerce Trend Analysis Platform

## Key Components

- Python 3.11.9
- FastAPI + Pydantic v2
- Local dataset ingestion from CSV / XLSX
- Prophet, XGBoost, scikit-learn
- Pandas for feature engineering
- MLflow-compatible experiment hooks
- Docker-based deployment
- Simplified module separation for API and ML logic

## Simplified Structure

- `backend/app/` — app code
- `backend/app/api/` — public API endpoints
- `backend/app/core/` — config and environment settings
- `backend/app/ml/` — data loading, feature engineering, pipeline
- `backend/app/schemas/` — request/response models
- `backend/data/` — dataset files
- `notebooks/` — notebook experiments and reports
- `backend/Dockerfile` — Docker build config
- `docker-compose.yml` — local Docker orchestration
