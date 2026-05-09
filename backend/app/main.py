from fastapi import FastAPI

from backend.app.api.v1.router import api_router
from backend.app.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(api_router, prefix="/api/v1")


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
