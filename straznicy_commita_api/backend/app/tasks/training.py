from __future__ import annotations

from celery import shared_task


@shared_task
def train_market_models() -> dict[str, str]:
    return {"status": "started"}
