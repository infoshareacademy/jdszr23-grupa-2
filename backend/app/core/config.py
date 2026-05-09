from __future__ import annotations

from pathlib import Path

from pydantic import BaseSettings, Field


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class AppSettings(BaseSettings):
    app_name: str = Field(default="ecommerce_trend_platform")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    data_path: Path = Field(default=BASE_DIR / "data" / "products.csv")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = AppSettings()
