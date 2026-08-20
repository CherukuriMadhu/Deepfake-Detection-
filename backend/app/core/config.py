"""
app/core/config.py

Application-wide settings loaded from environment variables (with .env fallback).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ─────────────────────────────────────────────────────────────────────
    APP_NAME: str = "Deepfake Detector API"
    DEBUG: bool = False

    # ── Storage ─────────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    CACHE_DIR: str = "dataset_cache"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ── Database ────────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./deepfake.db"   # swap for postgresql://... in prod

    # ── Redis / Celery ──────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ── ML Models ───────────────────────────────────────────────────────────────
    VIDEO_MODEL_ARCH: str = "xception"
    VIDEO_MODEL_CKPT: str = "checkpoints/best_xception_video.pt"
    AUDIO_MODEL_CKPT: str = "checkpoints/best_audio_model.pt"
    NUM_FRAMES: int = 32
    FUSION_ALPHA: float = 0.5    # weight for video branch in fusion

    # ── Reports ─────────────────────────────────────────────────────────────────
    REPORTS_DIR: str = "reports"


settings = Settings()

# Ensure required dirs exist at import time
for _d in [settings.UPLOAD_DIR, settings.CACHE_DIR, settings.REPORTS_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)
