"""
app/core/celery_app.py

Celery application instance.  Workers import this module to discover tasks.
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "deepfake_detector",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.detection_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Retry failed tasks once after 30 s
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
