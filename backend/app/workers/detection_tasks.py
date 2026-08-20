"""
app/workers/detection_tasks.py

Celery tasks for asynchronous deepfake detection.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from app.core.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal
from app.db import models


def _md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@celery_app.task(bind=True, max_retries=1)
def run_detection_task(self, job_id: str, video_path: str):
    """
    Celery task: runs the full detection pipeline for a given job.
    Updates the DB with status and results.
    """
    db = SessionLocal()
    try:
        # ── Mark as running ────────────────────────────────────────────────────
        job = db.query(models.DetectionJob).filter_by(id=job_id).first()
        if not job:
            return {"error": f"Job {job_id} not found"}

        job.status     = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # ── Check cache (duplicate detection by file hash) ─────────────────────
        upload = job.upload
        file_hash = _md5(video_path)
        upload.file_hash = file_hash
        db.commit()

        # ── Run inference ──────────────────────────────────────────────────────
        # Import here to avoid loading models at import time
        from app.services.detection_service import run_detection
        result_data = run_detection(video_path, job_id)

        # ── Store result ───────────────────────────────────────────────────────
        detection_result = models.DetectionResult(
            job_id          = job_id,
            predicted_class = result_data["predicted_class"],
            confidence      = result_data["confidence"],
            is_fake         = result_data["is_fake"],
            fake_probability = result_data["fake_probability"],
            video_fake_prob = result_data["video_fake_prob"],
            audio_fake_prob = result_data["audio_fake_prob"],
            class_probs     = result_data["class_probs"],
            frame_probs     = result_data.get("frame_probs"),
            heatmap_frames  = result_data.get("heatmap_frames"),
            caveat          = result_data["caveat"],
        )
        db.add(detection_result)

        job.status       = "complete"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        return {"job_id": job_id, "status": "complete"}

    except Exception as exc:
        db.rollback()
        job = db.query(models.DetectionJob).filter_by(id=job_id).first()
        if job:
            job.status    = "failed"
            job.error_msg = str(exc)
            db.commit()
        raise self.retry(exc=exc, countdown=30)

    finally:
        db.close()
