"""
app/api/routes/upload.py

POST /api/upload — accept video file, validate, store, and queue detection job.
"""

import os
import uuid
import hashlib
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
import aiofiles
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db import models
from app.models.schemas import UploadResponse
from app.workers.detection_tasks import run_detection_task

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/webm", "video/x-matroska",
}
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _partial_md5(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """Fast partial hash for duplicate detection (first 1 MB)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: UploadFile = File(...),
    db:   Session    = Depends(get_db),
):
    """
    Accept a video file upload.  Validates format and size, persists to disk,
    creates an Upload + DetectionJob row in the DB, and enqueues a Celery task.
    Returns the job_id for polling.
    """
    # ── Validate content type ──────────────────────────────────────────────────
    ct = file.content_type or ""
    if ct not in ALLOWED_CONTENT_TYPES:
        # Accept by extension as fallback
        ext = Path(file.filename or "").suffix.lower()
        if ext not in {".mp4", ".mpeg", ".mov", ".avi", ".webm", ".mkv"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {ct or ext}. Accepted: mp4, avi, mov, mkv, webm."
            )

    # ── Persist file ───────────────────────────────────────────────────────────
    upload_id = str(uuid.uuid4())
    upload_dir = Path(settings.UPLOAD_DIR) / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename or "upload").name
    dest_path = upload_dir / safe_filename

    total_bytes = 0
    async with aiofiles.open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 256):   # 256 KB chunks
            total_bytes += len(chunk)
            if total_bytes > MAX_BYTES:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max upload size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                )
            await out.write(chunk)

    # ── Duplicate check (fast hash) ────────────────────────────────────────────
    partial_hash = _partial_md5(str(dest_path))
    existing_upload = db.query(models.Upload).filter_by(file_hash=partial_hash).first()
    if existing_upload and existing_upload.job:
        existing_job = existing_upload.job
        if existing_job.status == "complete":
            # Return the cached result job
            return UploadResponse(
                job_id    = existing_job.id,
                upload_id = existing_upload.id,
                filename  = existing_upload.filename,
                status    = "complete",
                message   = "Duplicate upload detected — returning cached result.",
            )

    # ── DB: Upload row ─────────────────────────────────────────────────────────
    db_upload = models.Upload(
        id           = upload_id,
        filename     = safe_filename,
        filepath     = str(dest_path.absolute()),
        file_hash    = partial_hash,
        file_size    = total_bytes,
        content_type = ct,
    )
    db.add(db_upload)

    # ── DB: DetectionJob row ───────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    db_job = models.DetectionJob(id=job_id, upload_id=upload_id, status="queued")
    db.add(db_job)
    db.commit()

    # ── Enqueue Celery task ────────────────────────────────────────────────────
    task = run_detection_task.delay(job_id, str(dest_path.absolute()))
    db_job.celery_task_id = task.id
    db.commit()

    return UploadResponse(
        job_id    = job_id,
        upload_id = upload_id,
        filename  = safe_filename,
        status    = "queued",
        message   = "Video uploaded and queued for analysis. Poll /api/status/{job_id} for updates.",
    )
