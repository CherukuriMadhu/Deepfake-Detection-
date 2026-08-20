"""
app/db/models.py

SQLAlchemy ORM models for the deepfake detection system.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


class Upload(Base):
    __tablename__ = "uploads"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename    = Column(String, nullable=False)
    filepath    = Column(String, nullable=False)
    file_hash   = Column(String, index=True)       # MD5 for duplicate detection
    file_size   = Column(Integer)
    content_type = Column(String)
    created_at  = Column(DateTime, default=_utcnow)

    job         = relationship("DetectionJob", back_populates="upload", uselist=False)


class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_id   = Column(String, ForeignKey("uploads.id"), nullable=False)
    status      = Column(String, default="queued")  # queued | running | complete | failed
    celery_task_id = Column(String, nullable=True)
    created_at  = Column(DateTime, default=_utcnow)
    started_at  = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_msg   = Column(Text, nullable=True)

    upload      = relationship("Upload", back_populates="job")
    result      = relationship("DetectionResult", back_populates="job", uselist=False)


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id          = Column(String, ForeignKey("detection_jobs.id"), nullable=False, unique=True)

    # 4-class output
    predicted_class = Column(String)
    confidence      = Column(Float)
    is_fake         = Column(Boolean)
    fake_probability = Column(Float)

    # Per-modality
    video_fake_prob = Column(Float)
    audio_fake_prob = Column(Float)

    # Full JSON blobs
    class_probs     = Column(JSON)       # {class_name: prob}
    frame_probs     = Column(JSON)       # list of per-frame fake probabilities
    heatmap_frames  = Column(JSON)       # list of {frame_index, heatmap_b64}

    # Calibrated caveat (always populated)
    caveat          = Column(Text)

    created_at      = Column(DateTime, default=_utcnow)

    job             = relationship("DetectionJob", back_populates="result")


class Report(Base):
    __tablename__ = "reports"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id      = Column(String, ForeignKey("detection_jobs.id"), nullable=False)
    filepath    = Column(String, nullable=False)
    created_at  = Column(DateTime, default=_utcnow)
