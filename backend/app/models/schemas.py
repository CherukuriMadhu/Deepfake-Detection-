"""
app/models/schemas.py

Pydantic request/response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


# ── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    upload_id: str
    filename: str
    status: str
    message: str


# ── Job Status ───────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # queued | running | complete | failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_msg: Optional[str] = None


# ── Detection Result ─────────────────────────────────────────────────────────

class HeatmapFrame(BaseModel):
    frame_index: int
    heatmap_b64: str     # base64-encoded PNG heatmap overlay


class DetectionResultResponse(BaseModel):
    job_id: str
    status: str

    # 4-class classification
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    class_probs: Optional[Dict[str, float]] = None

    # Binary verdict
    is_fake: Optional[bool] = None
    fake_probability: Optional[float] = None

    # Per-modality breakdown
    video_fake_prob: Optional[float] = None
    audio_fake_prob: Optional[float] = None

    # Per-frame scores & Grad-CAM heatmaps
    frame_probs: Optional[List[float]] = None
    heatmap_frames: Optional[List[HeatmapFrame]] = None

    # Calibrated uncertainty caveat — always present in response
    caveat: str = Field(
        default=(
            "⚠️  No deepfake detector achieves 100% accuracy. "
            "This result reflects probabilistic model output and should not "
            "be treated as definitive forensic evidence."
        )
    )

    created_at: Optional[datetime] = None


# ── Report ───────────────────────────────────────────────────────────────────

class ReportResponse(BaseModel):
    job_id: str
    report_url: str
    message: str
