"""
app/api/routes/detect.py

GET /api/status/{job_id}  — poll job status
GET /api/results/{job_id} — get full detection result
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.models.schemas import JobStatusResponse, DetectionResultResponse, HeatmapFrame

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Poll job status: queued | running | complete | failed."""
    job = db.query(models.DetectionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(
        job_id       = job.id,
        status       = job.status,
        created_at   = job.created_at,
        started_at   = job.started_at,
        completed_at = job.completed_at,
        error_msg    = job.error_msg,
    )


@router.get("/results/{job_id}", response_model=DetectionResultResponse)
def get_results(job_id: str, db: Session = Depends(get_db)):
    """Return full detection result once the job is complete."""
    job = db.query(models.DetectionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    if job.status == "queued" or job.status == "running":
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job.status}. Poll /api/status/{job_id}."
        )
    if job.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.error_msg}"
        )

    result = job.result
    if not result:
        raise HTTPException(status_code=404, detail="Result not available yet.")

    heatmap_frames = None
    if result.heatmap_frames:
        heatmap_frames = [
            HeatmapFrame(
                frame_index=hf["frame_index"],
                heatmap_b64=hf["heatmap_b64"]
            )
            for hf in result.heatmap_frames
        ]

    return DetectionResultResponse(
        job_id           = job_id,
        status           = job.status,
        predicted_class  = result.predicted_class,
        confidence       = result.confidence,
        class_probs      = result.class_probs,
        is_fake          = result.is_fake,
        fake_probability = result.fake_probability,
        video_fake_prob  = result.video_fake_prob,
        audio_fake_prob  = result.audio_fake_prob,
        frame_probs      = result.frame_probs,
        heatmap_frames   = heatmap_frames,
        caveat           = result.caveat,
        created_at       = result.created_at,
    )
