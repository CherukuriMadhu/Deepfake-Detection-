"""
app/api/routes/reports.py

GET /api/reports/{job_id}/pdf — generate and return a downloadable PDF report.
"""

import io
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from app.core.config import settings
from app.db.session import get_db
from app.db import models

router = APIRouter()

CLASS_DESCRIPTIONS = {
    "real-real":            "Genuine video with genuine audio — no manipulation detected.",
    "fake_video-real_audio": "Face manipulation detected (FaceSwap / FSGAN style).",
    "real_video-fake_audio": "Voice cloning / synthetic audio detected (SV2TTS / Wav2Lip style).",
    "fake_video-fake_audio": "Both face manipulation AND voice cloning detected.",
}


def _build_pdf(job: models.DetectionJob, result: models.DetectionResult, out_path: str):
    """Build a PDF report using ReportLab."""
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=12)
    story.append(Paragraph("Deepfake Detection Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3b82f6")))
    story.append(Spacer(1, 0.4 * cm))

    # ── Metadata table ─────────────────────────────────────────────────────────
    upload   = job.upload
    meta_data = [
        ["Job ID",      job.id],
        ["Filename",    upload.filename if upload else "N/A"],
        ["Analysed at", result.created_at.strftime("%Y-%m-%d %H:%M UTC") if result.created_at else "N/A"],
        ["File size",   f"{upload.file_size / 1024:.1f} KB" if upload and upload.file_size else "N/A"],
    ]
    meta_table = Table(meta_data, colWidths=[5 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("PADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.6 * cm))

    # ── Verdict ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Verdict", styles["Heading2"]))

    verdict_color = colors.HexColor("#ef4444") if result.is_fake else colors.HexColor("#22c55e")
    verdict_text  = "⚠ DEEPFAKE DETECTED" if result.is_fake else "✓ AUTHENTIC"
    verdict_style = ParagraphStyle("verdict", parent=styles["Normal"],
                                   fontSize=16, textColor=verdict_color, spaceAfter=6)
    story.append(Paragraph(verdict_text, verdict_style))

    class_desc = CLASS_DESCRIPTIONS.get(result.predicted_class or "", "Unknown")
    story.append(Paragraph(f"<b>Classification:</b> {result.predicted_class}", styles["Normal"]))
    story.append(Paragraph(class_desc, styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # ── Confidence scores ──────────────────────────────────────────────────────
    story.append(Paragraph("Confidence Scores", styles["Heading2"]))

    score_data = [
        ["Metric", "Score"],
        ["Overall fake probability", f"{result.fake_probability:.1%}"],
        ["Video manipulation score", f"{result.video_fake_prob:.1%}"],
        ["Audio manipulation score", f"{result.audio_fake_prob:.1%}"],
        ["Predicted class confidence", f"{result.confidence:.1%}"],
    ]
    if result.class_probs:
        score_data.append(["", ""])
        score_data.append(["Class Probabilities", ""])
        for cls_name, prob in result.class_probs.items():
            score_data.append([f"  {cls_name}", f"{prob:.1%}"])

    score_table = Table(score_data, colWidths=[10 * cm, 7 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.6 * cm))

    # ── Caveat ─────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#fbbf24")))
    caveat_style = ParagraphStyle("caveat", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#92400e"),
                                   backColor=colors.HexColor("#fffbeb"),
                                   borderPadding=6, spaceAfter=4, spaceBefore=4)
    story.append(Paragraph(result.caveat or "", caveat_style))

    doc.build(story)


@router.get("/reports/{job_id}/pdf")
def get_pdf_report(job_id: str, db: Session = Depends(get_db)):
    """Generate (or return cached) PDF report for a completed detection job."""
    job = db.query(models.DetectionJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail=f"Job is not complete (status: {job.status}).")
    if not job.result:
        raise HTTPException(status_code=404, detail="No result available for this job.")

    # Check cache
    existing_report = db.query(models.Report).filter_by(job_id=job_id).first()
    if existing_report and Path(existing_report.filepath).exists():
        return FileResponse(
            existing_report.filepath,
            media_type="application/pdf",
            filename=f"deepfake_report_{job_id[:8]}.pdf",
        )

    # Generate PDF
    reports_dir = Path(settings.REPORTS_DIR)
    reports_dir.mkdir(exist_ok=True)
    pdf_path = str(reports_dir / f"report_{job_id}.pdf")
    _build_pdf(job, job.result, pdf_path)

    # Save to DB
    report = models.Report(job_id=job_id, filepath=pdf_path)
    db.add(report)
    db.commit()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"deepfake_report_{job_id[:8]}.pdf",
    )
