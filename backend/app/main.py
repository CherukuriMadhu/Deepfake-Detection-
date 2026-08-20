"""
app/main.py

FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import create_tables
from app.api.routes import upload, detect, reports

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Deepfake detection API using FakeAVCeleb-trained XceptionNet (video) "
        "and AudioCNN (audio) models with Grad-CAM explainability.\n\n"
        "**Important caveat**: No deepfake detector achieves 100% accuracy. "
        "Results should be treated as probabilistic evidence, not definitive proof."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    create_tables()


# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(upload.router,  prefix="/api", tags=["Upload"])
app.include_router(detect.router,  prefix="/api", tags=["Detection"])
app.include_router(reports.router, prefix="/api", tags=["Reports"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": settings.APP_NAME,
        "status":  "healthy",
        "docs":    "/docs",
        "caveat":  (
            "⚠️  No deepfake detector achieves 100% accuracy. "
            "Results reflect probabilistic model output."
        ),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
