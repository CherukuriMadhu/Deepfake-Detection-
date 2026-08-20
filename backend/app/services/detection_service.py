"""
app/services/detection_service.py

Orchestrates the full detection pipeline:
  1. Frame + face extraction
  2. Audio extraction + spectrogram
  3. Video model inference
  4. Audio model inference
  5. Fusion
  6. Grad-CAM on top-k suspicious frames
  7. Result packaging

This service is called by the Celery worker — it must NEVER be called
directly from the request thread.
"""

import os
import sys
import glob
import hashlib
import shutil
import numpy as np
import torch
import base64
import cv2
from pathlib import Path
from typing import Dict, Any

# Allow sibling imports when run from the backend dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from app.ml.video_model import build_video_model
from app.ml.audio_model import build_audio_model
from app.ml.fusion_model import fuse_and_decode
from app.ml.explainability import GradCAM, generate_heatmap_b64
from training.dataset import get_val_transforms, FakeAVCelebDataset
from training.preprocess import extract_frames_and_faces, extract_audio

import torchaudio
import soundfile as sf
import torchaudio.transforms as T


# ── Model singleton (loaded once per worker) ────────────────────────────────

_video_model = None
_audio_model = None
_mtcnn       = None
_device      = None


def _load_models():
    global _video_model, _audio_model, _mtcnn, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Video model
    _video_model = build_video_model(
        arch=settings.VIDEO_MODEL_ARCH,
        num_classes=4,
        num_frames=settings.NUM_FRAMES,
        pretrained=False,
    ).to(_device)

    if Path(settings.VIDEO_MODEL_CKPT).exists():
        ckpt = torch.load(settings.VIDEO_MODEL_CKPT, map_location=_device, weights_only=True)
        _video_model.load_state_dict(ckpt["model_state"])
        print(f"[DetectionService] Loaded video model from {settings.VIDEO_MODEL_CKPT}")
    else:
        print(f"[DetectionService] WARNING: No video checkpoint at {settings.VIDEO_MODEL_CKPT}. Using random weights.")

    _video_model.eval()

    # Audio model
    _audio_model = build_audio_model(num_classes=4).to(_device)

    if Path(settings.AUDIO_MODEL_CKPT).exists():
        ckpt = torch.load(settings.AUDIO_MODEL_CKPT, map_location=_device, weights_only=True)
        _audio_model.load_state_dict(ckpt["model_state"])
        print(f"[DetectionService] Loaded audio model from {settings.AUDIO_MODEL_CKPT}")
    else:
        print(f"[DetectionService] WARNING: No audio checkpoint at {settings.AUDIO_MODEL_CKPT}. Using random weights.")

    _audio_model.eval()

    # MTCNN face detector
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False, device=_device)


def get_models():
    global _video_model, _audio_model, _mtcnn, _device
    if _video_model is None:
        _load_models()
    return _video_model, _audio_model, _mtcnn, _device


# ── Preprocessing helpers ────────────────────────────────────────────────────

def _preprocess_video(video_path: str, job_id: str) -> Dict:
    """Extract face crops and audio from the uploaded video."""
    video_model, audio_model, mtcnn, device = get_models()

    work_dir = Path(settings.CACHE_DIR) / "jobs" / job_id
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    audio_path = str(work_dir / "audio.wav")

    # Frames
    frames_ok = extract_frames_and_faces(video_path, str(frames_dir), mtcnn, fps_extract=5)
    # Audio
    audio_ok  = extract_audio(video_path, audio_path)

    return {
        "frames_dir": str(frames_dir),
        "audio_path": audio_path if audio_ok else None,
        "frames_ok":  frames_ok,
        "audio_ok":   audio_ok,
        "work_dir":   str(work_dir),
    }


def _load_frames_tensor(frames_dir: str, num_frames: int) -> torch.Tensor:
    """Load and normalise face frames into (1, T, 3, 224, 224)."""
    transform = get_val_transforms()
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    if len(frame_files) == 0:
        return torch.zeros(1, num_frames, 3, 224, 224)

    if len(frame_files) >= num_frames:
        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    else:
        indices = np.pad(np.arange(len(frame_files)),
                         (0, num_frames - len(frame_files)), mode="wrap")

    frames = []
    for i in indices:
        img = cv2.imread(frame_files[i])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = transform(image=img)
        frames.append(aug["image"])

    return torch.stack(frames).unsqueeze(0)   # (1, T, C, H, W)


def _load_audio_tensor(audio_path: str, target_length: int = 16000 * 3) -> torch.Tensor:
    """Load audio WAV into log-mel spectrogram (1, 1, n_mels, T)."""
    try:
        waveform, _ = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
    except Exception:
        waveform = np.zeros(target_length)

    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    wave_t = torch.from_numpy(waveform).float().unsqueeze(0)
    mel = T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=256)(wave_t)
    log_mel = T.AmplitudeToDB()(mel)
    return log_mel.unsqueeze(0)   # (1, 1, n_mels, T)


# ── Main detection entry point ───────────────────────────────────────────────

def run_detection(video_path: str, job_id: str) -> Dict[str, Any]:
    """
    Full detection pipeline. Called from Celery worker.
    Returns a result dict suitable for storing in DetectionResult.
    """
    video_model, audio_model, mtcnn, device = get_models()

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    prep = _preprocess_video(video_path, job_id)

    # ── 2. Load tensors ────────────────────────────────────────────────────────
    frames_tensor = _load_frames_tensor(prep["frames_dir"], settings.NUM_FRAMES).to(device)
    audio_tensor  = _load_audio_tensor(prep["audio_path"]).to(device) if prep["audio_ok"] else \
                    torch.zeros(1, 1, 64, 188).to(device)

    # ── 3. Inference ───────────────────────────────────────────────────────────
    with torch.no_grad():
        v_logits, frame_probs = video_model(frames_tensor)   # (1,4), (1,T)
        a_logits              = audio_model(audio_tensor)     # (1,4)

    # ── 4. Fusion ──────────────────────────────────────────────────────────────
    results = fuse_and_decode(v_logits, a_logits, alpha=settings.FUSION_ALPHA)
    result  = results[0]

    frame_probs_list = frame_probs[0].cpu().tolist()

    # ── 5. Grad-CAM on top-3 most suspicious frames ────────────────────────────
    top_k = 3
    frame_probs_arr = np.array(frame_probs_list)
    top_frame_indices = np.argsort(frame_probs_arr)[::-1][:top_k].tolist()

    frame_files = sorted(glob.glob(os.path.join(prep["frames_dir"], "*.jpg")))
    heatmap_frames = []

    if len(frame_files) > 0:
        grad_cam = GradCAM(video_model)
        for fi in top_frame_indices:
            actual_fi = min(fi, len(frame_files) - 1)
            img_bgr = cv2.imread(frame_files[actual_fi])
            if img_bgr is None:
                continue
            hb64 = generate_heatmap_b64(grad_cam, frames_tensor, fi, img_bgr)
            heatmap_frames.append({"frame_index": actual_fi, "heatmap_b64": hb64})
        grad_cam.remove_hooks()

    # ── 6. Cleanup temp work dir ────────────────────────────────────────────────
    try:
        shutil.rmtree(prep["work_dir"])
    except Exception:
        pass

    return {
        **result,
        "frame_probs": frame_probs_list,
        "heatmap_frames": heatmap_frames,
    }
