"""
ml/fusion_model.py

Combines video-branch and audio-branch scores to produce:
  - Final 4-class logits (and binary real/fake verdict)
  - Per-modality confidence breakdown
  - Calibrated uncertainty caveat

Two fusion modes:
  'weighted_avg' : simple learnable weighted average of class probabilities
  'mlp'          : small MLP fusion head that takes concatenated embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = ["real-real", "fake_video-real_audio", "real_video-fake_audio", "fake_video-fake_audio"]


class WeightedFusionModel(nn.Module):
    """
    Learnable weighted average of softmax probabilities from video and audio branches.
    Very fast — no extra forward passes needed; just learns alpha ∈ (0,1).
    """

    def __init__(self, num_classes=4):
        super().__init__()
        # alpha=0.5 initially; learned during fine-tuning
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.num_classes = num_classes

    def forward(self, video_logits, audio_logits):
        """
        video_logits: (B, num_classes)
        audio_logits: (B, num_classes)
        Returns fused_logits: (B, num_classes)
        """
        alpha = torch.sigmoid(self.alpha)
        v_probs = F.softmax(video_logits, dim=1)
        a_probs = F.softmax(audio_logits, dim=1)
        fused_probs = alpha * v_probs + (1 - alpha) * a_probs
        # Return as logits (log-space) for CrossEntropyLoss compatibility
        return torch.log(fused_probs + 1e-8)


class MLPFusionModel(nn.Module):
    """
    Small MLP that takes concatenated video + audio embeddings (not logits)
    and produces joint class predictions.
    Requires the backbone to expose intermediate embeddings.
    """

    def __init__(self, video_embed_dim=512, audio_embed_dim=256, num_classes=4):
        super().__init__()
        in_dim = video_embed_dim + audio_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_embed, audio_embed):
        """
        video_embed: (B, video_embed_dim)
        audio_embed: (B, audio_embed_dim)
        """
        combined = torch.cat([video_embed, audio_embed], dim=1)
        return self.fusion(combined)


# ── Inference helper ────────────────────────────────────────────────────────────

def fuse_and_decode(video_logits, audio_logits, alpha=0.5):
    """
    Stateless inference fusion (no learned params).
    Returns a dict with class probabilities, binary verdict, and per-modality info.

    IMPORTANT: No detector is 100% reliable. This system may produce false
    positives or false negatives. Always treat results as probabilistic evidence,
    not as definitive proof.
    """
    v_probs = F.softmax(video_logits, dim=1)   # (B, 4)
    a_probs = F.softmax(audio_logits, dim=1)   # (B, 4)
    fused   = alpha * v_probs + (1 - alpha) * a_probs   # (B, 4)

    pred_class = fused.argmax(dim=1)           # (B,)
    confidence = fused.max(dim=1).values       # (B,)

    results = []
    for i in range(fused.shape[0]):
        cls_idx    = int(pred_class[i])
        cls_name   = CLASS_NAMES[cls_idx]
        conf       = float(confidence[i])

        # Binary: anything that is not "real-real" is considered fake
        is_fake       = cls_idx != 0
        fake_prob     = 1.0 - float(fused[i, 0])   # P(not real-real)

        results.append({
            "predicted_class":   cls_name,
            "class_index":       cls_idx,
            "confidence":        round(conf, 4),
            "is_fake":           is_fake,
            "fake_probability":  round(fake_prob, 4),
            "video_fake_prob":   round(float(v_probs[i, 1] + v_probs[i, 3]), 4),
            "audio_fake_prob":   round(float(a_probs[i, 2] + a_probs[i, 3]), 4),
            "class_probs": {
                CLASS_NAMES[j]: round(float(fused[i, j]), 4)
                for j in range(4)
            },
            "caveat": (
                "⚠️  No deepfake detector achieves 100% accuracy. "
                "This result reflects probabilistic model output and should not "
                "be treated as definitive forensic evidence."
            ),
        })

    return results


def build_fusion_model(mode="weighted_avg", **kwargs):
    if mode == "weighted_avg":
        return WeightedFusionModel(**kwargs)
    elif mode == "mlp":
        return MLPFusionModel(**kwargs)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
