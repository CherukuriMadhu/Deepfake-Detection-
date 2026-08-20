"""
ml/video_model.py

XceptionNet (and optional EfficientNet-B4) for per-frame face-forgery detection.
Aggregates per-frame scores with a small GRU temporal head or simple mean pooling
to produce a clip-level score.
"""

import torch
import torch.nn as nn
import timm


class XceptionVideoModel(nn.Module):
    """
    XceptionNet feature extractor with a GRU temporal head for clip-level classification.
    Input:  (B, T, C, H, W)  — batch of clips with T sampled frames
    Output: (B, num_classes) — clip-level logits
    """

    def __init__(self, num_classes=4, num_frames=32, pretrained=True, temporal_head="gru"):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_head = temporal_head

        # ── 1. Backbone (XceptionNet via timm) ─────────────────────────────────
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,   # remove the final classification head
        )
        feat_dim = self.backbone.num_features  # 2048 for Xception

        # ── 2. Temporal Aggregation ─────────────────────────────────────────────
        if temporal_head == "gru":
            self.temporal = nn.GRU(
                input_size=feat_dim,
                hidden_size=512,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
            )
            classifier_in = 512
        else:
            # Simple mean pooling — no learned temporal model
            self.temporal = None
            classifier_in = feat_dim

        # ── 3. Per-frame score head (used for Grad-CAM later) ──────────────────
        self.frame_head = nn.Linear(feat_dim, 1)  # per-frame fake probability

        # ── 4. Clip-level classifier ────────────────────────────────────────────
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_frames(self, x):
        """
        x: (B*T, C, H, W)
        Returns frame embeddings: (B*T, feat_dim)
        """
        return self.backbone(x)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # ── Flatten time into batch dim ─────────────────────────────────────────
        x_flat = x.view(B * T, C, H, W)
        feats = self.forward_frames(x_flat)          # (B*T, feat_dim)

        # ── Per-frame fake probabilities ────────────────────────────────────────
        frame_logits = self.frame_head(feats)        # (B*T, 1)
        frame_probs = torch.sigmoid(frame_logits).view(B, T)  # (B, T)

        # ── Temporal aggregation ────────────────────────────────────────────────
        feats_seq = feats.view(B, T, -1)            # (B, T, feat_dim)

        if self.temporal is not None:
            out, _ = self.temporal(feats_seq)       # (B, T, 512)
            # Use last hidden state
            agg = out[:, -1, :]                     # (B, 512)
        else:
            agg = feats_seq.mean(dim=1)             # (B, feat_dim)

        # ── Clip-level classification ───────────────────────────────────────────
        clip_logits = self.classifier(self.dropout(agg))  # (B, num_classes)

        return clip_logits, frame_probs


class EfficientNetVideoModel(nn.Module):
    """
    Optional EfficientNet-B4 variant — controlled by config flag.
    Same interface as XceptionVideoModel.
    """

    def __init__(self, num_classes=4, num_frames=32, pretrained=True):
        super().__init__()
        self.num_frames = num_frames

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        self.temporal = nn.GRU(
            input_size=feat_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.frame_head = nn.Linear(feat_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_frames(self, x):
        return self.backbone(x)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.forward_frames(x_flat)

        frame_logits = self.frame_head(feats)
        frame_probs = torch.sigmoid(frame_logits).view(B, T)

        feats_seq = feats.view(B, T, -1)
        out, _ = self.temporal(feats_seq)
        agg = out[:, -1, :]

        clip_logits = self.classifier(self.dropout(agg))
        return clip_logits, frame_probs


def build_video_model(arch="xception", num_classes=4, num_frames=32, pretrained=True):
    """Factory function — use config flag to switch backbones."""
    if arch == "xception":
        return XceptionVideoModel(num_classes=num_classes, num_frames=num_frames, pretrained=pretrained)
    elif arch == "efficientnet":
        return EfficientNetVideoModel(num_classes=num_classes, num_frames=num_frames, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown video arch: {arch}")
