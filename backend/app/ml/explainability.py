"""
app/ml/explainability.py

Grad-CAM implementation over the last convolutional layer of the video model.
Returns normalised heatmaps as base64 PNG overlays for the API response.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from typing import Optional


class GradCAM:
    """
    Hooks into the target conv layer of the video model backbone to compute
    Grad-CAM activation maps for any given input frame.
    """

    def __init__(self, model, target_layer_name: Optional[str] = None):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []

        # Auto-detect last conv layer in backbone if not specified
        target_layer = self._find_target_layer(model, target_layer_name)
        self._register_hooks(target_layer)

    def _find_target_layer(self, model, layer_name):
        """Walk the backbone to find the last conv layer."""
        backbone = getattr(model, "backbone", model)
        last_conv = None
        for name, module in backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("Could not find a Conv2d layer in model backbone.")
        return last_conv

    def _register_hooks(self, layer):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, frames_tensor: torch.Tensor, frame_idx: int) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given frame index in the clip.

        frames_tensor: (1, T, C, H, W)
        Returns: heatmap as (H, W) float array in [0, 1]
        """
        self.model.eval()
        frames_tensor = frames_tensor.requires_grad_(True)

        # Forward pass — needed to accumulate activations
        clip_logits, frame_probs = self.model(frames_tensor)

        # Backprop w.r.t. the fake class score (index 1, 2, or 3 — any non-real)
        # We use the max non-real class probability
        non_real_scores = clip_logits[0, 1:]          # classes 1,2,3
        score = non_real_scores.max()

        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        # Pool gradients over spatial dims → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H', W')
        cam = F.relu(torch.tensor(cam)).numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def generate_heatmap_b64(
    grad_cam: GradCAM,
    frames_tensor: torch.Tensor,
    frame_idx: int,
    original_bgr: np.ndarray,
) -> str:
    """
    Generate a Grad-CAM heatmap overlay on the original frame and return
    as a base64-encoded PNG string.
    """
    cam = grad_cam.compute(frames_tensor, frame_idx)

    h, w = original_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8   = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(original_bgr, 0.5, heatmap_bgr, 0.5, 0)

    # Encode to base64 PNG
    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64
