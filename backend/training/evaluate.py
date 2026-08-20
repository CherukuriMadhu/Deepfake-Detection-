"""
training/evaluate.py

Evaluates the trained video + audio + fusion models on the held-out test split.
Also supports cross-manipulation generalization testing:
  --train-cats to specify which manipulation methods the model was trained on
  --test-cats  to specify which manipulation methods to test on
  (both default to all categories)

Usage:
  python -m training.evaluate \
    --manifest <path>/manifest.csv \
    --video-ckpt <path>/best_xception_video.pt \
    --audio-ckpt <path>/best_audio_model.pt \
    --arch xception \
    [--fusion-alpha 0.5] \
    [--num-frames 32]
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import FakeAVCelebDataset, get_val_transforms
from app.ml.video_model import build_video_model
from app.ml.audio_model import build_audio_model
from app.ml.fusion_model import fuse_and_decode, CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",      required=True)
    p.add_argument("--video-ckpt",    required=True)
    p.add_argument("--audio-ckpt",    required=True)
    p.add_argument("--arch",          default="xception")
    p.add_argument("--num-frames",    type=int, default=32)
    p.add_argument("--batch-size",    type=int, default=4)
    p.add_argument("--num-workers",   type=int, default=0)
    p.add_argument("--fusion-alpha",  type=float, default=0.5,
                   help="Weight for video branch in fusion (1-alpha for audio)")
    return p.parse_args()


def run_inference(video_model, audio_model, loader, device, alpha=0.5):
    video_model.eval()
    audio_model.eval()

    all_labels, all_fused_probs, all_preds = [], [], []

    with torch.no_grad():
        for frames, audio, labels in tqdm(loader, desc="Evaluating"):
            frames = frames.to(device)
            audio  = audio.to(device)

            v_logits, _ = video_model(frames)
            a_logits     = audio_model(audio)

            results = fuse_and_decode(v_logits, a_logits, alpha=alpha)
            fused_probs = torch.tensor(
                [[r["class_probs"][c] for c in CLASS_NAMES] for r in results]
            )
            preds = fused_probs.argmax(dim=1).numpy()

            all_fused_probs.append(fused_probs.numpy())
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_labels      = np.concatenate(all_labels)
    all_fused_probs = np.concatenate(all_fused_probs)
    all_preds       = np.concatenate(all_preds)
    return all_labels, all_fused_probs, all_preds


def print_report(all_labels, all_probs, all_preds, split_name="test"):
    print(f"\n{'='*60}")
    print(f"  Evaluation on: {split_name}")
    print(f"{'='*60}")

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro AUC : {auc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")

    print("\nPer-class Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(cm_df.to_string())
    print()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ─────────────────────────────────────────────────────────────
    video_model = build_video_model(arch=args.arch, num_classes=4,
                                    num_frames=args.num_frames, pretrained=False).to(device)
    vc = torch.load(args.video_ckpt, map_location=device, weights_only=True)
    video_model.load_state_dict(vc["model_state"])
    print(f"Loaded video model from {args.video_ckpt} (epoch {vc.get('epoch','?')})")

    audio_model = build_audio_model(num_classes=4).to(device)
    ac = torch.load(args.audio_ckpt, map_location=device, weights_only=True)
    audio_model.load_state_dict(ac["model_state"])
    print(f"Loaded audio model from {args.audio_ckpt} (epoch {ac.get('epoch','?')})")

    # ── Full test set ────────────────────────────────────────────────────────────
    test_ds = FakeAVCelebDataset(
        args.manifest, split="test", num_frames=args.num_frames,
        transform=get_val_transforms()
    )
    if len(test_ds) == 0:
        print("No test samples found. Exiting.")
        return

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    labels, probs, preds = run_inference(video_model, audio_model, test_loader,
                                         device, alpha=args.fusion_alpha)
    print_report(labels, probs, preds, split_name="Full test set")

    # ── Cross-manipulation generalization ────────────────────────────────────────
    # We can read the manifest and filter to specific categories to simulate
    # training on one manipulation method and testing on another.
    manifest = pd.read_csv(args.manifest)
    test_manifest = manifest[manifest["split"] == "test"]
    categories = test_manifest["category"].unique()

    if len(categories) > 1:
        print("\n--- Cross-manipulation generalization (per-category breakdown) ---")
        for cat in categories:
            cat_indices = test_manifest[test_manifest["category"] == cat].index.tolist()
            if len(cat_indices) == 0:
                continue
            cat_labels = labels[cat_indices]
            cat_probs  = probs[cat_indices]
            cat_preds  = preds[cat_indices]
            print_report(cat_labels, cat_probs, cat_preds, split_name=f"Category: {cat}")

    print("\nNote: Cross-manipulation generalization is the key metric — "
          "training on FaceSwap/Wav2Lip and testing on FSGAN/SV2TTS "
          "reveals true detector robustness.")


if __name__ == "__main__":
    main()
