"""
training/train_video_model.py

End-to-end training loop for the XceptionNet video-branch classifier on FakeAVCeleb.
Supports:
  - Mixed-precision training (torch.cuda.amp)
  - Early stopping on val AUC
  - MLflow metric logging (accuracy, AUC, F1, per-class confusion matrix)
  - Checkpointing (best model by val AUC)

Usage:
  python train_video_model.py \
    --manifest <path>/manifest.csv \
    --output-dir <path>/checkpoints \
    [--arch xception|efficientnet] \
    [--epochs 30] \
    [--batch-size 4] \
    [--lr 1e-4] \
    [--num-frames 32]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import mlflow
from pathlib import Path
from tqdm import tqdm

# Allow importing from sibling directories
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import FakeAVCelebDataset, get_train_transforms, get_val_transforms, get_audio_transforms
from app.ml.video_model import build_video_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--arch",        default="xception", choices=["xception", "efficientnet"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num-frames",  type=int, default=32)
    p.add_argument("--patience",    type=int, default=5,
                   help="Early stopping patience (epochs without val AUC improvement)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def evaluate(model, loader, device, num_classes=4):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for frames, _, labels in tqdm(loader, desc="Evaluating", leave=False):
            frames = frames.to(device)
            labels = labels.to(device)

            clip_logits, _ = model(frames)
            probs = torch.softmax(clip_logits, dim=1).cpu().numpy()
            preds = clip_logits.argmax(dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0  # fallback if only 1 class present in batch

    return {"acc": acc, "auc": auc, "f1": f1, "cm": cm}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ────────────────────────────────────────────────────────────────
    train_ds = FakeAVCelebDataset(
        args.manifest, split="train", num_frames=args.num_frames,
        transform=get_train_transforms(), audio_transform=get_audio_transforms()
    )
    val_ds = FakeAVCelebDataset(
        args.manifest, split="val", num_frames=args.num_frames,
        transform=get_val_transforms()
    )

    if len(train_ds) == 0:
        print("No training samples found. Check manifest and split assignments.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = build_video_model(
        arch=args.arch,
        num_classes=4,
        num_frames=args.num_frames,
        pretrained=not args.no_pretrained
    ).to(device)
    print(f"Model: {args.arch}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # ── MLflow ──────────────────────────────────────────────────────────────────
    mlflow.set_experiment("deepfake_video_model")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        best_auc      = 0.0
        patience_cnt  = 0
        best_ckpt     = out_dir / f"best_{args.arch}_video.pt"

        for epoch in range(1, args.epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0

            for frames, _, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                frames = frames.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    clip_logits, _ = model(frames)
                    loss = criterion(clip_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────────
            if len(val_ds) > 0:
                metrics = evaluate(model, val_loader, device)
                print(
                    f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} "
                    f"val_acc={metrics['acc']:.4f} val_auc={metrics['auc']:.4f} "
                    f"val_f1={metrics['f1']:.4f}"
                )
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_acc":    metrics["acc"],
                    "val_auc":    metrics["auc"],
                    "val_f1":     metrics["f1"],
                }, step=epoch)

                # ── Early stopping & checkpointing ─────────────────────────────
                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    patience_cnt = 0
                    torch.save({
                        "epoch":       epoch,
                        "arch":        args.arch,
                        "model_state": model.state_dict(),
                        "best_auc":    best_auc,
                    }, best_ckpt)
                    print(f"  → Checkpoint saved (val_auc={best_auc:.4f})")
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"Early stopping at epoch {epoch} (no AUC improvement for {args.patience} epochs).")
                        break
            else:
                # No val set in subset — just save each epoch
                print(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f}  (no val split)")
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                torch.save({
                    "epoch":       epoch,
                    "arch":        args.arch,
                    "model_state": model.state_dict(),
                }, best_ckpt)

        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
        print(f"Best checkpoint saved to: {best_ckpt}")
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))


if __name__ == "__main__":
    main()
