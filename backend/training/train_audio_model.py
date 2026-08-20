"""
training/train_audio_model.py

Training loop for the audio-branch CNN classifier (AudioCNN).
Mirrors train_video_model.py in structure but consumes only the audio branch
of FakeAVCelebDataset.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import mlflow
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import FakeAVCelebDataset, get_val_transforms, get_audio_transforms
from app.ml.audio_model import build_audio_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def evaluate_audio(model, loader, device, num_classes=4):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for _, audio, labels in tqdm(loader, desc="Eval audio", leave=False):
            audio  = audio.to(device)
            labels = labels.to(device)
            logits = model(audio)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return {"acc": acc, "auc": auc, "f1": f1}


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ────────────────────────────────────────────────────────────────
    train_ds = FakeAVCelebDataset(
        args.manifest, split="train",
        transform=get_val_transforms(),       # no heavy video aug needed here
        audio_transform=get_audio_transforms()
    )
    val_ds = FakeAVCelebDataset(
        args.manifest, split="val",
        transform=get_val_transforms()
    )

    if len(train_ds) == 0:
        print("No training samples found.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ── Model ───────────────────────────────────────────────────────────────────
    model     = build_audio_model(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_ckpt = out_dir / "best_audio_model.pt"

    mlflow.set_experiment("deepfake_audio_model")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        best_auc     = 0.0
        patience_cnt = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            for _, audio, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                audio  = audio.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(audio)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            avg_loss = train_loss / max(len(train_loader), 1)
            scheduler.step()

            if len(val_ds) > 0:
                m = evaluate_audio(model, val_loader, device)
                print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
                      f"val_acc={m['acc']:.4f} val_auc={m['auc']:.4f} val_f1={m['f1']:.4f}")
                mlflow.log_metrics(
                    {"train_loss": avg_loss, "val_acc": m["acc"],
                     "val_auc": m["auc"], "val_f1": m["f1"]}, step=epoch)

                if m["auc"] > best_auc:
                    best_auc = m["auc"]
                    patience_cnt = 0
                    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                                "best_auc": best_auc}, best_ckpt)
                    print(f"  → Checkpoint saved (val_auc={best_auc:.4f})")
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break
            else:
                print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}  (no val split)")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_ckpt)

        print(f"\nDone. Best val AUC: {best_auc:.4f}")
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))


if __name__ == "__main__":
    main()
