import argparse
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import torchvision.transforms as T

from datasets import (
    FFPPFrameDataset,
    CelebDFImageDataset,
    RealFakeImageDataset,
    tri_collate_fn,
    single_collate_fn,
)
from models import TriAFN, XceptionNet, BaselineCNN
import features as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    model_variant: str
    train_root: str
    val_root: str
    test_root: str
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    augment: bool
    num_workers: int
    save_path: str


def set_image_size(img_size: int):
    F.SPATIAL_TRANSFORM = T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    F.SINGLE_CHANNEL_TRANSFORM = T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def build_model(model_variant: str):
    if model_variant == "tri_afn_b0":
        return TriAFN(model_name="efficientnet_b0", feature_dim=256)
    if model_variant == "tri_afn_b4":
        return TriAFN(model_name="tf_efficientnet_b4_ns", feature_dim=512)
    if model_variant == "xception_spatial":
        return XceptionNet(num_classes=1)
    if model_variant == "baseline_cnn_spatial":
        return BaselineCNN(num_classes=1)
    raise ValueError(f"Unknown model_variant: {model_variant}")


def make_loader(root: str, model_variant: str, img_size: int, batch_size: int, augment: bool, shuffle: bool, num_workers: int):
    if is_tri_stream(model_variant):
        set_image_size(img_size)
        ds = FFPPFrameDataset(root, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=tri_collate_fn)

    ds = RealFakeImageDataset(root, img_size=img_size, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=single_collate_fn)


def evaluate_on_loader(model: nn.Module, loader: DataLoader, desc: str) -> Dict[str, Any]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for (spatial_batch, freq_batch, noise_batch), labels in tqdm(loader, desc=desc, leave=False):
            spatial_batch = spatial_batch.to(device)
            freq_batch = freq_batch.to(device)
            noise_batch = noise_batch.to(device)
            labels = labels.view(-1).to(device)

            if is_tri_stream(cfg.model_variant):
                logits = model(spatial_batch, freq_batch, noise_batch).view(-1)
            else:
                logits = model(img_batch).view(-1)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if len(all_labels) == 0:
        return {
            "acc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": "",
            "n": 0,
        }

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, preds)
    report = classification_report(all_labels, preds)

    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n": int(len(all_labels)),
    }


def train_one_run(cfg: TrainConfig):
    train_loader = make_loader(cfg.train_root, cfg.img_size, cfg.batch_size, cfg.augment, True, cfg.num_workers)
    val_loader = make_loader(cfg.val_root, cfg.img_size, cfg.batch_size, cfg.augment, False, cfg.num_workers)
    test_loader = make_loader(cfg.test_root, cfg.img_size, cfg.batch_size, cfg.augment, False, cfg.num_workers)

    model = build_model(cfg.model_variant).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=True)

        for (spatial_batch, freq_batch, noise_batch), labels in pbar:
            spatial_batch = spatial_batch.to(device)
            freq_batch = freq_batch.to(device)
            noise_batch = noise_batch.to(device)
            labels = labels.view(-1).float().to(device)

            optimizer.zero_grad()
            logits = model(spatial_batch, freq_batch, noise_batch).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=running_loss / max(1, total),
                acc=correct / max(1, total),
            )

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_metrics = evaluate_on_loader(model, val_loader, desc="Validation")
        print(
            f"\nEpoch {epoch+1} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"| val_acc={val_metrics['acc']*100:.2f}% val_f1={val_metrics['f1']:.4f} "
            f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_auc": best_val_auc,
                },
                cfg.save_path,
            )
            print(f"Saved best checkpoint to {cfg.save_path} (val_auc={best_val_auc:.4f})")

    print("\nFinal evaluation on test set using last epoch model:")
    test_metrics = evaluate_on_loader(model, test_loader, desc="Test")
    print(
        f"test_acc={test_metrics['acc']*100:.2f}% test_f1={test_metrics['f1']:.4f} "
        f"test_prec={test_metrics['precision']:.4f} test_rec={test_metrics['recall']:.4f} "
        f"test_auc={test_metrics['auc']:.4f}"
    )
    print("\nTest classification report:")
    print(test_metrics["classification_report"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_variant", type=str, default="tri_afn_b0", choices=["tri_afn_b0", "tri_afn_b4"])
    p.add_argument("--train_root", type=str, required=True)
    p.add_argument("--val_root", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_path", type=str, default="best_model.pt")
    return p.parse_args()

def is_tri_stream(model_variant: str) -> bool:
    return model_variant.startswith("tri_afn")

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_variant=args.model_variant,
        train_root=args.train_root,
        val_root=args.val_root,
        test_root=args.test_root,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=args.augment,
        num_workers=args.num_workers,
        save_path=args.save_path,
    )
    print("Using device:", device)
    train_one_run(cfg)
