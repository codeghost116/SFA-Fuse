import argparse
import numpy as np
import torch
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

from datasets import (
    FFPPFrameDataset,
    CelebDFImageDataset,
    RealFakeImageDataset,
    tri_collate_fn,
    single_collate_fn,
)
from models import TriAFN, XceptionNet, BaselineCNN

import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_loader(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for (spatial_batch, freq_batch, noise_batch), labels in tqdm(loader, desc="Eval", leave=False):
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
        return {}

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(all_labels, preds)
    report = classification_report(all_labels, preds)

    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n": int(len(all_labels)),
    }


def build_loader_from_path(path: str, dataset_type: str, batch_size: int, num_workers: int):
    if dataset_type == "ffpp":
        ds = FFPPFrameDataset(path, augment=False)
    elif dataset_type == "celebdf":
        ds = CelebDFImageDataset(path, augment=False)
    else:
        raise ValueError("dataset_type must be 'ffpp' or 'celebdf'")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=tri_collate_fn)
    return loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--dataset_type", type=str, default="ffpp", choices=["ffpp", "celebdf"])
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def is_tri_stream(model_variant: str) -> bool:
    return model_variant.startswith("tri_afn")

if __name__ == "__main__":
    args = parse_args()

    utils.set_seed(42)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    model_variant = cfg.get("model_variant", "tri_afn_b0")

    if model_variant == "tri_afn_b0":
        model = TriAFN(model_name="efficientnet_b0", feature_dim=256)
    elif model_variant == "tri_afn_b4":
        model = TriAFN(model_name="tf_efficientnet_b4_ns", feature_dim=512)
    else:
        model = TriAFN()

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    loader = build_loader_from_path(args.dataset_root, args.dataset_type, args.batch_size, args.num_workers)
    metrics = eval_loader(model, loader)

    print("Evaluation results")
    print(metrics["n"], "samples")
    print(f"ACC {metrics['acc']:.4f} F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f}")
    print(metrics["classification_report"])
