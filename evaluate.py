import torch
from utils import compute_auc

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.cpu().numpy())
            labels.extend(y.numpy())

    return compute_auc(labels, preds)

