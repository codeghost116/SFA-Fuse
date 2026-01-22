import os
import json
import random
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: torch.nn.Module, cfg: Dict[str, Any], best_val: float):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "best_val": float(best_val),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
