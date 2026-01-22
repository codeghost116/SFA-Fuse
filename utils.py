import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

