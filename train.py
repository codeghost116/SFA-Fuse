import torch
from torch.utils.data import DataLoader
from models import TriAFN
from utils import set_seed

def train(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.squeeze(), y.float())
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TriAFN().to(device)
