import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TriAFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BaselineCNN()

    def forward(self, rgb, fft, noise):
        rgb_out = self.backbone(rgb)
        fft_out = self.backbone(fft)
        noise_out = self.backbone(noise)
        return (rgb_out + fft_out + noise_out) / 3
