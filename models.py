import torch
import torch.nn as nn
import timm


class TriAFN(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        feature_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.encoder_spatial = timm.create_model(
            model_name, pretrained=True, in_chans=3, num_classes=0
        )
        self.encoder_frequency = timm.create_model(
            model_name, pretrained=False, in_chans=1, num_classes=0
        )
        self.encoder_noise = timm.create_model(
            model_name, pretrained=False, in_chans=1, num_classes=0
        )

        encoder_feature_size = self.encoder_spatial.num_features

        self.spatial_proj = nn.Linear(encoder_feature_size, feature_dim)
        self.freq_proj = nn.Linear(encoder_feature_size, feature_dim)
        self.noise_proj = nn.Linear(encoder_feature_size, feature_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=transformer_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_fusion = nn.TransformerEncoder(
            transformer_layer, num_layers=transformer_layers
        )

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        spatial_img: torch.Tensor,
        freq_img: torch.Tensor,
        noise_img: torch.Tensor,
    ) -> torch.Tensor:
        v_spatial = self.encoder_spatial(spatial_img)
        v_frequency = self.encoder_frequency(freq_img)
        v_noise = self.encoder_noise(noise_img)

        v_spatial = self.spatial_proj(v_spatial)
        v_frequency = self.freq_proj(v_frequency)
        v_noise = self.noise_proj(v_noise)

        token_sequence = torch.stack([v_spatial, v_frequency, v_noise], dim=1)
        fused_sequence = self.transformer_fusion(token_sequence)
        fused_vector = fused_sequence.mean(dim=1)

        logits = self.classifier_head(fused_vector)
        return logits

class SimpleXceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

        self.depthwise1 = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pointwise1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.depthwise2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_ch,
            bias=False,
        )
        self.pointwise2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(x)
        out = self.depthwise1(out)
        out = self.pointwise1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.bn2(out)

        out = out + residual
        return out


class XceptionNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        self.block1 = SimpleXceptionBlock(64, 128, stride=2)
        self.block2 = SimpleXceptionBlock(128, 256, stride=2)
        self.block3 = SimpleXceptionBlock(256, 512, stride=2)

        self.tail = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.tail(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1)
