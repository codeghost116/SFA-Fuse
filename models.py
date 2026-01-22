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
