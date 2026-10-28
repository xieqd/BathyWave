"""Haar fusion, the BathyWave localizer, and masked waveform reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import WAVEFORM_LENGTH, ModelConfig


class HaarFusion1D(nn.Module):
    """Fuse local convolution with interpolated [A2, D1, D2] subbands.

    Input/output shapes are (batch, channels, time). No inverse transform is used.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 7, dilation: int = 1
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.local_branch = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False
        )
        subband_channels = 3 * in_channels
        self.wavelet_branch = nn.Sequential(
            nn.Conv1d(
                subband_channels,
                subband_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=subband_channels,
                bias=False,
            ),
            nn.BatchNorm1d(subband_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(subband_channels, out_channels, 1, bias=False),
        )
        self.fusion_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def haar_decompose(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] % 2:
            features = F.pad(features, (0, 1), mode="replicate")
        even, odd = features[..., 0::2], features[..., 1::2]
        scale = 2.0**-0.5
        return (even + odd) * scale, (even - odd) * scale

    def subband_features(self, features: torch.Tensor) -> torch.Tensor:
        approximation_1, detail_1 = self.haar_decompose(features)
        approximation_2, detail_2 = self.haar_decompose(approximation_1)
        return torch.cat(
            [
                F.interpolate(band, size=features.shape[-1], mode="linear", align_corners=False)
                for band in (approximation_2, detail_1, detail_2)
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        wavelet_features = self.wavelet_branch(self.subband_features(features))
        return self.activation(self.fusion_norm(self.local_branch(features) + wavelet_features))


class WaveletResidualBlock1D(nn.Module):
    """Haar fusion, Conv1D-BN, residual addition, then ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.haar_fusion = HaarFusion1D(in_channels, out_channels, kernel_size, dilation)
        self.refine = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.refine(self.haar_fusion(features))
        return self.dropout(self.activation(residual + self.shortcut(features)))


class WaveletConvEncoder(nn.Module):
    """WCE: a stem and four residual stages, preserving temporal length."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        config = self.config
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.in_channels,
                config.base_channels,
                config.kernel_size,
                padding=config.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(config.base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage_channels = tuple(config.base_channels * factor for factor in (1, 2, 4, 4))
        self.stage_dilations = (1, 2, 4, 8)
        stages = []
        current_channels = config.base_channels
        for channels, dilation in zip(self.stage_channels, self.stage_dilations):
            blocks = []
            for _ in range(config.blocks_per_stage):
                blocks.append(
                    WaveletResidualBlock1D(
                        current_channels, channels, config.kernel_size, dilation, config.dropout
                    )
                )
                current_channels = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.output_channels = current_channels

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Map (N, 1, T) waveforms to (N, output_channels, T) features."""
        return self.stages(self.stem(waveforms))


class LocalizationHead(nn.Sequential):
    """Pointwise convolutions producing surface and bottom logits."""

    def __init__(self, feature_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__(
            nn.Conv1d(feature_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )


class BathyWave(nn.Module):
    """Map (N, 1, T) normalized waveforms to (N, 2, T) localization logits.

    Defaults reproduce the 400-sample architecture. Channels are ordered as
    surface, bottom. The model does not apply sigmoid or position decoding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 32,
        kernel_size: int = 7,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = ModelConfig(
            in_channels, out_channels, base_channels, kernel_size, blocks_per_stage, dropout
        )
        self.encoder = WaveletConvEncoder(self.config)
        self.localization_head = LocalizationHead(
            self.encoder.output_channels, base_channels, out_channels
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.localization_head(self.encoder(waveforms))


class ReconstructionDecoder(nn.Sequential):
    def __init__(self, feature_channels: int, hidden_channels: int = 64):
        super().__init__(
            nn.Conv1d(feature_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, 1),
        )


class MaskedWaveformAutoencoder(nn.Module):
    """Reconstruct (N, 1, T) inputs from a Boolean mask of the same shape."""

    def __init__(self, encoder: WaveletConvEncoder):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1))
        self.decoder = ReconstructionDecoder(encoder.output_channels)

    def forward(self, waveforms: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_waveforms = torch.where(mask, self.mask_token.to(dtype=waveforms.dtype), waveforms)
        return self.decoder(self.encoder(masked_waveforms))


def patch_mask(
    batch_size: int,
    length: int = WAVEFORM_LENGTH,
    patch_length: int = 16,
    mask_ratio: float = 0.5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return a (N, 1, T) mask with True at reconstruction-loss positions."""
    patch_count = length // patch_length
    # Round-to-even gives 12 of 25 patches (48%) for the default configuration.
    masked_count = max(1, min(patch_count, round(patch_count * mask_ratio)))
    order = torch.rand(batch_size, patch_count, device=device).argsort(dim=1)
    mask = torch.zeros(batch_size, patch_count, dtype=torch.bool, device=device)
    mask.scatter_(1, order[:, :masked_count], True)
    return mask.repeat_interleave(patch_length, dim=1)[:, None, :]
