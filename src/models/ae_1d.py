"""A single-channel one-dimensional convolutional autoencoder.

The network is intentionally applied independently to every DAS channel.  This
provides a strong, simple baseline for local waveform anomalies and lets the
residual be interpreted directly at channel level.  Spatial correlation is
modelled by the spatio-temporal and spatial-masked networks instead.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .base_model import BaseTorchModel, require_torch


class _ChannelAutoencoder(nn.Module):
    """Conv1d encoder/decoder operating on one channel at a time."""

    def __init__(
        self,
        base_channels: int = 16,
        latent_channels: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        require_torch()
        padding = kernel_size // 2
        hidden = max(int(base_channels), 1)
        latent = max(int(latent_channels), 1)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size, stride=2, padding=padding),
            nn.GroupNorm(1, hidden * 2),
            nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size, stride=2, padding=padding),
            nn.GroupNorm(1, hidden * 4),
            nn.GELU(),
            nn.Conv1d(hidden * 4, latent, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, hidden * 4, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden * 4),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden * 4, hidden * 2, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden * 2, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size, padding=padding),
        )

    def forward(self, x):
        """``x``: ``(batch, 1, samples)`` -> reconstruction of the same shape."""

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        if reconstruction.shape[-1] != x.shape[-1]:
            # Strided convolutions round odd lengths up; interpolate back to the
            # exact acquisition length so residuals align sample-by-sample.
            reconstruction = nn.functional.interpolate(
                reconstruction, size=x.shape[-1], mode="linear", align_corners=False
            )
        return reconstruction


class AE1D(BaseTorchModel):
    """Channel-wise 1D autoencoder anomaly detector.

    Parameters
    ----------
    base_channels:
        Width of the first convolutional block.
    latent_channels:
        Width of the bottleneck representation.
    kernel_size:
        Odd temporal convolution kernel.
    dropout:
        Dropout applied at the bottleneck.
    epochs, batch_size, lr, ...:
        Training hyperparameters inherited from :class:`BaseTorchModel`.
    """

    name = "ae_1d"

    def __init__(
        self,
        base_channels: int = 16,
        latent_channels: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "auto",
        num_workers: int = 0,
        early_stopping_patience: int = 10,
        grad_clip_norm: float = 1.0,
        amp: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            num_workers=num_workers,
            early_stopping_patience=early_stopping_patience,
            grad_clip_norm=grad_clip_norm,
            amp=amp,
            seed=seed,
        )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve alignment")
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self._params.update(
            {
                "base_channels": self.base_channels,
                "latent_channels": self.latent_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
            }
        )

    def _create_network(self, n_channels: int, n_samples: int) -> nn.Module:
        """Create the shared per-channel network."""

        return _ChannelAutoencoder(
            base_channels=self.base_channels,
            latent_channels=self.latent_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

    def _forward(self, batch):
        """Treat every channel as an independent univariate time series."""

        if self.net is None:
            raise RuntimeError("Network has not been built")
        n_samples, n_channels, length = batch.shape
        folded = batch.reshape(n_samples * n_channels, 1, length)
        reconstruction = self.net(folded)
        return reconstruction.reshape(n_samples, n_channels, length)
