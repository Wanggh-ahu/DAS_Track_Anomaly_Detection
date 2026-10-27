"""Spatio-temporal autoencoder with a convolutional LSTM bottleneck.

The input pass is divided into ``seq_len`` consecutive time chunks.  A shared
2-D convolutional encoder compresses each ``(channels, chunk_samples)`` frame,
a ConvLSTM models how the spatial latent map evolves between chunks, and a
shared decoder reconstructs every frame.  Anomalies produce large residuals
either in space (localised track defects) or in time (transient impacts).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_model import BaseTorchModel


class ConvLSTMCell(nn.Module):
    """A minimal convolutional LSTM cell.

    All gates use a single 2-D convolution over the concatenation of the input
    map and the previous hidden map.  Shapes are ``(batch, channels, height,
    width)`` throughout.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvLSTM kernel_size must be odd")
        self.hidden_channels = int(hidden_channels)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            int(input_channels) + self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        if state is None:
            hidden = x.new_zeros((batch, self.hidden_channels, height, width))
            cell = x.new_zeros((batch, self.hidden_channels, height, width))
        else:
            hidden, cell = state
        gates = self.conv(torch.cat([x, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gates, 4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class _STAEConvLSTMNetwork(nn.Module):
    """Frame encoder + ConvLSTM + frame decoder."""

    def __init__(
        self,
        seq_len: int = 8,
        hidden_channels: int = 16,
        latent_channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = max(int(seq_len), 1)
        self.hidden_channels = max(int(hidden_channels), 1)
        self.latent_channels = max(int(latent_channels), 1)
        padding = kernel_size // 2
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, self.hidden_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels,
                self.latent_channels,
                kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.GroupNorm(1, self.latent_channels),
            nn.GELU(),
            nn.Conv2d(
                self.latent_channels,
                self.latent_channels,
                kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.GroupNorm(1, self.latent_channels),
            nn.GELU(),
        )
        self.conv_lstm = ConvLSTMCell(
            input_channels=self.latent_channels,
            hidden_channels=self.latent_channels,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.frame_decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.hidden_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, 1, channels, samples)`` -> same shape."""

        batch, _, n_channels, n_samples = x.shape
        seq_len = min(self.seq_len, max(n_samples, 1))
        chunk = (n_samples + seq_len - 1) // seq_len
        padded_samples = seq_len * chunk
        if padded_samples != n_samples:
            x = nn.functional.pad(x, (0, padded_samples - n_samples))

        # Split the time axis into consecutive chunks and fold them into the
        # batch dimension for the shared 2-D frame encoder.
        frames = (
            x.reshape(batch, 1, n_channels, seq_len, chunk)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch * seq_len, 1, n_channels, chunk)
        )
        encoded = self.frame_encoder(frames)
        _, latent_channels, height, width = encoded.shape
        sequence = encoded.reshape(batch, seq_len, latent_channels, height, width)

        hidden: torch.Tensor | None = None
        cell: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            state = None if hidden is None or cell is None else (hidden, cell)
            hidden, cell = self.conv_lstm(sequence[:, step], state)
            outputs.append(hidden)
        memory = torch.stack(outputs, dim=1)  # (B, seq_len, latent, H, W)
        memory = self.dropout(
            memory.reshape(batch * seq_len, latent_channels, height, width)
        )
        decoded = self.frame_decoder(memory)
        decoded = nn.functional.interpolate(
            decoded,
            size=(n_channels, chunk),
            mode="bilinear",
            align_corners=False,
        )
        reconstruction = (
            decoded.reshape(batch, seq_len, 1, n_channels, chunk)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, 1, n_channels, padded_samples)
        )
        return reconstruction[:, :, :, :n_samples]


class STAEConvLSTM(BaseTorchModel):
    """Convolutional LSTM spatio-temporal autoencoder.

    Parameters
    ----------
    seq_len:
        Number of temporal chunks processed by the ConvLSTM.  A pass with
        ``T`` samples is padded to a multiple of ``seq_len`` internally.
    hidden_channels, latent_channels:
        Widths of the frame encoder and latent sequence.
    kernel_size:
        Odd convolution kernel used throughout.
    dropout:
        Spatial dropout applied to the ConvLSTM output sequence.
    """

    name = "stae_convlstm"

    def __init__(
        self,
        seq_len: int = 8,
        hidden_channels: int = 16,
        latent_channels: int = 16,
        kernel_size: int = 3,
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
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.seq_len = int(seq_len)
        self.hidden_channels = int(hidden_channels)
        self.latent_channels = int(latent_channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self._params.update(
            {
                "seq_len": self.seq_len,
                "hidden_channels": self.hidden_channels,
                "latent_channels": self.latent_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
            }
        )

    def _create_network(self, n_channels: int, n_samples: int) -> nn.Module:
        return _STAEConvLSTMNetwork(
            seq_len=self.seq_len,
            hidden_channels=self.hidden_channels,
            latent_channels=self.latent_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

    def _forward(self, batch):
        """Add the single input-channel dimension expected by the 2-D encoder."""

        if self.net is None:
            raise RuntimeError("Network has not been built")
        reconstruction = self.net(batch.unsqueeze(1))
        return reconstruction.squeeze(1)
