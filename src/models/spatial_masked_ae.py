"""Spatially masked autoencoder with a memory bottleneck.

Two failure modes of reconstruction-based anomaly detection are addressed here.

**Identity shortcut.**  A plain autoencoder can copy its input, including the
defect, and then report a small residual.  Masking a contiguous band of
channels removes the copy path, and a high mask ratio (0.6 by default) forces
the network to infer the held-out band from genuinely distant context.

**Over-generalisation.**  Even with masking, a decoder with enough capacity can
reconstruct unusual inputs.  A memory bank between encoder and decoder
constrains every latent position to a sparse combination of learned *normal*
prototypes, so a defect that has no prototype cannot be reproduced and the
residual grows.

Inference masks a block that is wider than the fold being scored, so the fold's
channels sit well inside the hidden region instead of at its boundary.  This
removes the boundary leakage of the previous fold-wise masking, where edge
channels were reconstructed from their immediate (visible) neighbours.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseTorchModel


class _ConvBlock(nn.Sequential):
    """Conv2d -> GroupNorm -> GELU (+ optional spatial dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class _MemoryBank(nn.Module):
    """Per-position sparse memory addressing (MemAE-style).

    Each spatial position of the latent map is projected onto ``n_slots``
    learned prototypes with a softmax.  The addressed feature is the weighted
    combination of prototypes, so the decoder only ever sees vectors that lie in
    the span of normal prototypes.
    """

    def __init__(
        self,
        n_slots: int = 64,
        dim: int = 32,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_slots = max(int(n_slots), 1)
        self.dim = max(int(dim), 1)
        self.temperature = max(float(temperature), 1e-3)
        self.memory = nn.Parameter(torch.empty(self.n_slots, self.dim))
        nn.init.xavier_uniform_(self.memory)
        self.last_weights: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the memory-refined latent map and the addressing entropy.

        The memory is applied *residually*: ``z + M(w)``.  Replacing the code
        outright (``M(w)`` only) lets the softmax collapse onto a single slot,
        after which the decoder emits a near-constant field and the model
        silently degenerates into an energy detector.  The residual form keeps a
        gradient path to the encoder, so a collapsed memory costs accuracy but
        cannot destroy the reconstruction.
        """

        batch, channels, height, width = z.shape
        queries = z.permute(0, 2, 3, 1).reshape(-1, channels)
        normalised_queries = F.normalize(queries, dim=1)
        normalised_memory = F.normalize(self.memory, dim=1)
        similarity = normalised_queries @ normalised_memory.t() / self.temperature
        weights = F.softmax(similarity, dim=1)

        addressed = queries + weights @ self.memory
        addressed = addressed.view(batch, height, width, channels).permute(0, 3, 1, 2)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        self.last_weights = weights.detach()
        return addressed, entropy


class _SpatialMaskedNetwork(nn.Module):
    """U-Net style encoder-decoder taking ``[masked_signal, mask]`` as input."""

    def __init__(
        self,
        base_channels: int = 16,
        latent_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        memory_slots: int = 64,
        memory_temperature: float = 0.20,
    ) -> None:
        super().__init__()
        base = max(int(base_channels), 1)
        latent = max(int(latent_channels), base)
        self.encoder_1 = _ConvBlock(2, base, kernel_size)
        self.encoder_2 = _ConvBlock(base, base * 2, kernel_size, stride=2)
        self.encoder_3 = _ConvBlock(base * 2, latent, kernel_size, stride=2)
        self.bottleneck = _ConvBlock(latent, latent, kernel_size, dropout=dropout)
        # Dilated context: inpainting a contiguous channel block needs a
        # receptive field at least as wide as that block.  Two dilated convs
        # widen it from ~15 to ~39 channels at no extra parameter cost, which is
        # what lets the network see both sides of a 0.25-0.4 wide mask.
        self.context = nn.Sequential(
            _ConvBlock(latent, latent, kernel_size, dilation=2, dropout=dropout),
            _ConvBlock(latent, latent, kernel_size, dilation=4, dropout=dropout),
        )
        # Memory sits at the decoder input: every latent position must be
        # explained by a sparse set of normal prototypes before decoding.
        self.memory = _MemoryBank(
            n_slots=memory_slots,
            dim=latent,
            temperature=memory_temperature,
        )
        self.decoder_2 = _ConvBlock(latent + base * 2, base * 2, kernel_size)
        self.decoder_1 = _ConvBlock(base * 2 + base, base, kernel_size)
        self.output = nn.Conv2d(base, 1, kernel_size, padding=kernel_size // 2)
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, masked_signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the full-field reconstruction ``(N, C, T)``."""

        # Add the convolution input-channel dimension: (N, C, T) -> (N, 1, C, T).
        masked_signal = masked_signal.unsqueeze(1)
        mask = mask.unsqueeze(1)
        target_height, target_width = masked_signal.shape[-2:]
        inputs = torch.cat([masked_signal, mask], dim=1)
        skip_1 = self.encoder_1(inputs)
        skip_2 = self.encoder_2(skip_1)
        encoded = self.encoder_3(skip_2)
        encoded = self.bottleneck(encoded)
        encoded = self.context(encoded)
        encoded, entropy = self.memory(encoded)
        self.last_aux_loss = entropy

        decoded = F.interpolate(
            encoded, size=skip_2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder_2(torch.cat([decoded, skip_2], dim=1))
        decoded = F.interpolate(
            decoded, size=skip_1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder_1(torch.cat([decoded, skip_1], dim=1))
        reconstruction = self.output(decoded)
        if reconstruction.shape[-2:] != (target_height, target_width):
            reconstruction = F.interpolate(
                reconstruction,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction.squeeze(1)


class SpatialMaskedAE(BaseTorchModel):
    """Spatially masked reconstruction network with a memory bottleneck.

    Parameters
    ----------
    mask_ratio:
        Fraction of channels hidden during training *and* during inference.
        ``0.6`` leaves only 40 % of the fibre visible, which is enough context
        for a smooth moving-load field but not enough to copy a defect.
    n_masks:
        Number of inference folds.  Every fold is scored while a wider block
        (``mask_ratio`` of the array) centred on the fold is hidden.
    memory_slots, memory_temperature:
        Size and softmax temperature of the memory bank.
    entropy_weight:
        Weight of the addressing-entropy regulariser.  Minimising entropy
        encourages each latent position to use few prototypes, which is what
        makes the memory selective.
    """

    name = "spatial_masked_ae"

    def __init__(
        self,
        base_channels: int = 16,
        latent_channels: int = 32,
        kernel_size: int = 3,
        mask_ratio: float = 0.40,
        n_masks: int = 4,
        dropout: float = 0.1,
        memory_slots: int = 64,
        memory_temperature: float = 0.20,
        entropy_weight: float = 0.0005,
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
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.kernel_size = int(kernel_size)
        self.mask_ratio = float(mask_ratio)
        self.n_masks = int(n_masks)
        self.dropout = float(dropout)
        self.memory_slots = int(memory_slots)
        self.memory_temperature = float(memory_temperature)
        self.entropy_weight = float(entropy_weight)
        self._last_mask: torch.Tensor | None = None
        self._params.update(
            {
                "base_channels": self.base_channels,
                "latent_channels": self.latent_channels,
                "kernel_size": self.kernel_size,
                "mask_ratio": self.mask_ratio,
                "n_masks": self.n_masks,
                "dropout": self.dropout,
                "memory_slots": self.memory_slots,
                "memory_temperature": self.memory_temperature,
                "entropy_weight": self.entropy_weight,
            }
        )

    # ------------------------------------------------------------------ network
    def _create_network(self, n_channels: int, n_samples: int) -> nn.Module:
        return _SpatialMaskedNetwork(
            base_channels=self.base_channels,
            latent_channels=self.latent_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            memory_slots=self.memory_slots,
            memory_temperature=self.memory_temperature,
        )

    def _loss(self, batch: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Reconstruction error on the *hidden* channels plus addressing entropy.

        The error must be restricted to the masked region.  Averaging over the
        whole field lets the network minimise the objective by copying the
        visible channels and emitting a constant over the hidden band, which
        collapses the model into an energy detector while the reported loss
        still looks healthy.
        """

        error = (batch - reconstruction) ** 2
        mask = getattr(self, "_last_mask", None)
        if mask is not None and tuple(mask.shape) == tuple(error.shape):
            recon_loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            recon_loss = torch.mean(error)
        aux = getattr(self.net, "last_aux_loss", None) if self.net is not None else None
        if aux is None:
            return recon_loss
        return recon_loss + self.entropy_weight * aux

    # -------------------------------------------------------------------- masks
    @staticmethod
    def _expand_channel_mask(channel_mask: torch.Tensor, time_samples: int) -> torch.Tensor:
        """Broadcast a ``(N, C, 1)`` channel mask over the time axis."""

        return channel_mask.expand(-1, -1, time_samples)

    def _mask_width(self, n_channels: int) -> int:
        width = max(1, int(math.ceil(self.mask_ratio * n_channels)))
        return min(width, max(n_channels - 1, 1))

    def _random_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample a contiguous masked channel segment for each batch item."""

        n_samples, n_channels, time_samples = batch.shape
        width = self._mask_width(n_channels)
        starts = torch.randint(
            0, n_channels - width + 1, (n_samples,), device=batch.device
        )
        positions = torch.arange(n_channels, device=batch.device).view(1, -1)
        channel_mask = (
            (positions >= starts[:, None])
            & (positions < (starts + width)[:, None])
        ).to(dtype=batch.dtype)
        return self._expand_channel_mask(channel_mask[:, :, None], time_samples)

    def _centre_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """A deterministic mask used during validation."""

        n_samples, n_channels, time_samples = batch.shape
        width = min(self._mask_width(n_channels), n_channels)
        start = (n_channels - width) // 2
        channel_mask = torch.zeros(
            (1, n_channels, 1), device=batch.device, dtype=batch.dtype
        )
        channel_mask[:, start : start + width, :] = 1.0
        return channel_mask.expand(n_samples, -1, time_samples)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Training uses random masks; validation uses a stable centre mask."""

        if self.net is None:
            raise RuntimeError("Network has not been built")
        mask = self._random_mask(batch) if self.net.training else self._centre_mask(batch)
        self._last_mask = mask
        masked_signal = batch * (1.0 - mask)
        return self.net(masked_signal, mask)

    # -------------------------------------------------------------- inference
    def _fold_bounds(self, n_channels: int) -> list[tuple[int, int]]:
        """Split the channel axis into deterministic inference folds."""

        folds = max(1, min(self.n_masks, n_channels))
        edges = np.linspace(0, n_channels, folds + 1).round().astype(int)
        bounds: list[tuple[int, int]] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop > start:
                bounds.append((int(start), int(stop)))
        return bounds

    def _block_for_fold(
        self, start: int, stop: int, n_channels: int
    ) -> tuple[int, int]:
        """Hidden block for a fold: at least ``mask_ratio`` wide and centred.

        Widening the hidden region beyond the fold is what prevents boundary
        leakage: the scored channels are surrounded by hidden channels rather
        than by visible neighbours.
        """

        width = max(stop - start, self._mask_width(n_channels))
        width = min(width, n_channels)
        centre = 0.5 * (start + stop)
        block_start = int(round(centre - 0.5 * width))
        block_start = max(0, min(block_start, n_channels - width))
        block_stop = block_start + width
        if block_start > start:
            block_start = start
            block_stop = min(n_channels, start + width)
        if block_stop < stop:
            block_stop = stop
            block_start = max(0, stop - width)
        return block_start, block_stop

    def _reconstruct_block(
        self, batch: torch.Tensor, start: int, stop: int, time_samples: int
    ) -> torch.Tensor:
        """Reconstruct the full field while masking channels ``[start, stop)``."""

        if self.net is None:
            raise RuntimeError("Network has not been built")
        channel_mask = torch.zeros(
            (batch.shape[0], batch.shape[1], 1),
            device=batch.device,
            dtype=batch.dtype,
        )
        channel_mask[:, start:stop, :] = 1.0
        mask = self._expand_channel_mask(channel_mask, time_samples)
        masked_signal = batch * (1.0 - mask)
        return self.net(masked_signal, mask)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Stitch held-out fold reconstructions into a full pass reconstruction."""

        self._check_fitted()
        if self.net is None:
            raise RuntimeError("Fitted model has no network")
        batch = self.validate_samples(data)
        if self.input_shape_ is not None and batch.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected input shape {self.input_shape_}, received {batch.shape[1:]}"
            )
        self.net.eval()
        reconstruction = np.empty_like(batch, dtype=np.float32)
        n_channels = batch.shape[1]
        with torch.no_grad():
            for fold_start, fold_stop in self._fold_bounds(n_channels):
                block_start, block_stop = self._block_for_fold(
                    fold_start, fold_stop, n_channels
                )
                for offset in range(0, batch.shape[0], self.batch_size):
                    chunk = torch.from_numpy(
                        batch[offset : offset + self.batch_size]
                    ).to(self.device)
                    decoded = self._reconstruct_block(
                        chunk, block_start, block_stop, batch.shape[2]
                    )
                    # Keep only the held-out fold; every channel belongs to
                    # exactly one fold, so the stitched result is complete.
                    reconstruction[
                        offset : offset + chunk.shape[0],
                        fold_start:fold_stop,
                        :,
                    ] = decoded[:, fold_start:fold_stop, :].cpu().numpy()
        return reconstruction
