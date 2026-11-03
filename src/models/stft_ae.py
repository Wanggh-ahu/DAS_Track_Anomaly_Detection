"""Log-magnitude spectrogram autoencoder with an explicit frequency axis.

Why this model exists
---------------------
The benchmark showed that a fixed 80-160 Hz band-pass detector matches or beats
every waveform-domain model, and that a bank of filters cannot recover that
performance because combining bands blindly costs more than the adaptivity
buys.  The missing capability is therefore not "more capacity" but **choosing
where in frequency to look**.

A waveform bottleneck (`ae_1d`) compresses each 128-sample channel into a
vector, which destroys the correspondence between latent units and frequency
bins.  Here the reconstruction error is computed on a time-frequency map, so the
residual keeps an explicit frequency axis: "which band is unexplained" is part
of the output, and the anomaly score can be read at the frequency of the
defect instead of being diluted over the whole band.

The window is scored per channel exactly like :class:`AE1D`; the only change is
the representation, which keeps the comparison to the waveform baseline clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn

from .base_model import BaseTorchModel, require_torch


class _SpectrogramAutoencoder(nn.Module):
    """Small 2-D convolutional autoencoder over a log-magnitude spectrogram."""

    def __init__(
        self,
        n_freq: int,
        n_frames: int,
        n_channels: int = 1,
        base_channels: int = 16,
        latent_channels: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        window: Any = None,
        mu: Any = None,
        sigma: Any = None,
        magnitude_scale: float = 1.0,
        n_fft: int = 64,
        hop_length: int = 8,
    ) -> None:
        super().__init__()
        require_torch()
        import torch

        padding = kernel_size // 2
        hidden = max(int(base_channels), 1)
        latent = max(int(latent_channels), 1)
        self.n_freq = int(n_freq)
        self.n_frames = int(n_frames)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

        self.register_buffer(
            "window", torch.hann_window(self.n_fft) if window is None else window
        )
        self.register_buffer(
            "magnitude_scale",
            torch.tensor(max(float(magnitude_scale), 1e-6), dtype=torch.float32),
        )
        self.register_buffer("mu", torch.zeros(1, 1, self.n_freq, 1) if mu is None else mu)
        self.register_buffer(
            "sigma", torch.ones(1, 1, self.n_freq, 1) if sigma is None else sigma
        )
        # Per-(channel, frequency) statistics of the *residual* on clean data,
        # filled after training by ``calibrate_normal``.  Until then
        # ``residual_ready`` is 0 and scoring falls back to the raw residual.
        self.register_buffer(
            "residual_mu", torch.zeros(int(n_channels), self.n_freq)
        )
        self.register_buffer(
            "residual_sigma", torch.ones(int(n_channels), self.n_freq)
        )
        self.register_buffer("residual_ready", torch.zeros(()))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size, stride=2, padding=padding),
            nn.GroupNorm(1, hidden * 2),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size, stride=2, padding=padding),
            nn.GroupNorm(1, hidden * 4),
            nn.GELU(),
            nn.Conv2d(hidden * 4, latent, kernel_size, stride=2, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent, hidden * 4, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden * 4),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden * 4, hidden * 2, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden * 2, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, 1, kernel_size, padding=padding),
        )

    def forward(self, spectrogram):
        """``spectrogram``: normalised log-magnitude ``(N, 1, F, T)`` -> same shape."""

        import torch.nn.functional as functional

        latent = self.encoder(spectrogram)
        reconstruction = self.decoder(latent)
        if reconstruction.shape[-2:] != spectrogram.shape[-2:]:
            reconstruction = functional.interpolate(
                reconstruction, size=spectrogram.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        return reconstruction


class STFTLogMagnitudeAE(BaseTorchModel):
    """Anomaly detection on the log-magnitude spectrogram of each channel.

    Parameters
    ----------
    n_fft, hop_length:
        STFT geometry.  The defaults give 33 frequency bins (15.6 Hz resolution)
        and 17 frames (8 ms) for the standard 128-sample window, which resolves
        the 65 Hz load line, the 120 Hz resonance and an 18 ms impact.
    base_channels, latent_channels:
        Encoder width and bottleneck width.  The bottleneck is
        ``latent_channels x 5 x 3`` = 120 values for a 33 x 17 input, i.e. a
        compression ratio comparable to ``ae_1d``.
    """

    name = "stft_ae"

    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 8,
        base_channels: int = 16,
        latent_channels: int = 8,
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
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve alignment")
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self._params.update(
            {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "base_channels": self.base_channels,
                "latent_channels": self.latent_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
            }
        )
        self.magnitude_scale_: float = 1.0
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        # The base class only passes (batch, reconstruction) to ``_loss``; the
        # reconstruction is a waveform, so the spectrogram prediction is cached
        # here by ``_forward`` and consumed immediately by ``_loss``.
        self._last_prediction: Any = None

    # ------------------------------------------------------------- transforms
    def _n_frames(self, n_samples: int) -> int:
        return 1 + int(n_samples) // self.hop_length

    def _stft(self, batch):
        """Return ``(log_magnitude, unit_phasor)`` for a ``(N, C, T)`` batch."""

        import torch

        n_samples, n_channels, length = batch.shape
        flat = batch.reshape(n_samples * n_channels, length)
        spectrum = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.net.window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        magnitude = spectrum.abs()
        phasor = spectrum / magnitude.clamp_min(1e-8)
        log_magnitude = torch.log1p(magnitude / self.net.magnitude_scale)
        frames = log_magnitude.shape[-1]
        return (
            log_magnitude.reshape(n_samples, n_channels, -1, frames),
            phasor.reshape(n_samples, n_channels, -1, frames),
        )

    def _normalise(self, log_magnitude):
        return (log_magnitude - self.net.mu) / self.net.sigma

    # ---------------------------------------------------------------- network
    def _create_network(self, n_channels: int, n_samples: int) -> nn.Module:
        import torch

        n_freq = self.n_fft // 2 + 1
        n_frames = self._n_frames(n_samples)
        mu = (
            torch.as_tensor(self.mu_, dtype=torch.float32).reshape(1, 1, n_freq, 1)
            if self.mu_ is not None
            else torch.zeros(1, 1, n_freq, 1)
        )
        sigma = (
            torch.as_tensor(self.sigma_, dtype=torch.float32).reshape(1, 1, n_freq, 1)
            if self.sigma_ is not None
            else torch.ones(1, 1, n_freq, 1)
        )
        return _SpectrogramAutoencoder(
            n_freq=n_freq,
            n_frames=n_frames,
            n_channels=int(n_channels),
            base_channels=self.base_channels,
            latent_channels=self.latent_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            mu=mu,
            sigma=sigma,
            magnitude_scale=self.magnitude_scale_,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def _compute_statistics(self, train: np.ndarray) -> None:
        """Per-frequency normalisation and the magnitude scale.

        Without a per-bin standardisation the loss is dominated by the loud
        low-frequency bins (the 65 Hz load line), and the network spends its
        capacity there instead of on the defect band.
        """

        import torch

        probe = torch.from_numpy(train[: min(len(train), 512)])
        magnitude_scale = max(float(np.median(np.abs(train))) * 4.0, 1e-6)
        self.magnitude_scale_ = magnitude_scale
        net = torch.hann_window(self.n_fft)
        flat = probe.reshape(-1, probe.shape[-1])
        spectrum = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=net,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        log_magnitude = torch.log1p(spectrum.abs() / magnitude_scale)
        mu = log_magnitude.mean(dim=(0, 2)).numpy()
        sigma = log_magnitude.std(dim=(0, 2)).numpy()
        # Bins outside the acquisition band carry almost no signal, so their
        # log-magnitude variance is tiny.  Dividing by that tiny variance
        # amplifies pure numerical noise and makes the residual peak at 400 Hz
        # where nothing happens; floor each bin's scale against the median so an
        # empty bin cannot outvote the informative ones.
        floor = max(float(np.median(sigma)) * 0.25, 1e-3)
        self.mu_ = mu.astype(np.float32)
        self.sigma_ = np.maximum(sigma, floor).astype(np.float32)

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "STFTLogMagnitudeAE":
        train = self.validate_samples(train_data)
        self._compute_statistics(train)
        return super().fit(train, val_data=val_data, **kwargs)

    # -------------------------------------------------------------- forwards
    def _forward(self, batch):
        """Return a waveform reconstruction; cache the spectrogram prediction."""

        import torch

        log_magnitude, phasor = self._stft(batch)
        n_samples, n_channels, n_freq, n_frames = log_magnitude.shape
        normalised = self._normalise(log_magnitude).reshape(
            n_samples * n_channels, 1, n_freq, n_frames
        )
        prediction = self.net(normalised)
        self._last_prediction = prediction.reshape(
            n_samples, n_channels, n_freq, n_frames
        )
        predicted_log = self._last_prediction * self.net.sigma + self.net.mu
        predicted_magnitude = torch.expm1(predicted_log).clamp_min(0.0)
        predicted_magnitude = predicted_magnitude * self.net.magnitude_scale
        spectrum = predicted_magnitude * phasor
        flat = spectrum.reshape(n_samples * n_channels, n_freq, n_frames)
        length = batch.shape[-1]
        waveform = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.net.window,
            center=True,
            length=length,
        )
        return waveform.reshape(n_samples, n_channels, length)

    def _loss(self, batch, reconstruction):
        """Squared error in the normalised log-magnitude domain."""

        import torch

        if self._last_prediction is None:
            raise RuntimeError("_loss must follow _forward on the same batch")
        target, _ = self._stft(batch)
        target = self._normalise(target)
        return torch.mean(torch.square(target - self._last_prediction))

    # ---------------------------------------------------------------- scoring
    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Not used: the anomaly score is built in the spectrogram domain.

        Returning a waveform would force the evaluator to score a
        time-domain residual, which averages the frequency axis away and throws
        out exactly the information this model exists to keep.
        """

        raise NotImplementedError(
            "STFTLogMagnitudeAE scores in the spectrogram domain; use channel_scores"
        )

    def calibrate_normal(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> "STFTLogMagnitudeAE":
        """Fit per-(channel, frequency) residual statistics on clean windows.

        The attribution experiment showed the model can localise a defect in
        frequency but still loses to a five-band filterbank at telling two
        defects apart.  The gap was not resolution — it was *calibration*: the
        filterbank normalises every (channel, band) against its own normal-pass
        distribution, while the autoencoder's residual is not comparable across
        channels or bins.  This method borrows exactly that trick at bin
        resolution and nothing else.

        Must be called on data disjoint from training; ``scripts/train.py``
        passes its calibration split.
        """

        import torch

        self._check_fitted()
        if self.net is None:
            raise RuntimeError("Fitted model has no network")
        _, error = self.spectrogram_error(data)
        sustained = error.mean(axis=-1)  # (N, C, F)
        median = np.median(sustained, axis=0)
        deviation = np.median(np.abs(sustained - median), axis=0)
        scale = 1.4826 * deviation
        fallback = sustained.std(axis=0)
        scale = np.where(scale > 1e-6, scale, fallback)
        scale = np.maximum(scale, 1e-6)
        if median.shape != (self.net.residual_mu.shape[0], self.net.residual_mu.shape[1]):
            raise ValueError(
                f"calibration shape {median.shape} does not match the network "
                f"({tuple(self.net.residual_mu.shape)})"
            )
        with torch.no_grad():
            self.net.residual_mu.copy_(torch.as_tensor(median, dtype=torch.float32))
            self.net.residual_sigma.copy_(torch.as_tensor(scale, dtype=torch.float32))
            self.net.residual_ready.fill_(1.0)
        self.metadata_["residual_calibration"] = {
            "windows": int(data.shape[0]),
            "channels": int(median.shape[0]),
            "bins": int(median.shape[1]),
            "median_scale": float(np.median(scale)),
        }
        return self

    def channel_scores(
        self,
        data: np.ndarray,
        sampling_rate_hz: float = 1000.0,
        frequency_reduction: str = "max",
    ) -> np.ndarray:
        """Per-channel anomaly score from the time-frequency residual.

        The residual is (a) restricted to the acquisition band, (b) smoothed
        across three neighbouring frequency bins, and (c) reduced over
        frequency with ``max``, then averaged over frames.  Steps (a) and (b)
        are not cosmetic: a single bin's residual is noise dominated, and the
        benchmark already showed that taking a bare maximum over noisy bands
        costs more than it gains (see ``FilterbankBaseline``).

        ``frequency_reduction="mean"`` reproduces the waveform-residual
        behaviour and is kept as the ablation.

        The order matters: frames are averaged *before* frequency is reduced.
        A defect is a short event, but reducing over frequency first lets one
        noisy bin of one frame dominate the whole window, which measured worse
        than either reduction taken alone (0.645 versus 0.741 channel AUC).

        When :meth:`calibrate_normal` has run, the sustained residual is
        standardised per (channel, bin) before smoothing, so a quiet bin cannot
        outvote an informative one simply by being quiet.
        """

        target, error = self.spectrogram_error(data)
        frequencies = self.frequency_axis(sampling_rate_hz)
        band = (frequencies >= 5.0) & (frequencies <= 200.0)
        if not band.any():
            band = np.ones_like(frequencies, dtype=bool)
        profile = self.frequency_profile(data, sampling_rate_hz=sampling_rate_hz)
        selected = profile[:, :, band]
        if str(frequency_reduction) == "mean":
            reduced = selected.mean(axis=2)
        else:
            reduced = selected.max(axis=2)
        return reduced.astype(np.float32)

    def frequency_profile(
        self, data: np.ndarray, sampling_rate_hz: float = 1000.0
    ) -> np.ndarray:
        """Sustained, calibrated residual per ``(window, channel, bin)``.

        This is the scoring feature itself, kept with its frequency axis intact
        so the attribution benchmark measures exactly what detection uses.
        Calibration and the 3-bin smoothing are applied here, not in the caller.
        """

        _, error = self.spectrogram_error(data)
        sustained = error.mean(axis=-1)
        if self.net is not None and float(self.net.residual_ready) > 0.0:
            centre = self.net.residual_mu.detach().cpu().numpy()[None]
            scale = self.net.residual_sigma.detach().cpu().numpy()[None]
            sustained = (sustained - centre) / scale
        padded = np.pad(sustained, ((0, 0), (0, 0), (1, 1)), mode="edge")
        return ((padded[:, :, :-2] + padded[:, :, 1:-1] + padded[:, :, 2:]) / 3.0).astype(
            np.float32
        )

    def score(self, data: np.ndarray) -> np.ndarray:
        """Window-level score: mean over channels of the per-channel score."""

        return self.channel_scores(data).mean(axis=1)

    def reconstruction_quality(self, data: np.ndarray) -> tuple[float, float]:
        """Return ``(median correlation, std ratio)`` of the waveform round trip.

        Collapse check for the diagnostics: a model that outputs a constant or a
        blurred copy shows up immediately here.
        """

        import torch

        self._check_fitted()
        batch = self.validate_samples(data)
        if self.net is None:
            raise RuntimeError("Fitted model has no network")
        self.net.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], self.batch_size):
                chunk = torch.from_numpy(batch[start : start + self.batch_size]).to(
                    self.device
                )
                outputs.append(self._forward(chunk).detach().cpu().numpy())
        reconstruction = np.concatenate(outputs, axis=0)
        correlations = [
            np.corrcoef(batch[index].ravel(), reconstruction[index].ravel())[0, 1]
            for index in range(min(len(batch), 200))
        ]
        return float(np.median(correlations)), float(
            reconstruction.std() / max(batch.std(), 1e-8)
        )

    # ------------------------------------------------------------ diagnostics
    def spectrogram_error(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(target, error)`` log-magnitude maps of shape ``(N, C, F, T)``.

        ``error`` is the absolute difference in the *normalised* log-magnitude
        domain, which is what the anomaly score is built from.  Keeping the
        frequency axis is the entire point of this model, so the diagnostic
        tooling reads it directly instead of collapsing to a scalar.
        """

        import torch

        self._check_fitted()
        batch = self.validate_samples(data)
        if self.net is None:
            raise RuntimeError("Fitted model has no network")
        self.net.eval()
        targets: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], self.batch_size):
                chunk = torch.from_numpy(batch[start : start + self.batch_size]).to(
                    self.device
                )
                log_magnitude, _ = self._stft(chunk)
                n_samples, n_channels, n_freq, n_frames = log_magnitude.shape
                normalised = self._normalise(log_magnitude)
                prediction = self.net(
                    normalised.reshape(n_samples * n_channels, 1, n_freq, n_frames)
                ).reshape(n_samples, n_channels, n_freq, n_frames)
                targets.append(normalised.detach().cpu().numpy())
                errors.append(
                    (prediction - normalised).abs().detach().cpu().numpy()
                )
        return np.concatenate(targets), np.concatenate(errors)

    def frequency_axis(self, sampling_rate_hz: float = 1000.0) -> np.ndarray:
        """Centre frequency in Hz of every STFT bin."""

        return np.arange(self.n_fft // 2 + 1) * (
            float(sampling_rate_hz) / self.n_fft
        )
