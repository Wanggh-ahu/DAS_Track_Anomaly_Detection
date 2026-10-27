"""Common model interfaces, a PyTorch training base and a model factory.

Every anomaly detector in this project follows the same contract:

``fit(train_data, val_data=None)``
    Fit on normal pass samples with shape ``(N, channels, samples)``.

``score(data)``
    Return one anomaly score per pass, shape ``(N,)``.  Larger is more anomalous.

``channel_scores(data)``
    Return one score per channel and pass, shape ``(N, channels)``, for
    localisation.

``reconstruct(data)``
    Return a reconstruction with the same shape as the input (reconstruction
    models only).

Models are deliberately fed numpy arrays by the scripts; device movement and
mini-batching remain an implementation detail of each model.
"""

from __future__ import annotations

import inspect
import logging
import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # PyTorch is only needed by neural models, but is a project requirement.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - helps importing statistical baselines
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


def require_torch() -> None:
    """Raise a clear error when a neural model is used without PyTorch."""

    if torch is None:
        raise ImportError(
            "PyTorch is required for neural models. Install dependencies with "
            "`pip install -r requirements.txt`."
        )


class BaseModel(ABC):
    """Abstract base class shared by statistical and neural detectors."""

    name: ClassVar[str] = "base"

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = dict(params)
        self._is_fitted = False
        self.metadata_: dict[str, Any] = {}
        self.training_history_: dict[str, list[float]] = {}

    # ------------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before inference")

    def get_params(self) -> dict[str, Any]:
        """Return JSON-friendly constructor parameters."""

        return dict(self._params)

    def __repr__(self) -> str:
        params = ", ".join(f"{key}={value!r}" for key, value in self._params.items())
        return f"{self.__class__.__name__}({params})"

    # -------------------------------------------------------------------- data
    @staticmethod
    def validate_samples(
        data: np.ndarray | "torch.Tensor",
        *,
        allow_2d: bool = True,
        dtype: np.dtype | str = np.float32,
    ) -> np.ndarray:
        """Validate and normalise a pass batch to ``(N, C, T)`` float arrays."""

        if torch is not None and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        array = np.asarray(data)
        if allow_2d and array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(
                "DAS samples must have shape (batch, channels, samples) or "
                f"(channels, samples); received {array.shape}"
            )
        if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
            raise ValueError(f"empty DAS sample array with shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("DAS samples contain NaN or infinite values")
        return np.ascontiguousarray(array, dtype=dtype)

    # -------------------------------------------------------------------- API
    @abstractmethod
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """Fit the model using normal (anomaly-free) pass samples."""

    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return an anomaly score per sample; higher means more anomalous."""

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct input samples.  Not all baselines support this."""

        raise NotImplementedError(f"{self.__class__.__name__} has no reconstruction")

    def channel_scores(self, data: np.ndarray) -> np.ndarray:
        """Return per-channel scores derived from reconstruction residuals.

        Reconstruction models get this for free.  Models such as Isolation
        Forest override the method when a different localisation strategy is
        required.
        """

        batch = self.validate_samples(data)
        reconstruction = self.reconstruct(batch)
        residual = np.square(batch - reconstruction)
        return residual.mean(axis=2)

    def predict(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Binary anomaly prediction at a user-supplied threshold."""

        return (self.score(data) >= float(threshold)).astype(np.int8)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Persist a statistical model with pickle.

        Neural models override this method with a ``torch.save`` checkpoint.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "model_name": self.name,
            "model": self,
            "params": self.get_params(),
            "metadata": metadata or {},
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path, **_: Any) -> "BaseModel":
        """Load a model saved by :meth:`save`."""

        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = payload["model"]
        if not isinstance(model, BaseModel):
            raise TypeError(f"File {path!r} does not contain a BaseModel")
        return model


class BaseTorchModel(BaseModel):
    """Base class providing a compact, reusable autoencoder training loop.

    Subclasses only need to implement :meth:`_create_network` and, when input
    reshaping is required, :meth:`_forward`.  The default objective is mean
    squared reconstruction error over the full ``(channels, samples)`` sample.
    """

    TRAINING_DEFAULTS: ClassVar[dict[str, Any]] = {
        "epochs": 50,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "device": "auto",
        "num_workers": 0,
        "early_stopping_patience": 10,
        "grad_clip_norm": 1.0,
        "amp": False,
        "seed": None,
    }

    def __init__(
        self,
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
        **extra_params: Any,
    ) -> None:
        require_torch()
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
            **extra_params,
        )
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = self._resolve_device(device)
        self.num_workers = int(num_workers)
        self.early_stopping_patience = int(early_stopping_patience)
        self.grad_clip_norm = float(grad_clip_norm)
        self.amp = bool(amp)
        self.seed = seed
        self.net: nn.Module | None = None
        self.input_shape_: tuple[int, int] | None = None
        self.optimizer_: Any = None

    # ----------------------------------------------------------------- device
    @staticmethod
    def _resolve_device(device: str) -> "torch.device":
        require_torch()
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return resolved

    def _set_seed(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    # ---------------------------------------------------------------- network
    @abstractmethod
    def _create_network(self, n_channels: int, n_samples: int) -> "nn.Module":
        """Construct and return the PyTorch network for the observed shape."""

    def _ensure_network(self, n_channels: int, n_samples: int) -> "nn.Module":
        require_torch()
        if self.net is None:
            self.net = self._create_network(n_channels, n_samples).to(self.device)
            self.input_shape_ = (int(n_channels), int(n_samples))
        return self.net

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        """Map an input batch to a reconstruction with the same shape."""

        if self.net is None:
            raise RuntimeError("Network has not been built")
        return self.net(batch)

    @staticmethod
    def _loss(batch: "torch.Tensor", reconstruction: "torch.Tensor") -> "torch.Tensor":
        """Default reconstruction objective."""

        return torch.mean(torch.square(batch - reconstruction))

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "BaseTorchModel":
        """Train the autoencoder with early stopping on validation loss.

        Extra keyword arguments are accepted for forward compatibility; the
        recognised training hyperparameters are constructor arguments.
        """

        require_torch()
        train = self.validate_samples(train_data)
        val = self.validate_samples(val_data) if val_data is not None else None
        n_channels, n_samples = train.shape[1:]
        network = self._ensure_network(n_channels, n_samples)
        if val is not None and val.shape[1:] != (n_channels, n_samples):
            raise ValueError(
                f"Validation shape {val.shape[1:]} differs from training shape "
                f"{(n_channels, n_samples)}"
            )

        self._set_seed()
        dataset = TensorDataset(torch.from_numpy(train))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        self.optimizer_ = torch.optim.Adam(
            network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        amp_enabled = self.amp and self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            network.train()
            total_loss = 0.0
            total_items = 0
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True)
                self.optimizer_.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type, enabled=amp_enabled
                ):
                    reconstruction = self._forward(batch)
                    loss = self._loss(batch, reconstruction)
                scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    scaler.unscale_(self.optimizer_)
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), self.grad_clip_norm
                    )
                scaler.step(self.optimizer_)
                scaler.update()
                total_loss += float(loss.detach()) * batch.shape[0]
                total_items += batch.shape[0]

            train_loss = total_loss / max(total_items, 1)
            val_loss = self._validation_loss(val, loader) if val is not None else train_loss
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            LOGGER.info(
                "[%s] epoch %03d/%03d train_loss=%.6f val_loss=%.6f",
                self.name,
                epoch,
                self.epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if (
                    self.early_stopping_patience > 0
                    and epochs_without_improvement >= self.early_stopping_patience
                ):
                    LOGGER.info(
                        "[%s] early stopping at epoch %d (best=%.6f)",
                        self.name,
                        epoch,
                        best_loss,
                    )
                    break

        if best_state is not None:
            network.load_state_dict(best_state)
        self._is_fitted = True
        self.training_history_ = history
        return self

    def _validation_loss(self, val: np.ndarray, train_loader: Any) -> float:
        """Compute deterministic reconstruction loss on a validation array."""

        if self.net is None:
            return float("nan")
        self.net.eval()
        total = 0.0
        total_items = 0
        with torch.no_grad():
            for start in range(0, val.shape[0], self.batch_size):
                batch = torch.from_numpy(val[start : start + self.batch_size]).to(self.device)
                reconstruction = self._forward(batch)
                loss = self._loss(batch, reconstruction)
                total += float(loss) * batch.shape[0]
                total_items += batch.shape[0]
        self.net.train()
        return total / max(total_items, 1)

    # --------------------------------------------------------------- inference
    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct all samples in batches."""

        self._check_fitted()
        batch = self.validate_samples(data)
        if self.net is None:
            raise RuntimeError("Fitted model has no network")
        if self.input_shape_ is not None and batch.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected input shape {self.input_shape_}, received {batch.shape[1:]}"
            )
        self.net.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], self.batch_size):
                chunk = torch.from_numpy(batch[start : start + self.batch_size]).to(
                    self.device
                )
                reconstruction = self._forward(chunk)
                outputs.append(reconstruction.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def score(self, data: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per pass."""

        batch = self.validate_samples(data)
        reconstruction = self.reconstruct(batch)
        return np.square(batch - reconstruction).mean(axis=(1, 2))

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        require_torch()
        self._check_fitted()
        if self.net is None or self.input_shape_ is None:
            raise RuntimeError("Cannot save an unfitted network")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "model_name": self.name,
            "model_class": self.__class__.__name__,
            "model_params": self.get_params(),
            "input_shape": list(self.input_shape_),
            "state_dict": self.net.state_dict(),
            "history": self.training_history_,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "BaseTorchModel":
        """Load a checkpoint produced by :meth:`save`."""

        require_torch()
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(**checkpoint["model_params"])
        n_channels, n_samples = checkpoint["input_shape"]
        model._ensure_network(int(n_channels), int(n_samples))
        model.net.load_state_dict(checkpoint["state_dict"])
        model._is_fitted = True
        model.training_history_ = checkpoint.get("history", {})
        model.metadata_ = checkpoint.get("metadata", {})
        return model


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
_SUPPORTED_MODELS = {
    "pca",
    "energy",
    "isolation_forest",
    "ae_1d",
    "stae_convlstm",
    "spatial_masked_ae",
}


def _load_model_class(name: str) -> type[BaseModel]:
    """Import a model class lazily to avoid circular imports."""

    if name == "pca":
        from .statistical import PCABaseline

        return PCABaseline
    if name == "energy":
        from .energy import EnergyThresholdBaseline

        return EnergyThresholdBaseline
    if name == "isolation_forest":
        from .statistical import IsolationForestBaseline

        return IsolationForestBaseline
    if name == "ae_1d":
        from .ae_1d import AE1D

        return AE1D
    if name == "stae_convlstm":
        from .stae_convlstm import STAEConvLSTM

        return STAEConvLSTM
    if name == "spatial_masked_ae":
        from .spatial_masked_ae import SpatialMaskedAE

        return SpatialMaskedAE
    raise ValueError(
        f"Unknown model {name!r}; choose one of {sorted(_SUPPORTED_MODELS)}"
    )


def build_model(
    name: str,
    model_params: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    **overrides: Any,
) -> BaseModel:
    """Instantiate a model from a name and configuration dictionaries.

    Parameters are filtered against the target constructor, so unrelated
    training keys (for example ``epochs`` for PCA) are ignored safely.
    """

    if name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {name!r}; choose one of {sorted(_SUPPORTED_MODELS)}"
        )
    values: dict[str, Any] = {}
    values.update(training_config or {})
    values.update(model_params or {})
    values.update(overrides)
    # The validation split is a data-loader concern, not a model argument.
    values.pop("validation_split", None)

    model_class = _load_model_class(name)
    parameters = inspect.signature(model_class.__init__).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        kwargs = values
    else:
        kwargs = {key: value for key, value in values.items() if key in parameters}
    return model_class(**kwargs)


def available_models() -> list[str]:
    """Return supported model identifiers for CLI help."""

    return sorted(_SUPPORTED_MODELS)
