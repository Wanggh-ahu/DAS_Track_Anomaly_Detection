"""Unsupervised anomaly-detection models and shared interfaces."""

from .ae_1d import AE1D
from .base_model import (
    BaseModel,
    BaseTorchModel,
    available_models,
    build_model,
)
from .energy import EnergyThresholdBaseline
from .spatial_masked_ae import SpatialMaskedAE
from .stae_convlstm import STAEConvLSTM
from .statistical import IsolationForestBaseline, PCABaseline

MODEL_REGISTRY = {
    "pca": PCABaseline,
    "energy": EnergyThresholdBaseline,
    "isolation_forest": IsolationForestBaseline,
    "ae_1d": AE1D,
    "stae_convlstm": STAEConvLSTM,
    "spatial_masked_ae": SpatialMaskedAE,
}

__all__ = [
    "BaseModel",
    "BaseTorchModel",
    "build_model",
    "available_models",
    "PCABaseline",
    "EnergyThresholdBaseline",
    "IsolationForestBaseline",
    "AE1D",
    "STAEConvLSTM",
    "SpatialMaskedAE",
    "MODEL_REGISTRY",
]
