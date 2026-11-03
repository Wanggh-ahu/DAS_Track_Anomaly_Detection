"""Unsupervised anomaly-detection models and shared interfaces."""

from .ae_1d import AE1D
from .bandpass import BandEnergyBaseline
from .base_model import (
    BaseModel,
    BaseTorchModel,
    available_models,
    build_model,
)
from .energy import EnergyThresholdBaseline
from .filterbank import FilterbankBaseline
from .kurtogram import KurtogramBaseline, LoudestBandBaseline
from .spatial_masked_ae import SpatialMaskedAE
from .stae_convlstm import STAEConvLSTM
from .statistical import IsolationForestBaseline, PCABaseline
from .stft_ae import STFTLogMagnitudeAE
from .spatial_profile import SpatialProfileBaseline
from .timescale import EnvelopeSpectrumBaseline, TrajectoryBaseline

MODEL_REGISTRY = {
    "pca": PCABaseline,
    "energy": EnergyThresholdBaseline,
    "bandpass": BandEnergyBaseline,
    "filterbank": FilterbankBaseline,
    "kurtogram": KurtogramBaseline,
    "bandselect": LoudestBandBaseline,
    "envspec": EnvelopeSpectrumBaseline,
    "tftraj": TrajectoryBaseline,
    "spatial": SpatialProfileBaseline,
    "isolation_forest": IsolationForestBaseline,
    "ae_1d": AE1D,
    "stae_convlstm": STAEConvLSTM,
    "spatial_masked_ae": SpatialMaskedAE,
    "stft_ae": STFTLogMagnitudeAE,
}

__all__ = [
    "BaseModel",
    "BaseTorchModel",
    "build_model",
    "available_models",
    "PCABaseline",
    "EnergyThresholdBaseline",
    "BandEnergyBaseline",
    "FilterbankBaseline",
    "KurtogramBaseline",
    "LoudestBandBaseline",
    "EnvelopeSpectrumBaseline",
    "TrajectoryBaseline",
    "SpatialProfileBaseline",
    "IsolationForestBaseline",
    "AE1D",
    "STAEConvLSTM",
    "SpatialMaskedAE",
    "STFTLogMagnitudeAE",
    "MODEL_REGISTRY",
]
