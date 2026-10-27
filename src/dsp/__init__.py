"""Digital signal-processing building blocks for DAS recordings."""

from .features import (
    extract_feature_matrix,
    rms_feature,
    stft_feature,
    wavelet_packet_features,
)
from .filters import (
    bandpass_filter,
    preprocess_recording,
    spatial_common_mode_removal,
)

__all__ = [
    "bandpass_filter",
    "spatial_common_mode_removal",
    "preprocess_recording",
    "rms_feature",
    "stft_feature",
    "wavelet_packet_features",
    "extract_feature_matrix",
]
