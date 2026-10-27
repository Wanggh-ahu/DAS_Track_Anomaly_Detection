"""Data ingestion, event slicing and synthetic acquisition helpers."""

from .bin_reader import DASMetadata, HDF5DASReader, load_das_recording
from .pass_slicer import PassEvent, PassSlicer
from .simulator import SimulationConfig, TrainPassSimulator, generate_synthetic_dataset

__all__ = [
    "DASMetadata",
    "HDF5DASReader",
    "load_das_recording",
    "PassEvent",
    "PassSlicer",
    "SimulationConfig",
    "TrainPassSimulator",
    "generate_synthetic_dataset",
]
