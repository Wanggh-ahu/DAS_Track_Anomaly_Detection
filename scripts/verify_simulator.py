"""Sanity checks for the physics-based defect generator.

Verifies three properties of :mod:`src.data_io.simulator`:

1. Defect injection is additive: it does not rescale normal channels.
2. The requested local SNR is met at the detector input after preprocessing,
   measured over the defect's active time support.
3. The two mechanisms behave differently: resonance rings longer than impact.

Run with::

    python scripts/verify_simulator.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io.simulator import SimulationConfig, TrainPassSimulator  # noqa: E402
from src.dsp.filters import preprocess_recording  # noqa: E402


def _support_mask(
    labels: np.ndarray,
    n_samples: int,
    active_ranges: list[list[float]],
    sampling_rate_hz: float,
) -> np.ndarray:
    """Rebuild the (channel, sample) support used to define the local SNR."""

    mask = np.zeros((labels.size, n_samples), dtype=bool)
    channels = np.flatnonzero(labels)
    if channels.size == 0 or not active_ranges:
        return mask
    start = max(int(channels.min()) - 1, 0)
    stop = min(int(channels.max()) + 2, labels.size)
    for start_s, stop_s in active_ranges:
        first = max(int(np.floor(start_s * sampling_rate_hz)), 0)
        last = min(int(np.ceil(stop_s * sampling_rate_hz)), n_samples)
        mask[start:stop, first:last] = True
    return mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the synthetic defect generator.")
    parser.add_argument("--snr", nargs="*", type=float, default=[-15.0, -10.0, -5.0, 0.0, 5.0])
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(argv)

    base = dict(seed=args.seed, n_channels=96, duration_s=8.0)
    rows: list[dict[str, Any]] = []

    for anomaly_type in ("impact", "resonance", "mixed"):
        for snr in args.snr:
            simulator = TrainPassSimulator(
                SimulationConfig(anomaly_type=anomaly_type, snr_db=snr, **base)
            )
            data, labels, meta = simulator.simulate(anomaly=True)
            support = simulator.last_support_
            simulator.rng = np.random.default_rng(args.seed)
            clean, _, _ = simulator.simulate(anomaly=False)

            defect = meta["defect"]
            if not labels.any():
                raise RuntimeError(f"{anomaly_type}@{snr}dB produced no labels")

            defect_channels = np.flatnonzero(labels)
            healthy = np.setdiff1d(np.arange(labels.size), defect_channels)
            delta = np.abs(data - clean)
            if float(delta[healthy].max()) > 1e-6:
                raise AssertionError(
                    f"{anomaly_type}@{snr}dB modified healthy channels "
                    f"(max delta {float(delta[healthy].max()):.3e})"
                )

            if support is None:
                raise AssertionError(f"{anomaly_type}@{snr}dB did not expose a support mask")
            processed = preprocess_recording(
                data,
                sampling_rate_hz=simulator.config.sampling_rate_hz,
                bandpass_hz=[5.0, 200.0],
                order=4,
                common_mode_window=5,
            )
            processed_clean = preprocess_recording(
                clean,
                sampling_rate_hz=simulator.config.sampling_rate_hz,
                bandpass_hz=[5.0, 200.0],
                order=4,
                common_mode_window=5,
            )
            component = processed - processed_clean
            noise_power = float(np.mean(np.square(processed_clean[support])))
            signal_power = float(np.mean(np.square(component[support])))
            measured = 10.0 * np.log10(signal_power / max(noise_power, 1e-12))

            active_seconds = float(
                sum(stop - start for start, stop in defect["active_time_ranges_s"])
            )
            rows.append(
                {
                    "type": anomaly_type,
                    "target": snr,
                    "processed": defect["snr_db_processed"],
                    "raw": defect["snr_db_raw"],
                    "measured": measured,
                    "n_channels": int(labels.sum()),
                    "active_s": active_seconds,
                    "peak_ratio_db": 20.0
                    * np.log10(
                        max(float(np.abs(component[defect_channels]).max()), 1e-12)
                        / max(float(np.abs(processed_clean).std()), 1e-12)
                    ),
                }
            )

    header = (
        f"{'type':>10} {'target':>7} {'processed':>10} {'raw':>9} "
        f"{'measured':>9} {'width':>6} {'active_s':>9} {'peak_dB':>8}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['type']:>10} {row['target']:>7.1f} {row['processed']:>10.3f} "
            f"{row['raw']:>9.3f} {row['measured']:>9.3f} {row['n_channels']:>6d} "
            f"{row['active_s']:>9.3f} {row['peak_ratio_db']:>8.2f}"
        )

    errors = [abs(row["measured"] - row["target"]) for row in rows]
    print(f"\nmax |measured - target| = {max(errors):.3f} dB")

    # Mechanism check: a resonance must ring longer than an impact at equal SNR.
    impact_active = np.mean([r["active_s"] for r in rows if r["type"] == "impact"])
    resonance_active = np.mean([r["active_s"] for r in rows if r["type"] == "resonance"])
    print(
        f"mean active duration: impact={impact_active:.3f}s "
        f"resonance={resonance_active:.3f}s"
    )
    if resonance_active <= impact_active:
        raise AssertionError("resonance should ring longer than an impact")

    if max(errors) > 1.0:
        raise AssertionError("processed SNR deviates from the requested value by > 1 dB")
    print("simulator checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
