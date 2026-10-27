"""Logging helpers for reproducible experiments."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _json_default(value: Any) -> Any:
    """Serialise common scientific Python objects for JSON logs."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive fallback
        return "<unserialisable>"


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: str | int = logging.INFO,
    console: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Create or configure a named logger without duplicating handlers.

    Parameters
    ----------
    name:
        Logger name, usually the experiment or module name.
    log_file:
        Optional file path.  Parent directories are created automatically.
    level:
        Logging level as a name or numeric constant.
    console:
        Whether to emit messages to stderr.
    propagate:
        Forward records to the root logger.  Defaults to ``False`` to avoid
        duplicate output in notebooks.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    logger.propagate = propagate
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console and not any(
        isinstance(handler, logging.StreamHandler)
        and getattr(handler, "_das_console", False)
        for handler in logger.handlers
    ):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream._das_console = True  # type: ignore[attr-defined]
        logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path)
        existing = {
            getattr(handler, "baseFilename", None) for handler in logger.handlers
        }
        if resolved not in existing:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._das_file = True  # type: ignore[attr-defined]
            logger.addHandler(file_handler)
    return logger


class ExperimentLogger:
    """A lightweight experiment logger with a JSONL metric stream.

    Examples
    --------
    >>> exp = ExperimentLogger("train", "outputs/logs", config={"lr": 0.001})
    >>> exp.log_metrics({"train_loss": 0.12, "epoch": 1}, step=1)
    >>> exp.close()
    """

    def __init__(
        self,
        name: str,
        log_dir: str | Path,
        config: Mapping[str, Any] | None = None,
        level: str = "INFO",
        console: bool = True,
    ) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{name}_{timestamp}.log"
        self.metrics_path = self.log_dir / f"{name}_{timestamp}_metrics.jsonl"
        self.config_path = self.log_dir / f"{name}_{timestamp}_config.json"
        self.logger = get_logger(
            f"das.{name}.{timestamp}", log_file=self.log_path, level=level, console=console
        )
        if config is not None:
            self.save_config(config)

    # ------------------------------------------------------------- delegation
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(message, *args, **kwargs)

    # --------------------------------------------------------------- artefacts
    def save_config(self, config: Mapping[str, Any]) -> Path:
        """Write a timestamped JSON snapshot of the resolved configuration."""

        self.config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return self.config_path

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Append one JSON object to the metric stream and the text log."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if step is not None:
            payload["step"] = int(step)
        for key, value in metrics.items():
            payload[f"{prefix}.{key}" if prefix else str(key)] = value
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        self.info("metrics: %s", json.dumps(payload, ensure_ascii=False, default=_json_default))

    def copy_file(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Copy an artefact into the experiment log directory."""

        source_path = Path(source)
        target = Path(destination) if destination is not None else self.log_dir / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        return target

    def close(self) -> None:
        """Flush and detach handlers created for this experiment."""

        for handler in list(self.logger.handlers):
            if getattr(handler, "_das_console", False) or getattr(handler, "_das_file", False):
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
