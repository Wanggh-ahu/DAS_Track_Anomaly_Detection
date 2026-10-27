"""DAS track-structure unsupervised anomaly detection package.

The package is intentionally organised around a small, explicit data contract:

* one recording has shape ``(n_channels, n_samples)``;
* one model sample has shape ``(n_channels, n_samples)``;
* a batch of model samples has shape ``(batch, n_channels, n_samples)``.

Keeping this convention in every module makes it possible to freely swap the
statistical baselines and the neural reconstruction models.
"""

__version__ = "0.1.0"
