"""Data layer: versioned cricket datasets.

A data version is a frozen recipe, so the same version always produces the exact same
rows (checked with a fingerprint hash). In a real platform these would be DVC / lakeFS
snapshots; the idea is the same: every training run points at one immutable version.
"""
import hashlib

import numpy as np

from .config import SEED

#           batting_avg  strike_rate  boundary_pct  away_avg
LOW = np.array([10, 60, 0.10, 8])
HIGH = np.array([65, 160, 0.75, 60])

VERSIONS = {
    "innings@2024.06": {"seed": SEED, "batting_shift": 0.0},
    "innings@2024.09": {"seed": SEED + 1, "batting_shift": 0.0},  # same season, new rows
    "innings@2025.01": {"seed": SEED + 2, "batting_shift": 15.0},  # new season: averages jump
}
DEFAULT_VERSION = "innings@2024.06"


def latest_version() -> str:
    return list(VERSIONS)[-1]


def label_rule(X):
    """Ground truth: is the player 'in form'?"""
    return ((0.03 * X[:, 0] + 0.02 * X[:, 1] + 1.5 * X[:, 2]) > 4.5).astype(int)


class Dataset:
    """Rows are in time order: first 80% = training, last 20% = recent evaluation slice."""

    def __init__(self, version: str, n: int = 600):
        if version not in VERSIONS:
            raise KeyError(f"Unknown data version {version!r}. Known: {list(VERSIONS)}")
        spec = VERSIONS[version]
        rng = np.random.default_rng(spec["seed"])
        X = rng.uniform(LOW, HIGH, size=(n, 4))
        X[:, 0] += spec["batting_shift"]
        self.version = version
        self.X, self.y = X, label_rule(X)
        self.group = rng.integers(0, 2, n)  # protected attribute: never a model feature
        self.fingerprint = hashlib.sha256(
            X.tobytes() + self.y.tobytes() + self.group.tobytes()
        ).hexdigest()[:12]

        cut = int(0.8 * n)
        self.X_train, self.y_train, self.g_train = X[:cut], self.y[:cut], self.group[:cut]
        self.X_eval, self.y_eval, self.g_eval = X[cut:], self.y[cut:], self.group[cut:]
