"""Governance: explanations, API defences (rate limit + input validation) and the audit trail."""
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .config import FEATURES, OOD_MARGIN, RATE_LIMIT_PER_SEC


def explain(model, x, top_k: int = 3) -> list:
    """Why did the tree decide this? Walk the decision path and credit each split's feature
    with how much it moved P(in_form). Exact for decision trees, no SHAP install needed."""
    tree = model.tree_
    counts = tree.value[:, 0, :]
    p_in_form = counts[:, 1] / counts.sum(axis=1)
    path = model.decision_path(np.asarray([x], dtype=float)).indices  # root -> leaf
    contrib = np.zeros(len(FEATURES))
    for parent, child in zip(path[:-1], path[1:]):
        contrib[tree.feature[parent]] += p_in_form[child] - p_in_form[parent]
    top = np.argsort(-np.abs(contrib))[:top_k]
    return [{"feature": FEATURES[i], "contribution": round(float(contrib[i]), 2)} for i in top]


class RateLimiter:
    """At most `per_second` requests per client in any 1-second window."""

    def __init__(self, per_second: int = RATE_LIMIT_PER_SEC, clock=time.monotonic):
        self.per_second, self.clock = per_second, clock
        self.hits = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        now, hits = self.clock(), self.hits[client_id]
        while hits and now - hits[0] > 1.0:
            hits.popleft()
        if len(hits) >= self.per_second:
            return False
        hits.append(now)
        return True


def validate(x, low, high):
    """Return an error code, or None if x is a sane, in-distribution feature vector."""
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return "invalid_input"
    if x.shape != (len(FEATURES),) or not np.all(np.isfinite(x)):
        return "invalid_input"
    margin = OOD_MARGIN * (high - low)
    if np.any(x < low - margin) or np.any(x > high + margin):
        return "out_of_distribution"
    return None


class AuditLog:
    """Append-only JSON-lines log of every decision and every governance action."""

    def __init__(self, home):
        self.path = Path(home) / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields):
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")

    def read(self, event: str = None) -> list:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        return [r for r in rows if event is None or r["event"] == event]
