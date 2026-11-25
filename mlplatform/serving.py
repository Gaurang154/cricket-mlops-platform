"""Serving: load models by STAGE, guard the API, explain + audit every decision, canary support."""
import random
from collections import Counter

from .config import SEED
from .data import Dataset
from .governance import RateLimiter, explain, validate


class ServingLayer:
    def __init__(self, registry, audit, canary_fraction: float = 0.0, rate_limiter=None):
        self.registry, self.audit = registry, audit
        self.canary_fraction = canary_fraction
        self.limiter = rate_limiter or RateLimiter()
        self.rng = random.Random(SEED)
        self.counters = Counter()
        self.reload()

    def reload(self):
        """Pick up whatever is in Production right now (and Staging, for canary)."""
        self.prod, self.prod_info = self.registry.load_model("Production")
        if self.prod is None:
            raise RuntimeError("No model in Production yet. Train one and approve it first.")
        self.canary, self.canary_info = (self.registry.load_model("Staging")
                                         if self.canary_fraction > 0 else (None, None))
        X = Dataset(self.prod_info["data_version"]).X_train  # valid input range = training range
        self.low, self.high = X.min(axis=0), X.max(axis=0)

    def predict(self, client_id: str, x, group=None) -> dict:
        self.counters["requests_total"] += 1

        # 1. Defend: rate limit, then validate the input.
        error = "rate_limited" if not self.limiter.allow(client_id) else validate(x, self.low, self.high)
        if error:
            self.counters[f"blocked_{error}_total"] += 1
            self.audit.record("blocked", client=client_id, reason=error)
            return {"error": error}

        # 2. Route: a small slice of traffic goes to the canary (Staging) model.
        use_canary = self.canary is not None and self.rng.random() < self.canary_fraction
        model, info = (self.canary, self.canary_info) if use_canary else (self.prod, self.prod_info)

        # 3. Predict, explain, audit.
        x = [float(v) for v in x]
        pred = int(model.predict([x])[0])
        self.counters[f"predictions_v{info['version']}_total"] += 1
        self.audit.record("prediction", client=client_id, model_version=info["version"],
                          x=x, pred=pred, group=group)
        return {"in_form": bool(pred),  # coarse yes/no only: no scores to help model extraction
                "model_version": info["version"],
                "top_reasons": explain(model, x)}
