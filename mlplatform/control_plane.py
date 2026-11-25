"""Control plane = the self-service SDK.

Users speak in datasets, models and deployments. The platform wires tracking, lineage,
the fairness + accuracy gates, approval, monitoring, explanations and the audit trail
underneath, by default. One opt-in escape hatch exists for genuine exceptions.
"""
from pathlib import Path

from .config import MAX_FAIRNESS_GAP
from .data import DEFAULT_VERSION, Dataset, latest_version
from .governance import AuditLog
from .monitoring import monitor_report
from .pipeline import train_and_track
from .registry import Registry
from .serving import ServingLayer
from .tracking import Tracker


class Platform:
    def __init__(self, home=".mlops", escape_hatch: bool = False,
                 max_fairness_gap: float = MAX_FAIRNESS_GAP):
        self.home = Path(home)
        self.audit = AuditLog(self.home)
        self.tracker = Tracker(self.home)
        self.registry = Registry(self.home, self.tracker, self.audit)
        self.escape_hatch = escape_hatch
        self.max_fairness_gap = max_fairness_gap

    # ---------------------------------------------------------------- paved road
    def train(self, data_version: str = DEFAULT_VERSION, max_depth: int = 4) -> dict:
        """Train (tracked) and send the run through the promotion gate."""
        run_id = train_and_track(self.tracker, data_version, max_depth)
        return self.registry.promote_if_better(run_id, max_gap=self.max_fairness_gap)

    def approve(self, approver: str) -> int:
        """Human sign-off: Staging -> Production. If the candidate has served canary
        traffic, that live traffic must also pass the fairness check."""
        staged = self.registry.get("Staging")
        if staged is None:
            raise RuntimeError("Nothing in Staging to approve.")
        canary = self.monitor(version=staged["version"])
        if canary["fairness_ok"] is False:
            raise PermissionError(f"Canary fairness gap {canary['fairness_gap']} is too high.")
        self.registry.transition(staged["version"], "Production", by=approver, reason="approved")
        return staged["version"]

    def serve(self, canary_fraction: float = 0.0) -> ServingLayer:
        """A defended, explained, audited endpoint for whatever is in Production."""
        return ServingLayer(self.registry, self.audit, canary_fraction)

    def monitor(self, version: int = None) -> dict:
        """Drift + fairness report for a model version (default: Production)."""
        info = self.registry.get("Production") if version is None else self.registry.get_version(version)
        if info is None:
            raise RuntimeError("No such model version.")
        decisions = [d for d in self.audit.read("prediction") if d["model_version"] == info["version"]]
        reference = Dataset(info["data_version"]).X_train
        return {"model_version": info["version"], **monitor_report(reference, decisions)}

    def maybe_retrain(self, data_version: str = None) -> dict:
        """Closed loop: if Production is drifting, retrain on the newest data and re-gate."""
        report = self.monitor()
        if not report["drift"]:
            return {"retrained": False, "monitor": report}
        self.audit.record("retrain_triggered", drifted_features=report["drifted_features"])
        return {"retrained": True, "monitor": report,
                "gate": self.train(data_version or latest_version())}

    def rollback(self, by: str) -> int:
        return self.registry.rollback(by)

    def lineage(self, stage: str = "Production") -> dict:
        """Model version -> run -> params (data version, fingerprint, seed) -> metrics -> history."""
        info = self.registry.get(stage)
        if info is None:
            raise RuntimeError(f"No model in {stage}.")
        run = self.tracker.get_run(info["run_id"])
        return {"model": info["name"], "version": info["version"], "stage": stage,
                "run_id": info["run_id"], **run["params"], "metrics": run["metrics"],
                "history": info["history"]}

    def bootstrap(self, approver: str = "bootstrap"):
        """Make sure something is in Production (used by the API container on first boot)."""
        if self.registry.get("Production") is None:
            result = self.train()
            if result["decision"] != "PROMOTED":
                raise RuntimeError(f"Bootstrap model failed the gate: {result}")
            self.approve(approver)

    # ---------------------------------------------------------------- escape hatch
    def force_promote(self, run_id: str, by: str, reason: str) -> dict:
        """Skip the 'must beat Production' check (e.g. an urgent hotfix).
        Must be enabled per project, fairness is STILL enforced, and it is always audited."""
        if not self.escape_hatch:
            raise PermissionError("Escape hatch is disabled for this project.")
        self.audit.record("escape_hatch_used", run_id=run_id, by=by, reason=reason)
        return self.registry.promote_if_better(run_id, max_gap=self.max_fairness_gap,
                                               compare=False, by=by)
