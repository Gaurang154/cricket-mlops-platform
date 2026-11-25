"""Model registry: numbered versions, stages and lineage, stored in one JSON file.

Stages: Staging -> Production -> Archived. Only the promotion gate can put a model in
Staging, and only a named human approver (via the control plane) can put it in Production.
"""
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score

from .config import MAX_FAIRNESS_GAP, MODEL_NAME
from .data import Dataset
from .monitoring import equal_opportunity_gap

SINGLE_STAGES = ("Staging", "Production")  # at most one version in each of these


class Registry:
    def __init__(self, home, tracker, audit):
        self.path = Path(home) / "registry.json"
        self.tracker, self.audit = tracker, audit

    # --- reading ------------------------------------------------------------
    def versions(self) -> list:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def get(self, stage: str):
        """The version currently in `stage`, or None."""
        return next((v for v in self.versions() if v["stage"] == stage), None)

    def get_version(self, version: int):
        return next((v for v in self.versions() if v["version"] == version), None)

    def load_model(self, stage: str):
        """Load by STAGE reference (like models:/Name/Production). Returns (model, info)."""
        info = self.get(stage)
        return (self.tracker.load_model(info["run_id"]), info) if info else (None, None)

    # --- writing ------------------------------------------------------------
    def _save(self, versions):
        self.path.write_text(json.dumps(versions, indent=2))

    def transition(self, version: int, stage: str, by: str, reason: str):
        versions = self.versions()
        for v in versions:
            if v["version"] != version and v["stage"] == stage and stage in SINGLE_STAGES:
                v["stage"] = "Archived"  # the old holder of this stage steps down
                v["history"].append({"ts": time.time(), "stage": "Archived", "by": by,
                                     "reason": f"replaced by v{version}"})
            if v["version"] == version:
                v["stage"] = stage
                v["history"].append({"ts": time.time(), "stage": stage, "by": by, "reason": reason})
        self._save(versions)
        self.audit.record("stage_transition", model=MODEL_NAME, version=version, stage=stage,
                          by=by, reason=reason)

    def register(self, run_id: str, by: str, reason: str) -> int:
        versions = self.versions()
        run = self.tracker.get_run(run_id)
        version = len(versions) + 1
        versions.append({"name": MODEL_NAME, "version": version, "run_id": run_id,
                         "data_version": run["params"]["data_version"],
                         "stage": "None", "history": []})
        self._save(versions)
        self.transition(version, "Staging", by, reason)
        return version

    def rollback(self, by: str) -> int:
        """Put the most recent previously-Production version back into Production."""
        current = self.get("Production")
        previous = [v for v in self.versions()
                    if v["stage"] == "Archived" and any(h["stage"] == "Production" for h in v["history"])]
        if not current or not previous:
            raise RuntimeError("Nothing to roll back to.")
        target = max(previous, key=lambda v: v["version"])
        self.transition(target["version"], "Production", by, f"rollback from v{current['version']}")
        return target["version"]

    # --- the promotion gate -------------------------------------------------
    def promote_if_better(self, run_id: str, max_gap: float = MAX_FAIRNESS_GAP,
                          compare: bool = True, by: str = "promotion-gate") -> dict:
        """Fairness MUST pass. Candidate MUST beat the incumbent on the recent eval slice
        (unless compare=False, the audited escape hatch). The first model auto-promotes."""
        run = self.tracker.get_run(run_id)
        ds = Dataset(run["params"]["data_version"])
        candidate = self.tracker.load_model(run_id)
        pred = candidate.predict(ds.X_eval)
        cand_acc = accuracy_score(ds.y_eval, pred)
        gap = equal_opportunity_gap(pred, ds.y_eval, ds.g_eval)
        result = {"run_id": run_id, "data_version": ds.version,
                  "candidate_accuracy": round(cand_acc, 3), "fairness_gap": round(gap, 3)}

        incumbent, info = self.load_model("Production")
        if incumbent is not None:  # judged on the SAME recent slice
            inc_acc = accuracy_score(ds.y_eval, incumbent.predict(ds.X_eval))
            result["incumbent_accuracy"] = round(inc_acc, 3)
            result["incumbent_version"] = info["version"]

        if gap > max_gap:
            result.update(decision="BLOCKED", reason=f"fairness gap {gap:.3f} > {max_gap}")
        elif compare and incumbent is not None and cand_acc <= inc_acc:
            result.update(decision="HOLD", reason="not better than the Production model")
        else:
            reason = "first model" if incumbent is None else "passed accuracy + fairness gates"
            if not compare:
                reason = "escape hatch: accuracy comparison skipped"
            result.update(decision="PROMOTED", reason=reason,
                          version=self.register(run_id, by, reason), stage="Staging")
        self.audit.record("promotion_gate", **result)
        return result
