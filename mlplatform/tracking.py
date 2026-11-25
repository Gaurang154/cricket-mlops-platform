"""Experiment tracking: every run gets a folder with run.json (params + metrics) and model.pkl.

A deliberately tiny stand-in for MLflow tracking: same ideas, zero servers to set up.
"""
import json
import pickle
import time
import uuid
from pathlib import Path


class Tracker:
    def __init__(self, home):
        self.root = Path(home) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def log_run(self, params: dict, metrics: dict, model) -> str:
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.root / run_id
        run_dir.mkdir()
        record = {"run_id": run_id, "created": time.time(), "params": params, "metrics": metrics}
        (run_dir / "run.json").write_text(json.dumps(record, indent=2))
        with open(run_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        return run_id

    def get_run(self, run_id: str) -> dict:
        return json.loads((self.root / run_id / "run.json").read_text())

    def load_model(self, run_id: str):
        with open(self.root / run_id / "model.pkl", "rb") as f:
            return pickle.load(f)
