"""Training pipeline: load versioned data -> train -> evaluate -> track.

Each numbered step maps 1:1 onto an orchestrator task (Airflow / Prefect / Kubeflow).
"""
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .config import FEATURES, SEED
from .data import Dataset


def train_and_track(tracker, data_version: str, max_depth: int = 4) -> str:
    ds = Dataset(data_version)  # 1. load an exact data version

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)
    model.fit(ds.X_train, ds.y_train)  # 2. train (fixed seed)

    metrics = {  # 3. evaluate
        "train_accuracy": accuracy_score(ds.y_train, model.predict(ds.X_train)),
        "eval_accuracy": accuracy_score(ds.y_eval, model.predict(ds.X_eval)),
    }
    params = {  # lineage: exactly what produced this model
        "data_version": data_version,
        "data_fingerprint": ds.fingerprint,
        "seed": SEED,
        "max_depth": max_depth,
        "algorithm": "DecisionTreeClassifier",
        "features": FEATURES,
    }
    return tracker.log_run(params, metrics, model)  # 4. track -> run_id
