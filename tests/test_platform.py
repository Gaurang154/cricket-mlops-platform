"""Tests for every rubric item. Run with:  pytest -q"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app
from mlplatform import Platform
from mlplatform.data import Dataset
from mlplatform.governance import RateLimiter
from mlplatform.monitoring import equal_opportunity_gap, monitor_report


@pytest.fixture
def platform(tmp_path):
    return Platform(tmp_path)


def send_traffic(serving, data_version, n=300):
    ds = Dataset(data_version)
    for i in range(n):
        serving.predict(f"client-{i}", ds.X[i].tolist(), group=int(ds.group[i]))


# --- 1. Full loop -------------------------------------------------------------
def test_end_to_end_loop(platform):
    first = platform.train("innings@2024.06")
    assert first["decision"] == "PROMOTED" and first["stage"] == "Staging"
    assert platform.registry.get("Production") is None  # Production always needs approval
    assert platform.approve("head-of-analytics") == 1

    serving = platform.serve()
    out = serving.predict("coach", [53.6, 131.0, 0.61, 48.0], group=1)
    assert set(out) == {"in_form", "model_version", "top_reasons"}
    assert len(out["top_reasons"]) == 3

    send_traffic(serving, "innings@2025.01")  # new season -> drift
    loop = platform.maybe_retrain()
    assert loop["retrained"] and loop["monitor"]["drifted_features"] == ["batting_average"]
    assert loop["gate"]["decision"] == "PROMOTED"
    assert platform.approve("head-of-analytics") == 2

    lineage = platform.lineage()
    assert lineage["data_version"] == "innings@2025.01"
    assert lineage["data_fingerprint"] == Dataset("innings@2025.01").fingerprint
    assert lineage["seed"] == 1983


# --- 2. Lineage + reproducibility --------------------------------------------
def test_same_data_version_gives_same_model(tmp_path):
    a, b = Platform(tmp_path / "a"), Platform(tmp_path / "b")
    run_a, run_b = a.train()["run_id"], b.train()["run_id"]
    X = Dataset("innings@2024.06").X
    assert (a.tracker.load_model(run_a).predict(X) == b.tracker.load_model(run_b).predict(X)).all()
    assert a.tracker.get_run(run_a)["metrics"] == b.tracker.get_run(run_b)["metrics"]


# --- 3. Gated promotion ------------------------------------------------------
def test_candidate_must_beat_incumbent(platform):
    platform.train()
    platform.approve("lead")
    again = platform.train()  # identical model -> not better
    assert again["decision"] == "HOLD"
    assert platform.registry.get("Staging") is None


class BiasedModel:
    """Stub model with disparate impact: it only ever approves group-0 players."""

    def __init__(self, ds):
        self.y, self.g = ds.y_eval, ds.g_eval

    def predict(self, X):
        return np.where(self.g == 0, self.y, 0)


def test_fairness_gate_blocks_biased_model(platform):
    ds = Dataset("innings@2024.06")
    run_id = platform.tracker.log_run({"data_version": ds.version}, {}, BiasedModel(ds))
    result = platform.registry.promote_if_better(run_id)
    assert result["decision"] == "BLOCKED"
    assert result["fairness_gap"] == 1.0
    assert platform.registry.versions() == []  # never even registered


def test_equal_opportunity_gap():
    y_true = np.array([1, 1, 1, 1])
    group = np.array([0, 0, 1, 1])
    assert equal_opportunity_gap(np.array([1, 1, 1, 1]), y_true, group) == 0.0
    assert equal_opportunity_gap(np.array([1, 1, 0, 1]), y_true, group) == 0.5


def test_escape_hatch_is_opt_in_and_audited(tmp_path):
    with pytest.raises(PermissionError):
        Platform(tmp_path / "locked").force_promote("any", by="me", reason="please")

    p = Platform(tmp_path / "open", escape_hatch=True)
    p.train()
    p.approve("lead")
    held = p.train()
    assert held["decision"] == "HOLD"
    forced = p.force_promote(held["run_id"], by="on-call", reason="urgent hotfix")
    assert forced["decision"] == "PROMOTED"
    assert p.audit.read("escape_hatch_used")[0]["reason"] == "urgent hotfix"


# --- 4. Serving safety --------------------------------------------------------
def test_serving_guards(platform):
    platform.train()
    platform.approve("lead")
    serving = platform.serve()
    assert serving.predict("u", [500, 120, 0.4, 30]) == {"error": "out_of_distribution"}
    assert serving.predict("u", [1, 2]) == {"error": "invalid_input"}
    assert serving.predict("u", ["abc", 2, 3, 4]) == {"error": "invalid_input"}

    flood = [serving.predict("bot", [40, 120, 0.4, 30]) for _ in range(10)]
    assert sum(r.get("error") == "rate_limited" for r in flood) == 5
    assert len(platform.audit.read("blocked")) == 8  # every block is audited


def test_rate_limiter_window_resets():
    now = [0.0]
    limiter = RateLimiter(per_second=2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    now[0] = 1.5
    assert limiter.allow("a")


# --- 5. Monitoring ------------------------------------------------------------
def test_drift_detection():
    reference = Dataset("innings@2024.06").X_train
    same = [{"x": x.tolist(), "pred": 0} for x in Dataset("innings@2024.09").X[:200]]
    shifted = [{"x": x.tolist(), "pred": 0} for x in Dataset("innings@2025.01").X[:200]]
    assert monitor_report(reference, same)["drift"] is False
    assert monitor_report(reference, shifted)["drifted_features"] == ["batting_average"]
    assert monitor_report(reference, same[:10])["status"] == "warming_up"


# --- 6. Safe rollout ----------------------------------------------------------
def test_canary_then_rollback(platform):
    platform.train("innings@2024.06")
    platform.approve("lead")
    assert platform.train("innings@2025.01")["decision"] == "PROMOTED"

    canary = platform.serve(canary_fraction=0.5)
    send_traffic(canary, "innings@2025.01", n=100)
    assert canary.counters["predictions_v1_total"] > 0
    assert canary.counters["predictions_v2_total"] > 0

    assert platform.approve("lead") == 2
    assert platform.rollback(by="on-call") == 1
    assert platform.registry.get_version(2)["stage"] == "Archived"


# --- 7. HTTP API --------------------------------------------------------------
def test_http_api(tmp_path):
    client = TestClient(create_app(Platform(tmp_path)))  # bootstraps v1 on first boot
    assert client.get("/health").json() == {"status": "ok", "model_version": 1}

    player = {"batting_average": 53.6, "strike_rate": 131.0, "boundary_pct": 0.61, "away_average": 48.0}
    r = client.post("/predict", json=player, headers={"X-Client-Id": "tester"})
    assert r.status_code == 200 and len(r.json()["top_reasons"]) == 3

    r = client.post("/predict", json={**player, "batting_average": 500})
    assert r.status_code == 422 and r.json()["detail"] == "out_of_distribution"

    assert "mlops_drift_detected 0" in client.get("/metrics").text
    assert client.get("/lineage").json()["data_version"] == "innings@2024.06"
