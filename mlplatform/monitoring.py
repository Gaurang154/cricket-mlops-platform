"""Monitoring: data drift (KS test) and fairness, computed from logged decisions."""
import numpy as np
from scipy.stats import ks_2samp

from .config import (DRIFT_ALPHA, FEATURES, LIVE_FAIRNESS_GAP, MIN_MONITOR_SAMPLES,
                     MONITOR_WINDOW)


def equal_opportunity_gap(y_pred, y_true, group) -> float:
    """Largest gap in true-positive rate between groups. Used by the promotion gate."""
    y_pred, y_true, group = map(np.asarray, (y_pred, y_true, group))
    tprs = [y_pred[(group == g) & (y_true == 1)].mean()
            for g in np.unique(group) if ((group == g) & (y_true == 1)).any()]
    return float(max(tprs) - min(tprs)) if tprs else 0.0


def selection_rate_gap(y_pred, group) -> float:
    """Largest gap in positive-prediction rate between groups. Needs no labels -> works live."""
    y_pred, group = np.asarray(y_pred), np.asarray(group)
    rates = [y_pred[group == g].mean() for g in np.unique(group)]
    return float(max(rates) - min(rates)) if rates else 0.0


def monitor_report(reference_X, decisions: list) -> dict:
    """Compare recent live decisions with the model's training data."""
    report = {"n": len(decisions), "drift": False, "drifted_features": [],
              "fairness_gap": None, "fairness_ok": None}
    if len(decisions) < MIN_MONITOR_SAMPLES:
        report["status"] = "warming_up"
        return report
    report["status"] = "ok"
    recent = decisions[-MONITOR_WINDOW:]

    live = np.array([d["x"] for d in recent])
    ref = np.asarray(reference_X)
    for i, name in enumerate(FEATURES):
        if ks_2samp(ref[:, i], live[:, i]).pvalue < DRIFT_ALPHA:
            report["drifted_features"].append(name)
    report["drift"] = bool(report["drifted_features"])

    with_group = [d for d in recent if d.get("group") is not None]
    if with_group:
        gap = selection_rate_gap([d["pred"] for d in with_group], [d["group"] for d in with_group])
        report["fairness_gap"] = round(gap, 3)
        report["fairness_ok"] = gap <= LIVE_FAIRNESS_GAP
    return report
