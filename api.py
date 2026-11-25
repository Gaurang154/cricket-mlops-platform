"""HTTP API for the platform.

    uvicorn api:create_app --factory --port 8000

POST /predict   score a player (send an X-Client-Id header to identify yourself)
GET  /health    liveness + which model version is serving
GET  /lineage   full lineage of the Production model
GET  /metrics   Prometheus metrics: traffic, blocks, drift, fairness
"""
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from mlplatform import Platform


class PlayerStats(BaseModel):
    batting_average: float
    strike_rate: float
    boundary_pct: float
    away_average: float
    group: Optional[int] = None  # protected attribute: logged for fairness checks, never a model input


def create_app(platform: Platform = None) -> FastAPI:
    platform = platform or Platform(os.environ.get("MLOPS_HOME", ".mlops"))
    platform.bootstrap()  # first boot: train + gate + approve a model so there is something to serve
    serving = platform.serve()
    app = FastAPI(title="Cricket MLOps Platform")

    @app.post("/predict")
    def predict(stats: PlayerStats, x_client_id: str = Header(default="anonymous")):
        x = [stats.batting_average, stats.strike_rate, stats.boundary_pct, stats.away_average]
        out = serving.predict(x_client_id, x, group=stats.group)
        if "error" in out:
            status = 429 if out["error"] == "rate_limited" else 422
            raise HTTPException(status_code=status, detail=out["error"])
        return out

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": serving.prod_info["version"]}

    @app.get("/lineage")
    def lineage():
        return platform.lineage()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        report = platform.monitor()
        lines = [f"mlops_{name} {value}" for name, value in sorted(serving.counters.items())]
        lines += [f"mlops_model_version {report['model_version']}",
                  f"mlops_monitored_decisions {report['n']}",
                  f"mlops_drift_detected {int(report['drift'])}",
                  f"mlops_fairness_gap {report['fairness_gap'] or 0}"]
        return "\n".join(lines) + "\n"

    return app
