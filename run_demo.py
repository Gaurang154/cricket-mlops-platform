"""One command, one full lap of the MLOps loop:

    python run_demo.py

raw data -> tracked training -> gated registry -> approval -> defended + explained serving
-> monitoring -> drift -> automatic retraining -> canary -> approval -> lineage -> rollback
"""
import json
import shutil

from mlplatform import Platform
from mlplatform.data import Dataset

HOME = ".mlops-demo"


def step(n, title):
    print(f"\n\033[1m[{n}] {title}\033[0m")


def show(obj):
    print(json.dumps(obj, indent=2, default=str))


def send_traffic(serving, data_version, n=300):
    """Simulate analysts scoring players from one season of data."""
    ds = Dataset(data_version)
    for i in range(n):
        serving.predict(f"analyst-{i}", ds.X[i].tolist(), group=int(ds.group[i]))


def main():
    shutil.rmtree(HOME, ignore_errors=True)  # fresh start every run
    platform = Platform(HOME)

    step(1, "Train on data version innings@2024.06 -> tracked run -> promotion gate")
    show(platform.train("innings@2024.06"))

    step(2, "Human approval moves it Staging -> Production")
    print("Production is now v", platform.approve("head-of-analytics"), sep="")

    step(3, "Serve it: validated, rate-limited, explained, audited")
    serving = platform.serve()
    show(serving.predict("coach", [53.6, 131.0, 0.61, 48.0], group=1))
    print("Absurd input  ->", serving.predict("coach", [500, 131.0, 0.61, 48.0]))
    print("Request flood ->", [serving.predict("scraper", [40, 120, 0.4, 30]).get("error", "ok")
                               for _ in range(8)])

    step(4, "Same-season traffic (innings@2024.09) -> monitor")
    send_traffic(serving, "innings@2024.09")
    show(platform.monitor())
    print("Retrain needed? ->", platform.maybe_retrain()["retrained"])

    step(5, "New season arrives (innings@2025.01): batting averages jump -> monitor")
    send_traffic(serving, "innings@2025.01")
    show(platform.monitor())

    step(6, "Closed loop: drift -> automatic retrain on newest data -> same gates")
    result = platform.maybe_retrain()
    show(result["gate"])
    assert result["retrained"] and result["gate"]["decision"] == "PROMOTED", result

    step(7, "Canary: 20% of traffic goes to the new Staging model, then approval")
    canary = platform.serve(canary_fraction=0.2)
    send_traffic(canary, "innings@2025.01")
    print("Traffic split:", dict(canary.counters))
    print("Production is now v", platform.approve("head-of-analytics"), sep="")

    step(8, "Lineage: Production model -> run -> data version -> seed -> metrics")
    lineage = platform.lineage()
    lineage.pop("history")
    show(lineage)

    step(9, "Something looks wrong? Roll back in one call")
    print("Production is now v", platform.rollback(by="on-call"), sep="")

    step(10, "Audit trail")
    events = [e["event"] for e in platform.audit.read()]
    print({e: events.count(e) for e in dict.fromkeys(events)}, f"-> {platform.audit.path}")

    print("\n\033[1;32mFull MLOps loop completed.\033[0m")


if __name__ == "__main__":
    main()
