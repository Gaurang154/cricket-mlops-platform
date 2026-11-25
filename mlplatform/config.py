"""Platform-wide settings. Every component reads from here, so policy lives in one place."""

SEED = 1983  # one fixed seed everywhere -> every run is reproducible
MODEL_NAME = "PlayerFormClassifier"
FEATURES = ["batting_average", "strike_rate", "boundary_pct", "away_average"]

# --- Governance policy (the "paved road" defaults) ---------------------------
MAX_FAIRNESS_GAP = 0.15  # max equal-opportunity gap a model may have to be promoted
LIVE_FAIRNESS_GAP = 0.20  # max selection-rate gap allowed on live / canary traffic
DRIFT_ALPHA = 0.001  # KS-test p-value below this on any feature = drift (strict: 4 tests at once)
MIN_MONITOR_SAMPLES = 100  # don't judge drift/fairness on fewer decisions than this
MONITOR_WINDOW = 200  # judge only the most recent N decisions
RATE_LIMIT_PER_SEC = 5  # requests per second per client
OOD_MARGIN = 0.2  # inputs further than 20% outside the training range are rejected
