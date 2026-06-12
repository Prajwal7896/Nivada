import json
import time
import os

LOG_FILE = "mlops/monitoring/predictions_log.jsonl"


def track_prediction(confidence, extra=None):
    os.makedirs("mlops/monitoring", exist_ok=True)

    log_entry = {
        "timestamp": time.time(),
        "confidence": float(confidence),
        "extra": extra or {}
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry