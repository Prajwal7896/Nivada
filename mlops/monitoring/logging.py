import datetime
import json
import os

LOG_FILE = "mlops/monitoring/events.log"


def log_event(event, data):
    """
    Logs events for MLOps monitoring + debugging + retraining analysis
    """

    os.makedirs("mlops/monitoring", exist_ok=True)

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": event,
        "data": data
    }

    # 1. Console output (for dev)
    print(f"[{log_entry['timestamp']}] {event} → {data}")

    # 2. Persistent log file (for ML monitoring)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry