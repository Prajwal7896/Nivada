import time
from mlops.tracking.mlflow_tracker import log_metrics
from model import compute_metrics 

start = time.time()

# fake values
confidence = 0.87

time.sleep(1)  # simulate latency

latency = time.time() - start

log_metrics(confidence, latency)

print("Logged to MLflow")