import mlflow

def log_metrics(confidence, latency):
    with mlflow.start_run():
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("latency", latency)