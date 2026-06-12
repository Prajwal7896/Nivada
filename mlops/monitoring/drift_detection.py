from mlops.monitoring.monitor import Monitor

monitor = Monitor()

def add_prediction(score):
    monitor.log(score)

    if monitor.drift_detected():
        print("🚨 ALERT: Data drift detected!")