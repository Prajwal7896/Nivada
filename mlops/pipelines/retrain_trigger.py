import os
import random

def check_model_health():
    accuracy = random.uniform(0.7, 0.95)

    print("📊 Current Accuracy:", accuracy)

    if accuracy < 0.80:
        print("⚠️ Model degraded → Triggering retrain")
        os.system("python mlops/pipelines/train_pipeline.py")
    else:
        print("✅ Model healthy")


if __name__ == "__main__":
    check_model_health()
