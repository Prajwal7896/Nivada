import os

def retrain():
    print("🔁 Retraining model triggered...")
    os.system("python mlops/pipelines/train_pipeline.py")


if __name__ == "__main__":
    retrain()