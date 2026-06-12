import os
import subprocess
import time

def train_model():
    print("🚀 Starting training pipeline...")

    start = time.time()

    result = subprocess.run(["python", "model.py"], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Training successful")
    else:
        print("❌ Training failed")
        print(result.stderr)

    print(f"⏱ Time taken: {time.time() - start:.2f}s")


if __name__ == "__main__":
    train_model()