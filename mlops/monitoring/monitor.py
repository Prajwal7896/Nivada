import numpy as np

class Monitor:
    def __init__(self):
        self.scores = []

    def log(self, score):
        self.scores.append(score)

    def drift_detected(self):
        if len(self.scores) < 20:
            return False

        recent = np.mean(self.scores[-10:])
        past = np.mean(self.scores[:10])

        drift = abs(recent - past)

        print("📉 Drift score:", drift)

        return drift > 0.15