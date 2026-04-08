import numpy as np
from .config import DISTANCE_MODEL_PATH


class DistanceEstimator:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

        try:
            self.numpy_weights = np.load(DISTANCE_MODEL_PATH)
            self.weights = np.asarray(self.numpy_weights["weights"], dtype=np.float32).reshape(-1)

            if "bias" in self.numpy_weights:
                self.bias = float(np.asarray(self.numpy_weights["bias"]).reshape(-1)[0])
        except FileNotFoundError:
            print("Distance model weights not found.")
        except KeyError as e:
            print(f"Distance model weights missing key: {e}")

    def estimate_distance(self, pixel_width, pixel_height):
        if self.weights is None or self.weights.size < 2:
            return -1.0

        features = np.array([pixel_width, pixel_height], dtype=np.float32)
        distance = float(np.dot(self.weights[:2], features) + self.bias)
        return max(distance, 0.0)
