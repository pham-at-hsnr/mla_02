import numpy as np 
from .config import DISTANCE_MODEL_PATH


class DistanceEstimator:
    def __init__(self):
        self.numpy_weights = np.load(DISTANCE_MODEL_PATH)

    def estimate_distance(self, pixel_width, pixel_height):
        # TODO implement the distance estimation logic here
        # use self.numpy_weights to access the trained weights of your model
        pass