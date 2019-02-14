import numpy as np


class SupportVectorMachine:

    def __init__(self):
        self.W = None
        self.b = 0.0

    def cost(self):
        pass

    def fit(self):
        pass

    def predict(self, x):
        return np.sign(self.W @ x + self.b)
