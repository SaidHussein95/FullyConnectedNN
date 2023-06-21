import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        return np.where(label_tensor == 1, -np.log(self.prediction_tensor+np.finfo(float).eps), 0).sum()

    def backward(self, label_tensor):
        return np.where(label_tensor == 1, -1 / self.prediction_tensor+np.finfo(float).eps, 0)
