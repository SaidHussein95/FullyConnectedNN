import numpy as np
from Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()  # Super_constructor

    def forward(self, input_tensor):
        sub = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))  # Subtracts each row with its max value, numerical stability
        self.softmax_forward = sub / np.sum(sub, axis=1, keepdims=True) # returns sum of each row and keeps same dims
        return self.softmax_forward

    def backward(self, error_tensor):
        subtract = error_tensor - np.sum(error_tensor * self.softmax_forward, axis=1, keepdims=True)
        softmax_backward = self.softmax_forward * subtract
        return softmax_backward

