import numpy as np
from Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()  # Super_constructor

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        derivative_tensor = np.where(self.input_tensor > 0, error_tensor, 0)
        return derivative_tensor






