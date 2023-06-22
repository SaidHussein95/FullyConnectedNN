# Deep Learning Framework

This repository contains a basic implementation of a deep learning framework. The framework includes various components such as optimizers, layers, and loss functions that can be used to build and train neural networks.

## Optimizers

### Stochastic Gradient Descent (SGD)

The `Sgd` class implements the basic Stochastic Gradient Descent optimizer. It performs gradient descent updates based on the provided learning rate.

## Layers

### Base Layer

The `BaseLayer` class serves as the base class for all layers in the framework. It provides the fundamental operations required during training and testing. Each layer must implement the `forward` and `backward` methods.

### Fully Connected Layer

The `FullyConnected` class is a trainable layer that performs a linear operation on its input. It inherits from the `BaseLayer` class and implements the `forward` and `backward` methods. It also includes a property for the optimizer.

### Rectified Linear Unit (ReLU)

The `ReLU` class implements the Rectified Linear Unit activation function. It inherits from the `BaseLayer` class and provides the `forward` and `backward` methods.

### SoftMax Layer

The `SoftMax` class implements the SoftMax activation function, which is commonly used for classification tasks. It inherits from the `BaseLayer` class and includes the `forward` and `backward` methods.

## Loss Functions

### Cross Entropy Loss

The `CrossEntropyLoss` class implements the Cross Entropy Loss function, typically used in classification tasks. It computes the loss value and provides the `forward` and `backward` methods.

## Neural Network

The `NeuralNetwork` class represents a neural network architecture. It manages the forward and backward propagation through the layers, as well as training and testing functionalities. The network can be constructed by appending layers to it, and it supports optimization using a specified optimizer.

## Install the required dependencies:
pip install numpy
from Base import BaseLayer
from copy import deepcopy

## Contributing
Contributions to this deep learning framework are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Together, we can enhance the functionality and usability of the framework.
