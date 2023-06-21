from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []  # loss value after calling train
        self.layers = []
        self.data_layer = None  # contains input data and labels
        self.loss_layer = None  # contains loss and prediction

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)  # calculate the output of the last layer (loss)

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        backward_layers = self.layers[::-1]  # reverse layers
        for layer in backward_layers:
            error = layer.backward(error)
        return error

    def append_layer(self, layer):  # for customizing the optimization setting and save intermediate values
        if layer.trainable:
            optimizer_copy = deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):  # apply forward and backward to calculate loss
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):  # apply input to predict
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor


