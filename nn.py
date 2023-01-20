import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        
        self.layer1 = layer_sizes[0] # num of neurons in input layer
        self.layer2 = layer_sizes[1] # num of neurons in hidden layer
        self.layer3 = layer_sizes[2] # num of neurons in output layer

        self.w1 = np.random.normal(size=(self.layer2, self.layer1))
        self.w2 = np.random.normal(size=(self.layer3, self.layer2))

        self.b1 = np.zeros((self.layer2, 1))
        self.b2 = np.zeros((self.layer3, 1))

    def activation(self, x):
        
        # TODO
        return 1 / (np.exp(-x) + 1) #sigmoid function

    def forward(self, x):
        
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        x1 = self.activation(self.w1 @ x + self.b1)
        y = self.activation(self.w2 @ x1 + self.b2)
        return y
