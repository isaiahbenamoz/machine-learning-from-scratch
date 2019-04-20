import numpy as np
from scipy.special import softmax


class RecurrentNeuralNetwork:

    def __init__(self, nodes, activation='tanh', loss='mean_squared_error'):
        self.nodes = nodes
        self.activations = activation
        self.loss = loss
        self.parameters = {}

    def train(self):
        pass

    def forward_propagate(self):
        pass

    def initialize_parameters(self):
        self.parameters['w'] = np.random.randn(self.nodes[2], self.nodes[1])
        self.parameters['u'] = np.random.randn(self.nodes[2], self.nodes[0])
        self.parameters['v'] = np.random.randn(self.nodes[2], self.nodes[2])

    @staticmethod
    def linear_forward(b, w, h_prev, u, x):
        return w @ h_prev + u @ x + b

    @staticmethod
    def nonlinear_forward(a, activation="tanh"):
        if activation == 'tanh':
            return np.tanh(a)

    @staticmethod
    def output_forward(V, h, c):
        return softmax(V @ h + c)

