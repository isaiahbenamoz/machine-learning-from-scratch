from data import generate
from deep_learning.neural_network import NeuralNetwork
import numpy as np

x = generate.normal(100, 20, 1.5, 1.45)
x_train, y_train = x[:, 0:5].T, x[:, 10:12].T
y_train = (1 / (1 + np.exp(-y_train)))

nn = NeuralNetwork([5, 10, 20, 50, 50, 20, 10, 2], [None, 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
                   'mean_squared_error')

nn.train(x_train, y_train, learning_rate=0.01)
nn.plot_model(line_width=0.45, node_size=10)
