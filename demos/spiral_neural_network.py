from deep_learning.neural_network import NeuralNetwork
from data import generate
from math import tanh
import numpy as np
from visualization.transformations import TransformationVisualizer

x1 = -generate.spiral(num=1000)
x1[0] -= 0.5
x1[1] += 0.0

x2 = generate.spiral(num=1000)
x2[0] += 0.5
x2[1] -= 0.0

x = np.concatenate((x1, x2), axis=1)

y1 = np.zeros((2, x1.shape[1])) - 3
y2 = np.zeros((2, x1.shape[1])) + 3

y = np.concatenate((y2, y1), axis=1)

L = 5

nn = NeuralNetwork([2 for _ in range(L)], [None] + ['tanh' for _ in range(L - 1)])
nn.train(x, y, epochs=10000)

ws = [nn.parameters['w' + str(i)] for i in range(1, L)]
bs = [nn.parameters['b' + str(i)] for i in range(1, L)]

t = []
for w, b in zip(ws, bs):
    t.append(w)
    # t.append(b)
    t.append(tanh)

grid_ = TransformationVisualizer.initialize_grid(start=-5.0, stop=5.0, num_lines=80, points_per_line=500)
lines_ = [x1, x2]

gt = TransformationVisualizer(t, grid_, lines_)
gt.animate(plot_range=(-3, 3), save_loc='./graphs/spiral_neural_network.gif')