from deep_learning.neural_network import NeuralNetwork
from data import generate
from math import tanh, cos, sin, pi, sqrt
import numpy as np
from visualization.transformations import TransformationVisualizer
import matplotlib.pyplot as plt


e = pi / 1.15

x1 = generate.spiral(end=e, num=10000)
x1[0] += sqrt(3) / 4
x1[1] += 1 / 4

t2 = 2 * pi / 3
x2 = np.array([[cos(t2), -sin(t2)], [sin(t2), cos(t2)]]) @ generate.spiral(end=e, num=10000)
x2[0] += -sqrt(3) / 4
x2[1] += 1 / 4

t3 = 4 * pi / 3
x3 = np.array([[cos(t3), -sin(t3)], [sin(t3), cos(t3)]]) @ generate.spiral(end=e, num=10000)
x3[0] += 0.0
x3[1] += -1 / 2


plt.plot(x1[0], x1[1])
plt.plot(x2[0], x2[1])
plt.plot(x3[0], x3[1])
plt.axis('equal')
plt.show()

x = np.concatenate((x1, x2, x3), axis=1)

y1 = np.zeros((2, x1.shape[1]))
y1[0] += 1.0
y1[1] += 1.0

y2 = np.zeros((2, x1.shape[1]))
y2[0] += -1.0
y2[1] += -1.0

y3 = np.zeros((2, x1.shape[1]))
y1[0] += 0.0
y1[1] += 0.0

y = np.concatenate((y1, y2, y3), axis=1)

L = 6

nn = NeuralNetwork([2 for _ in range(L)], [None] + ['tanh' for _ in range(L - 1)])
nn.train(x, y, epochs=5000, learning_rate=0.75)

ws = [nn.parameters['w' + str(i)] for i in range(1, L)]
bs = [nn.parameters['b' + str(i)] for i in range(1, L)]

t = []
for w, b in zip(ws, bs):
    t.append(w)
    t.append(b)
    t.append(tanh)

grid_ = TransformationVisualizer.initialize_grid(start=-5.0, stop=5.0, num_lines=80, points_per_line=500)
lines_ = [x1, x2, x3]

gt = TransformationVisualizer(t, grid_, lines_)
gt.animate(plot_range=(-3, 3), num_frames=20, save_loc='./results/3_spiral_neural_network.gif')