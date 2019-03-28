from deep_learning.neural_network import NeuralNetwork
from data import generate
from math import tanh, cos, sin, pi, sqrt
import numpy as np
from visualization.transformations import TransformationVisualizer
import matplotlib.pyplot as plt
e = pi / 2

x1 = generate.spiral(end=e, num=1000)
x1[0] += sqrt(2) / 4
x1[1] += sqrt(2) / 4

t2 = pi / 2
x2 = np.array([[cos(t2), -sin(t2)], [sin(t2), cos(t2)]]) @ generate.spiral(end=e, num=1000)
x2[0] += -sqrt(2) / 4
x2[1] += sqrt(2) / 4

t3 = pi
x3 = np.array([[cos(t3), -sin(t3)], [sin(t3), cos(t3)]]) @ generate.spiral(end=e, num=1000)
x3[0] += -sqrt(2) / 4
x3[1] += -sqrt(2) / 4

t4 = 3 * pi / 2
x4 = np.array([[cos(t4), -sin(t4)], [sin(t4), cos(t4)]]) @ generate.spiral(end=e, num=1000)
x4[0] += sqrt(2) / 4
x4[1] += -sqrt(2) / 4


plt.plot(x1[0], x1[1])
plt.plot(x2[0], x2[1])
plt.plot(x3[0], x3[1])
plt.plot(x4[0], x4[1])
plt.axis('equal')
plt.show()

x = np.concatenate((x1, x2, x3, x4), axis=1)

y1 = np.zeros((2, x1.shape[1]))
y1[0] += 1.0
y1[1] += 1.0

y2 = np.zeros((2, x1.shape[1]))
y2[0] += -1.0
y2[1] += 1.0

y3 = np.zeros((2, x1.shape[1]))
y1[0] += -1.0
y1[1] += -1.0

y4 = np.zeros((2, x1.shape[1]))
y1[0] += 1.0
y1[1] += -1.0

y = np.concatenate((y1, y2, y3, y4), axis=1)

L = 8

nn = NeuralNetwork([2 for _ in range(L)], [None] + ['tanh' for _ in range(L - 1)])
nn.train(x, y, epochs=10000, learning_rate=0.75)

ws = [nn.parameters['w' + str(i)] for i in range(1, L)]
bs = [nn.parameters['b' + str(i)] for i in range(1, L)]

t = []
for w, b in zip(ws, bs):
    t.append(w)
    t.append(b)
    t.append(tanh)

grid_ = TransformationVisualizer.initialize_grid(start=-5.0, stop=5.0, num_lines=80, points_per_line=500)
lines_ = [x1, x2, x3, x4]

gt = TransformationVisualizer(t, grid_, lines_)
gt.animate(plot_range=(-3, 3), num_frames=20, save_loc='./results/4_spiral_neural_network.gif')