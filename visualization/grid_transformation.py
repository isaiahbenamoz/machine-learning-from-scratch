import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def linear_transformation(W, grid, num_frames):
    """
    :param W: a 2 x 2 weight matrix
    :param num_frames: the number of frames in the transformation
    :return: a num_frames x num_points x 2 matrix that specifies the gird transformation
    """
    frames = []
    for frame in range(num_frames + 1):
        intermediate = np.eye(2) + frame / num_frames * (W - np.eye(2))

        frames.append([intermediate @ line for line in grid])

    return frames



def shift(b):
    pass


def nonlinear_transformation(func):
    pass


def create_grid(minimum, maximum, num_lines, resolution):
    lines = []

    for y_value in np.linspace(minimum, maximum, num_lines):
        x = np.linspace(maximum, minimum, resolution).reshape(1, -1)
        y = y_value * np.ones((1, resolution))
        lines.append(np.concatenate((x, y), axis=0))

    for x_value in np.linspace(minimum, maximum, num_lines):
        x = x_value * np.ones((1, resolution))
        y = np.linspace(maximum, minimum, resolution).reshape(1, -1)
        lines.append(np.concatenate((x, y), axis=0))

    return lines

def plot_grid(grid):
    fig = plt.figure()
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.tick_params()
    for line in grid:
        plt.plot(line[0], line[1], color='blue', linewidth=0.5)

    fig.savefig('test.png')



grid = create_grid(-5, 5, 30, 50)
plot_grid(grid)

W = np.array([[0.5, 1], [1, 0.5]])
frame1, frame2 = linear_transformation(W, grid, 1)
plot_grid(frame2)

