import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import exp, tanh, cosh


class GridTransformer:
    """
    A class to organize grid transformations and create animations for them.
    """
    def __init__(self, transformations, lines=None, plot_range=(-1.0, 1.0), line_range=(-1.0, 1.0),
                 num_lines=10, points_per_line=100):
        """
        :param transformations:
        :param plot_range:
        :param line_range:
        """
        self.transformations = transformations
        self.plot_range = plot_range
        self.line_range = line_range
        self.num_lines = num_lines
        self.points_per_line = points_per_line

        # initialize the grid
        self.animation_grids = []
        self.animation_lines = []
        self.initialize_grid()

        # initialize the plot
        self.figure, self.axis = plt.subplots()
        self.plotted_lines = [[], []]
        self.initialize_plot()

    def initialize_grid(self):

        # initialize the starting grid
        starting_grid = []

        # iterate over the y values
        for y_value in np.linspace(*self.line_range, self.num_lines):

            # create the x and y coordinates
            x = np.linspace(*self.line_range, self.points_per_line).reshape(1, -1)
            y = y_value * np.ones_like(x)

            # save the horizontal line coordinates
            starting_grid.append(np.concatenate((x, y), axis=0))

        # iterate over the x values
        for x_value in np.linspace(*self.line_range, self.num_lines):

            # create the x and y values
            y = np.linspace(*self.line_range, self.points_per_line).reshape(1, -1)
            x = x_value * np.ones_like(y)

            # save the vertical line coordinates
            starting_grid.append(np.concatenate((x, y), axis=0))

        # initialize the starting functions
        starting_functions = []

        # if extra functions are defined
        if self.functions is not None:

            # iterate over each function
            for function, maximum, minimum in self.functions:

                # create the function line
                function_line = np.array([[i for i in np.linspace(minimum, maximum, self.points_per_line)],
                                          [function(i) for i in np.linspace(minimum, maximum, self.points_per_line)]])

                # add the function line to the functions grid
                starting_functions.append(function_line)

        # save the initial grid
        self.animation_grids.append((starting_grid, starting_functions))

    def linear_transformation(self, W, num_frames):
        for i in self.sigmoid_range(num=num_frames):
            W_i = (1 - i) * np.eye(2) + i * W
            self.animation_grids.append([W_i @ line for line in self.animation_grids[-1]])

    def shift_transformation(self, b, num_frames):
        for i in self.sigmoid_range(num=num_frames):
            self.animation_grids.append([line + b * i for line in self.animation_grids[-1]])

    def nonlinear_transformation(self, function, num_frames):
        function = np.vectorize(function)

        # get the initial and final grids
        start_grid = self.animation_grids[-1]
        final_grid = [function(line) for line in self.animation_grids[-1]]

        # iterate over each frame
        for i in self.sigmoid_range(num=num_frames):

            # create a current grid
            curr_grid = []

            # iterate over the starting and final grids
            for start_line, final_line in zip(start_grid, final_grid):
                curr_grid.append(start_line * (1 - i) + i * final_line)

            # add the current grid to the animation grids
            self.animation_grids.append(curr_grid)



    def initialize_plot(self, title=''):
        # set the title of the plot
        plt.title(title)

        # set the limits of the plot
        plt.ylim(*self.plot_range)
        plt.xlim(*self.plot_range)

        # iterate over each line in the initial grid
        for line in self.animation_grids[0][0]:

            # plot the line
            plotted_line, = self.axis.plot(line[0], line[1], color='blue', lw=0.5)

            # save the line
            self.plotted_lines[0].append(plotted_line)

        # iterate over the extra functions
        for function in self.animation_grids[0][1]:

            # plot the function
            plotted_function, = self.axis.plot(function[0], function[1], lw=1.0)

            # save the function
            self.plotted_lines[1].append(plotted_function)

    def animate_transformations(self, num_frames=30, dpi=150, save_loc='test.gif'):

        # iterate over each transformation
        for transformation in self.transformations:

            # if the transformation is a function
            if callable(transformation):
                self.nonlinear_transformation(transformation, num_frames)

            # if the transformation is a shift
            elif transformation.shape == (2, 1):
                self.shift_transformation(transformation, num_frames)

            # if the transformation is linear
            elif transformation.shape == (2, 2):
                self.linear_transformation(transformation, num_frames)

        # create the update function
        def update(curr_grid):
            # iterate over each line
            for line, plotted_line in zip(curr_grid[0], self.plotted_lines[0]):

                # update the x and y data
                plotted_line.set_xdata(line[0][0])
                plotted_line.set_ydata(line[0][1])

            for line, plotted_line in zip(curr_grid[1], self.plotted_lines[1]):

                # update the x and y data
                plotted_line.set_xdata(line[1][0])
                plotted_line.set_ydata(line[1][1])

        # create animation to save
        animation = FuncAnimation(self.figure, update, frames=self.animation_grids, interval=50)

        # save the animation if specified
        if save_loc:
            animation.save(save_loc, dpi=dpi, writer='imagemagick')

    @staticmethod
    def sigmoid_range(start=-5.0, end=5.0, num=50):
        for i in np.linspace(start, end, num):
            yield 1 / (1 + exp(-i))


if __name__ == '__main__':

    from deep_learning.neural_network import NeuralNetwork
    from data import generate
    from math import tanh

    x1 = np.array([[i for i in np.linspace(-1, -0.5, 20)],
                  [cosh(i) for i in np.linspace(-1, -0.5, 20)]])

    x2 = np.array([[i for i in np.linspace(0.5, 1, 20)],
                   [cosh(i) for i in np.linspace(0.5, 1, 20)]])

    x = np.concatenate((x1, x2), axis=1)

    y1 = np.zeros((2, x1.shape[1]))
    y2 = np.ones((2, x2.shape[1]))

    y = np.concatenate((y2, y1), axis=1)

    lines = [(tanh, -1.0, -0.5), ]

    nn = NeuralNetwork([2, 2, 2, 2, 2], [None, 'relu', 'relu', 'relu', 'relu'])
    nn.train(x, y)

    print(nn.parameters)

    ws = [nn.parameters['w' + str(i)] for i in range(1, 5)]
    bs = [nn.parameters['b' + str(i)] for i in range(1, 5)]

    t = []
    for w, b in zip(ws, bs):
        t.append(w)
        t.append(b)
        t.append(lambda x: max(x, 0))

    gt = GridTransformer(t, line_range=(-5.0, 5.0), num_lines=50)
    gt.animate_transformations()
