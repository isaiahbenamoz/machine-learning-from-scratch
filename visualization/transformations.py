import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data import generate
from math import tanh, cosh


class TransformationVisualizer:
    """
    A class to organize grid transformations and create animations for them.
    """
    def __init__(self, transformations, grid, lines, points):

        # initialize the transformations
        self.transformations = transformations

        # initialize the grid, lines and points
        self.grid = grid
        self.lines = lines
        self.points = points

    @staticmethod
    def initialize_grid(start=-1.0, stop=1.0, num_lines=10, points_per_line=50):

        # initialize the grid
        grid = []

        # iterate over the y values
        for y_value in np.linspace(start, stop, num_lines):

            # create the x and y coordinates
            x = np.linspace(start, stop, points_per_line).reshape(1, -1)
            y = y_value * np.ones_like(x)

            # save the horizontal line coordinates
            grid.append(np.concatenate((x, y), axis=0))

        # iterate over the x values
        for x_value in np.linspace(start, stop, num_lines):

            # create the x and y values
            y = np.linspace(start, stop, points_per_line).reshape(1, -1)
            x = x_value * np.ones_like(y)

            # save the vertical line coordinates
            grid.append(np.concatenate((x, y), axis=0))

        # return the grid
        return np.array(grid)

    @staticmethod
    def linear_transformation(W, points, num_frames):

        if points is None:
            return [None for _ in range(num_frames)]

        # initialize the frames list
        frames = []

        # iterate over the linear transformation
        for i in generate.sigmoid(num=num_frames):

            # calculate the intermediate W
            W_i = (1 - i) * np.eye(2) + i * W

            # add the frame to the frames list
            frames.append(W_i @ points)

        # return the frames list
        return frames

    @staticmethod
    def shift_transformation(b, points, num_frames):

        if points is None:
            return [None for _ in range(num_frames)]

        # initialize the frames list
        frames = []

        # iterate over the shift transformation
        for i in generate.sigmoid(num=num_frames):

            # add the frame to the frames list
            frames.append(points + i * b)

        # return the frames list
        return frames

    @staticmethod
    def nonlinear_transformation(function, points, num_frames):

        if points is None:
            return [None for _ in range(num_frames)]

        # vectorize the function
        function = np.vectorize(function)

        # get the final points
        final_points = function(points)

        # initialize the frames
        frames = []

        # iterate over each frame
        for i in generate.sigmoid(num=num_frames):

            # save the current frame
            frames.append(points * (1 - i) + i * final_points)

        # return the frames
        return frames

    @staticmethod
    def create_plot(title, plot_range):

        # get the plot's figure and axis
        figure, axis = plt.subplots()

        # set the title of the plot
        plt.title(title)

        # set the limits of the plot
        plt.ylim(*plot_range)
        plt.xlim(*plot_range)

        # return the figure and axis
        return figure, axis

    @staticmethod
    def plot_line(axis, line, color, lw=0.5):

        # plot the grid line
        plotted_line, = axis.plot(line[0], line[1], color=color, lw=lw)

        # return the plotted grid
        return plotted_line

    @staticmethod
    def plot_points(axis, points, color):

        # initialize a list to store lines
        plotted_points = []

        # plot the grid line
        plotted_line, = axis.scatter(points[0], points[1], color=color)

        # save the line
        plotted_points.append(plotted_line)

        # return the plotted grid
        return plotted_points

    def get_frames(self, num_frames):

        # initialize the animation grids
        animation_grids = [(self.grid, self.lines, self.points)]

        # iterate over each transformation for the grid
        for transformation in self.transformations:

            # if the transformation is a function
            if callable(transformation):
                grids = self.nonlinear_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.nonlinear_transformation(transformation, animation_grids[-1][1], num_frames)
                points = self.nonlinear_transformation(transformation, animation_grids[-1][2], num_frames)

            # if the transformation is a shift
            elif transformation.shape == (2, 1):
                grids = self.shift_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.shift_transformation(transformation, animation_grids[-1][1], num_frames)
                points = self.shift_transformation(transformation, animation_grids[-1][2], num_frames)

            # if the transformation is linear
            else:
                grids = self.linear_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.linear_transformation(transformation, animation_grids[-1][1], num_frames)
                points = self.linear_transformation(transformation, animation_grids[-1][2], num_frames)

            animation_grids.extend([(grid, line, point) for grid, line, point in zip(grids, lines, points)])

        return animation_grids

    def animate(self, title=None, plot_range=(-1.0, 1.0),
                num_frames=30, dpi=150, save_loc='test.gif'):

        frames = self.get_frames(num_frames)

        # creat the plot
        figure, axis = self.create_plot(title, plot_range)

        # initialize the plotted grid list
        plotted_grid = []

        if self.grid is not None:
            # iterate over each line in the grid
            for line in self.grid:

                # add the line to the plotted grid list
                plotted_grid.append(self.plot_line(axis, line, 'grey'))

        # initialize plotted lines list
        plotted_lines = []

        if self.lines is not None:
            # iterate over each line in this list of lines
            for line in self.lines:

                # add the plotted line to the list
                plotted_lines.append(self.plot_line(axis, line, None))

        # initialize the plotted points list
        plotted_points = []

        if self.points is not None:
            # iterate over each point in the points list
            for points in self.points:

                # add the scatter plot to the list
                plotted_points.append(self.plot_points(axis, points, None))

        # create the update function
        def update(curr_grid):

            # iterate over each line
            if plotted_grid:
                for line, plotted_line in zip(curr_grid[0], plotted_grid):

                    # update the x and y data
                    plotted_line.set_xdata(line[0])
                    plotted_line.set_ydata(line[1])

            if plotted_lines:
                for line, plotted_line in zip(curr_grid[1], plotted_lines):

                    # update the x and y data
                    plotted_line.set_xdata(line[0])
                    plotted_line.set_ydata(line[1])

            if plotted_points:
                for points, plotted_points_ in zip(curr_grid[2], plotted_points):

                    # update the x and y data
                    plotted_points_.set_xdata(points[0])
                    plotted_points_.set_ydata(points[1])

        # create animation to save
        animation = FuncAnimation(figure, update, frames=frames, interval=50)

        # save the animation if specified
        if save_loc:
            animation.save(save_loc, dpi=dpi, writer='imagemagick')


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

    nn = NeuralNetwork([2, 2, 2, 2], [None, 'tanh', 'tanh', 'tanh'])
    nn.train(x, y)

    ws = [nn.parameters['w' + str(i)] for i in range(1, 4)]
    bs = [nn.parameters['b' + str(i)] for i in range(1, 4)]

    t = []
    for w, b in zip(ws, bs):
        t.append(w)
        t.append(b)
        t.append(tanh)

    grid_ = TransformationVisualizer.initialize_grid()
    lines_ = [x1, x2]
    points_ = None

    gt = TransformationVisualizer(t, grid_, lines_, points_)
    gt.animate()
