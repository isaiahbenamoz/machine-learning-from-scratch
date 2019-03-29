import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data import generate


class TransformationVisualizer:
    """
    A class to organize grid transformations and create animations for them.
    """
    def __init__(self, transformations, grid, lines):
        """
        :param transformations: a list of transformations to animate
        :param grid: a 3D numpy array of grid lines
        :param lines: a 3D numpy array of lines
        """

        # initialize the transformations
        self.transformations = transformations

        # initialize the grid, lines and points
        self.grid = grid
        self.lines = lines

    @staticmethod
    def initialize_grid(start=-1.0, stop=1.0, num_lines=10, points_per_line=50):
        """ Creates grid lines.
        :param start: the minimum value where grid lines are created
        :param stop: the maximum value where grid lines are created
        :param num_lines: the number of grid lines
        :param points_per_line: the number of points per line
        :return: the created grid
        """

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
        """ Creates frames for a linear transformation.
        :param W: the weight matrix
        :param points: the points to be transformed
        :param num_frames: the number of frames
        :return: the generated frames
        """

        # if there are no points
        if points is None:

            # return a None array
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
        """ Creates frames a shift transformation.
        :param b: the bias vector
        :param points: the points to be transformed
        :param num_frames: the number of frames
        :return: the generated frames
        """

        # if the points array is None
        if points is None:

            # return a None array
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
        """ Creates frames for a nonlinear transformation.
        :param function: the function to be animated
        :param points: the points to be transformed
        :param num_frames: the number of frames
        :return: the generated frames
        """

        # if the points array is None
        if points is None:

            # return a None array
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
            frames.append((1 - i) * points + i * final_points)

        # return the frames
        return frames

    @staticmethod
    def create_plot(title, plot_range, x_ticks, y_ticks):
        """ Creates the plot to be animated.
        :param title: the title of the plot
        :param plot_range: the area of the plot shown
        :param x_ticks: the tick marks for the x axis
        :param y_ticks: the tick marks for the y axis
        :return: the figure and axis of the plot
        """

        # get the plot's figure and axis
        figure, axis = plt.subplots()

        # set the title of the plot
        plt.title(title)
        axis.set_aspect('equal', 'box')

        # set the limits of the plot
        plt.ylim(*plot_range)
        plt.xlim(*plot_range)

        # ticks are provided
        if x_ticks is not None:
            # plot the ticks
            plt.xticks(x_ticks)

        # if ticks are provided
        if y_ticks is not None:
            # plot the ticks
            plt.yticks(y_ticks)

        # return the figure and axis
        return figure, axis

    @staticmethod
    def plot_line(axis, line, color, lw=0.5):
        """ Plot a line of points.
        :param axis: the axis to plot the line on
        :param line: the line of points
        :param color: the color of the line
        :param lw: the width of the line
        :return: the plotted line
        """

        # plot the grid line
        plotted_line, = axis.plot(line[0], line[1], color=color, lw=lw)

        # return the plotted grid
        return plotted_line

    def get_frames(self, num_frames):
        """ Create the frames for the animation.
        :param num_frames: the number of frames
        :return: the grids to be animated
        """

        # initialize the animation grids
        animation_grids = [(self.grid, self.lines)]

        # iterate over each transformation for the grid
        for transformation in self.transformations:

            # if the transformation is a function
            if callable(transformation):
                grids = self.nonlinear_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.nonlinear_transformation(transformation, animation_grids[-1][1], num_frames)

            # if the transformation is a shift
            elif transformation.shape == (2, 1):
                grids = self.shift_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.shift_transformation(transformation, animation_grids[-1][1], num_frames)

            # if the transformation is linear
            else:
                grids = self.linear_transformation(transformation, animation_grids[-1][0], num_frames)
                lines = self.linear_transformation(transformation, animation_grids[-1][1], num_frames)

            # add the grid line and lines to the animation grids
            animation_grids.extend([(grid, line) for grid, line in zip(grids, lines)])

        return animation_grids

    def animate(self, title=None, plot_range=(-1.0, 1.0), x_ticks=None, y_ticks=None,
                num_frames=30, interval=75, dpi=150, still_frames=5, save_loc='test.gif'):
        """ Create and save the animation.
        :param title: the title of the plot
        :param plot_range: the dimensions of the plot
        :param num_frames: the number of frames per transformation
        :param interval: the amount of time between each frames in ms
        :param dpi: the pixel density of the plot
        :param still_frames: the number of still frames at the beginning and end of the animation
        :param save_loc: the save location
        :param x_ticks: the tick marks for the x axis
        :param y_ticks: the tick marks for the y axis
        """

        # create the frames to be animated
        frames = self.get_frames(num_frames)

        # create the plot
        figure, axis = self.create_plot(title, plot_range, x_ticks, y_ticks)

        # initialize the plotted grid list
        plotted_grid = []

        # if the grid is defined
        if self.grid is not None:

            # iterate over each line in the grid
            for line in self.grid:

                # add the line to the plotted grid list
                plotted_grid.append(self.plot_line(axis, line, 'grey'))

        # initialize plotted lines list
        plotted_lines = []

        # if the lines are defined
        if self.lines is not None:

            # iterate over each line in this list of lines
            for line in self.lines:

                # add the plotted line to the list
                plotted_lines.append(self.plot_line(axis, line, None, lw=2.0))

        # create the update function
        def update(curr_grid):

            # if the plotted grid is not empty
            if plotted_grid:

                # iterate over each line
                for line, plotted_line in zip(curr_grid[0], plotted_grid):

                    # update the x and y data
                    plotted_line.set_xdata(line[0])
                    plotted_line.set_ydata(line[1])

            # if the plotted lines is not empty
            if plotted_lines:

                # iterate over each line
                for line, plotted_line in zip(curr_grid[1], plotted_lines):

                    # update the x and y data
                    plotted_line.set_xdata(line[0])
                    plotted_line.set_ydata(line[1])

        # create the final frames
        final_frames = [frames[0] for _ in range(still_frames)] + frames[::2] + \
                       [frames[-1] for _ in range(still_frames)]

        # create animation to save
        animation = FuncAnimation(figure, update, frames=final_frames, interval=interval)

        # save the animation if specified
        if save_loc:

            # save the gif
            animation.save(save_loc, dpi=dpi, writer='imagemagick')
