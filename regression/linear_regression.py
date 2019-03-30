import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


class LinearRegression:
    """
    An implementation of linear regression using numpy.
    """

    def __init__(self, learning_rate=0.01):
        """
        :param learning_rate : float
            The rate at which our model tends to adjust it's parameters
        """
        self.learning_rate = learning_rate
        self.W = None
        self.b = 0.0
        self.history = []

    def predict(self, x):
        """ A method that runs our linear model
        x : np.ndarray (num_dimensions, num_examples)
            The m input examples for which we would like to predict a y value
        """
        return self.W @ x + self.b

    def cost(self, x, y):
        """
        :param x: np.ndarray (num_dimensions, num_examples)
            The m input examples for which we would like to predict a y value
        :param y: np.ndarray (num_examples, 1)
            The ground truth results
        :return: the cost of our model for a given input (x, y)
        """
        return ((self.predict(x) - y) ** 2).mean(axis=1, keepdims=True)

    def gradient(self, x, y):
        """
        :param x: np.ndarray (num_dimensions, num_examples)
            The m input examples for which we would like to predict a y value
        :param y: np.ndarray (num_examples, 1)
            The ground truth results
        """
        y_hat = self.predict(x)
        dW = ((y_hat - y) * x).mean()
        db = (y_hat - y).mean()
        return dW, db

    def fit(self, x, y, epochs=1000):
        """
        :param x: np.ndarray (num_dimensions, num_examples)
            The m input examples for which we would like to predict a y value
        :param y: np.ndarray (num_examples, 1)
            The ground truth results
        :param epochs: int
            The number of times our model iterates over the entire training set
        """

        # initialize W
        self.W = np.zeros((x.shape[0], 1))

        # iterate over the number of epochs
        for i in range(epochs):

            # add the weights and bias to the model history
            self.history.append(self.predict(x))

            # print one of ten cost evaluations
            if i % (epochs // 10) == 0:
                print('Epoch #' + str(i), 'cost =', self.cost(x, y)[0][0])

            # compute the gradient with respect to w and b
            dw, db = self.gradient(x, y)

            # adjust w and b as to minimize the cost function
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def solve(self, x, y):
        """
        :param x: np.ndarry (num_dimensions, num_examples)
            The m input examples for which we would like to predict a y value
        :param y: np.ndarray (num_examples, 1)
            The ground truth results
        """
        # extract the shape of x
        m, n = x.shape

        # concatenate a row of ones with our training data
        x = np.concatenate((np.ones((1, n)), x), axis=0)

        # plug in x and y to the normal equation
        w = np.linalg.inv((x @ x.T)) @ (x @ y.T)

        # extract the w and b values from the calculation
        self.W, self.b = w[1:, ].reshape(1, m), float(w[0, ])

    def animate(self, x, y, show=True, save_loc=None, dpi=150):
        """
        Animate the model training.
        :param x: the input we are using to predict y
        :param y: the ground truth values associated with our training data
        :param show: if this true, then the plot is shown
        :param save_loc: the location to save the plot
        :param dpi: the pixel density of the plot
        """

        # get the subplots for plotting
        fig, ax = plt.subplots()

        # set the aspect ratio for the graph to equal
        ax.set_aspect('equal', adjustable='box')

        # set the x and y limits
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

        # set the x and y ticks
        plt.xticks(np.linspace(-1, 1, 5))
        plt.yticks(np.linspace(-1, 1, 5))

        # Plot a scatter that persists (isn't redrawn) and the initial line.
        sns.scatterplot(x[0], y[0])

        # initialize a line with our first prediction
        line, = ax.plot(x[0], self.history[0][0], 'red')

        def update(pred):
            # Update the line and the axes (with a new xlabel). Return a tuple of
            # "artists" that have to be redrawn for this frame.
            line.set_ydata(pred[0])
            return line, ax

        # create the animation
        anim = FuncAnimation(fig, update, frames=self.history[::4], interval=50)

        # if there is a save location, save the animation
        if save_loc:
            anim.save(save_loc, dpi=dpi, writer='imagemagick')

        # if show is true, show the plot
        if show:
            plt.show()

    def plot_2d(self, x, y, show=True, save_loc=None, dpi=150):
        """
        :param x: the input we are using to predict y
        :param y: the ground truth values associated with our training data
        :param show: if this true, then the plot is shown
        :param save_loc: the location to save the plot
        :param dpi: the pixel density of the plot
        """
        # create a new figure for saving
        fig = plt.figure()

        # create a scatter plot
        plt.scatter(x, y)

        # plot the line of prediction
        plt.plot(x[0], self.predict(x)[0], 'r-')

        # if show is true, show the plot
        if show:
            plt.show()

        # if the save location is specified, save the plot there
        if save_loc:
            fig.savefig(save_loc, dpi=dpi)
