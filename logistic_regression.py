import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


class LogisticRegression:
    """
    This class implements logistic regression. Given an input, x, it will predict y, a value between 0 and 1 via
    sigmoid(w @ x + b). This value is very useful for binary classification tasks (e.g. predicting whether a tumor is
    malignant or non-malignant).
    """

    def __init__(self, lambda_=0.0, epochs=1000, learning_rate=0.003):
        """
        :param lambda_: the weight assigned to the regularized portion of the cost function
        :param epochs: the number of iterations that gradient descent will run
        :param learning_rate: a scalar that adjusts how quickly W and b are adjusted
        """
        self.lambda_ = lambda_
        self.W = None
        self.b = 0.0
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = 1e-10
        self.parameters = []

    def cost(self, x, y):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :return: the log loss cost function of our model
        """
        y_pred = self.predict(x)
        cost = -(y * np.log(y_pred + self.epsilon) + (1 - y) * np.log(1 - y_pred + self.epsilon)).mean()
        regularize = (self.lambda_ / 2) * (self.W ** 2).mean()
        return cost + regularize

    def gradient(self, x, y):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :return: the gradient of the cost function with respect to W and b
        """
        y_hat = self.predict(x)
        dW = -(x * (y - y_hat)).mean(axis=1) + (self.lambda_ * self.W)
        db = -(y - y_hat).mean()
        return dW, db

    def fit(self, x, y):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :return: a list of the Ws and bs for all epochs
        """
        # initialize W to the correct dimensions
        self.W = np.zeros((1, x.shape[0]))

        # initialize dW and db
        dW, db = np.zeros((1, x.shape[0])), 0.0

        # run gradient descent for the given number of epochs
        for epoch in range(self.epochs):

            # if the current epoch number is a multiple of the number of epochs / 10
            if epoch % (self.epochs // 10) == 0:
                # print the cost function with out current model
                print('Epoch #' + str(epoch) + ' loss = ' + str(self.cost(x, y)))

            # add the parameters to the parameters list
            self.parameters.append((self.W.copy(), self.b))

            # compute the gradients given our training data
            dW, db = self.gradient(x, y)

            # adjust W and b according to the gradient of the cost function
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        print('\nW:', self.W)
        print('b:', self.b)
        print('c:', self.cost(x, y), '\n')

    def predict(self, x, W=None, b=None):
        """
        :param x: the data which we would like to make a prediction about
        :param W: an optional parameter to specify W for the prediction
        :param b: an optional parameter to specify b for the prediction
        :return: the model's prediction, y_hat, for the given x data
        """
        # if W and b are defined, use them to make the prediction
        if not (W is None or b is None):
            return self.sigmoid(W @ x + b)

        # if W is not defined because the model has not been trained
        if self.W is None:
            # raise a RuntimeError
            raise RuntimeError('Training is required before a prediction can be made.')

        # otherwise, use the parameters from the saved model
        return self.sigmoid(self.W @ x + self.b)

    @staticmethod
    def sigmoid(z):
        """
        :param z: the value which we would like to compute the sigmoid of
        :return: the result of the sigmoid applied to z
        """
        return 1 / (1 + np.exp(-z))

    def plot_1d(self, x, y, show=True, save_loc='', dpi=150):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :param show: a boolean value indicating whether to show the plot
        :param save_loc: the location where the the user would like to save the graph
        :param dpi: the pixel density of the graph being saved
        """
        if x.shape[0] != 1:
            raise RuntimeError('In order to create this plot x must 1 dimensional.')

        # create a figure for saving
        fig = plt.figure()

        # create a scatter plot with the given x and y coordinate data
        sns.scatterplot(x[0], y[0])

        # create the domain over which the logistic regression will be plotted
        domain = np.linspace(np.min(x) - 1, np.max(x) + 1, 500).reshape(1, -1)

        # plot the predicted sigmoid curve
        sns.lineplot(domain[0], self.predict(domain)[0], color='red')

        # label the x and y coordinates and add the title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Logistic Regression')

        # show the plot if specified
        if show:
            plt.show()

        # if the user gave a save location, save the plot
        if save_loc:
            fig.savefig(save_loc, dpi=dpi)

    def plot_2d(self, x, y, show=True, save_loc='', dpi=150):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :param show: a boolean value indicating whether to show the plot
        :param save_loc: the location where the the user would like to save the graph
        :param dpi: the pixel density of the graph being saved
        """
        if x.shape[0] != 2:
            raise RuntimeError('In order to create this plot, x must be two dimensional.')

        # create a figure for saving
        fig = plt.figure()

        import pandas as pd

        df = pd.DataFrame()
        df['x1'] = x[0, :]
        df['x2'] = x[1, :]
        df['y'] = y[0]

        # create a scatter plot with the given x and y coordinate data
        sns.scatterplot(data=df, x='x1', y='x2', hue='y')

        # define the function to find x2 from x1
        def calc_x2(x1_):
            num = -self.W[0][0] * x1_ - self.b
            denom = self.W[0][1]
            return num / denom

        # create the domain over which the logistic regression will be plotted
        x1 = np.linspace(np.min(x[0, :]), np.max(x[0, :]), 500).reshape(1, -1)

        # plot the predicted sigmoid curve
        sns.lineplot(x1[0], calc_x2(x1)[0], color='green')

        # label the x and y coordinates and add the title
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Logistic Regression')

        # show the plot if specified
        if show:
            plt.show()

        # if the user gave a save location, save the plot
        if save_loc:
            fig.savefig(save_loc, dpi=dpi)

    def animate_1d(self, x, y, show=True, save_loc='', dpi=150):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :param show: a boolean value indicating whether to show the animation
        :param save_loc: the location where the the user would like to save the animation
        :param dpi: the pixel density of the graph being saved
        """
        if x.shape[0] != 1:
            raise RuntimeError('In order to create this plot x must 1 dimensional.')

        # create the figure we'll use to create the animation
        fig, ax = plt.subplots()

        # create a scatter plot of our data
        sns.scatterplot(x[0], y[0])

        # label the x and y axises and add a title to the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Logistic Regression')

        # create the domain over which the logistic regression will be plotted
        domain = np.linspace(np.min(x) - 1, np.max(x) + 1, 500).reshape(1, -1)

        # plot an initial red line and save it as a variable
        line, = ax.plot(domain[0], self.predict(domain, *self.parameters[0])[0], 'r-')

        # create an update function that
        def update(parameters):
            line.set_ydata(self.predict(domain, *parameters))
            return line, ax

        # if a save location is defined
        if save_loc:
            anim = FuncAnimation(fig, update, frames=self.parameters[::10], interval=50)
            anim.save(save_loc, dpi=dpi, writer='imagemagick')

        if show:
            plt.show()


if __name__ == '__main__':

    from sklearn.datasets.samples_generator import make_blobs
    x_, y_ = make_blobs(n_samples=200, centers=2, n_features=2, random_state=19, cluster_std=1.0)
    x_ = x_.T.reshape(2, -1)
    y_ = y_.reshape(1, -1)

    lr = LogisticRegression(lambda_=0.0, learning_rate=0.1)
    lr.fit(x_, y_)
    lr.plot_2d(x_, y_, save_loc='logistic.png')

