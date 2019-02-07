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

    def __init__(self, c=1.0, epochs=1000, learning_rate=0.0025):
        """
        :param c: the weight assigned to the non-regularized portion of the cost function
        :param epochs: the number of iterations that gradient descent will run
        :param learning_rate: a scalar that adjusts how quickly W and b are adjusted
        """
        self.C = c
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
        return self.C * (y * np.log(y_pred + self.epsilon) + (1 - y) * np.log(1 - y_pred + self.epsilon)).mean()

    def gradient(self, x, y):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :return: the gradient of the cost function with respect to W and b
        """
        y_hat = self.predict(x)
        dW = self.C * (x * (y - y_hat)).mean(axis=1)
        db = (y - y_hat).mean()
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
            # compute the gradients given our training data
            dW, db = self.gradient(x, y)

            # adjust W and b according to the gradient of the cost function
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # add the parameters to the parameters list
            self.parameters.append((self.W, self.b))

            # if the current epoch number is a multiple of the number of epochs / 10
            if epoch % (self.epochs // 10) == 0:
                # print the cost function with out current model
                print('Epoch #' + str(epoch) + ' loss = ' + str(self.cost(x, y)))

        print('dW:', dW)
        print('db:', db)

    def predict(self, x, W=None, b=None):
        """
        :param x: the data which we would like to make a prediction about
        :param W: an optional parameter to specify W for the prediction
        :param b: an optional parameter to specify b for the prediction
        :return: the model's prediction, y_hat, for the given x data
        """
        # if W and b are defined, use them to make the prediction
        if W and b:
            return self.sigmoid(W @ x + b)

        # if W is not defined because the model has not been trained
        if not self.W:
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

    def plot(self, x, y, show=True, save_loc='', dpi=150):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :param show: a boolean value indicating whether to show the plot
        :param save_loc: the location where the the user would like to save the graph
        :param dpi: the pixel density of the graph being saved
        """
        # create a figure for saving
        fig = plt.figure()

        # create a scatter plot with the given x and y coordinate data
        sns.scatterplot(x[0], y[0])

        # plot the predicted sigmoid curve
        sns.lineplot(x[0], self.predict(x)[0], color='red')

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

    def animate(self, x, y, show=True, save_loc='', dpi=150):
        """
        :param x: the data which we would like to make a prediction about
        :param y: the ground truth labels corresponding to our input data
        :param show: a boolean value indicating whether to show the animation
        :param save_loc: the location where the the user would like to save the animation
        :param dpi: the pixel density of the graph being saved
        """
        # create the figure we'll use to create the animation
        fig, ax = plt.subplots()

        # create a scatter plot of our data
        sns.scatterplot(x[0], y[0])

        # label the x and y axises and add a title to the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Logistic Regression')

        # plot an initial red line and save it as a variable
        line, = ax.plot(x[0], self.predict(x[0]), 'r-')

        # create an update function that
        def update(parameters):
            line.set_ydata(self.predict(x, *parameters))
            return line, ax

        # if a save location is defined
        if save_loc:
            predictions = np.ndarray([self.predict(x, y, W, b) for W, b in self.parameters])
            print(predictions)
            anim = FuncAnimation(fig, update, frames=predictions, interval=50)
            anim.save(save_loc, dpi=dpi, writer='imagemagick')

        if show:
            plt.show()


if __name__ == '__main__':
    # The desired mean values of the sample.
    mu = np.array([0.0, 1.0])

    # The desired covariance matrix.
    r = np.array([
            [1.00, 0.99],
            [0.99, 1.00]
        ])

    # Generate the random samples.
    data = np.random.multivariate_normal(mu, r, size=1000)
    x_ = data[:, 0].reshape(1, -1)
    y_ = data[:, 1].reshape(1, -1)
    print(x_.shape, y_.shape)

    lr = LogisticRegression(1.0)
    lr.fit(x_, y_ > 0.5)
    lr.plot(x_, y_ > 0.5)
    lr.animate(x_, y_, show=True, save_loc='logistic.gif', dpi=150)
