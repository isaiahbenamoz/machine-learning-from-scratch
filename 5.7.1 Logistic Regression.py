import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
        """
        # initialize W to the correct dimensions
        self.W = np.zeros((1, x.shape[0]))

        # run gradient descent for the given number of epochs
        for epoch in range(self.epochs):
            # compute the gradients given our training data
            dW, db = self.gradient(x, y)

            # adjust W and b according to the gradient of the cost function
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # if the current epoch number is a multiple of the number of epochs / 10
            if epoch % (self.epochs // 10) == 0:
                # print the cost function with out current model
                print('Epoch #' + str(epoch) + ' loss = ' + str(self.cost(x, y)))

        print('dW:', dW)
        print('db:', db)

    def predict(self, x):
        """
        :param x: the data which we would like to make a prediction about
        :return: the model's prediction, y_hat, for the given x data
        """
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
        sns.scatterplot(x.reshape(1, -1)[0], y.reshape(1, -1)[0])

        # plot the predicted sigmoid curve
        sns.lineplot(x[0], self.predict(x)[0])

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

    def animate(self):
        pass


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
    lr.fit(x_, y_)
    lr.plot(x_, y_)
