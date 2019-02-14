import numpy as np
from scipy.special import expit
sigmoid = np.vectorize(expit)


class NeuralNetwork:
    """
    A neural network implementation that can have any number of layers.
    """

    def __init__(self, nodes, activations, loss='mean_squared_error'):
        """
        :param nodes: a list containing the number of nodes in each layer
        :type nodes: list[int]
        :param activations: a list containing the activation functions used in each layer
        :type activations: list[str]
        :param loss: a string indicating the loss function to be used
        :type loss: str
        :rtype: None
        """
        # if any model input constraints are not met, then raise an error
        if len(nodes) != len(activations):
            raise RuntimeError('The length of nodes must be equal to the length of activations.')

        if activations[0] is not None:
            raise RuntimeError('The activation function for the first layer should be None.')

        if not set(activations).issubset({None, 'relu', 'sigmoid', 'tanh'}):
            raise RuntimeError(str(set(activations) - {None, 'relu', 'sigmoid', 'tanh'}) +
                               ' is / are not valid activation functions.')

        if loss not in ['mean_squared_error', 'cross_entropy']:
            raise RuntimeError(str(loss) + ' is not a valid loss function.')

        # initialize the user's input parameters
        self.num_layers = len(nodes) - 1
        self.nodes = nodes
        self.activations = activations
        self.loss = loss

        # crete dictionaries to store the model parameters and cache
        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        # initialize the parameters for the model at each layer
        self.initialize_parameters()

    def predict(self, x):
        """ Use the model to predict y_hat for the given x values.

        :param x: the data for which y_hat would like to predicted
        :type x: np.ndarray
        :returns: the model's predictions, y_hat
        :rtype: np.ndarray
        """
        # initialize the first activation layer to be x
        a_l_prev = x

        # iterate over each layer of the model
        for layer in range(1, self.num_layers + 1):
            # extract the weights and biases for the current layer
            w_l = self.parameters['w' + str(layer)]
            b_l = self.parameters['b' + str(layer)]

            # calculate z for the given layer (z = w @ a + b)
            z_l = self.linear_forward(w_l, a_l_prev, b_l)

            # apply the given activation function (activation(z))
            a_l = self.nonlinear_forward(z_l, self.activations[layer])

            # move to the next layer
            a_l_prev = a_l

        return a_l_prev

    def train(self, x, y, epochs=1000, learning_rate=0.01):
        """ Trains the model on the data (x, y).

        :param x: the input training data that predicts y
        :type x: np.ndarray
        :param y: the ground truth values corresponding to x
        :type y: np.ndarray
        :param epochs: the number of times iterated over the dataset
        :type epochs: int
        :param learning_rate: a scalar that determines how fast the model parameters change
        :type learning_rate: float
        """
        # iterate for the number of epochs
        for epoch in range(epochs):

            # every one tenth of the training session, print the loss
            if epoch % (epochs // 10) == 0:
                # calculate the cost for all training examples
                cost = self.loss_forward(self.predict(x), y, self.loss)

                # print out the cost for the current epoch
                print('Epoch #', epoch, 'cost:', cost)

            # run forward propagation
            self.forward_propagate(x)

            # run backward propagation
            self.backward_propagate(y)

            # update the parameters with the given learning rate
            self.update_parameters(learning_rate)

        # compute the cost for the final model
        cost = self.loss_forward(self.predict(x), y, self.loss)
        # print out the cost for the last epoch
        print('Epoch #', epochs, 'cost:', cost)

    def initialize_parameters(self):
        """ Initialize the models parameters for the relu activation function. """
        # iterate over each layer as defined the length of num_nodes
        for layer in range(1, self.num_layers + 1):
            # initialize w_i and b_i
            w_l = np.random.randn(self.nodes[layer], self.nodes[layer - 1]) * np.sqrt(2 / self.nodes[layer])
            b_l = self.parameters['b' + str(layer)] = np.zeros((self.nodes[layer], 1))

            # save w_l and b_l to the parameters dictionary
            self.parameters['b' + str(layer)] = b_l
            self.parameters['w' + str(layer)] = w_l

    def update_parameters(self, learning_rate):
        """ Update the parameters according to the derivatives calculated during back propagation.

        :param learning_rate: a scalar that determines how fast the model parameters change
        :type learning_rate: float
        """
        # iterate over each layer and adjust the parameters
        for layer in range(1, self.num_layers + 1):
            self.parameters['w' + str(layer)] -= learning_rate * self.gradients['dw' + str(layer)]
            self.parameters['b' + str(layer)] -= learning_rate * self.gradients['db' + str(layer)]

    def forward_propagate(self, x):
        """ Runs forward propagation storing z and a for each layer.

        :param x: the input training data that predicts y
        :type x: np.ndarray
        """
        # initialize the first activation layer to x
        self.cache = {'a0': x}

        # iterate over each layer of the model
        for layer in range(1, self.num_layers + 1):
            # extract the necessary values
            w_l = self.parameters['w' + str(layer)]
            b_l = self.parameters['b' + str(layer)]
            a_l_prev = self.cache['a' + str(layer - 1)]

            # forward propagate for the given layer
            z_l = self.linear_forward(w_l, a_l_prev, b_l)
            a_l = self.nonlinear_forward(z_l, self.activations[layer])

            # save the activation to the cache
            self.cache['z' + str(layer)] = z_l
            self.cache['a' + str(layer)] = a_l

    @staticmethod
    def linear_forward(w_l, a_l_prev, b_l):
        """ Compute z for the current layer.

        :param w_l: the weights for the current layer
        :type w_l: np.ndarray
        :param a_l_prev: the previous layer's activation
        :type a_l_prev: np.ndarray
        :param b_l: the bias terms for the current layer
        :type b_l: np.ndarray
        :return: the z value for the current layer
        :rtype: np.ndarray
        """
        return w_l @ a_l_prev + b_l

    @staticmethod
    def nonlinear_forward(z_l, activation):
        """ Compute the activation for the current layer.

        :param z_l: the value computed by linear forward propagation
        :type z_l: np.ndarray
        :param activation: the activation function specified
        :type activation: str
        :return a_l: the activation for the current layer
        :rtype: np.ndarray
        """
        if activation == 'relu':
            return np.where(z_l > 0, z_l, z_l * 0.01)
        elif activation == 'sigmoid':
            return sigmoid(z_l)
        elif activation == 'tanh':
            return np.tanh(z_l)

    @staticmethod
    def loss_forward(y_hat, y, loss):
        """ Compute the model's loss.

        :param y_hat: the model's predictions for y
        :type y_hat: np.ndarray
        :param y: the data's ground truth values
        :type y: np.ndarray
        :param loss: the loss function specified
        :type loss: str
        :return L: the model's loss
        :rtype: np.ndarray
        """
        if loss == 'mean_squared_error':
            return (0.5 * (y_hat - y) ** 2).mean()
        if loss == 'cross_entropy':
            return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

    def backward_propagate(self, y):
        """ Run backward propagation to calculate how to shift W and b for each layer.

        :param y: the data's ground truth values
        :type y: np.ndarray
        """
        # extract the models predictions from the cache
        y_hat = self.cache['a' + str(self.num_layers)]

        # calculate da for the last layer
        da_l = self.loss_backward(y_hat, y, self.loss)

        # save da_l in the gradients cache
        self.gradients['da' + str(self.num_layers)] = da_l

        # iterate backward over each layer of the model
        for layer in range(self.num_layers, 0, -1):
            # extract the necessary parameters
            w_l = self.parameters['w' + str(layer)]
            z_l = self.cache['z' + str(layer)]

            # extract the necessary values and gradients
            a_l_prev = self.cache['a' + str(layer - 1)]

            da_l = self.gradients['da' + str(layer)]

            # back propagate to obtain dz_l
            dz_l = self.nonlinear_backward(da_l, z_l, self.activations[layer])

            # back propagate to obtain dw_l, db_l, and da_l_prev
            dw_l, db_l, da_l_prev = self.linear_backward(dz_l, w_l, a_l_prev)

            # save the gradients for adjusting dw and db
            self.gradients['dw' + str(layer)] = dw_l
            self.gradients['db' + str(layer)] = db_l

            # save da for further back propagation
            self.gradients['da' + str(layer - 1)] = da_l_prev

    @staticmethod
    def loss_backward(y_hat, y, loss):
        """ Calculate the derivative of the loss function with respect to y_hat.

        :param y_hat: the model's predictions for y
        :type y_hat: np.ndarray
        :param y: the data's ground truth values
        :type y: np.ndarray
        :param loss: the loss function specified
        :type loss: str
        :returns: the derivative of the loss function with respect to y_hat
        :rtype: np.ndarray
        """
        if loss == 'mean_squared_error':
            return y_hat - y
        if loss == 'cross_entropy':
            return -(y / y_hat) - ((1 - y) / (1 - y_hat))

    @staticmethod
    def nonlinear_backward(da_l, z_l, activation):
        """ Calculate the derivative of the loss function with respect to z for the current layer.

        :param da_l: the derivative of the loss function with respect to da at the current layer
        :type da_l: np.ndarray
        :param z_l: the z value for the current layer
        :type z_l: np.ndarray
        :param activation: the activation function used at the current layer
        :type activation: str
        :return: the derivative of the loss function with respect to z for the current layer
        :rtype: np.ndarray
        """
        if activation == 'relu':
            g = np.ones_like(z_l)
            g[z_l < 0] = 0.01
            return da_l * g
        elif activation == 'sigmoid':
            g = sigmoid(z_l)
            return da_l * (g * (1.0 - g))
        elif activation == 'tanh':
            return da_l * (1.0 - np.tanh(z_l) ** 2)

    @staticmethod
    def linear_backward(dz_l, w_l, a_l_prev):
        """ Calculate the derivative of the loss function with respect to w, b, and a_prev.

        :param dz_l: the derivative of the loss function with respect to z at layer l
        :type dz_l: np.ndarray
        :param w_l: the weights for layer l
        :type w_l: np.ndarray
        :param a_l_prev: the activation value for layer l
        :type a_l_prev: np.ndarray
        :return: dw and db for the current layer; da for the previous layer
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        # calculate dw for the current layer
        dw_l = (dz_l @ a_l_prev.T) / a_l_prev.shape[1]

        # calculate db for the current layer
        db_l = dz_l.mean(axis=1, keepdims=True)

        # calculate da for the previous layer
        da_l_prev = w_l.T @ dz_l

        # return the three calculations
        return dw_l, db_l, da_l_prev


if __name__ == '__main__':
    def generate_data(num_samples=100):
        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0, 0.0])

        a, b = 1.50, 1.49
        # The desired covariance matrix.
        r = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a]
        ])

        # Generate the random samples.
        y = np.random.multivariate_normal(mu, r, size=num_samples)

        x = y[:, 0:2].T
        y = y[:, 2].T

        return x, y

    x_train, y_train = generate_data(200)
    y_train = (1 / (1 + np.exp(-y_train))).reshape(1, -1)

    # x = np.array([
    #     [0, 0, 1, 1],
    #     [0, 1, 0, 1]
    # ])
    #
    # y = np.array([[0, 1, 1, 0]])

    import matplotlib.pyplot as plt

    plt.scatter(x_train[0], x_train[1], c=y_train[0])
    plt.title('ground')
    plt.show()

    print(type(x_train))
    nn = NeuralNetwork([2, 20, 20, 1], [None, 'relu', 'relu', 'sigmoid'], 'mean_squared_error')
    nn.train(x_train, y_train, learning_rate=0.01)
    y_pred = nn.predict(x_train)
    
    plt.title('custom')
    plt.scatter(x_train[0], x_train[1], c=y_pred[0])
    plt.show()
