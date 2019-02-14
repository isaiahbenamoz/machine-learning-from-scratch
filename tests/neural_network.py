from deep_learning.neural_network import NeuralNetwork
import numpy as np
from scipy.special import expit
from sklearn.metrics import mean_squared_error
import unittest
expit = np.vectorize(expit)


class NeuralNetworkTester(unittest.TestCase):

    def test_linear_forward(self):
        np.random.seed(0)

        w_l = np.random.randn(10, 10)
        a_l_prev = np.random.randn(10, 30)
        b_l = np.random.rand(10, 1)

        z_l = NeuralNetwork.linear_forward(w_l, a_l_prev, b_l)
        self.assertTrue(np.allclose(z_l, np.dot(w_l, a_l_prev) + b_l))

    def test_nonlinear_forward(self):
        np.random.seed(0)

        z_l = np.random.randn(10, 20)

        a_l = NeuralNetwork.nonlinear_forward(z_l, 'sigmoid')
        self.assertTrue(np.allclose(a_l, expit(z_l)))

        a_l = NeuralNetwork.nonlinear_forward(z_l, 'relu')
        self.assertTrue(np.allclose(a_l, np.where(z_l > 0, z_l, z_l * 0.01)))

        a_l = NeuralNetwork.nonlinear_forward(z_l, 'tanh')
        self.assertTrue(np.allclose(a_l, np.tanh(z_l)))

    def test_loss_forward(self):
        np.random.seed(0)

        y = np.random.randn(2, 20)
        y_hat = np.random.randn(2, 20)

        mse = NeuralNetwork.loss_forward(y_hat, y, 'mean_squared_error')
        self.assertAlmostEqual(mse, 0.5 * mean_squared_error(y, y_hat))

    def test_loss_backward(self):
        np.random.seed(0)

        y = np.random.randn(2, 20)
        y_hat = np.random.randn(2, 20)

        da_l = NeuralNetwork.loss_backward(y_hat, y, 'mean_squared_error')
        self.assertEqual(da_l.shape, y_hat.shape, y.shape)
        self.assertTrue(np.allclose(da_l, y_hat - y))

    def test_nonlinear_backward(self):
        # TODO: implement tests for nonlinear backward propagation
        pass

    def test_linear_backward(self):
        # TODO: implement tests for linear backward propagation
        pass


if __name__ == '__main__':
    unittest.main()
