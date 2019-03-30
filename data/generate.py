import numpy as np
from math import exp, pi, cos, sin


def normal(num=100, dimensions=3, a=1.0, b=0.5, mu=None):
        # generate the means of the distribution
        mu = np.array([0.0 for _ in range(dimensions)]) if mu is None else np.array(mu)

        # generate the desired covariance matrix
        r = np.array([[a if col == row else b for col in range(dimensions)] for row in range(dimensions)])

        # Generate the random samples
        return np.random.multivariate_normal(mu, r, size=num)


def sigmoid(start=-5.0, end=5.0, num=50):
        for i in np.linspace(start, end, num):
            yield 1 / (1 + exp(-i))

        
def spiral(start=0.0, end=3.14 * 1.5, mul=1.0, num=50):
        x = []
        y = []
        for t in np.linspace(start, end, num):
                r = mul * t
                x.append(r * cos(t))
                y.append(r * sin(t))

        return np.array([x, y])
