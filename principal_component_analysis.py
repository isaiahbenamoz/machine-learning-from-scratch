import numpy as np
import matplotlib.pyplot as plt


class PrincipalComponentAnalysis:
    """
    A technique to reduce the dimensions of your data while maintaining the most information.
    """

    def __init__(self, num_components):
        """
        :param num_components: the number of components that are maintained during a transform
        """
        self.num_components = num_components
        self.components = np.zeros((num_components, num_components))
        self.explained_variance = None

    def fit(self, x):
        """
        Fit the transformation to the given x data.
        :param x: the data that our model fits to
        """
        # normalize the rows (features) of x to have mean zero
        z = self.normalize(x)

        # calculate the covariance matrix for z
        z = self.covariance(z)
        w, v = np.linalg.eig(z)

        # sort by the magnitude of the eigenvalues
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        # save the calculated principal components
        self.components = v.copy()

        # save the eigenvalues
        self.explained_variance = w

    def transform(self, x):
        """
        Transforms the data onto a lower dimensional space as to maintain maximum information
        :param x: the data to be transformed
        :return: the data in a k-dimensional space where k is less than the original dimension of x
        """
        # normalize x
        z = self.normalize(x)
        return self.components.T[:self.num_components, :] @ z

    def plot_variance(self):
        """
        Plot the amount of variance that each principal component explains.
        """
        # find the total of the variance
        total_variance = self.explained_variance.sum()

        # find the proportion of the variance explained by each eigen-vector
        prop_variance = [v / total_variance for v in self.explained_variance]

        # create a bar chart with the variances
        plt.bar(range(len(self.explained_variance)), prop_variance, alpha=0.9,
                align='center', label='individual explained variance')

        # create a step plot of the variance explained by each eigen-vector
        plt.step(range(len(self.explained_variance)), np.cumsum(prop_variance), where='mid', label='cumulative explained variance')

        # add x and y labels to the chart; adjust other looks
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')

        # show the plot
        plt.show()

    def plot_2d(self, x):
        """
        Plot a scatter with the orthogonal principal components.
        :param x: the data to be plotted
        """
        # plot the scatter of x
        plt.scatter(x[0], x[1])

        # plot the eigen-vectors that were previously calculated
        plt.quiver([0, 0], [0, 0], self.components[0, :], self.components[1, :], color=['r'], scale=21)

        # show the plot
        plt.show()

    @staticmethod
    def normalize(x):
        return x - x.mean(axis=1).reshape(x.shape[0], 1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    @staticmethod
    def covariance(z):
        """
        Calculate the covariance of matrix z.
        :param z: the matrix for which we would like to find the covariance matrix
        :return: the covariance matrix
        """
        return (z @ z.T) / (z.shape[1] - 1)


if __name__ == '__main__':
    def generate_data(num_samples=100):
        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0, 0.0])

        # The desired covariance matrix.
        r = np.array([
            [1.50, 1.25, 1.25],
            [1.25, 1.50, 1.25],
            [1.25, 1.25, 1.50]
        ])

        # Generate the random samples.
        y = np.random.multivariate_normal(mu, r, size=num_samples)

        x = y[:, 0:2].T
        y = y[:, 2].T

        return x, y

    x, y = generate_data()

    pca = PrincipalComponentAnalysis(2)
    z = pca.fit_transform(x)
    pca.plot_2d(x)
    plt.scatter(z[0], z[1])

    from sklearn.decomposition import pca
    z = pca.PCA(2).fit_transform(x.T)
    plt.scatter(z[:, 0], z[:, 1])
    plt.show()
