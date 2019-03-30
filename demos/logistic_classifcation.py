from sklearn.datasets.samples_generator import make_blobs
from regression.logistic_regression import LogisticRegression

# create blobs to plot
x, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=20, cluster_std=1.0)

# reshape the blobs
x = x.T.reshape(2, -1)
y = y.reshape(1, -1)

# run logistic regression
lr = LogisticRegression(lambda_=0.0, learning_rate=0.1)
lr.fit(x, y)

# plot the result and save
lr.plot_2d(x, y, save_loc='results/logistic_classification.png')
