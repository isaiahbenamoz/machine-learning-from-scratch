from regression.linear_regression import LinearRegression
from data.generate import normal

# create the correlated data
data = normal(mu=[0.1, -0.3, 0.4], a=1.0, b=0.98)

# extract x and y
x = data[:, 0].reshape(1, -1)
y = data[:, 1].reshape(1, -1)

# create the linear regression model
lr = LinearRegression()

# fit the model to the data
lr.fit(x, y, epochs=500)

# animate the fitting
lr.animate(x, y, save_loc='./results/linear_regression_train.gif')
