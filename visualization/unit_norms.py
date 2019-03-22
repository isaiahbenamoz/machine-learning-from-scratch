from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# get the subplots for plotting
fig, ax = plt.subplots()

# add labels for x1 and x2
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

# set the aspect ratio for the graph to equal
ax.set_aspect('equal', adjustable='box')

ax.set_ylim([-1.2, 1.2])
ax.set_xlim([-1.2, 1.2])

# plot the data
line = ax.plot([0], [0], alpha=0.75)


def update(p):
    """ Update the line plot.

    :param p: the norm of the current iteration
    :return: the line and ax of the new graph
    """

    # initialize values for when y is positive
    x1 = np.concatenate((np.linspace(-1, 0.0, 250), np.linspace(0.00502513, 1, 250)), axis=0)

    # find the corresponding x2 values
    x2 = (1 - (np.abs(x1) ** p)) ** (1 / p)

    # add the values for when y is negative
    x1 = np.concatenate((x1, np.flip(x1)), axis=0).reshape(1, 1000)
    x2 = np.concatenate((x2, -x2), axis=0).reshape(1, 1000)

    # iterative over each line and update its coordinates
    for l, a, b in zip(line, x1, x2):
        l.set_xdata(x1)
        l.set_ydata(x2)

    # update the plots title
    plt.title('All vectors with norm = 1 and p = {:1.3f}'.format(p))

    # return the plots updated values
    return line, ax


# create the p ranges for the graph
ps = np.concatenate((np.logspace(-1.0, 2.0, 50), np.flip(np.logspace(-1.0, 2.0, 50))), axis=0)

# create the animation
anim = FuncAnimation(fig, update, frames=ps, interval=1)
anim.save('unit_norms.gif', dpi=150, writer='imagemagick')

plt.show()
