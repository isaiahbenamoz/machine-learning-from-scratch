from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

# get the subplots for plotting
fig, ax = plt.subplots()

# plot the data
c = plt.plot(alpha=0.75)


def update(p):
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.

    # initialize values for when y is positive
    x1 = np.linspace(-1, 1, 500)
    x2 = (1 - (np.abs(x1) ** p)) ** (1 / p)

    # add the values for when y is negative
    x1 = np.concatenate((x1, np.flip(x1)), axis=0).reshape(1, 1000)
    x2 = np.concatenate((x2, -x2), axis=0).reshape(1, 1000)

    c.set_xdata(x1)
    c.set_ydata(x2)
    plt.title('Matrix norm: p = ' + str(p))

    return c, ax

# create the animation
anim = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 100), interval=50)
anim.save('unit_norms.gif', dpi=150, writer='imagemagick')

plt.show()
