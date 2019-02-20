from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

# get the subplots for plotting
fig, ax = plt.subplots()

ax.set_aspect('equal', adjustable='box')

ax.set_ylim([-1.2, 1.2])
ax.set_xlim([-1.2, 1.2])

# plot the data
line = ax.plot([0], [0], alpha=0.75)


def update(p):

    print('{:1.3f}'.format(p))
    # initialize values for when y is positive
    x1 = np.linspace(-1, 1, 500)
    x2 = (1 - (np.abs(x1) ** p)) ** (1 / p)

    # add the values for when y is negative
    x1 = np.concatenate((x1, np.flip(x1)), axis=0).reshape(1, 1000)
    x2 = np.concatenate((x2, -x2), axis=0).reshape(1, 1000)

    for l, a, b in zip(line, x1, x2):
        l.set_xdata(x1)
        l.set_ydata(x2)

    plt.title('All vectors with norm = 1 and p = {:1.3f}'.format(p))

    return line, ax


# create the animation
# ps = np.concatenate((np.linspace(0.0, 1.0, 50), np.linspace(1.01, 5.0, 50), np.linspace(5.01, 50.0, 50)), axis=0)
anim = FuncAnimation(fig, update, frames=np.logspace(-1.0, 2.0, 30), interval=1)
anim.save('unit_norms.gif', dpi=150, writer='imagemagick')

plt.show()
