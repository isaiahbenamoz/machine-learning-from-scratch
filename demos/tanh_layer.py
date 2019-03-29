from math import sqrt, tanh
import numpy as np
from visualization.transformations import TransformationVisualizer

t = [
    np.array([[2, sqrt(2)], [1, sqrt(2)]]),
    np.array([[-1], [-1]]),
    tanh
]

grid = TransformationVisualizer.initialize_grid(-5, 5, 80, 500)

TransformationVisualizer(t, grid, None).animate(save_loc='./results/tanh_layer.gif', x_ticks=np.linspace(-1, 1, 5),
                                                y_ticks=np.linspace(-1, 1, 5))
