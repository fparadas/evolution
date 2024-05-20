import numpy as np


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


rosen_boundaries = lambda x: [(-2.048, 2.048)] * x
