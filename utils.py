import numpy as np


def repeat(n, func, *args, **kwargs):
    return np.asarray([func(*args, **kwargs) for _ in range(n)])
