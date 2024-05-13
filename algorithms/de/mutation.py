import numpy as np


def rand(idxs, fitness, NP, DIM, y):
    a = np.random.choice(idxs, NP)

    bs, cs = np.random.choice(idxs, (y, DIM, NP) if y > 1 else (DIM, NP))
    return a, bs, cs


def best(idxs, fitness, NP, DIM, y):
    a = np.argmin(fitness)
    bs, cs = np.random.choice(idxs[idxs != a], (y, DIM, NP) if y > 1 else (DIM, NP))
    return a, bs, cs


def curr_to_best(idxs, fitness, NP, DIM, y):
    a, cs = idxs, idxs
    bs = np.argsort(fitness)[:y].reshape((y, 1))
    return a, bs, cs
