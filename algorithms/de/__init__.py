import numpy as np
import algorithms.de.mutation as mutation


def minimize(fobj, boundaries, NP, F, Cr, x=mutation.rand, y=1, max_iter=100):
    # step 0: setup utility functions
    fobj_vectorized = np.vectorize(fobj, signature="(n)->()")

    DIM = len(boundaries)
    min_boundaries, max_boundaries = np.asarray(boundaries).T
    diff = np.fabs(min_boundaries - max_boundaries)

    denorm = lambda x: (x * diff) + min_boundaries
    denorm_and_eval = lambda x: fobj_vectorized(denorm(x))

    # step 1: init population
    population = np.random.uniform(-1, 1, (NP, DIM))
    fitness = denorm_and_eval(population)

    for _ in range(max_iter):
        # step 2: mutation
        idxs = np.arange(NP)
        a, bs, cs = x(idxs, fitness, NP, DIM, y)
        mutants = np.clip(
            population[a] + np.sum((F / y) * (population[bs] - population[cs]), axis=0),
            -1,
            1,
        )

        # step 3: crossover
        cross_points = np.random.rand(NP, DIM) < Cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, DIM)] = True
        trial = np.where(cross_points, mutants, population[idxs])

        f_trial = denorm_and_eval(trial)

        improved = f_trial < fitness
        population[improved] = trial[improved]
        fitness[improved] = f_trial[improved]

        best = np.argmin(fitness)

        yield denorm(population[best]), fitness[best]
