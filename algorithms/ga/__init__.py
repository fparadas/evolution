import numpy as np
import algorithms.ga.mutation as mutation
import algorithms.ga.selection as selection
from utils import repeat


def minimize(
    fobj,
    boundaries,
    NP,
    F,
    Cr,
    selection_function=selection.tournament,
    mutation_function=mutation.uniform,
    max_iter=100,
):
    fobj_vectorized = np.vectorize(fobj, signature="(n)->()")

    DIM = len(boundaries)
    min_boundaries, max_boundaries = np.asarray(boundaries).T
    diff = np.fabs(min_boundaries - max_boundaries)

    denorm = lambda x: (x * diff) + min_boundaries
    denorm_and_eval = lambda x: fobj_vectorized(denorm(x))

    # step 1: init population
    population = np.random.uniform(-1, 1, (NP, len(boundaries)))
    fitness = denorm_and_eval(population)

    # step 2: main loop
    for i in range(max_iter):
        selection_size = int(NP * Cr)
        selected = repeat(selection_size, selection_function, population, fitness)

        # step 3: crossover
        parents = np.random.choice(
            len(selected), size=(int(selection_size / 2), 2), replace=False
        )
        offspring = np.mean(selected[parents], axis=1)

        population = np.concatenate((population, offspring))

        # step 4: mutation
        mutation_size = int(F * len(population))
        mutation_candidates = np.random.choice(len(population), size=mutation_size)
        mutants = mutation_function(population[mutation_candidates])

        population[mutation_candidates] = mutants

        # step 5: cut and eval
        fitness = denorm_and_eval(population)
        population = population[np.argsort(fitness)][:NP]
        fitness = fitness[np.argsort(fitness)][:NP]

        yield denorm(population[0]), fitness[0]
