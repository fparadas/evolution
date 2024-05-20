import numpy as np


def roullete(P, F):
    """
    Seleciona um indivíduo de uma população usando a estratégia de roleta.
    f(x) x é um indivíduo da população
    Parâmetros:
    P (np.array): Um vetor da população, onde cada elemento é um indivíduo.
    F (np.array): Um vetor de valores de fitness correspondentes aos indivíduos em P.

    Retorna:
    object: O indivíduo selecionado.
    """
    inverse_fitness = 1 / F
    total_inverse_fitness = np.sum(inverse_fitness)
    if total_inverse_fitness == 0:
        probabilities = np.ones(len(inverse_fitness)) / len(inverse_fitness)
    else:
        probabilities = inverse_fitness / total_inverse_fitness

    selected_index = np.random.choice(len(P), p=probabilities)

    return P[selected_index]


def tournament(P, F):
    """
    Seleciona um indivíduo da população usando a estratégia de torneio.

    Parâmetros:
    P (np.array): Um vetor da população, onde cada elemento é um indivíduo.
    F (np.array): Um vetor de valores de fitness correspondentes aos indivíduos em P.

    Retorna:
    object: O indivíduo selecionado.
    """
    indices = np.random.choice(len(P), size=2, replace=False)

    if F[indices[0]] > F[indices[1]]:
        winner_index = indices[0]
    else:
        winner_index = indices[1]

    return P[winner_index]
