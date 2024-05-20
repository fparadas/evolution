import numpy as np


def minimize(fobj, boundaries, n_particles, w, c1, c2, max_iter):
    """
    Minimize the objective function using the Particle Swarm Optimization algorithm.

    Parameters
    ----------
    fobj : function
        The objective function to be minimized. It should take a 1-D numpy array as input.
    boundaries : list of tuples
        The min and max values for each dimension of the input space.
    n_particles : int
        The number of particles in the swarm.
    w : float
        The inertia weight. [0, 2]
    c1 : float
        The cognitive parameter. [0, 1]
    c2 : float
        The social parameter. [0, 1]
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    tuple
        The global best input and the global best value.
    """
    # step 0: setup utility functions
    fobj_vectorized = np.vectorize(fobj, signature="(n)->()")

    DIM = len(boundaries)
    min_boundaries, max_boundaries = np.asarray(boundaries).T
    diff = np.fabs(min_boundaries - max_boundaries)

    denorm = lambda x: (x * diff) + min_boundaries
    denorm_and_eval = lambda x: fobj_vectorized(denorm(x))

    # step 1: initialize particles
    particles = np.random.uniform(-1, 1, (n_particles, DIM))
    velocity = np.zeros_like(particles)

    points = particles
    pbest = denorm_and_eval(points)
    gbest = points[np.argmin(pbest)]
    gbest_val = np.min(pbest)

    for _ in range(max_iter):
        # step 2: update velocity
        r1, r2 = np.random.rand(2, DIM)
        velocity = (
            w * velocity
            + c1 * r1 * (points - particles)
            + c2 * r2 * (gbest - particles)
        )

        # step 3: update particles
        particles = np.clip(particles + velocity, -1, 1)

        # step 4: evaluate particles and update bests
        f = denorm_and_eval(particles)
        points[f < pbest] = particles[f < pbest]
        pbest[f < pbest] = f[f < pbest]
        gbest = points[np.argmin(pbest)]
        gbest_val = np.min(pbest)

        yield denorm(gbest), gbest_val
