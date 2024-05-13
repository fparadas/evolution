import numpy as np

gaussian = lambda x: np.clip(x + np.random.normal(0, 1, x.shape), -1, 1)
uniform = lambda x: np.clip(x + np.random.uniform(-1, 1, x.shape), -1, 1)
