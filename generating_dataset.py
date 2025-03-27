import numpy as np
def generate_dataset(p, n, d, g, seed=42):
    np.random.seed(seed)
    indices = np.linspace(0, d - 1, d, dtype=int)
    S = g**np.abs(np.expand_dims(indices, 0) - np.expand_dims(indices, 1))
    mu0 = np.zeros(d)
    mu1 = np.array([1 / (i + 1) for i in range(d)])
    mu1[0] = 1

    y = np.random.binomial(1, p, size=n)
    X = np.array([np.random.multivariate_normal(mu0, S) if label == 0 
                else np.random.multivariate_normal(mu1, S) for label in y])

    return X, y