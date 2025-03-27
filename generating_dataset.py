import numpy as np
def generate_dataset(p, n, d, g, seed=42):
    np.random.seed(seed)
    indices = np.arange(d)
    S = g ** np.abs(np.subtract.outer(indices, indices))
    mu0 = np.zeros(d)
    mu1 = np.array([1 / (i + 1) for i in range(d)])
    mu1[0] = 1 
    y = np.random.binomial(1, p, size=n)
    X = np.empty((n, d))
    for i in range(n):
        if y[i] == 0:
            X[i, : ]=np.random.multivariate_normal(mu0, S)
        else:
            X[i, :]=np.random.multivariate_normal(mu1, S)
    
    return X, y