import numpy as np
def generate_dataset(p, n, d, g, seed=42):
    """
    generates a binary labeled dataset where features follow d-dimensional multivariate 
    normal distribution with class-specific means and a shared covariance matrix

    :param p: probability of a sample belonging to class 1
    :param n: number of samples to generate
    :param d: dimensions of a feature space
    :param g: used to construct covariance matrix, determines how quickly the correlation between features decreases as their distance increases
    :param seed: 
    :returns: X, y, where X is an array of features and y is an array of binary classes
    """
    np.random.seed(seed)
    indices = np.linspace(0, d - 1, d, dtype=int)
    S = g**np.abs(np.expand_dims(indices, 0) - np.expand_dims(indices, 1))
    mu0 = np.zeros(d)
    mu1 = np.array([1 / (i + 1) for i in range(d)])
    mu1[0] = 1

    y = np.random.binomial(1, p, size=n)
    X = np.array([np.random.multivariate_normal(mu0, S) if label == 0 else np.random.multivariate_normal(mu1, S) for label in y])
    return X, y