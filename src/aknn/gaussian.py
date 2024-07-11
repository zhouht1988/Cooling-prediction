import numpy as np


def gaussian(x, mu, cov):
    """
    Calculates the value for correlated hyper-rellipsodial Gaussian kernel

    Args:
        :param x: The input row vector
        :param mu: Position vector
        :param cov: Covariance matrix

    Returns:
        :return: Gaussian value
    """
    m = len(x)
    x = np.asmatrix(x)
    mu = np.asmatrix(mu)
    cov = np.asmatrix(cov)
    K = np.exp((-0.5) * (x - mu) * np.linalg.inv(cov) * (x - mu).T) / (
                (pow(2 * np.pi, m / 2)) * np.sqrt(np.linalg.det(cov)))

    return K[0, 0]
