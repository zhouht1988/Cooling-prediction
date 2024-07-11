import numpy as np


def normalize_all(data):
    """
    Normalizes the input data set to a range of [0, 1]

    Args:
        :param data: The input data set to be normalized

    Returns
        :return: s: Normalized data
        :return: xmax: Maximum values in each column of input data
        :return: xmin: Minimum values in each column of input data
    """

    [m, n] = data.shape
    attributes_max = np.zeros(n)
    attributes_min = np.zeros(n)
    s = np.zeros(data.shape)

    for i in range(0, n):
        attributes_max[i] = max(data[:, i])
        attributes_min[i] = min(data[:, i])

    for j in range(0, n):
        for i in range(0, m):
            s[i, j] = (data[i, j] - attributes_min[j]) / (attributes_max[j] - attributes_min[j])

    return s


def normalize(data, attributes_max, attributes_min):
    """
    Normalizes a single row  data based on previous analytic information

    Args:
        :param data: The single row input data to be normalized
        :param attributes_max: The maximum values in each column of previously used data
        :param attributes_min: The minimum values in each column of previously used data

    Returns:
        :return: normalized data
    """

    return (data - attributes_min) / (attributes_max - attributes_min)
