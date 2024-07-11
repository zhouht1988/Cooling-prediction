import numpy as np
from src.aknn.pattern import Pattern


def pattern_initialize(data):
    '''
    Initialize the data and record the maximum, minimum values for each column of data.
    :param data: The data to be initialized.
    :return: The maximum and minimum values for each column of data and target columns.
    '''
    [_, n] = data.shape
    attributes_max = np.zeros(n)
    attributes_min = np.zeros(n)

    for i in range(0, n):
        attributes_max[i] = max(data[:, i])
        attributes_min[i] = min(data[:, i])

    target_min = np.min(data[:, - 1])
    target_max = np.max(data[:, - 1])

    return Pattern(attributes_max, attributes_min, target_max, target_min)