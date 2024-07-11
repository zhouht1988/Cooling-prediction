from scipy.io import loadmat
import numpy as np
import pandas as pd


def read_excel(path, sheet_number=0):
    s = pd.read_excel(path, sheet_number)
    return s.to_numpy()


def read_mat(path, mat_name):
    s = loadmat(path)
    s = np.asarray(s[mat_name])
    return s


def read_csv(path):
    s = pd.read_csv(path)
    return s.to_numpy()


def get_concatenated_data(data_previous, data_current):
    """
    Concatenate previous outputs from previous data set to a single current data;
    This function can be used to prepare the predict data before prediction phase

    Args:
        :param data_previous: Previous data set
        :param data_current: A single row of data

    Returns:
        :return: Concatenated data
    """
    scalar_outputs_prev = data_previous[:, -1]
    scalar_outputs_prev = scalar_outputs_prev[:: -1]

    if data_current.size != 0:
        s = np.concatenate((data_current, scalar_outputs_prev), axis=0)
    else:
        s = scalar_outputs_prev
    return np.array(s, dtype=float)


def get_concatenated_data_set(data_raw, k):
    """
    Extracts previous outputs from the raw data and concatenates them to each new data;
    This function can be used to prepare the data set used for model initialization

    Args:
        :param data_raw: Raw data as a numpy array
        :param k: Size of previous targets to be concatenated to each new data vector

    Returns:
        :return: Concatenated training data set
    """
    attributes = data_raw[k: len(data_raw), 0: -1]
    scalar_outputs = data_raw[k: len(data_raw), -1]
    scalar_outputs = scalar_outputs.reshape(len(scalar_outputs), 1)
    scalar_outputs_pres = np.zeros([len(data_raw) - k, k - 1])  # previous load data

    # extracts previous outputs
    for i in range(1, k):
        tmp_l = data_raw[(k - i): (len(data_raw) - i), -1]
        scalar_outputs_pres[:, i - 1] = tmp_l

    s = np.concatenate((attributes, scalar_outputs_pres, scalar_outputs), axis=1)
    debug = True

    return s


def get_concatenated_data_set_target(data_raw, k, target):
    """
    Extracts previous outputs from the raw data and concatenates them to each new data;
    This function can be used to prepare the data set used for model initialization

    Args:
        :param data_raw: Raw data as a numpy array
        :param k: Size of previous targets to be concatenated to each new data vector
        :param target: Index of the column to be concatenated

    Returns:
        :return: Concatenated training data set
    """
    attributes_1 = data_raw[k - 1: len(data_raw) - 1, 0: target]
    attributes_2 = data_raw[k - 1: len(data_raw) - 1, target + 1:]

    scalar_outputs = data_raw[k - 1: len(data_raw) - 1, target]
    scalar_outputs = scalar_outputs.reshape(len(scalar_outputs), 1)

    scalar_outputs_pres = np.zeros([len(data_raw) - k, k - 1])  # previous load data

    # extracts previous outputs
    for i in range(2, k + 1):
        tmp_l = data_raw[(k - i): (len(data_raw) - i), target]
        scalar_outputs_pres[:, i - 2] = tmp_l

    s = np.concatenate((attributes_1, scalar_outputs_pres, scalar_outputs, attributes_2), axis=1)

    return s


def rmse(predictions, targets):
    """
    Returns rmse between two data sets

    Args:
        :param predictions: Prediction outcomes
        :param targets: Actual values

    Returns:
        :return: The rmse
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def accurray(y_true, y_pred):
    """
    Returns the accuracy between two data sets

    Args:
        :param y_pred: Prediction values
        :param y_true: Actual values

    Returns:
        :return: The accuracy
    """
    y_true, y_pred = np.array(y_true, dtype='float'), np.array(y_pred, dtype='float')
    return 1 - ((np.abs(y_true - y_pred)) / y_true).mean()


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype='float'), np.array(y_pred, dtype='float')
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
