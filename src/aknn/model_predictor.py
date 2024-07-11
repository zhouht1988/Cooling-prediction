import numpy as np
from src.aknn import gaussian as gs



def get_kernal_features(data, sigmas, mu_ker, nk, num_k):
    m = len(data)
    for i in range(0, num_k):
        mean_tmp = mu_ker[i][0: m]
        sigma_tmp = sigmas[i][0:m, 0:m]
        mu_i = (mean_tmp * nk[0] + data) / (nk[i] + 1)

        diff_data = np.asmatrix(data - mu_i)
        diff_mean = np.asmatrix(mean_tmp - mu_i)
        sigma_i = (sigma_tmp * nk[i] + diff_data.T * diff_data + nk[i] * diff_mean.T * diff_mean) / (nk[i] + 1)
        k_tmp = gs.gaussian(data, mu_i, sigma_i)
        sigma_inv = np.linalg.inv(sigmas[i])
        aj = sigma_inv[-1]

        mu = mu_ker[i][-1] - np.sum((data - mu_ker[i][0: m]) * aj[0: m]) / aj[-1]

    return mu, aj[-1]


def model_predict(data, model, pattern, data_current, df_train, df_pred):
    """
    Predicts the target of input data with the given model

    Args:
        :param data: Data to be predicted
        :param model: Trained model

    Returns:
        :return: The predicted result (Expected Conditional Mean of the output space)
    """


    sigmas = model.sigmas
    mu_ker = model.mu_ker
    nk = model.nk
    num_k = model.num_k

    m = len(data)
    sum_k = 0
    mu_k = 0

    for i in range(0, num_k):
        mean_tmp = mu_ker[i][0: m]
        sigma_tmp = sigmas[i][0:m, 0:m]
        mu_i = (mean_tmp * nk[0] + data) / (nk[i] + 1)

        diff_data = np.asmatrix(data - mu_i)
        diff_mean = np.asmatrix(mean_tmp - mu_i)
        sigma_i = (sigma_tmp * nk[i] + diff_data.T * diff_data + nk[i] * diff_mean.T * diff_mean) / (nk[i] + 1)
        k_tmp = gs.gaussian(data, mu_i, sigma_i)
        sigma_inv = np.linalg.inv(sigmas[i])
        aj = sigma_inv[-1]

        mu = mu_ker[i][-1] - np.sum((data - mu_ker[i][0: m]) * aj[0: m]) / aj[-1]

        sum_k += k_tmp
        mu_k += mu * k_tmp

    return mu_k / sum_k

