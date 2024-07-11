import time
import src.aknn as aknn
from src.aknn.model import Model
import numpy as np
from src.aknn import gaussian
import copy
import pandas as pd
import matplotlib.pyplot as plt



def model_initialize(data, rolling, entropy_delta):

    '''
    Initializes the model using a clustering data set
    :param data: Clustering data set.
    :param rolling: Length of historical data to review.
    :param entropy_delta: Length of historical data to review.
    :return: return Clustering model
    '''
    nk_train_data = {}
    nk_train_data[0] = data[0+rolling, :]
    nk_train_data[1] = data[1+rolling, :]

    [n, m] = data.shape
    # Initialization proces
    initial_sigma = 0.5 * (np.linalg.norm(data[0+rolling, :] - data[1+rolling, :])) * np.eye(m)  # take distances as initial sigma
    sigmas = np.array([initial_sigma, initial_sigma])
    mu_ker = np.array([data[0+rolling, :], data[1+rolling, :]])
    nk = np.ones(2)
    num_k = 2
    model = Model(sigmas, mu_ker, nk, num_k, nk_train_data)

    # Clutering set
    st = time.time()
    for i in range(2, n):
        model = model_train(data[i, :], i, copy.deepcopy(model), rolling, entropy_delta)
    print('Clustering run:', (time.time() - st)/n)

    return model

def plot_clustering_distribution(data, model, start_time_clustering, end_time_clustering):
    '''
    Plot clustering data distribution.
    :param data: Data to be clustered.
    :param model: Record data clustering result info in model.
    :param start_time_clustering: The start time of the clustering data.
    :param end_time_clustering: The end time of the clustering data.
    :return: display the data clustering results distribution
    '''
    data_cluster = data.copy()
    nk_train_data = model.nk_train_data
    category_dict = {}
    for key, value in nk_train_data.items():
        for v in value:
            category_dict[v] = key

    data_cluster['index'] = data_cluster.index
    data_cluster['category'] = data_cluster['index'].map(category_dict)
    data_cluster.fillna(method='ffill', inplace=True)
    data_cluster.drop('index', axis=1, inplace=True)
    data_cluster.dropna(inplace=True)
    data_cluster['category'] = data_cluster['category'].astype(int)

    legends = []
    date_min = data_cluster["datetime"].min()
    date_max = data_cluster["datetime"].max()
    date_range = pd.date_range(date_min, date_max, freq="H")
    df_range = pd.DataFrame({"datetime": date_range})
    plt.figure(figsize=(20, 6))
    for i, group in data_cluster.groupby("category"):
        df_ploti = df_range.merge(group, on='datetime', how='left')
        plt.plot(df_ploti["datetime"], df_ploti["hvac"], 'o-', label='Proposed method')
        legends.append(f"category{i}")
    plt.xlim(pd.to_datetime(start_time_clustering), pd.to_datetime(end_time_clustering))
    plt.legend(legends)
    plt.title('AKNN Clustering')
    plt.show()


def model_train(s, index, model, rolling, entropy_delta):
    """
    This function aims to train the AKNN model

    Args:
        :param s: A 1*m vector which combined of the input data and the scalar output
        :param model: Model object to be trained
        :param rolling: Length of historical data to review
        :param entropy_delta: Parameter for constraining the number of AKNN kernel clusters
    """
    m = len(s)
    nomination_flag = True

    sigmas = model.sigmas
    mu_ker = model.mu_ker
    nk = model.nk
    num_k = model.num_k
    nk_train_data = model.nk_train_data

    psk = np.zeros(num_k)  # conditional probability of P(s|fi_k), obtained by Gaussian kernel
    pk = np.zeros(num_k)  # ration of the total number of samples that are clustered into each kernel

    """kernel nomination"""
    for i in range(0, num_k):
        psk[i] = gaussian.gaussian(s, mu_ker[i], sigmas[i])
        pk[i] = nk[i] / np.sum(nk)

    ps = np.sum(psk * pk) + 1e-9  # conditional probability P(s)
    prob_k = (psk * pk) / ps  # conditional probability P(pi_k|s), based on bayes function


    while nomination_flag:
        which_k = prob_k.argmax()

        """kernel confirmation"""
        nk_j = nk[which_k] + 1
        mu_j = (mu_ker[which_k] * nk[which_k] + s) / nk_j
        sigma_j = ((sigmas[which_k] * nk[which_k]) +
                   np.asmatrix(s - mu_j).T * np.asmatrix(s - mu_j) +
                   nk[which_k] * np.asmatrix(mu_ker[which_k] - mu_j).T * np.asmatrix(mu_ker[which_k] - mu_j)) / nk_j
        #
        h_j_old = 0.5 * np.log(pow(2 * np.pi * np.e, m) * np.linalg.det(sigmas[which_k]))  # previous entropy
        h_j_new = 0.5 * np.log(pow(2 * np.pi * np.e, m) * np.linalg.det(sigma_j))  # new entropy

        nk_j_data_train = nk_train_data[which_k]

        if h_j_new - h_j_old < entropy_delta:
            """kernel update"""
            nomination_flag = False
            nk[which_k] = nk_j
            mu_ker[which_k] = mu_j
            sigmas[which_k] = np.asarray(sigma_j)
            nk_train_data[which_k] = np.append(nk_j_data_train, index+rolling)
        else:
            prob_k[which_k] = 0  # resets the probability of k-th kernel and renominate

        if np.sum(prob_k) == 0:  # if no kernel matched, initialize a new kernel
            num_k += 1
            nk = np.append(nk, 1)
            mu_ker = np.vstack((mu_ker, s))
            nk_train_data[len(nk)-1] = np.array([index+rolling])

            """generates covariance matrix for the new kernel"""
            """paper reference: DOI 10.1007/s00500-013-1181-9"""
            Igama = np.zeros([2, num_k - 1], dtype=np.dtype(np.ndarray))
            Agama = np.zeros([2, num_k - 1])
            for j in range(0, num_k - 1):
                norm = np.linalg.norm(np.asmatrix(mu_ker[j] - s))
                sqrt = np.lib.scimath.sqrt(np.asmatrix(sigmas[j]))
                Igama[0, j] = sigmas[j] + norm * sqrt
                Igama[1, j] = sigmas[j] - norm * sqrt
                Agama[0, j] = np.linalg.norm(Igama[0, j])
                Agama[1, j] = np.linalg.norm(Igama[1, j])

            gama = np.min(Agama)
            sigmas = np.vstack((sigmas, gama * np.eye(m)[np.newaxis]))
            nomination_flag = False

    return Model(sigmas, mu_ker, nk, num_k, nk_train_data)


def aknn_features(data_clustering, model, data_clustering_normalized, rolling):
    """
       This function aims to extract features of each kernel in AKNN

       Args:
           :param data_clustering: Data to be clustered
           :param model: Model object to be trained
           :param data_clustering_normalized: Data to be normalized
           :param rolling: rolling: Length of historical data to review
    """
    sigmas = model.sigmas
    mu_ker = model.mu_ker
    nk = model.nk
    num_k = model.num_k

    psk = np.zeros(num_k)  # conditional probability of P(s|fi_k), obtained by Gaussian kernel
    pk = np.zeros(num_k)  # ration of the total number of samples that are clustered into each kernel

    """kernel nomination"""
    [n, m] = data_clustering_normalized.shape

    kernel_list = []
    for i in range(0, num_k):
        data_clustering[f"kernel_{i}"] = np.nan
        kernel_list.append(f"kernel_{i}")
    for d_s in range(0, n):
        s = data_clustering_normalized[d_s, :]
        for i in range(0, num_k):
            psk[i] = gaussian.gaussian(s, mu_ker[i], sigmas[i])
            pk[i] = nk[i] / np.sum(nk)
        ps = np.sum(psk * pk) + 1e-9  # conditional probability P(s)
        prob_k = (psk * pk) / ps
        data_clustering.loc[d_s + rolling, kernel_list] = prob_k
    return data_clustering


def aknn_clustering(dataframe, start_time_clustering, end_time_clustering, col_target, rolling, entropy_delta, is_plot):
    '''
    This function aims to cluster data by AKNN algorithm
    :param dataframe: The data to be clustered.
    :param start_time_clustering: The start time of the clustering data.
    :param end_time_clustering: The end time of the clustering data.
    :param col_target: The index of the target column to be predicted.
    :param rolling: Length of historical data to review
    :param entropy_delta: Length of historical data to review
    :param is_plot: whether plot the clustering distribution
    :return: The features extracted from each sample across all kernels.
    '''

    tmp_df = dataframe[(dataframe["datetime"] >= pd.to_datetime(start_time_clustering)) & (
                dataframe["datetime"] < pd.to_datetime(end_time_clustering))]

    tmp_df.dropna(inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    train_end = len(tmp_df)
    df = tmp_df[["hvac", 'air_temp', 'relative_humidity_set_1',
                 'solar_radiation_set_1', 'dew_point_temperature_set_1d',
                 'dayofweek', 'hourofday']]
    df_time = tmp_df[["datetime", "hvac", 'air_temp', 'relative_humidity_set_1',
                      'solar_radiation_set_1', 'dew_point_temperature_set_1d',
                      'dayofweek', 'hourofday']]
    data_clustering = df_time.iloc[:train_end, :]

    cols = list(range(1, df.shape[1]))
    data = df.to_numpy()
    data = np.array(data, dtype='float')
    data_train = data[:train_end]

    cols.append(col_target)
    tmp = data_train[:, cols]

    data_concatenated = aknn.data_processor.get_concatenated_data_set(tmp, rolling)
    data_train_normalized = aknn.normalizer.normalize_all(data_concatenated)
    p = aknn.pattern_initializer.pattern_initialize(data_concatenated)
    m = aknn.model_cluster.model_initialize(data_train_normalized, rolling, entropy_delta)
    if is_plot:
        aknn.model_cluster.plot_clustering_distribution(data_clustering, m, start_time_clustering, end_time_clustering)
    data_clustering = aknn.model_cluster.aknn_features(data_clustering, m, data_train_normalized, rolling)
    return data_clustering

