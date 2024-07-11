import numpy as np
import json
import pickle


class Model:
    """
    This class maintains important parameters for the model

    Attributes:
        :param sigmas: Covariance matrices for each Gaussian kernel
        :param mu_ker: Position vectors(2D arrays) for each Gaussian kernel
        :param nk: Number of clustered samples for each Gaussian kernel
        :param num_k: Total number of kernels
    """

    def __init__(self, sigmas: np.ndarray, mu_ker: np.ndarray, nk: np.ndarray, num_k: int, nk_train_data: dict, rolling: int):
        self.__sigmas = sigmas
        self.__mu_ker = mu_ker
        self.__nk = nk
        self.__num_k = num_k
        self.__nk_train_data = nk_train_data
        self.__rolling = rolling

    def __to_dic(self):
        """
        Transfer the object to a dictionary

        Returns:
            :return: Transferred dictionary
        """
        return {
            'sigmas': self.__sigmas.tolist(),
            'mu_ker': self.__mu_ker.tolist(),
            'nk': self.__nk.tolist(),
            'num_k': self.__num_k,
            'nk_train_data': str(self.__nk_train_data),
            'rolling': self.__rolling
        }

    def to_json(self):
        """
        Serialized the Model object to a json file

        Returns:
            :return: Json object
       """
        js = json.dumps(self.__to_dic(), indent=4)

        return js

    def to_json_and_save(self, store_path, store_name):
        """
        Serialized the Model object to a json file, and save it to the disk

        Args:
            :param store_path: Path of storing place
            :param store_name: Name of json file

        Returns:
            :return: Json object
        """
        js = json.dumps(self.__to_dic(), indent=5)

        try:
            with open(store_path + store_name + '.json', 'w', encoding='utf-8') as f:
                f.write(js)
        except IOError as e:
            print(e)

        return js

    @staticmethod
    def load_json(load_path, load_name):
        """
        Reads the json file and deserialized it to a Model object

        Args:
            :param load_path: Path of the json file
            :param load_name: Name of the json file

        Returns:
            :return: Deserialized Model object
        """
        try:
            with open(load_path + load_name + '.json', 'r', encoding='utf-8') as f:
                tmp = json.load(f)

                sigmas = np.array(tmp['sigmas'])
                mu_ker = np.array(tmp['mu_ker'])
                nk = np.array(tmp['nk'])
                num_k = tmp['num_k']
        except IOError as e:
            print(e)

        return Model(sigmas, mu_ker, nk, num_k)

    def to_pickle(self):
        """
        Serialized the Model object to a pickle object

        Returns:
            :return: Pickle object
        """
        return pickle.dumps(self)

    def to_pickle_and_save(self, store_path, store_name):
        """
        Serialized the Model object to a pickle object, and save it to the disk

        Args:
            :param store_path: Path of file place
            :param store_name: Name of the saving file

        Returns:
            :return: Pickle object
        """
        try:
            with open(store_path + store_name + '.pickle', 'wb') as f:
                pickle.dump(self, f)
                f.close()
        except IOError as e:
            print(e)

    @staticmethod
    def load_pickle(load_path, load_name):
        """
        Reads the pickle binary file and deserialized it to a Model object

        Args:
            :param load_path: Path of the pickle file
            :param load_name: Name of the pickle file

        Returns:
            :return: Deserialized Model object
        """
        try:
            with open(load_path + load_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
                f.close()
        except IOError as e:
            print(e)
        return model

    @property
    def sigmas(self):
        return self.__sigmas

    @property
    def mu_ker(self):
        return self.__mu_ker

    @property
    def nk(self):
        return self.__nk

    @property
    def num_k(self):
        return self.__num_k

    @property
    def nk_train_data(self):
        return self.__nk_train_data

    @property
    def rolling(self):
        return self.__rolling

    @sigmas.setter
    def sigmas(self, sigmas):
        self.__sigmas = sigmas

    @mu_ker.setter
    def mu_ker(self, mu_ker):
        self.__mu_ker = mu_ker

    @nk.setter
    def nk(self, nk):
        self.__nk = nk

    @num_k.setter
    def num_k(self, num_k):
        self.__num_k = num_k

    @nk_train_data.setter
    def nk_train_data(self, nk_train_data):
        self.__nk_train_data = nk_train_data

    @rolling.setter
    def rolling(self, rolling):
        self.__rolling = rolling