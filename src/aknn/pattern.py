import numpy as np
import json
import pickle


class Pattern(object):
    """
    This class maintains vital data of the engine initialization process

    Attributes:
        :param attributes_max: Maximum values in each attributes of input data set
        :param attributes_min: Minimum values in each attributes of input data set
        :param target_max: Maximum target value
        :param target_min: Minimum target value
    """

    def __init__(self, attributes_max: np.ndarray, attributes_min: np.ndarray, target_max: float, target_min: float):
        self.__attributes_max = attributes_max
        self.__attributes_min = attributes_min
        self.__target_max = target_max
        self.__target_min = target_min

    def __to_dic(self):
        """
        Transfer the object to a dictionary

        Returns:
            :return: Transferred dictionary
        """
        return {
            'attributes_max': self.__attributes_max.tolist(),
            'attributes_min': self.__attributes_min.tolist(),
            'target_max': self.__target_max,
            'target_min': self.__target_min
        }

    def to_json(self):
        """
        Serialized the Pattern object to a json file

        Returns:
            :return: Json object
       """
        js = json.dumps(self.__to_dic())
        return js

    def to_json_and_save(self, store_path, store_name):
        """
        Serialized the Pattern object to a json file, and save it to the disk

        Args:
            :param store_path: Path of storing place
            :param store_name: Name of json file

        Returns:
            :return: Json object
        """
        js = json.dumps(self.__to_dic(), indent=4)

        try:
            with open(store_path + store_name + '.json', 'w', encoding='utf-8') as f:
                f.write(js)
        except IOError as e:
            print(e)

        return js

    @staticmethod
    def load_json(load_path, load_name):
        """
        Reads the json file and deserialized it to a Pattern object

        Args:
            :param load_path: Path of the json file
            :param load_name: Name of the json file

        Returns:
            :return: Deserialized Pattern object
        """
        try:
            with open(load_path + load_name + '.json', 'r', encoding='utf-8') as f:
                tmp = json.load(f)

                attributes_max = np.array(tmp['attributes_max'])
                attributes_min = np.array(tmp['attributes_min'])
                target_max = tmp['target_max']
                target_min = tmp['target_min']
        except IOError as e:
            print(e)

        return Pattern(attributes_max, attributes_min, target_max, target_min)

    def to_pickle(self):
        """
        Serialized the Patten object to a pickle object

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
        with open(store_path + store_name, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load_pickle(load_path, load_name):
        """
        Reads the pickle binary file and deserialized it to a Pattern object

        Args:
            :param load_path: Path of the pickle file
            :param load_name: Name of the pickle file

        Returns:
            :return: Deserialized Pattern object
        """
        with open(load_path + load_name, 'rb') as f:
            pic = pickle.load(f)
            f.close()
        return pic

    @property
    def attributes_max(self):
        return self.__attributes_max

    @property
    def attributes_min(self):
        return self.__attributes_min

    @property
    def target_max(self):
        return self.__target_max

    @property
    def target_min(self):
        return self.__target_min

    @attributes_max.setter
    def attributes_max(self, attributes_max):
        self.__attributes_max = attributes_max

    @attributes_min.setter
    def attributes_min(self, attributes_min):
        self.__attributes_min = attributes_min

    @target_max.setter
    def target_max(self, target_max):
        self.__target_max = target_max

    @target_min.setter
    def target_min(self, target_min):
        self.__target_min = target_min
