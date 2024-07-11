from src.aknn import normalizer
from src.aknn.aknn_cluster import model_train


def model_update(data, model, pattern):
    """
    This function is used to re-train the model with the new input data

    Args:
        :param data: New data combined of the input vector and the scalar output
        :param model: Current model
        :param pattern: Current pattern

    Returns:
        :return: Updated model
    """
    s = normalizer.normalize(data, pattern.attributes_max, pattern.attributes_min)
    model_train(s, model)
