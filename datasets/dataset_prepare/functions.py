import numpy as np
def euclidean_dist( test_matrix, train_matrix):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    num_test = test_matrix.shape[0]
    num_train = train_matrix.shape[0]
    dists = np.zeros((num_test, num_train))
    d1 = -2 * np.dot(test_matrix, train_matrix.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(test_matrix), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(train_matrix), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting

    return dists


def generate_cycle_mask( height, width):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    # ellipse mask
    mask = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask.dtype = 'uint8'
    return mask


def average_del_min(data_list):
    if len(data_list) == 0:
        return 0
    if len(data_list) > 2:
        data_list.remove(min(data_list))
        # data_list.remove(max(data_list))
        average_data = float(sum(data_list)) / len(data_list)
        return average_data
    elif len(data_list) <= 2:
        average_data = float(sum(data_list)) / len(data_list)
        return average_data