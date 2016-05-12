"""
    Some tools to work on matrices
"""
import numpy as np


def check_zero_matrix(mat_list):
    """
    Return the list of matrices ids which are non empty
    :param mat_list: List of matrices, usually covariance matrices
    :return: list of ids of non empty matrices
    """
    non_zero_list = []
    for i in range(len(mat_list)):
        if np.count_nonzero(mat_list[i]) is not 0:
            non_zero_list.append(i)
    return non_zero_list
